# agentes/agente_analitico.py
from __future__ import annotations
import builtins
import io
import logging
import re
import time
import traceback
from contextlib import redirect_stdout
from typing import Any, Dict, TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel
    from memoria import MemoriaSessao

try:
    from langchain_core.messages import SystemMessage, HumanMessage
except Exception as _e:
    raise ImportError(
        "agentes/agente_analitico.py requer langchain-core instalado: `pip install langchain-core`"
    ) from _e


log = logging.getLogger("agente_fiscal.agentes")

# =============================================================================
# Sandbox / Segurança
# =============================================================================
class SecurityException(Exception):
    """Erro de segurança: tentativa de uso de recurso proibido no sandbox."""


# OBS: Estes são módulos que permitimos o *código gerado* importar.
# O próprio agente (este arquivo) pode importar o que precisar,
# mas a função `solve` gerada pela LLM só poderá importar estes:
ALLOWED_IMPORTS = {"pandas", "numpy", "matplotlib", "plotly", "traceback"}

def _restricted_import(name: str, *args, **kwargs):
    """Hook de import para o código do usuário (solve).
    Bloqueia qualquer import fora da whitelist ALLOWED_IMPORTS.
    """
    root_module = (name or "").split(".")[0]
    if root_module not in ALLOWED_IMPORTS:
        raise SecurityException(f"Importação proibida no sandbox: {name}")
    return builtins.__import__(name, *args, **kwargs)


# Conjunto de builtins seguros expostos ao código do usuário
SAFE_BUILTINS = {
    k: getattr(builtins, k)
    for k in (
        "abs", "all", "any", "bool", "dict", "enumerate", "float", "int", "isinstance",
        "len", "list", "max", "min", "print", "range", "round", "set", "sorted",
        "str", "sum", "tuple", "type", "zip",
    )
}
# Substitui o import por nossa função restrita
SAFE_BUILTINS["__import__"] = _restricted_import


# Limites defensivos para as saídas de tabela
MAX_DF_ROWS = 5000
MAX_DF_COLS = 50


def _sanitize_df(df: pd.DataFrame | None) -> pd.DataFrame | None:
    """Garante que a tabela retornada pelo solve não exploda o front.
    - Mantém no máximo MAX_DF_ROWS linhas e MAX_DF_COLS colunas.
    - Converte tipos problemáticos de forma passiva (sem alterar semântica).
    """
    if df is None:
        return None
    try:
        if not isinstance(df, pd.DataFrame):
            return None
        # recortes
        if df.shape[0] > MAX_DF_ROWS:
            df = df.head(MAX_DF_ROWS).copy()
        if df.shape[1] > MAX_DF_COLS:
            df = df.iloc[:, :MAX_DF_COLS].copy()
        # nomes seguros (evita colunas sem nome)
        df.columns = [str(c) if (c is not None and c != "") else f"col_{i}" for i, c in enumerate(df.columns)]
        return df
    except Exception as e:
        log.debug(f"_sanitize_df falhou: {e}")
        return None


def _extract_function_code(raw: str) -> str:
    """Extrai de forma robusta a função `solve(catalog, question)` do texto do LLM.
    Aceita:
      - bloco ```python ... ```
      - resposta sem cercas, contendo apenas a função
    """
    if not raw:
        return ""

    # 1) Preferência: bloco ```python ... ```
    m = re.search(r"```python\s*(.*?)```", raw, flags=re.S | re.I)
    if m:
        snippet = m.group(1).strip()
        # pode vir explicação antes, então capture a partir do def
        mdef = re.search(r"(def\s+solve\s*\(\s*catalog\s*,\s*question\s*\)\s*:.*)", snippet, flags=re.S)
        if mdef:
            return mdef.group(1).strip()
        return snippet

    # 2) Fallback: localizar def solve direto no texto
    m2 = re.search(r"(def\s+solve\s*\(\s*catalog\s*,\s*question\s*\)\s*:.*)", raw, flags=re.S)
    if m2:
        return m2.group(1).strip()

    return ""

# =============================================================================
# Agente Analítico
# =============================================================================
class AgenteAnalitico:
    """Gera e executa código Python via LLM, em sandbox restrito, com auto-correção."""

    def __init__(self, llm: "BaseChatModel", memoria: "MemoriaSessao"):
        self.llm = llm
        self.memoria = memoria
        self.last_code: str = ""     # último código gerado/corrigido
        self.last_error: str = ""    # último erro (para depuração)

    # --------------------------- Prompting ---------------------------------
    def _prompt_inicial(self, catalog: Dict[str, pd.DataFrame]) -> SystemMessage:
        """Prompt-base para a *geração* da função `solve`."""
        schema_lines: list[str] = []
        if catalog:
            for t, df in catalog.items():
                try:
                    schema_lines.append(
                        f"- Tabela `{t}` ({df.shape[0]} linhas): Colunas: `{', '.join(map(str, df.columns))}`"
                    )
                except Exception:
                    schema_lines.append(f"- Tabela `{t}` (meta indisponível)")
        schema = "\n".join(schema_lines) or "- (Nenhum dado carregado)"
        history = ""
        try:
            history = self.memoria.resumo()
        except Exception:
            pass

        prompt = f"""
        Você é um agente de análise de dados de elite, expert em **Python + Pandas**. Gere **APENAS** a função `solve(catalog, question)`,
        com código completo e executável no **sandbox**. Siga à risca:

        REGRAS CRÍTICAS DO SANDBOX
        1) **Todos os imports** DEVEM ficar **dentro** da função `solve`.
        2) Imports permitidos: {', '.join(sorted(ALLOWED_IMPORTS))}. Qualquer outro causará bloqueio.
        3) Não use `open`, `eval`, `exec`, `os`, `sys`, `subprocess`, `pathlib`, `pickle`, `requests`, nem I/O de arquivos.
        4) Acesse dados pelo `catalog['tabela']` e chame **sempre** `.copy()` antes de manipular.
        5) Retorne exatamente: `(texto: str, tabela: pd.DataFrame | None, figura: plt.Figure | go.Figure | None)`.
        6) Seja **defensivo**:
        - Verifique colunas: `if 'col' in df.columns: ...`
        - Converta numéricos: `pd.to_numeric(df['col'], errors='coerce')`
        - Datas: `pd.to_datetime(df['data'], errors='coerce')`
        - Trate `NaN` adequadamente.
        7) Gráficos:
        - Preferir `plotly.express as px` ou `plotly.graph_objects as go`.
        - Opcionalmente `matplotlib.pyplot as plt` (use `fig, ax = plt.subplots(figsize=(10,6))` e `plt.tight_layout()`).
        - Não use seaborn.
        8) O texto deve **resumir o resultado**. Não converta DataFrame para string.
        9) A tabela retornada poderá ser truncada pelo host (máx. {MAX_DF_ROWS} linhas / {MAX_DF_COLS} colunas).
        10) Não altere o estado global, não defina *threads*, *processes* ou *sockets*.

        ESQUEMA DISPONÍVEL
        {schema}

        HISTÓRICO (contexto resumido)
        {history}

        ESTRUTURA OBRIGATÓRIA
        ```python
        def solve(catalog, question):
            # imports DENTRO da função
            import pandas as pd
            import numpy as np
            import matplotlib.pyplot as plt
            import plotly.express as px
            import plotly.graph_objects as go

            text_output = "Análise não pôde ser concluída."
            table_output = None
            figure_output = None

            # Exemplo de acesso seguro
            if len(catalog) == 0:
                return ("Não há tabelas no catálogo.", None, None)

            # Escolha uma tabela relevante com validação:
            first_tbl = next(iter(catalog.keys()))
            df = catalog[first_tbl].copy()

            try:
                # seu código robusto aqui...
                text_output = "# Título\\nResumo curto e objetivo."
                # table_output = df_resultado
                # figure_output = fig
            except Exception as e:
                import traceback
                text_output = f"Erro durante a análise: {{type(e).__name__}}: {{e}}\\n" + traceback.format_exc(limit=1)

            return (text_output, table_output, figure_output)

        ```
        Gere APENAS o código Python completo da função `solve`, nada antes ou depois.
        """
        return SystemMessage(content=prompt.strip())

    def _prompt_correcao(self, failed_code: str, error_message: str) -> SystemMessage:
        """Constrói o prompt para a correção de código (Código Completo)."""
        prompt = f"""
        O código Python gerado anteriormente falhou durante a execução no sandbox seguro. Analise o erro e o código, e reescreva APENAS a função `solve` corrigida.

        **ERRO OCORRIDO:**
        {error_message}

        **CÓDIGO QUE FALHOU:**
        ```python
        {failed_code}
        ```

        **INSTRUÇÕES PARA CORREÇÃO:**
        1.  Coloque todos os `import` DENTRO de `solve`.
        2.  Use apenas: {', '.join(ALLOWED_IMPORTS)}.
        3.  Use `.copy()` nos DataFrames do catálogo.
        4.  Valide colunas e tipos (use `pd.to_numeric`, `pd.to_datetime` com `errors='coerce'`).
        5.  Não use built-ins inseguros.
        6.  Retorne `(texto, tabela, figura)`.
        7.  Trate exceções dentro de `solve`.

        Reescreva APENAS o código Python completo da função `solve` corrigida.
        """
        return SystemMessage(content=prompt.strip())

    # --------------------- Geração / Correção via LLM -----------------------
    def _gerar_codigo(self, pergunta: str, catalog: Dict[str, pd.DataFrame]) -> str:
        sys = self._prompt_inicial(catalog)
        hum = HumanMessage(content=f"Pergunta do usuário: {pergunta}")
        resp_text = ""
        try:
            resp_text = self.llm.invoke([sys, hum]).content or ""
            code = _extract_function_code(resp_text.strip())
            if not code or not code.lstrip().startswith("def solve("):
                raise ValueError("A resposta do LLM não contém a função `solve` válida.")
            self.last_code = code
            return code
        except Exception as e:
            log.error(f"Erro ao invocar LLM para gerar código: {e}")
            raise RuntimeError(f"Falha na geração de código com LLM: {e}\nResposta do LLM:\n{resp_text}") from e

    def _corrigir_codigo(self, failed_code: str, erro: str) -> str:
        sys = self._prompt_correcao(failed_code, erro)
        hum = HumanMessage(content="Corrija a função `solve` conforme instruções.")
        resp_text = ""
        try:
            resp_text = self.llm.invoke([sys, hum]).content or ""
            code = _extract_function_code(resp_text.strip())
            if not code or not code.lstrip().startswith("def solve("):
                raise ValueError("A correção do LLM não contém a função `solve` válida.")
            self.last_code = code
            return code
        except Exception as e:
            log.error(f"Erro ao invocar LLM para corrigir código: {e}")
            raise RuntimeError(f"Falha na correção com LLM: {e}\nResposta do LLM:\n{resp_text}") from e

    # ------------------------------ Execução --------------------------------
    def _executar_sandbox(self, code: str, pergunta: str, catalog: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Compila/executa o código gerado em escopo controlado e retorna as saídas normalizadas."""
        # Escopo isolado com builtins restritos
        scope: Dict[str, Any] = {"__builtins__": SAFE_BUILTINS.copy()}

        # 1) Compilar/registrar a função `solve`
        try:
            exec(code, scope)
        except SecurityException as se:
            log.error(f"Tentativa de execução insegura bloqueada: {se}")
            raise
        except Exception as e_comp:
            log.error(f"Erro de compilação no código gerado:\n{code}\nErro: {e_comp}")
            raise SyntaxError(f"Erro de compilação do código gerado: {e_comp}") from e_comp

        if "solve" not in scope or not callable(scope["solve"]):
            raise RuntimeError("A função `solve` não foi definida corretamente pelo LLM.")

        solve_fn = scope["solve"]

        # 2) Executar `solve` capturando stdout para anexar ao texto
        start_ts = time.time()
        stdout_buf = io.StringIO()
        try:
            with redirect_stdout(stdout_buf):
                texto, tabela, fig = solve_fn({k: v for k, v in catalog.items()}, pergunta)
        except Exception as e_runtime:
            tb_str = traceback.format_exc(limit=3)
            self.last_error = f"{type(e_runtime).__name__}: {e_runtime}\n{tb_str}"
            log.error(f"Erro durante a execução de solve:\n{self.last_error}")
            raise RuntimeError(f"Erro na execução de 'solve': {e_runtime}\n{tb_str}") from e_runtime
        finally:
            duration = round(time.time() - start_ts, 3)

        # 3) Normalizações das saídas
        # 3.1 texto
        if not isinstance(texto, str):
            log.warning(f"'texto' não é str ({type(texto)}). Convertendo para str.")
            texto = str(texto)
        # anexa stdout se houver conteúdo útil
        captured = stdout_buf.getvalue().strip()
        if captured:
            texto = f"{texto.rstrip()}\n\n--- stdout ---\n{captured}"

        # 3.2 tabela
        tabela = _sanitize_df(tabela)

        # 3.3 figura (validação leve para matplotlib/plotly)
        if fig is not None:
            try:
                import importlib
                is_ok = False
                try:
                    mf = importlib.import_module("matplotlib.figure")
                    if isinstance(fig, getattr(mf, "Figure")):
                        is_ok = True
                except Exception:
                    pass
                try:
                    go = importlib.import_module("plotly.graph_objects")
                    if isinstance(fig, getattr(go, "Figure")):
                        is_ok = True
                except Exception:
                    pass
                if not is_ok:
                    log.warning(f"'fig' não reconhecida ({type(fig)}). Será descartada.")
                    fig = None
            except Exception:
                fig = None

        return {
            "texto": texto,
            "tabela": tabela,
            "figuras": [fig] if fig is not None else [],
            "duracao_s": duration,
            "code": code,
        }

    # ------------------------------ Orquestração -----------------------------
    def responder(self, pergunta: str, catalog: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Gera, executa e tenta auto-corrigir o código até 3 tentativas (1 geração + 2 correções)."""
        max_retries = 2
        code_to_run = ""
        try:
            # 1) Geração inicial
            code_to_run = self._gerar_codigo(pergunta, catalog)
            if not code_to_run.strip():
                raise ValueError("LLM não gerou nenhum código.")

            # 2) Tentativas de execução + correção
            for attempt in range(max_retries + 1):
                try:
                    log.info(f"[AgenteAnalitico] Execução tentativa {attempt + 1} para: '{pergunta[:60]}...'")
                    out = self._executar_sandbox(code_to_run, pergunta, catalog)

                    # memória (best-effort)
                    try:
                        self.memoria.salvar(pergunta, out.get("texto", ""), duracao_s=out.get("duracao_s", 0.0))
                    except Exception:
                        log.debug("Falha ao salvar memória (ignorado).")

                    out["agent_name"] = f"AgenteAnalitico (Tentativa {attempt + 1})"
                    out["summary"] = f"Código executado com sucesso."
                    return out

                except (SecurityException, SyntaxError, RuntimeError, TypeError, ValueError, KeyError, IndexError, AttributeError) as e1:
                    err_msg = f"{type(e1).__name__}: {e1}"
                    self.last_error = err_msg
                    log.warning(f"[AgenteAnalitico] Falha na tentativa {attempt + 1}: {err_msg}")

                    if attempt < max_retries:
                        # solicitar correção
                        try:
                            code_to_run = self._corrigir_codigo(code_to_run, err_msg)
                            if not code_to_run.strip():
                                raise ValueError("LLM não gerou correção de código.")
                        except Exception as e_corr:
                            log.error(f"Falha ao obter correção do LLM: {e_corr}")
                            raise RuntimeError("Falha ao obter correção do LLM.") from e_corr
                    else:
                        # estourou o limite de tentativas
                        raise

        except Exception as e_final:
            log.error(f"[AgenteAnalitico] Falha irrecuperável: {type(e_final).__name__}: {e_final}")
            try:
                self.memoria.salvar(pergunta, f"Erro: {type(e_final).__name__}: {e_final}", duracao_s=0.0)
            except Exception:
                pass

            return {
                "texto": (
                    f"Ocorreu um erro irrecuperável ao tentar analisar sua pergunta: "
                    f"{type(e_final).__name__}: {e_final}"
                ),
                "tabela": None,
                "figuras": [],
                "duracao_s": 0.0,
                "code": self.last_code or code_to_run or "",
                "agent_name": "AgenteAnalitico (Falha Irrecuperável)",
                "summary": f"Falha final na geração/correção para: '{pergunta[:60]}...'",
            }

__all__ = ["AgenteAnalitico", "SecurityException", "ALLOWED_IMPORTS", "SAFE_BUILTINS"]
