# agentes/analitico.py

from __future__ import annotations
import builtins
import logging
import re
import time
import traceback
from typing import Any, Dict
import pandas as pd
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel  # noqa: F401
    from memoria import MemoriaSessao  # ajuste o caminho se for diferente

try:
    # LangChain messages (mantido fora de TYPE_CHECKING porque são usados em runtime)
    from langchain_core.messages import SystemMessage, HumanMessage
except Exception as _e:
    raise ImportError(
        "agentes/analitico.py requer langchain-core instalado: `pip install langchain-core`"
    ) from _e


log = logging.getLogger("projeto_fiscal.agentes")

# ------------------------------ Sandbox seguro ------------------------------
class SecurityException(Exception):
    pass

ALLOWED_IMPORTS = {"pandas", "numpy", "matplotlib", "plotly", "traceback"}

def _restricted_import(name: str, *args, **kwargs):
    """Função de import restrita para o sandbox."""
    root_module = name.split(".")[0]
    if root_module not in ALLOWED_IMPORTS:
        raise SecurityException(f"Importação proibida: {name}")
    return builtins.__import__(name, *args, **kwargs)

# SAFE_BUILTINS final e correto, definido uma vez no nível do módulo.
SAFE_BUILTINS = {k: getattr(builtins, k) for k in (
    "abs", "all", "any", "bool", "dict", "enumerate", "float", "int", "isinstance",
    "len", "list", "max", "min", "print", "range", "round", "set", "sorted",
    "str", "sum", "tuple", "type", "zip",
)}
SAFE_BUILTINS["__import__"] = _restricted_import  # Sobrescreve o import padrão


# ------------------------------ Agente Analítico ------------------------------
class AgenteAnalitico:
    """Gera e executa código Python via LLM com auto-correção."""

    def __init__(self, llm: "BaseChatModel", memoria: "MemoriaSessao"):
        self.llm = llm
        self.memoria = memoria
        self.last_code: str = ""  # Armazena o último código tentado

    # ---------- Prompting ----------
    def _prompt_inicial(self, catalog: Dict[str, pd.DataFrame]) -> SystemMessage:
        """Constrói o prompt inicial para a geração de código (Código Completo)."""
        schema_lines = []
        example_table_name = 'documentos' if 'documentos' in catalog else next(iter(catalog.keys()), 'tabela_exemplo')

        for t, df in catalog.items():
            schema_lines.append(f"- Tabela `{t}` ({df.shape[0]} linhas): Colunas: `{', '.join(map(str, df.columns))}`")
        schema = "\n".join(schema_lines) or "- (Nenhum dado carregado)"
        history = self.memoria.resumo()

        prompt = f"""
        Você é um agente de análise de dados de elite expert em Python. Sua tarefa é gerar código Python robusto e bem formatado para uma função 'solve'.

        **REGRAS CRÍTICAS DE EXECUÇÃO:**
        1.  **CRÍTICO:** Todas as declarações de `import` DEVEM estar DENTRO da função `solve`.
        2.  Imports permitidos: {', '.join(ALLOWED_IMPORTS)}. NENHUM OUTRO será permitido pelo sandbox.
        3.  Use APENAS funções built-in seguras. O sandbox bloqueará outras. Funções como `open()`, `eval()`, `exec()` são PROIBIDAS.
        4.  Acesse dados via `catalog['nome_tabela']`. **SEMPRE** use `.copy()` ao pegar um DataFrame do catalog (ex: `df = catalog['documentos'].copy()`).
        5.  Retorne uma tupla: `(texto: str, tabela: pd.DataFrame | None, figura: plt.Figure | go.Figure | None)`.
        6.  Seja DEFENSIVO: Use `pd.to_numeric(df['coluna'], errors='coerce')` para conversões numéricas. Use `.fillna(0)` ou `.dropna()` apropriadamente. Verifique se as colunas existem antes de usá-las (`if 'coluna' in df.columns:`).
        7.  GRÁFICOS: Prefira `plotly.express as px`. Use `fig.update_layout(width=800, height=500)` para ajustar o tamanho. Para `matplotlib.pyplot as plt`, use `fig, ax = plt.subplots(figsize=(10, 6))` e `plt.tight_layout()` antes de retornar `fig`.
        8.  Se retornar uma `tabela` (DataFrame), o `texto` deve ser um resumo ou título, NÃO a tabela convertida para string (`.to_string()`).
        9.  Manipule datas com `pd.to_datetime(df['coluna_data'], errors='coerce')`.

        **ESQUEMA DISPONÍVEL:**
        {schema}

        **HISTÓRICO RECENTE (para contexto):**
        {history}

        **ESTRUTURA OBRIGATÓRIA DA FUNÇÃO:**
        ```python
        def solve(catalog, question):
            # Imports AQUI dentro da função
            import pandas as pd
            import numpy as np
            import matplotlib.pyplot as plt
            import plotly.express as px
            import plotly.graph_objects as go

            # Variáveis de resultado padrão
            text_output = "Análise não pôde ser concluída."
            table_output = None
            figure_output = None

            # Exemplo de acesso seguro aos dados
            if '{example_table_name}' in catalog:
                df = catalog['{example_table_name}'].copy()
            else:
                 return ("Tabela '{example_table_name}' não encontrada no catálogo.", None, None)

            # --- SEU CÓDIGO ROBUSTO DE ANÁLISE VEM AQUI ---
            try:
                # Ex.: df['valor_total'] = pd.to_numeric(df['valor_total'], errors='coerce').fillna(0)
                text_output = "# Título da Análise\\nDescrição dos resultados..."
                # table_output = df_resultado
                # figure_output = fig
            except Exception as e:
                import traceback
                error_details = traceback.format_exc(limit=1)
                text_output = f"Erro durante a análise: {{type(e).__name__}}: {{e}}\\nDetalhe: ...{{error_details.splitlines()[-1]}}"

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

    # ---------- Geração / Correção ----------
    def _gerar_codigo(self, pergunta: str, catalog: Dict[str, pd.DataFrame]) -> str:
        sys = self._prompt_inicial(catalog)
        hum = HumanMessage(content=f"Pergunta do usuário: {pergunta}")
        try:
            resp = self.llm.invoke([sys, hum]).content.strip()
            code_match = re.search(r"```python\n(.*?)\n```", resp, re.DOTALL)
            if code_match:
                code = code_match.group(1).strip()
            else:
                log.warning("LLM não retornou bloco ```python```. Tentando usar a resposta inteira.")
                code = resp.strip()
                if not code.startswith("def solve(catalog, question):"):
                    raise ValueError("Resposta do LLM não parece conter a função 'solve'.")
            self.last_code = code
            return code
        except Exception as e:
            log.error(f"Erro ao invocar LLM para gerar código: {e}")
            raise RuntimeError(f"Falha na comunicação com LLM: {e}") from e

    def _corrigir_codigo(self, failed_code: str, erro: str) -> str:
        sys = self._prompt_correcao(failed_code, erro)
        hum = HumanMessage(content="Por favor, corrija a função `solve` baseada no erro e no código fornecido.")
        try:
            resp = self.llm.invoke([sys, hum]).content.strip()
            code_match = re.search(r"```python\n(.*?)\n```", resp, re.DOTALL)
            if code_match:
                code = code_match.group(1).strip()
            else:
                log.warning("LLM não retornou bloco ```python``` na CORREÇÃO. Usando resposta inteira.")
                code = resp.strip()
                if not code.startswith("def solve(catalog, question):"):
                    raise ValueError("Correção do LLM não parece conter a função 'solve'.")
            self.last_code = code
            return code
        except Exception as e:
            log.error(f"Erro ao invocar LLM para corrigir código: {e}")
            raise RuntimeError(f"Falha na comunicação com LLM durante correção: {e}") from e

    # ---------- Execução ----------
    def _executar_sandbox(self, code: str, pergunta: str, catalog: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        scope = {"__builtins__": SAFE_BUILTINS.copy()}
        scope["__builtins__"]["__import__"] = _restricted_import  # reforça

        try:
            exec(code, scope)
        except SecurityException as se:
            log.error(f"Tentativa de execução insegura bloqueada: {se}")
            raise se
        except Exception as e_comp:
            log.error(f"Erro de compilação/execução do código gerado:\n{code}\nErro: {e_comp}")
            raise SyntaxError(f"Erro ao executar código gerado: {e_comp}") from e_comp

        if "solve" not in scope or not callable(scope["solve"]):
            raise RuntimeError("A função `solve` não foi definida corretamente no código gerado.")

        solve_fn = scope["solve"]
        t0 = time.time()
        try:
            texto, tabela, fig = solve_fn({k: v for k, v in catalog.items()}, pergunta)
        except Exception as e_runtime:
            log.error(f"Erro durante a execução de 'solve':\n{code}\nErro: {e_runtime}")
            tb_str = traceback.format_exc(limit=3)
            raise RuntimeError(f"Erro na execução da lógica de 'solve': {e_runtime}\n{tb_str}") from e_runtime

        dt = time.time() - t0

        # Normalização dos retornos
        if not isinstance(texto, str):
            log.warning(f"'texto' não é string ({type(texto)}). Convertendo para str.")
            texto = str(texto)
        if tabela is not None and not isinstance(tabela, pd.DataFrame):
            log.warning(f"'tabela' não é DataFrame/None ({type(tabela)}). Ignorando tabela.")
            tabela = None
        if fig is not None:
            # valida de forma defensiva sem depender estático de libs
            try:
                import importlib
                matplotlib_figure = None
                try:
                    matplotlib_figure = importlib.import_module("matplotlib.figure")
                except Exception:
                    pass
                go = None
                try:
                    go = importlib.import_module("plotly.graph_objects")
                except Exception:
                    pass

                is_matplotlib_fig = bool(matplotlib_figure) and isinstance(fig, getattr(matplotlib_figure, "Figure"))
                is_plotly_fig = bool(go) and isinstance(fig, getattr(go, "Figure"))
                if not (is_matplotlib_fig or is_plotly_fig):
                    log.warning(f"'figura' não reconhecida: {type(fig)}. Removendo.")
                    fig = None
            except Exception:
                log.warning("Não foi possível validar a figura (libs gráficas ausentes).")
                fig = None

        return {
            "texto": texto,
            "tabela": tabela,
            "figuras": [fig] if fig is not None else [],
            "duracao_s": round(dt, 3),
            "code": code,
        }

    # ---------- Orquestração ----------
    def responder(self, pergunta: str, catalog: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Gera, executa e tenta auto-corrigir o código até 3 tentativas."""
        max_retries = 2
        code_to_run = ""
        try:
            code_to_run = self._gerar_codigo(pergunta, catalog)
            if not code_to_run.strip():
                raise ValueError("LLM não gerou nenhum código.")
            for attempt in range(max_retries + 1):
                try:
                    log.info(f"Tentativa {attempt + 1} de executar código para: '{pergunta[:50]}...'")
                    out = self._executar_sandbox(code_to_run, pergunta, catalog)
                    # registra memória
                    try:
                        self.memoria.salvar(pergunta, out.get("texto", ""), duracao_s=out.get("duracao_s", 0.0))
                    except Exception:
                        log.debug("Falha ao salvar memória (ignorado).")
                    out["agent_name"] = f"AgenteAnalitico (Tentativa {attempt + 1})"
                    out["summary"] = f"Executou código com sucesso para: '{pergunta[:50]}...'"
                    log.info(f"Execução bem-sucedida na tentativa {attempt + 1}.")
                    return out
                except (SyntaxError, SecurityException, RuntimeError, TypeError, ValueError, KeyError, IndexError, AttributeError) as e1:
                    error_message = f"{type(e1).__name__}: {e1}"
                    log.warning(f"Falha na tentativa {attempt + 1}: {error_message}")
                    if attempt < max_retries:
                        log.info(f"Solicitando correção ao LLM (tentativa {attempt + 2}/{max_retries + 1}).")
                        try:
                            code_to_run = self._corrigir_codigo(code_to_run, error_message)
                            if not code_to_run.strip():
                                raise ValueError("LLM não gerou nenhum código de correção.")
                        except Exception as e_corr:
                            log.error(f"Erro ao obter correção do LLM: {e_corr}")
                            raise RuntimeError("Falha ao obter correção do LLM.") from e_corr
                    else:
                        log.error("Número máximo de tentativas excedido. Falha final.")
                        raise
        except Exception as e_final:
            log.error(f"Falha irrecuperável no AgenteAnalitico: {type(e_final).__name__}: {e_final}")
            try:
                self.memoria.salvar(pergunta, f"Erro: {type(e_final).__name__}: {e_final}", duracao_s=0.0)
            except Exception:
                pass
            return {
                "texto": (
                    f"Ocorreu um erro irrecuperável ao tentar analisar sua pergunta após {max_retries + 1} "
                    f"tentativas. Detalhe: {type(e_final).__name__}: {e_final}"
                ),
                "tabela": None,
                "figuras": [],
                "duracao_s": 0.0,
                "code": self.last_code or code_to_run or "",
                "agent_name": "AgenteAnalitico (Falha Irrecuperável)",
                "summary": f"Falha final na geração ou auto-correção para: '{pergunta[:50]}...'",
            }


__all__ = ["AgenteAnalitico", "SecurityException", "ALLOWED_IMPORTS", "SAFE_BUILTINS"]
