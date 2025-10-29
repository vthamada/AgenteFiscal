# agentes/agente_analitico.py

from __future__ import annotations
import builtins
import hashlib
import io
import json
import logging
import re
import time
import traceback
from contextlib import redirect_stdout
from typing import Any, Dict, TYPE_CHECKING, Tuple
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


# Módulos que o *código gerado* pode importar (dentro de solve)
ALLOWED_IMPORTS = {"pandas", "numpy", "matplotlib", "plotly", "traceback"}

def _restricted_import(name: str, *args, **kwargs):
    """Hook de import para o código do usuário (solve)."""
    root_module = (name or "").split(".")[0]
    if root_module not in ALLOWED_IMPORTS:
        raise SecurityException(f"Importação proibida no sandbox: {name}")
    return builtins.__import__(name, *args, **kwargs)


SAFE_BUILTINS = {
    k: getattr(builtins, k)
    for k in (
        "abs", "all", "any", "bool", "dict", "enumerate", "float", "int", "isinstance",
        "len", "list", "max", "min", "print", "range", "round", "set", "sorted",
        "str", "sum", "tuple", "type", "zip",
    )
}
SAFE_BUILTINS["__import__"] = _restricted_import

MAX_DF_ROWS = 5000
MAX_DF_COLS = 50


def _sanitize_df(df: pd.DataFrame | None) -> pd.DataFrame | None:
    if df is None:
        return None
    try:
        if not isinstance(df, pd.DataFrame):
            return None
        if df.shape[0] > MAX_DF_ROWS:
            df = df.head(MAX_DF_ROWS).copy()
        if df.shape[1] > MAX_DF_COLS:
            df = df.iloc[:, :MAX_DF_COLS].copy()
        df.columns = [str(c) if (c is not None and c != "") else f"col_{i}" for i, c in enumerate(df.columns)]
        return df
    except Exception as e:
        log.debug(f"_sanitize_df falhou: {e}")
        return None


def _extract_function_code(raw: str) -> str:
    """Extrai solve(catalog, question) do texto do LLM."""
    if not raw:
        return ""
    m = re.search(r"```python\s*(.*?)```", raw, flags=re.S | re.I)
    if m:
        snippet = m.group(1).strip()
        mdef = re.search(r"(def\s+solve\s*\(\s*catalog\s*,\s*question\s*\)\s*:.*)", snippet, flags=re.S)
        if mdef:
            return mdef.group(1).strip()
        return snippet
    m2 = re.search(r"(def\s+solve\s*\(\s*catalog\s*,\s*question\s*\)\s*:.*)", raw, flags=re.S)
    if m2:
        return m2.group(1).strip()
    return ""

def _norm_question(q: str) -> str:
    return re.sub(r"\s+", " ", (q or "").strip().lower())

def _schema_fingerprint(catalog: Dict[str, pd.DataFrame]) -> str:
    parts = []
    for tname in sorted(catalog.keys()):
        try:
            cols = ",".join(map(str, catalog[tname].columns))
            parts.append(f"{tname}:{cols}")
        except Exception:
            parts.append(f"{tname}:<indisponivel>")
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]
    return digest

# =============================================================================
# Agente Analítico
# =============================================================================
class AgenteAnalitico:
    """
    Gera e executa código Python via LLM, em sandbox restrito, com auto-correção.
    Agora com:
      - roteamento cognitivo de intenção (LLM) para guiar a geração do solve;
      - InsightCard opcional (quarto retorno) -> __meta__.insight_card;
      - cache leve por (intencao, pergunta, schema_fp).
    """

    # Intenções suportadas (labels que o LLM deve usar)
    INTENTS = ("anomalias_impostos", "variacao_fornecedor", "preco_item_ncm", "tendencia_mensal", "livre")

    def __init__(self, llm: "BaseChatModel", memoria: "MemoriaSessao"):
        self.llm = llm
        self.memoria = memoria
        self.last_code: str = ""
        self.last_error: str = ""

        # Cache simples: {(intent, norm_q, schema_fp): output_dict}
        self._cache: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        self._cache_max = 32

    # --------------------------- Intenção (LLM) ------------------------------
    def _classificar_intencao(self, pergunta: str, catalog: Dict[str, pd.DataFrame]) -> str:
        """Usa LLM para classificar a pergunta numa das INTENTS (sem fazer cálculo)."""
        schema_lines: list[str] = []
        for t, df in catalog.items():
            try:
                schema_lines.append(f"- {t}: {', '.join(map(str, df.columns))}")
            except Exception:
                schema_lines.append(f"- {t}: <indisponivel>")
        schema = "\n".join(schema_lines) or "(vazio)"

        sys = SystemMessage(content=(
            "Você é um roteador de intenção para análise fiscal. "
            f"Escolha APENAS UMA label dentre {self.INTENTS}. "
            "Não explique. Responda somente a label.\n\n"
            "Regras:\n"
            "- Se a pergunta falar de 'anomalia', 'inconsistência', 'fora da média' de impostos (ICMS/IPI/PIS/COFINS) => anomalias_impostos\n"
            "- Se falar de variação por fornecedor/fornecedores, sequência, janela, comparações => variacao_fornecedor\n"
            "- Se falar de preço por item/NCM ou distribuição de preço => preco_item_ncm\n"
            "- Se falar de séries no tempo, mensal, tendência => tendencia_mensal\n"
            "- Caso contrário => livre\n"
        ))
        hum = HumanMessage(content=f"PERGUNTA: {pergunta}\n\nESQUEMA:\n{schema}\n\nResponda somente a label.")
        try:
            label = (self.llm.invoke([sys, hum]).content or "").strip().lower()
            label = re.sub(r"[^a-z_]", "", label)
            if label in self.INTENTS:
                return label
        except Exception as e:
            log.debug(f"Classificação de intenção falhou (fallback 'livre'): {e}")
        return "livre"

    # --------------------------- Prompting ---------------------------------
    def _prompt_inicial(self, catalog: Dict[str, pd.DataFrame], intent: str) -> SystemMessage:
        """Prompt-base para gerar `solve`. Inclui contrato de InsightCard opcional."""
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

        # Dicas de método por intenção (apenas *guidance*, não preset)
        intent_guidance = {
            "anomalias_impostos": "Prefira métodos estatísticos como z-score, IQR, quantis ou EWMA. Justifique no texto final.",
            "variacao_fornecedor": "Use janelas móveis (rolling) e comparações vs média histórica por fornecedor.",
            "preco_item_ncm": "Compare distribuição de preço unitário por NCM/descrição (boxplot/quantis).",
            "tendencia_mensal": "Agregue por mês (resample/groupby) e calcule evolução (MoM/YoY) quando possível.",
            "livre": "Escolha método apropriado e explique sucintamente."
        }.get(intent, "Escolha método apropriado e explique sucintamente.")

        prompt = f"""
        Você é um agente de análise fiscal de elite que GERA e EXECUTA **APENAS** a função `solve(catalog, question)` em **sandbox**.
        Intenção: **{intent}**. {intent_guidance}

        REGRAS CRÍTICAS DO SANDBOX
        1) **Todos os imports** DENTRO de `solve`.
        2) Imports permitidos: {', '.join(sorted(ALLOWED_IMPORTS))}.
        3) Sem I/O, rede, arquivos, eval/exec, subprocess, etc.
        4) Acesse dados via `catalog['tabela']` e sempre `.copy()` antes de manipular.
        5) Retorne exatamente:
           - `(texto: str, tabela: pd.DataFrame|None, figura: plt.Figure|go.Figure|None)` **ou**
           - `(texto, tabela, figura, insight_card: dict)`  ← **opcional** (ver abaixo).
        6) Seja **defensivo** (validar colunas, `pd.to_numeric(..., errors='coerce')`, `pd.to_datetime(..., errors='coerce')`, tratar NaN).
        7) Gráficos: usar plotly (px/go) ou matplotlib (sem seaborn). `plt.tight_layout()` se usar matplotlib.
        8) O texto deve ser **curto, objetivo e explicativo**, citando o **método estatístico** e os **parâmetros** usados.
        9) Se gerar `insight_card` (recomendado), use este **schema**:
           ```python
           insight_card = {{
               "insight_type": "{intent}",
               "rationale": "explica a evidência (ex.: z>3 por 3 notas seguidas)",
               "metrics": {{"n": 0, "z_atual": 0.0}},   # chaves livres
               "trust_score": 0.0,                      # 0..1
               "data_used": {{"rows": 0, "period": ""}}
           }}
           ```
           Inclua **somente chaves simples** (serializáveis). O host pode truncar a tabela; por isso, sumarize na narrativa.

        ESQUEMA DISPONÍVEL
        {schema}

        HISTÓRICO (contexto resumido)
        {history}

        ESTRUTURA OBRIGATÓRIA (exemplo-base)
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
            insight_card = None  # opcional

            if len(catalog) == 0:
                return ("Não há tabelas no catálogo.", None, None)

            # Selecione tabelas relevantes de forma segura:
            first_tbl = next(iter(catalog.keys()))
            df = catalog[first_tbl].copy()

            try:
                # ... sua análise robusta aqui (em função da intenção) ...
                text_output = "# Título\\nResumo curto, cite o método e o porquê."
                # table_output = df_resultado
                # figure_output = fig
                # insight_card = {{ "insight_type": "{intent}", "rationale": "…", "metrics": {{}}, "trust_score": 0.8, "data_used": {{"rows": int(df.shape[0])}} }}
            except Exception as e:
                import traceback
                text_output = f"Erro durante a análise: {{type(e).__name__}}: {{e}}\\n" + traceback.format_exc(limit=1)

            # Retorne 3 ou 4 elementos (se tiver insight_card)
            if insight_card is not None:
                return (text_output, table_output, figure_output, insight_card)
            return (text_output, table_output, figure_output)
        ```
        Gere APENAS o código Python completo da função `solve`, nada antes ou depois.
        """
        return SystemMessage(content=prompt.strip())

    def _prompt_correcao(self, failed_code: str, error_message: str, intent: str) -> SystemMessage:
        prompt = f"""
        O código Python gerado anteriormente falhou na execução (intenção: {intent}). Analise o erro e reescreva APENAS `solve`.

        **ERRO:**
        {error_message}

        **CÓDIGO QUE FALHOU:**
        ```python
        {failed_code}
        ```

        **REGRAS:**
        - Imports somente dentro de `solve` e apenas: {', '.join(ALLOWED_IMPORTS)}.
        - Use `.copy()` nos DataFrames.
        - Valide colunas/tipos; trate NaN; converta numéricos e datas com `errors='coerce'`.
        - Sem I/O/rede/arquivos.
        - Retorne `(texto, tabela, figura)` ou `(texto, tabela, figura, insight_card: dict)`.
        - O texto deve citar o método estatístico e parâmetros.

        Reescreva APENAS o código Python completo da função `solve` corrigida.
        """
        return SystemMessage(content=prompt.strip())

    # --------------------- Geração / Correção via LLM -----------------------
    def _gerar_codigo(self, pergunta: str, catalog: Dict[str, pd.DataFrame], intent: str) -> str:
        sys = self._prompt_inicial(catalog, intent)
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

    def _corrigir_codigo(self, failed_code: str, erro: str, intent: str) -> str:
        sys = self._prompt_correcao(failed_code, erro, intent)
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
        raw_ret = None
        try:
            with redirect_stdout(stdout_buf):
                raw_ret = solve_fn({k: v for k, v in catalog.items()}, pergunta)
        except Exception as e_runtime:
            tb_str = traceback.format_exc(limit=3)
            self.last_error = f"{type(e_runtime).__name__}: {e_runtime}\n{tb_str}"
            log.error(f"Erro durante a execução de solve:\n{self.last_error}")
            raise RuntimeError(f"Erro na execução de 'solve': {e_runtime}\n{tb_str}") from e_runtime
        finally:
            duration = round(time.time() - start_ts, 3)

        # 3) Normalizações das saídas (aceita 3 ou 4 elementos)
        texto: str = ""
        tabela: pd.DataFrame | None = None
        fig: Any | None = None
        insight_card: Dict[str, Any] | None = None

        if isinstance(raw_ret, (list, tuple)) and (3 <= len(raw_ret) <= 4):
            texto = raw_ret[0]
            tabela = raw_ret[1]
            fig = raw_ret[2]
            if len(raw_ret) == 4 and isinstance(raw_ret[3], dict):
                # Sanitiza insight_card (somente tipos simples)
                try:
                    json.dumps(raw_ret[3])
                    insight_card = raw_ret[3]
                except Exception:
                    insight_card = None
        else:
            raise RuntimeError("A função `solve` não retornou um tuple/list com 3 ou 4 elementos.")

        # 3.1 texto
        if not isinstance(texto, str):
            log.warning(f"'texto' não é str ({type(texto)}). Convertendo para str.")
            texto = str(texto)

        captured = stdout_buf.getvalue().strip()
        if captured:
            texto = f"{texto.rstrip()}\n\n--- stdout ---\n{captured}"

        # 3.2 tabela
        tabela = _sanitize_df(tabela)

        # 3.3 figura (validação leve)
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

        out: Dict[str, Any] = {
            "texto": texto,
            "tabela": tabela,
            "figuras": [fig] if fig is not None else [],
            "duracao_s": duration,
            "code": self.last_code or "",
        }
        if insight_card is not None:
            out["__meta__"] = {"insight_card": insight_card}
        return out

    # ------------------------------ Orquestração -----------------------------
    def responder(self, pergunta: str, catalog: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Gera, executa e tenta auto-corrigir o código até 3 tentativas (1 geração + 2 correções).
        Agora com:
          - roteamento cognitivo de intenção (LLM);
          - cache leve (reduz custo/latência em perguntas repetidas).
        """
        # 0) Intenção (LLM-only)
        intent = self._classificar_intencao(pergunta, catalog)
        schema_fp = _schema_fingerprint(catalog)
        cache_key = (intent, _norm_question(pergunta), schema_fp)

        # 0.1) Cache
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            # Anexa nota de cache (sem alterar contrato)
            cached = dict(cached)
            cached.setdefault("__meta__", {})
            cached["__meta__"]["cache_hit"] = True
            cached["agent_name"] = f"AgenteAnalitico (cache,{intent})"
            return cached

        max_retries = 2
        code_to_run = ""
        try:
            # 1) Geração inicial orientada por intenção
            code_to_run = self._gerar_codigo(pergunta, catalog, intent)
            if not code_to_run.strip():
                raise ValueError("LLM não gerou nenhum código.")

            # 2) Tentativas de execução + correção
            for attempt in range(max_retries + 1):
                try:
                    log.info(f"[AgenteAnalitico] ({intent}) Execução tentativa {attempt + 1} para: '{pergunta[:60]}...'")
                    out = self._executar_sandbox(code_to_run, pergunta, catalog)

                    # memória (best-effort)
                    try:
                        self.memoria.salvar(pergunta, out.get("texto", ""), duracao_s=out.get("duracao_s", 0.0))
                    except Exception:
                        log.debug("Falha ao salvar memória (ignorado).")

                    out["agent_name"] = f"AgenteAnalitico (Tentativa {attempt + 1}, {intent})"
                    out["summary"] = "Código executado com sucesso."

                    # Atualiza cache (LRU simplificada)
                    if len(self._cache) >= self._cache_max:
                        # remove um item arbitrário (FIFO simples)
                        try:
                            self._cache.pop(next(iter(self._cache.keys())))
                        except Exception:
                            self._cache.clear()
                    out_to_cache = dict(out)
                    self._cache[cache_key] = out_to_cache

                    return out

                except (SecurityException, SyntaxError, RuntimeError, TypeError, ValueError, KeyError, IndexError, AttributeError) as e1:
                    err_msg = f"{type(e1).__name__}: {e1}"
                    self.last_error = err_msg
                    log.warning(f"[AgenteAnalitico] ({intent}) Falha na tentativa {attempt + 1}: {err_msg}")

                    if attempt < max_retries:
                        # solicitar correção
                        try:
                            code_to_run = self._corrigir_codigo(code_to_run, err_msg, intent)
                            if not code_to_run.strip():
                                raise ValueError("LLM não gerou correção de código.")
                        except Exception as e_corr:
                            log.error(f"Falha ao obter correção do LLM: {e_corr}")
                            raise RuntimeError("Falha ao obter correção do LLM.") from e_corr
                    else:
                        # estourou o limite de tentativas
                        raise

        except Exception as e_final:
            log.error(f"[AgenteAnalitico] ({intent}) Falha irrecuperável: {type(e_final).__name__}: {e_final}")
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
                "agent_name": f"AgenteAnalitico (Falha Irrecuperável, {intent})",
                "summary": f"Falha final na geração/correção para: '{pergunta[:60]}...'",
            }

__all__ = ["AgenteAnalitico", "SecurityException", "ALLOWED_IMPORTS", "SAFE_BUILTINS"]
