# orchestrator.py
from __future__ import annotations

import json
import time
import logging
import re
import os
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable, Tuple

import pandas as pd

log = logging.getLogger(__name__)

# ------------------------------------------------------------
# Imports principais (centralizados via pacote `agentes`)
# ------------------------------------------------------------
try:
    from validacao import ValidadorFiscal
    from agentes import (
        AgenteXMLParser,
        AgenteOCR,
        AgenteNLP,
        AgenteAnalitico,
        AgenteNormalizadorCampos,
        AgenteAssociadorXML,
        MetricsAgent,
    )
    from memoria import MemoriaSessao
    from banco_de_dados import BancoDeDados
except Exception as e:
    raise ImportError(f"Falha ao importar módulos núcleo: {e}") from e

# ------------------------------------------------------------
# LangChain (tools/agents) - opcional
# ------------------------------------------------------------
_LC_AVAILABLE = False
try:
    from langchain_core.language_models.chat_models import BaseChatModel  # type: ignore
    try:
        from langchain_core.tools import Tool  # type: ignore
    except Exception:  # pragma: no cover
        from langchain.tools import Tool  # type: ignore
    try:
        from langchain.agents import initialize_agent, AgentType  # type: ignore
    except Exception:
        initialize_agent = None  # type: ignore
        AgentType = None  # type: ignore
    _LC_AVAILABLE = True
except Exception:
    BaseChatModel = object  # type: ignore

# ------------------------------------------------------------
# LangGraph - orquestração cognitiva (novo)
# ------------------------------------------------------------
_LG_AVAILABLE = False
try:
    # langgraph>=0.2.x
    from langgraph.graph import StateGraph, END  # type: ignore
    _LG_AVAILABLE = True
except Exception:
    # Executa em modo determinístico se não houver LangGraph
    pass


# ============================================================
# Orchestrator
# ============================================================
class Orchestrator:
    """
    Coordena ingestão e processamento fiscal (XML/OCR→NLP→normalização→associação→validação),
    com grafo cognitivo (LangGraph) se disponível. Fallback determinístico seguro se não.
    Agora com:
      - Blackboard compartilhado
      - Decisões adaptativas baseadas em confiança/cobertura/sanidade
      - Logs explicativos e ciclo de feedback com memória/métricas
    """

    db: "BancoDeDados"
    validador: "ValidadorFiscal"
    memoria: "MemoriaSessao"
    llm: Optional["BaseChatModel"] = None
    metrics_agent: Optional["MetricsAgent"] = None

    # ---------- Configs centrais (ENV override) ----------
    COVERAGE_THRESHOLD: float = float(os.getenv("NLP_COVERAGE_MIN", "0.6"))
    OCR_CONF_STRONG: float = float(os.getenv("OCR_CONF_STRONG", "0.90"))
    OCR_CONF_MEDIUM: float = float(os.getenv("OCR_CONF_MEDIUM", "0.70"))
    NORMALIZER_SANITY_MIN: float = float(os.getenv("NORMALIZER_SANITY_MIN", "0.80"))

    # Campos críticos p/ considerar “útil”
    CRITICAL_FIELDS: Tuple[str, ...] = ("emitente_cnpj", "valor_total", "data_emissao")

    def __init__(
        self,
        db: "BancoDeDados",
        validador: Optional["ValidadorFiscal"] = None,
        memoria: Optional["MemoriaSessao"] = None,
        llm: Optional["BaseChatModel"] = None,
    ):
        self.db = db
        self.validador = validador or ValidadorFiscal()
        # FIX: MemoriaSessao requer db
        self.memoria = memoria or MemoriaSessao(self.db)
        self.llm = llm
        self.metrics_agent = MetricsAgent(llm=self.llm)

        # Agentes (núcleo)
        self.xml_agent = AgenteXMLParser(self.db, self.validador, self.metrics_agent)
        self.ocr_agent = AgenteOCR(llm=self.llm)
        self.nlp_agent = AgenteNLP(llm=self.llm)
        # FIX: não passar keep_context_copy= no call (bug histórico)
        self.normalizador = AgenteNormalizadorCampos(
            llm=self.llm,
            enable_llm=True if self.llm else False,
            drop_general_fields=True
        )
        self.associador = AgenteAssociadorXML(self.db)
        self.analitico = AgenteAnalitico(self.llm, self.memoria) if (self.llm and _LC_AVAILABLE) else None

        # Blackboard compartilhado (memória de execução do documento)
        self.blackboard: Dict[str, Dict[str, Any]] = {
            "ocr": {},
            "nlp": {},
            "normalizer": {},
            "validator": {},
            "associador": {},
            "decisions": {"log": []},
        }

        # Tools/Planner (LangChain) - opcional
        self._lc_tools: List[Any] = []
        self._lc_agent = None
        self._init_langchain_layer()

        # Grafo (LangGraph) - opcional
        self._graph = None
        if _LG_AVAILABLE:
            try:
                self._graph = self._build_langgraph()
                log.info("Orchestrator: LangGraph ATIVADO.")
            except Exception as e:
                log.warning(f"Orchestrator: Falha ao construir LangGraph, usando fallback determinístico. Erro: {e}")
                self._graph = None
        else:
            log.info("Orchestrator: LangGraph indisponível. Usando pipeline determinístico.")

        if self.llm and _LC_AVAILABLE:
            log.info("Orchestrator: camada de inteligência (LangChain) ATIVADA.")
        elif self.llm and not _LC_AVAILABLE:
            log.warning("Orchestrator: LLM fornecido, mas LangChain não está disponível. Usando pipeline sem planner.")
        else:
            log.info("Orchestrator: executando sem LLM (modo determinístico/robusto).")

    # --------------------------------------------------------
    # Lista branca de colunas válidas (fallback)
    # --------------------------------------------------------
    _DOC_COLS_BASE = {
        "id",
        "status",
        "motivo_rejeicao",
        "meta_json",

        # Identificação básica
        "tipo",
        "chave_acesso",
        "modelo",
        "serie",
        "numero_nota",
        "natureza_operacao",

        # Datas/Horas
        "data_emissao",
        "data_saida",
        "hora_emissao",
        "hora_saida",

        # Emitente
        "emitente_cnpj",
        "emitente_cpf",
        "emitente_nome",
        "emitente_ie",
        "emitente_im",
        "emitente_endereco",
        "emitente_municipio",
        "emitente_uf",

        # Destinatário
        "destinatario_cnpj",
        "destinatario_cpf",
        "destinatario_nome",
        "destinatario_ie",
        "destinatario_im",
        "destinatario_endereco",
        "destinatario_municipio",
        "destinatario_uf",

        # Totais (cabeçalho)
        "valor_total",
        "total_produtos",
        "total_servicos",
        "total_icms",
        "total_ipi",
        "total_pis",
        "total_cofins",
        "valor_iss",
        "valor_descontos",
        "valor_outros",
        "valor_frete",
        "valor_seguro",
        "valor_liquido",

        # Pagamento (novo)
        "condicao_pagamento",
        "meio_pagamento",
        "bandeira_cartao",
        "valor_troco",

        # Transporte / volumes (opcional)
        "modalidade_frete",
        "placa_veiculo",
        "uf_veiculo",
        "peso_bruto",
        "peso_liquido",
        "qtd_volumes",

        # Arquivo & identificação
        "caminho_arquivo",
        "nome_arquivo",
        "origem",
        "ambiente",

        # XML e autorização (novo)
        "caminho_xml",
        "versao_schema",
        "protocolo_autorizacao",
        "data_autorizacao",
        "cstat",
        "xmotivo",
        "responsavel_tecnico",

        # Rótulos úteis (mantenha apenas se usa no cabeçalho)
        "cfop",
        "ncm",
        "cst",
        "cnpj_autorizado",
        "observacoes",
    }

    # --------------------------------------------------------
    # Utils internos
    # --------------------------------------------------------
    def _campos_permitidos_documentos(self) -> set:
        """Tenta descobrir as colunas reais via PRAGMA; senão usa fallback fixo."""
        try:
            cur = self.db.conn.execute("PRAGMA table_info(documentos)")
            cols = {row[1] for row in cur.fetchall()}
            if cols:
                return cols
        except Exception:
            pass
        return set(self._DOC_COLS_BASE)

    @staticmethod
    def _filtrar_campos_validos(d: Dict[str, Any], permitidas: set) -> Dict[str, Any]:
        """Remove chaves internas (__meta__ etc.) e ignora campos fora do schema."""
        if not d:
            return {}
        limpo: Dict[str, Any] = {}
        for k, v in d.items():
            if not k:
                continue
            kstr = str(k)
            if kstr.startswith("__"):
                continue
            if kstr in permitidas:
                limpo[kstr] = v
        return limpo

    @staticmethod
    def _retry(fn: Callable[[], Tuple[Any, ...]], attempts: int = 2, backoff_s: float = 0.6) -> Tuple[Any, ...]:
        """Retry com backoff linear simples. Retorna a última exceção se todas falharem."""
        last_exc: Optional[Exception] = None
        for i in range(attempts):
            try:
                return fn()
            except Exception as e:
                last_exc = e
                if i < attempts - 1:
                    time.sleep(backoff_s * (i + 1))
        if last_exc:
            raise last_exc
        raise RuntimeError("Falha inesperada no retry helper.")

    @staticmethod
    def _safe_json_loads(text: Optional[str]) -> Dict[str, Any]:
        if not text:
            return {}
        try:
            data = json.loads(text)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _merge_meta_json(self, doc_id: int, patch: Dict[str, Any]) -> None:
        """Funde dicionário em meta_json do documento, com tolerância a falhas."""
        try:
            atual = self.db.get_documento(doc_id) or {}
            base_meta = self._safe_json_loads(atual.get("meta_json"))

            def _deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
                out = dict(a or {})
                for k, v in (b or {}).items():
                    if isinstance(v, dict) and isinstance(out.get(k), dict):
                        out[k] = _deep_merge(out[k], v)
                    else:
                        out[k] = v
                return out

            novo = _deep_merge(base_meta, patch or {})
            self.db.atualizar_documento_campos(doc_id, meta_json=json.dumps(novo, ensure_ascii=False))
        except Exception as e:
            log.debug(f"Falha ao atualizar meta_json doc_id={doc_id}: {e}")

    def _persistir_detalhes_contexto(self, doc_id: int, contexto: Dict[str, Any]) -> None:
        """
        Grava pares chave/valor em documentos_detalhes quando disponível.
        Aceita níveis (ex.: context['fallbacks']).
        """
        try:
            cur = self.db.conn.cursor()
            cur.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='documentos_detalhes' LIMIT 1")
            if not cur.fetchone():
                return

            def _flat(prefix: str, obj: Any):
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        yield from _flat(f"{prefix}.{k}" if prefix else str(k), v)
                else:
                    yield prefix, obj

            for chave, valor in _flat("", contexto):
                try:
                    self.db.conn.execute(
                        "INSERT INTO documentos_detalhes (documento_id, chave, valor) VALUES (?, ?, ?)",
                        (
                            doc_id,
                            chave,
                            json.dumps(valor, ensure_ascii=False)
                            if isinstance(valor, (dict, list))
                            else (str(valor) if valor is not None else None),
                        ),
                    )
                except Exception:
                    pass
            self.db.conn.commit()
        except Exception as e:
            log.debug("Falha ao persistir documentos_detalhes doc_id=%d: %s", doc_id, e)

    # --------------------------------------------------------
    # Blackboard helpers
    # --------------------------------------------------------
    def _bb_write(self, section: str, data: Dict[str, Any]) -> None:
        self.blackboard.setdefault(section, {})
        # deep merge simples
        for k, v in (data or {}).items():
            if isinstance(v, dict) and isinstance(self.blackboard[section].get(k), dict):
                self.blackboard[section][k].update(v)
            else:
                self.blackboard[section][k] = v

    def _bb_decision(self, message: str, **extra) -> None:
        entry = {"ts": time.time(), "msg": message}
        if extra:
            entry.update(extra)
        self.blackboard["decisions"]["log"].append(entry)
        log.info("[CognitiveDecision] %s", message)

    # --------------------------------------------------------
    # Heurísticas / checagens
    # --------------------------------------------------------
    def _faltando_campos_criticos(self, d: Dict[str, Any]) -> bool:
        if not d:
            return True
        for k in self.CRITICAL_FIELDS:
            v = d.get(k)
            if v in (None, "", "None"):
                return True
        return False

    @staticmethod
    def _coverage(meta_nlp: Dict[str, Any]) -> float:
        try:
            return float(meta_nlp.get("coverage", 0.0))
        except Exception:
            return 0.0

    def _precisa_llm(self, campos: Dict[str, Any], meta_nlp: Dict[str, Any]) -> bool:
        return self._faltando_campos_criticos(campos) or (self._coverage(meta_nlp) < self.COVERAGE_THRESHOLD)

    def _decidir_rota(self, conf_ocr: float, campos: Dict[str, Any], xml_encontrado: bool) -> Dict[str, str]:
        """Regra de decisão tolerante para status final."""
        ok_cnpj = bool(campos.get("emitente_cnpj"))
        ok_valor = campos.get("valor_total") is not None
        ok_data = bool(campos.get("data_emissao"))

        # Novo: identificação forte alternativa
        ok_chave = bool(campos.get("chave_acesso"))
        id_tripla = bool(campos.get("modelo")) and bool(campos.get("serie")) and bool(campos.get("numero_nota"))

        completo = (ok_valor and ok_data) and (ok_cnpj or ok_chave or id_tripla)

        if xml_encontrado and completo:
            return {"status": "processado", "fonte": "xml+associacao"}
        if conf_ocr >= self.OCR_CONF_STRONG and completo:
            return {"status": "processado", "fonte": "ocr/nlp"}
        if conf_ocr >= self.OCR_CONF_MEDIUM and (ok_valor or ok_data):
            return {"status": "revisao_pendente", "fonte": "ocr/nlp_parcial"}
        return {"status": "revisao_pendente", "fonte": "baixa_confianca"}

    def _regex_minima_completar(self, texto: str, campos: Dict[str, Any]) -> Dict[str, Any]:
        """
        Último recurso: captura pistas soltas (IE/município/endereço) via regex simples.
        Não grava no cabeçalho; coloca em __context__.fallbacks.* para auditoria/aproveitamento posterior.
        """
        try:
            def _rx(p, flags=re.I | re.M):
                m = re.search(p, texto or "", flags)
                return m.group(1).strip() if m else None

            fallbacks = {
                "inscricao_estadual": _rx(r"inscri[çc][aã]o\s+estadual[:\s]*([0-9\.\/\-]+)"),
                "municipio": _rx(r"munic[ií]pio[:\s]*([A-ZÀ-Ü][A-Za-zÀ-ÿ\s\-']+)"),
                "endereco": _rx(r"endere[çc]o[:\s]*(.+)"),
            }
            fallbacks = {k: v for k, v in fallbacks.items() if v}

            if fallbacks:
                ctx = campos.get("__context__", {})
                ctx.setdefault("fallbacks", {}).update(fallbacks)
                campos["__context__"] = ctx
        except Exception as e:
            log.warning("Regex mínima de fallback falhou: %s", e)
        return campos

    def _processar_com_llm_fallback(self, entrada: Any, motivo: str = "desconhecido") -> Dict[str, Any]:
        """Fallback cognitivo global: tenta extrair o cabeçalho fiscal somente com LLM."""
        if not self.llm:
            return {"__meta__": {"source": f"llm_fallback_inexistente({motivo})"}}

        schema = getattr(self.nlp_agent, "schema_campos", [])
        schema_json = json.dumps(schema, ensure_ascii=False)

        prompt = (
            f"O documento não pôde ser processado normalmente ({motivo}).\n"
            "Extraia os campos fiscais estruturados de uma nota brasileira (NFe, NFCe, NFSe, CTe) a partir do texto.\n"
            "Regras:\n"
            "- Responda APENAS com um JSON válido, usando exatamente as chaves do schema fornecido.\n"
            "- Campos ausentes => null.\n"
            "- Datas em YYYY-MM-DD; valores numéricos (sem 'R$').\n"
            "- Inclua __meta__ com confiança 0..1 por campo.\n\n"
            f"Schema: {schema_json}\n\n"
            "Texto:\n"
        )
        if isinstance(entrada, (bytes, bytearray)):
            prompt += "[Arquivo binário PDF/imagem — use apenas seu conhecimento fiscal para estruturar caso impossível ler o conteúdo]"
        else:
            prompt += str(entrada or "")[:6000]

        try:
            resp = self.llm.invoke(prompt)  # type: ignore[attr-defined]
            txt = getattr(resp, "content", None) or str(resp)
            parsed = self.nlp_agent._safe_parse_json(txt)
            if isinstance(parsed, dict):
                parsed["__meta__"] = {"source": f"llm_fallback({motivo})", **(parsed.get("__meta__", {}) or {})}
                return parsed
        except Exception as e:
            log.error("Falha no LLM fallback: %s", e)

        return {"__meta__": {"source": f"erro_fallback({motivo})"}}

    # --------------------------------------------------------
    # LangChain Tools (opcional)
    # --------------------------------------------------------
    def _init_langchain_layer(self) -> None:
        if not (_LC_AVAILABLE and self.llm):
            return

        def _tool_parse_input(input_str: str) -> Dict[str, Any]:
            try:
                if not input_str:
                    return {}
                data = json.loads(input_str)
                if not isinstance(data, dict):
                    return {}
                return data
            except Exception:
                return {}

        def _t_ocr(input_str: str) -> str:
            try:
                data = _tool_parse_input(input_str)
                nome_arquivo = data.get("nome_arquivo") or "arquivo"
                caminho = data.get("caminho")
                if not caminho or not Path(caminho).exists():
                    raise ValueError("Caminho do arquivo é obrigatório e deve existir.")
                content = Path(caminho).read_bytes()
                texto, conf = self.ocr_agent.reconhecer(nome_arquivo, content)
                return json.dumps({"ok": True, "texto": texto[:200000], "conf": conf}, ensure_ascii=False)
            except Exception as e:
                return json.dumps({"ok": False, "erro": f"OCR: {e}"}, ensure_ascii=False)

        def _t_nlp(input_str: str) -> str:
            try:
                data = _tool_parse_input(input_str)
                texto_ocr = data.get("texto_ocr") or ""
                campos = self.nlp_agent.extrair_campos({
                    "texto_ocr": texto_ocr or "",
                    "ocr_meta": data.get("ocr_meta") or {}
                })
                itens = campos.pop("itens_ocr", []) or []
                impostos = campos.pop("impostos_ocr", []) or []
                return json.dumps({"ok": True, "campos": campos, "itens": itens, "impostos": impostos}, ensure_ascii=False)
            except Exception as e:
                return json.dumps({"ok": False, "erro": f"NLP: {e}"}, ensure_ascii=False)

        def _t_normalizar(input_str: str) -> str:
            try:
                data = _tool_parse_input(input_str)
                campos = data.get("campos") or {}
                out = self.normalizador.normalizar(
                    dict(campos)  # FIX: sem keep_context_copy kwarg
                )
                return json.dumps({"ok": True, "campos": out}, ensure_ascii=False)
            except Exception as e:
                return json.dumps({"ok": False, "erro": f"Normalizador: {e}"}, ensure_ascii=False)

        def _t_associar(input_str: str) -> str:
            try:
                data = _tool_parse_input(input_str)
                doc_id = int(data.get("doc_id"))
                campos = data.get("campos") or {}
                texto_ocr = data.get("texto_ocr") or ""
                out = self.associador.tentar_associar_pdf(doc_id, dict(campos), texto_ocr=texto_ocr)
                return json.dumps({"ok": True, "campos": out}, ensure_ascii=False)
            except Exception as e:
                return json.dumps({"ok": False, "erro": f"AssociadorXML: {e}"}, ensure_ascii=False)

        def _t_validar(input_str: str) -> str:
            try:
                data = _tool_parse_input(input_str)
                doc_id = int(data.get("doc_id"))
                self.validador.validar_documento(doc_id=doc_id, db=self.db)
                after = self.db.get_documento(doc_id) or {}
                return json.dumps({"ok": True, "status": after.get("status"), "motivo": after.get("motivo_rejeicao")}, ensure_ascii=False)
            except Exception as e:
                return json.dumps({"ok": False, "erro": f"Validação: {e}"}, ensure_ascii=False)

        def _t_metricas(input_str: str) -> str:
            try:
                data = _tool_parse_input(input_str)
                self.metrics_agent.registrar_metrica(
                    db=self.db,
                    tipo_documento=str(data.get("tipo")),
                    status=str(data.get("status")),
                    confianca_media=float(data.get("conf") or 0.0),
                    tempo_medio=float(data.get("duracao") or 0.0),
                )
                return json.dumps({"ok": True}, ensure_ascii=False)
            except Exception as e:
                return json.dumps({"ok": False, "erro": f"Métricas: {e}"}, ensure_ascii=False)

        self._lc_tools = []
        try:
            self._lc_tools.append(Tool(name="ocr_executar", description="Executa OCR no arquivo pelo caminho local. Args JSON: {nome_arquivo:str, caminho:str}", func=_t_ocr))
            self._lc_tools.append(Tool(name="nlp_extrair_campos", description="Roda NLP no texto OCR. Args JSON: {texto_ocr:str}", func=_t_nlp))
            self._lc_tools.append(Tool(name="normalizar_campos", description="Normaliza campos fiscais. Args JSON: {campos:dict}", func=_t_normalizar))
            self._lc_tools.append(Tool(name="associar_xml", description="Associa PDF a XML existente. Args JSON: {doc_id:int, campos:dict, texto_ocr:str?}", func=_t_associar))
            self._lc_tools.append(Tool(name="validar_documento", description="Valida o documento no banco. Args JSON: {doc_id:int}", func=_t_validar))
            self._lc_tools.append(Tool(name="registrar_metricas", description="Registra métricas agregadas. Args JSON: {tipo:str, status:str, conf:float, duracao:float}", func=_t_metricas))
        except Exception as e:
            log.warning(f"Falha ao criar Tools do LangChain: {e}")
            self._lc_tools = []

        self._lc_agent = None
        if self._lc_tools and initialize_agent and AgentType:
            try:
                self._lc_agent = initialize_agent(
                    tools=self._lc_tools,
                    llm=self.llm,  # type: ignore[arg-type]
                    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                    verbose=False,
                    handle_parsing_errors=True,
                    max_iteration=8,
                    early_stopping_method="force",
                )
            except Exception as e:
                log.warning(f"Falha ao inicializar Agent LangChain (planner): {e}")
                self._lc_agent = None

    # --------------------------------------------------------
    # Grafo cognitivo (LangGraph)
    # --------------------------------------------------------
    def _build_langgraph(self):
        """
        Define um grafo de estados resiliente com loops de correção:
        register -> ocr -> nlp -> (condicional) llm_refine -> normalize -> associate -> decide
        -> validate -> persist -> metrics -> END
        Alimenta o Blackboard e produz decisões explicativas.
        """

        # Estado compartilhado do grafo
        def new_state() -> Dict[str, Any]:
            return {
                "doc_id": -1,
                "nome": None,
                "origem": None,
                "conteudo": None,
                "caminho_arquivo": None,
                "tipo_doc": None,
                "texto_ocr": "",
                "layout_fp": None,
                "conf_ocr": 0.0,
                "campos": {},
                "itens_ocr": [],
                "impostos_ocr": [],
                "meta_nlp": {},
                "status": "processando",
                "fonte": "desconhecida",
                "erro": None,
                "start_time": time.time(),
                "loops": 0,
                "max_loops": 2,  # controla quantas vezes podemos voltar ao LLM_refine
            }

        graph = StateGraph(dict, name="AgenteFiscalGraph")

        # ------------- NODES -------------
        def n_register(state: Dict[str, Any]) -> Dict[str, Any]:
            nome = state["nome"]
            origem = state["origem"]
            conteudo = state["conteudo"]
            tipo_doc = Path(nome).suffix.lower().strip(".")
            try:
                doc_id = self.db.inserir_documento(
                    nome_arquivo=nome,
                    tipo=tipo_doc,
                    origem=origem,
                    hash=self.db.hash_bytes(conteudo),
                    status="processando",
                    data_upload=self.db.now(),
                    caminho_arquivo=str(self.db.save_upload(nome, conteudo)),
                )
                state["doc_id"] = doc_id
                state["tipo_doc"] = tipo_doc
                state["caminho_arquivo"] = (self.db.get_documento(doc_id) or {}).get("caminho_arquivo")
                self._bb_write("decisions", {"doc_id": doc_id})
                log.info("Grafo: registrado doc_id=%s '%s'", doc_id, nome)
            except Exception as e:
                state["erro"] = f"Falha ao registrar documento: {e}"
                log.exception(state["erro"])
            return state

        def n_ocr(state: Dict[str, Any]) -> Dict[str, Any]:
            if state.get("erro"):
                return state
            nome = state["nome"]
            conteudo = state["conteudo"]
            doc_id = state["doc_id"]

            def _do_ocr() -> Tuple[str, float]:
                return self.ocr_agent.reconhecer(nome, conteudo)

            try:
                t0 = time.time()
                texto, conf = self._retry(_do_ocr, attempts=2, backoff_s=0.6)
                dt = time.time() - t0
                state["texto_ocr"] = texto or ""
                state["conf_ocr"] = float(conf or 0.0)

                # layout fingerprint para memória cognitiva
                layout_fp = None
                try:
                    layout_fp = self.memoria.layout_fingerprint(texto or "")
                except Exception:
                    layout_fp = None
                state["layout_fp"] = layout_fp

                if (texto or "").strip():
                    self.db.inserir_extracao(
                        documento_id=doc_id,
                        agente="OCRAgent",
                        confianca_media=float(conf),
                        texto_extraido=texto[:200000] + ("..." if len(texto) > 200000 else ""),
                        linguagem="pt",
                        tempo_processamento=round(dt, 3),
                    )
                    ocr_tipo = "nativo" if conf >= 0.98 else "ocr"
                    self._merge_meta_json(doc_id, {"ocr": {"tipo": ocr_tipo, "conf": float(conf), "duracao_s": round(dt, 3)}})
                    # reforça meta detalhada do OCR quando disponível
                    try:
                        self._merge_meta_json(doc_id, {"ocr_meta": getattr(self.ocr_agent, "last_stats", {}) or {}})
                    except Exception:
                       pass
                    self._bb_write("ocr", {"data": {"texto": f"[{len(texto)} chars]"}, "__meta__": {"avg_confidence": conf, "pages": None, "llm_correction_applied": bool(getattr(self.ocr_agent, "last_stats", {}).get("llm_correction_applied", False))}})
                    # decisão adaptativa sobre reforço de OCR
                    if conf < self.OCR_CONF_MEDIUM:
                        self._bb_decision(f"OCR conf {conf:.2f} < {self.OCR_CONF_MEDIUM:.2f} → configurar NLP para LLM_full", stage="ocr")
                else:
                    pass
                    self.db.log("ocr_vazio", "AgenteOCR", f"doc_id={doc_id}|conf={conf:.2f}")
                    self._bb_decision("OCR retornou texto vazio → ativar LLM_full no NLP", stage="ocr")
                state["ocr_meta"] = getattr(self.ocr_agent, "last_stats", {}) or {}
            except Exception as e:
                state["erro"] = f"OCR falhou: {e}"
                log.exception(state["erro"])
            return state

        def n_nlp(state: Dict[str, Any]) -> Dict[str, Any]:
            if state.get("erro"):
                return state
            texto = state.get("texto_ocr") or ""
            doc_id = state["doc_id"]
            try:
                if texto:
                    def _do_nlp() -> Tuple[Dict[str, Any]]:
                        out = self.nlp_agent.extrair_campos({
                            "texto_ocr": texto or "",
                            "ocr_meta": state.get("ocr_meta") or {}
                        })
                        return (out,)
                    (campos,) = self._retry(_do_nlp, attempts=2, backoff_s=0.5)
                    campos = campos or {}
                    meta = campos.pop("__meta__", {}) if isinstance(campos, dict) else {}
                    state["meta_nlp"] = meta or {}
                    state["itens_ocr"] = campos.pop("itens_ocr", []) or []
                    state["impostos_ocr"] = campos.pop("impostos_ocr", []) or []
                    # Fusão incremental (mantém o que já tem, preenche ausentes)
                    fusion = self.normalizador.fundir(state.get("campos") or {}, campos)
                    state["campos"] = fusion

                    cov = float(state["meta_nlp"].get("coverage", 0.0) or 0.0)
                    self._bb_write("nlp", {"data": {"campos": list(fusion.keys())}, "__meta__": {"coverage": cov, "source": state["meta_nlp"].get("source", "desconhecido")}})
                    if cov < self.COVERAGE_THRESHOLD or self._faltando_campos_criticos(fusion):
                        self._bb_decision(f"NLP coverage {cov:.2f} < {self.COVERAGE_THRESHOLD:.2f} ou campos críticos faltando → ativar LLM_refine", stage="nlp", coverage=cov)
            except Exception as e:
                log.warning("NLP falhou no grafo doc_id=%d: %s", doc_id, e)
            return state

        def n_llm_refine(state: Dict[str, Any]) -> Dict[str, Any]:
            if state.get("erro") or not self.llm:
                return state
            if state.get("loops", 0) >= state.get("max_loops", 2):
                return state

            texto = state.get("texto_ocr") or ""
            doc_id = state["doc_id"]
            try:
                extrair_llm = getattr(self.nlp_agent, "_extrair_campos_llm", None)
                campos_llm = {}
                meta_llm = {}

                if callable(extrair_llm) and texto.strip():
                    campos_llm = extrair_llm(texto) or {}
                    meta_llm = campos_llm.pop("__meta__", {}) if isinstance(campos_llm, dict) else {}
                    try:
                        vals = [
                            float(v)
                            for v in (meta_llm.values() if isinstance(meta_llm, dict) else [])
                            if isinstance(v, (int, float, str))
                        ]
                        conf_llm = sum(pd.to_numeric(pd.Series(vals), errors="coerce").fillna(0.0)) / max(len(vals) or 1, 1)
                    except Exception:
                        conf_llm = None

                    self.db.inserir_extracao(
                        documento_id=doc_id,
                        agente="LLM-NLP",
                        confianca_media=conf_llm if isinstance(conf_llm, (int, float)) else None,
                        texto_extraido=json.dumps(meta_llm, ensure_ascii=False) if meta_llm else None,
                        linguagem="pt",
                        tempo_processamento=0.0,
                    )
                else:
                    # fallback global
                    gl = self._processar_com_llm_fallback(texto if texto else state.get("conteudo", b""), motivo="grafo_refine")
                    campos_llm = {k: v for k, v in gl.items() if k != "__meta__"}

                # Fusão
                base = state.get("campos") or {}
                fusion = self.normalizador.fundir(base, campos_llm)
                state["campos"] = fusion
                state["loops"] = int(state.get("loops", 0)) + 1

                # último recurso: regex mínima
                if self._faltando_campos_criticos(state["campos"]):
                    state["campos"] = self._regex_minima_completar(texto, state["campos"])

                self._bb_decision("LLM refine executado para completar campos críticos/baixa cobertura", stage="nlp")
            except Exception as e:
                log.warning("LLM refine falhou no grafo doc_id=%d: %s", doc_id, e)
            return state

        def n_normalize(state: Dict[str, Any]) -> Dict[str, Any]:
            if state.get("erro"):
                return state
            try:
                campos_norm = self.normalizador.normalizar(
                    self.normalizador.fundir(state.get("campos") or {})
                )
                # Se vier __context__, mandar para meta_json e (opcional) documentos_detalhes
                contexto = campos_norm.pop("__context__", None)
                if contexto:
                    try:
                        self._merge_meta_json(state["doc_id"], {"context": contexto})
                        self._persistir_detalhes_contexto(state["doc_id"], contexto)
                    except Exception:
                        pass
                state["campos"] = campos_norm

                # score de sanidade do normalizador (se exposto em __meta__)
                meta_norm = campos_norm.get("__meta__") if isinstance(campos_norm, dict) else None
                sanity = None
                if isinstance(meta_norm, dict):
                    sanity = meta_norm.get("sanity_score")
                # registra no blackboard
                self._bb_write("normalizer", {"__meta__": {"sanity_score": sanity}})
                if isinstance(sanity, (int, float)) and float(sanity) < self.NORMALIZER_SANITY_MIN:
                    self._bb_decision(f"Sanity {float(sanity):.2f} < {self.NORMALIZER_SANITY_MIN:.2f} → marcar revisão contextual", stage="normalize", sanity=float(sanity))
            except Exception as e:
                log.warning("Normalização falhou doc_id=%s: %s", state.get("doc_id"), e)
            return state

        def n_associate(state: Dict[str, Any]) -> Dict[str, Any]:
            if state.get("erro"):
                return state
            try:
                doc_id = state["doc_id"]
                texto = state.get("texto_ocr") or ""
                campos_associados = self.associador.tentar_associar_pdf(doc_id, state.get("campos") or {}, texto_ocr=texto)
                # Guardar assoc meta em meta_json e limpar do payload
                assoc_meta = (campos_associados or {}).pop("__assoc_meta__", None)
                if assoc_meta:
                    try:
                        self._merge_meta_json(doc_id, {"associacao": assoc_meta})
                    except Exception:
                        pass
                state["campos"] = campos_associados
                # blackboard
                self._bb_write("associador", {"__meta__": {"match_score": (assoc_meta or {}).get("match_score") if isinstance(assoc_meta, dict) else None}})
            except Exception as e:
                log.warning("Associação XML falhou doc_id=%s: %s", state.get("doc_id"), e)
            return state

        def n_decide(state: Dict[str, Any]) -> Dict[str, Any]:
            campos = state.get("campos") or {}
            tipo_detectado = (campos.get("tipo") or "").lower().strip()
            caminho_xml = (campos.get("caminho_xml") or "").strip()
            xml_encontrado = bool(
                caminho_xml
                or ("xml" in tipo_detectado)
                or tipo_detectado in {"nfe", "nfce", "cte", "nfse", "cfe"}
            )
            rota = self._decidir_rota(conf_ocr=float(state.get("conf_ocr") or 0.0), campos=campos, xml_encontrado=xml_encontrado)
            state["status"] = rota.get("status", "revisao_pendente")
            state["fonte"] = rota.get("fonte", "ocr/nlp")
            self._bb_decision(f"Rota final: status={state['status']} fonte={state['fonte']}", stage="decide")
            return state

        def n_validate(state: Dict[str, Any]) -> Dict[str, Any]:
            try:
                self.validador.validar_documento(doc_id=state["doc_id"], db=self.db)
                self._bb_write("validator", {"__meta__": {"executado": True}})
            except Exception as e:
                log.warning("Validação fiscal falhou doc_id=%s: %s", state.get("doc_id"), e)
            return state

        def n_persist(state: Dict[str, Any]) -> Dict[str, Any]:
            doc_id = state["doc_id"]
            campos = state.get("campos") or {}
            permitidas = self._campos_permitidos_documentos()
            campos_safe = self._filtrar_campos_validos(
                {k: v for k, v in campos.items() if k not in ("itens_ocr", "impostos_ocr")},
                permitidas,
            )
            if campos_safe:
                try:
                    self.db.atualizar_documento_campos(doc_id, **campos_safe)
                except Exception as e:
                    log.warning("Persistência parcial falhou doc_id=%s: %s", doc_id, e)
            else:
                try:
                    self.db.atualizar_documento_campos(
                        doc_id,
                        status="revisao_pendente",
                        motivo_rejeicao="Extração vazia/insuficiente após OCR/NLP e fallbacks (grafo)."
                    )
                    state["status"] = "revisao_pendente"
                    state["fonte"] = "fallback_incompleto"
                    self._bb_decision("Persistência sem campos úteis → marcado revisão_pendente", stage="persist")
                except Exception as e:
                    log.warning("Persistência mínima falhou doc_id=%s: %s", doc_id, e)

            # Itens / Impostos
            itens = state.get("itens_ocr") or []
            impostos = state.get("impostos_ocr") or []
            if itens:
                try:
                    itens = self.normalizador.normalizar_itens(itens)
                except Exception as e_norm_it:
                    log.warning("Falha ao normalizar itens doc_id=%d: %s", doc_id, e_norm_it)

                item_id_map: Dict[int, int] = {}
                for idx, item_data in enumerate(itens):
                    try:
                        item_id = self.db.inserir_item(documento_id=doc_id, **item_data)
                        item_id_map[idx] = item_id
                    except Exception as e_item:
                        log.warning("Falha ao inserir item OCR idx=%d doc_id=%d: %s", idx, doc_id, e_item)

                if impostos:
                    for imposto_data in impostos:
                        try:
                            item_ocr_idx = imposto_data.pop("item_idx", -1)
                            if item_ocr_idx in item_id_map:
                                self.db.inserir_imposto(item_id=item_id_map[item_ocr_idx], **imposto_data)
                            else:
                                log.warning("Imposto OCR sem item vinculado idx=%s, doc_id=%d", item_ocr_idx, doc_id)
                        except Exception as e_imp:
                            log.warning("Falha ao inserir imposto OCR doc_id=%d: %s", doc_id, e_imp)
            return state

        def n_metrics(state: Dict[str, Any]) -> Dict[str, Any]:
            doc_id = state["doc_id"]
            try:
                stats = getattr(self.ocr_agent, "last_stats", {}) or {}
                avg_conf = float(stats.get("avg_confidence", state.get("conf_ocr") or 0.0) or state.get("conf_ocr") or 0.0)
                fonte_nlp = (state.get("meta_nlp") or {}).get("source", "desconhecido")
                cov = float((state.get("meta_nlp") or {}).get("coverage", 0.0) or 0.0)

                # meta_json
                try:
                    self._merge_meta_json(doc_id, {
                        "nlp": {"source": fonte_nlp, "coverage": cov},
                        "pipeline": {"fonte_final": state.get("fonte"), "status_final": state.get("status")},
                        "blackboard": {"decisions": self.blackboard.get("decisions", {}).get("log", [])}
                    })
                except Exception:
                    pass

                # status + log
                try:
                    self.db.atualizar_documento_campo(doc_id, "status", state.get("status"))
                except Exception:
                    pass

                self.db.log(
                    "ingestao_midias_grafo",
                    "sistema",
                    f"doc_id={doc_id}|conf={avg_conf:.2f}|status={state.get('status')}|fonte={state.get('fonte')}|loops={state.get('loops')}",
                )

                # métricas agregadas
                processing_time = time.time() - float(state.get("start_time") or time.time())
                self.metrics_agent.registrar_metrica(
                    db=self.db,
                    tipo_documento=state.get("tipo_doc") or "desconhecido",
                    status=state.get("status") or "erro",
                    confianca_media=float(state.get("conf_ocr") or 0.0),
                    tempo_medio=float(processing_time),
                )

                # ciclo de feedback (memória cognitiva)
                sucesso = (state.get("status") in {"processado", "revisado"})
                receita = {
                    "ocr": getattr(self.ocr_agent, "last_stats", {}),
                    "nlp": {"coverage": cov, "source": fonte_nlp, "loops": state.get("loops")},
                    "decisions": self.blackboard.get("decisions", {}).get("log", []),
                }
                metricas = {
                    "coverage": cov,
                    "confidence": avg_conf,
                    "status": state.get("status"),
                    "fonte": state.get("fonte"),
                }
                layout_fp = state.get("layout_fp")
                try:
                    # grava receita por layout (com EMA)
                    if layout_fp:
                        self.memoria.registrar_layout_receita(
                            layout_fp=layout_fp,
                            receita=receita,
                            fonte="orchestrator",
                            sucesso=sucesso
                        )
                    # telemetria da execução
                    campos = state.get("campos") or {}
                    self.memoria.registrar_execucao(
                        doc_id=doc_id,
                        layout_fp=layout_fp,
                        cnpj_emitente=campos.get("emitente_cnpj"),
                        cnpj_destinatario=campos.get("destinatario_cnpj"),
                        receita=receita,
                        metricas=metricas,
                        sucesso=sucesso,
                    )
                except Exception as e:
                    log.debug("Falha no ciclo de feedback (memória) doc_id=%d: %s", doc_id, e)

            except Exception as e:
                log.warning("Falha métricas grafo doc_id=%s: %s", doc_id, e)
            return state

        # ------------- CONDIÇÕES -------------
        def c_post_nlp(state: Dict[str, Any]) -> str:
            # Se faltar campo crítico OU cobertura baixa, tenta refinar via LLM (até max_loops); senão segue
            can_loop = state.get("loops", 0) < state.get("max_loops", 2)
            if self.llm and self._precisa_llm(state.get("campos") or {}, state.get("meta_nlp") or {}) and can_loop:
                return "llm_refine"
            return "normalize"

        # ------------- MONTAGEM -------------
        graph.add_node("register", n_register)
        graph.add_node("ocr", n_ocr)
        graph.add_node("nlp", n_nlp)
        graph.add_node("llm_refine", n_llm_refine)
        graph.add_node("normalize", n_normalize)
        graph.add_node("associate", n_associate)
        graph.add_node("decide", n_decide)
        graph.add_node("validate", n_validate)
        graph.add_node("persist", n_persist)
        graph.add_node("metrics", n_metrics)

        graph.set_entry_point("register")
        graph.add_edge("register", "ocr")
        graph.add_edge("ocr", "nlp")
        graph.add_conditional_edges("nlp", c_post_nlp, {"llm_refine": "llm_refine", "normalize": "normalize"})
        graph.add_edge("llm_refine", "normalize")
        graph.add_edge("normalize", "associate")
        graph.add_edge("associate", "decide")
        graph.add_edge("decide", "validate")
        graph.add_edge("validate", "persist")
        graph.add_edge("persist", "metrics")
        graph.add_edge("metrics", END)

        compiled = graph.compile()

        # wrapper para invocação
        def run_graph(nome: str, conteudo: bytes, origem: str) -> int:
            # limpa blackboard por execução
            self.blackboard = {
                "ocr": {},
                "nlp": {},
                "normalizer": {},
                "validator": {},
                "associador": {},
                "decisions": {"log": []},
            }
            state = {
                **new_state(),
                "nome": nome,
                "conteudo": conteudo,
                "origem": origem,
            }
            out = compiled.invoke(state)  # executa o grafo
            return int(out.get("doc_id", -1) or -1)

        return run_graph

    # --------------------------------------------------------
    # Ingestão de arquivos (usa grafo quando disponível)
    # --------------------------------------------------------
    def ingestir_arquivo(self, nome: str, conteudo: bytes, origem: str = "web") -> int:
        """Processa um arquivo, retornando o ID do documento."""
        t_start = time.time()
        doc_id = -1
        status = "erro"
        doc_hash = self.db.hash_bytes(conteudo)
        ext = Path(nome).suffix.lower()
        tipo_doc = ext.strip(".") or "binario"

        try:
            existing_id = self.db.find_documento_by_hash(doc_hash)
            if existing_id:
                log.info("Doc '%s' (hash %s...) já existe ID %d. Ignorando.", nome, doc_hash[:8], existing_id)
                return existing_id

            if ext == ".xml":
                tipo_doc = "xml"
                doc_id = self.xml_agent.processar(nome, conteudo, origem)
            elif ext in {".pdf", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}:
                if callable(self._graph):
                    # —— NOVO: usa grafo cognitivo
                    doc_id = self._graph(nome, conteudo, origem)  # type: ignore
                else:
                    # fallback determinístico antigo (ainda robusto)
                    doc_id = self._processar_midias(nome, conteudo, origem)
                tipo_doc = Path(nome).suffix.lower().strip(".")
            else:
                motivo = f"Extensão '{ext}' não suportada."
                status = "quarentena"
                log.warning("Arquivo '%s' rejeitado: %s", nome, motivo)
                tipo_doc = "desconhecido"
                doc_id = self.db.inserir_documento(
                    nome_arquivo=nome,
                    tipo=tipo_doc,
                    origem=origem,
                    hash=doc_hash,
                    status=status,
                    data_upload=self.db.now(),
                    motivo_rejeicao=motivo,
                )
                self.metrics_agent.registrar_metrica(
                    db=self.db,
                    tipo_documento=tipo_doc,
                    status=status,
                    confianca_media=0.0,
                    tempo_medio=(time.time() - t_start),
                )

            if doc_id > 0:
                doc_info = self.db.get_documento(doc_id)
                if doc_info:
                    status = doc_info.get("status", status)

        except Exception as e:
            log.exception("Falha ingestão '%s': %s", nome, e)
            status = "erro"
            try:
                existing_id_on_error = self.db.find_documento_by_hash(doc_hash)
                if existing_id_on_error:
                    doc_id = existing_id_on_error
                    self.db.atualizar_documento_campos(doc_id, status="erro", motivo_rejeicao=str(e))
                elif doc_id > 0:
                    self.db.atualizar_documento_campos(doc_id, status="erro", motivo_rejeicao=str(e))
                else:
                    doc_id = self.db.inserir_documento(
                        nome_arquivo=nome,
                        tipo=tipo_doc,
                        origem=origem,
                        hash=doc_hash,
                        status="erro",
                        data_upload=self.db.now(),
                        motivo_rejeicao=str(e),
                    )
                self.metrics_agent.registrar_metrica(
                    db=self.db,
                    tipo_documento=tipo_doc,
                    status="erro",
                    confianca_media=0.0,
                    tempo_medio=(time.time() - t_start),
                )
            finally:
                pass
        finally:
            log.info("Ingestão '%s' (ID: %d, Status: %s) em %.2fs", nome, doc_id, status, time.time() - t_start)
        return doc_id

    # --------------------------------------------------------
    # Pipeline determinístico (fallback)
    # --------------------------------------------------------
    def _processar_midias(self, nome: str, conteudo: bytes, origem: str) -> int:
        """Processa PDF/Imagem via OCR→NLP→normalização→associação→validação→métricas (fallback)."""
        doc_id = -1
        t0 = time.time()
        status_final = "erro"
        conf = 0.0
        tipo_doc = Path(nome).suffix.lower().strip(".")
        fonte_final = "desconhecida"

        # reset do blackboard para este documento
        self.blackboard = {
            "ocr": {},
            "nlp": {},
            "normalizer": {},
            "validator": {},
            "associador": {},
            "decisions": {"log": []},
        }

        try:
            # 0) Registrar documento
            doc_id = self.db.inserir_documento(
                nome_arquivo=nome,
                tipo=tipo_doc,
                origem=origem,
                hash=self.db.hash_bytes(conteudo),
                status="processando",
                data_upload=self.db.now(),
                caminho_arquivo=str(self.db.save_upload(nome, conteudo)),
            )
            self._bb_write("decisions", {"doc_id": doc_id})
            log.info("Processando mídia '%s' (doc_id %d)", nome, doc_id)

            # 1) OCR com retry
            def _do_ocr() -> Tuple[str, float]:
                return self.ocr_agent.reconhecer(nome, conteudo)

            t_ocr_start = time.time()
            texto, conf = self._retry(_do_ocr, attempts=2, backoff_s=0.6)
            ocr_time = time.time() - t_ocr_start
            ocr_tipo = "nativo" if conf >= 0.98 else "ocr"
            ocr_meta = getattr(self.ocr_agent, "last_stats", {}) or {}

            layout_fp = None
            try:
                layout_fp = self.memoria.layout_fingerprint(texto or "")
            except Exception:
                pass

            if texto.strip():
                self.db.inserir_extracao(
                    documento_id=doc_id,
                    agente="OCRAgent",
                    confianca_media=float(conf),
                    texto_extraido=texto[:200000] + ("..." if len(texto) > 200000 else ""),
                    linguagem="pt",
                    tempo_processamento=round(ocr_time, 3),
                )
                self._merge_meta_json(doc_id, {"ocr": {"tipo": ocr_tipo, "conf": float(conf), "duracao_s": round(ocr_time, 3)}})
                self._bb_write("ocr", {"data": {"texto": f"[{len(texto)} chars]"}, "__meta__": {"avg_confidence": conf}})
                if conf < self.OCR_CONF_MEDIUM:
                    self._bb_decision(f"OCR conf {conf:.2f} < {self.OCR_CONF_MEDIUM:.2f} → preparar LLM_full no NLP", stage="ocr")
            else:
                self.db.log("ocr_vazio", "AgenteOCR", f"doc_id={doc_id}|conf={conf:.2f}")
                self._bb_decision("OCR retornou texto vazio → ativar LLM_full no NLP", stage="ocr")

            # 2) NLP (híbrido)
            campos_nlp: Dict[str, Any] = {}
            itens_ocr: List[Dict[str, Any]] = []
            impostos_ocr: List[Dict[str, Any]] = []
            meta_nlp: Dict[str, Any] = {}

            if texto:
                def _do_nlp() -> Tuple[Dict[str, Any]]:
                    out = self.nlp_agent.extrair_campos({
                        "texto_ocr": texto or "",
                        "ocr_meta": ocr_meta or {}
                    })
                    return (out,)
                try:
                    (campos,) = self._retry(_do_nlp, attempts=2, backoff_s=0.5)
                    campos_nlp = campos or {}
                    itens_ocr = campos_nlp.pop("itens_ocr", []) or []
                    impostos_ocr = campos_nlp.pop("impostos_ocr", []) or []
                    meta_nlp = campos_nlp.pop("__meta__", {}) if isinstance(campos_nlp, dict) else {}
                    cov = float(meta_nlp.get("coverage", 0.0) or 0.0)
                    self._bb_write("nlp", {"data": {"campos": list((campos_nlp or {}).keys())}, "__meta__": {"coverage": cov, "source": meta_nlp.get("source", "desconhecido")}})
                except Exception as e_nlp:
                    log.warning("NLP falhou doc_id %d: %s", doc_id, e_nlp)
                    campos_nlp = {}

            precisa_llm = self._precisa_llm(campos_nlp, meta_nlp)
            if precisa_llm:
                self._bb_decision("NLP insuficiente (coverage ou campos críticos) → executar LLM refine", stage="nlp")

            # 3) Fallback LLM
            if precisa_llm and self.llm:
                try:
                    extrair_llm = getattr(self.nlp_agent, "_extrair_campos_llm", None)
                    if callable(extrair_llm):
                        campos_llm = extrair_llm(texto) or {}
                        meta_llm = campos_llm.pop("__meta__", {}) if isinstance(campos_llm, dict) else {}
                        campos_nlp = self.normalizador.fundir(campos_nlp, campos_llm)
                        self.db.inserir_extracao(
                            documento_id=doc_id,
                            agente="LLM-NLP",
                            confianca_media=None,
                            texto_extraido=json.dumps(meta_llm, ensure_ascii=False) if meta_llm else None,
                            linguagem="pt",
                            tempo_processamento=0.0,
                        )
                        precisa_llm = self._faltando_campos_criticos(campos_nlp)
                except Exception as e:
                    log.warning("Fallback LLM-NLP falhou doc_id %d: %s", doc_id, e)

            if precisa_llm and self.llm:
                try:
                    llm_global = self._processar_com_llm_fallback(texto if texto else conteudo, motivo="coverage_baixa_ou_campos_faltando")
                    campos_nlp = self.normalizador.fundir(campos_nlp, {k: v for k, v in llm_global.items() if k != "__meta__"})
                    precisa_llm = self._faltando_campos_criticos(campos_nlp)
                except Exception as e:
                    log.warning("LLM fallback global falhou doc_id %d: %s", doc_id, e)

            if precisa_llm:
                campos_nlp = self._regex_minima_completar(texto, campos_nlp)

            # 4) Fusão & Normalização
            campos_norm = self.normalizador.normalizar(
                self.normalizador.fundir(campos_nlp)
            )
            contexto = campos_norm.pop("__context__", None)
            if contexto:
                try:
                    self._merge_meta_json(doc_id, {"context": contexto})
                    self._persistir_detalhes_contexto(doc_id, contexto)
                except Exception:
                    pass

            # 5) Associação a XML existente
            campos_associados = self.associador.tentar_associar_pdf(doc_id, campos_norm, texto_ocr=texto)
            # Assoc meta -> meta_json
            assoc_meta = (campos_associados or {}).pop("__assoc_meta__", None)
            if assoc_meta:
                try:
                    self._merge_meta_json(doc_id, {"associacao": assoc_meta})
                except Exception:
                    pass
            self._bb_write("associador", {"__meta__": {"match_score": (assoc_meta or {}).get("match_score") if isinstance(assoc_meta, dict) else None}})

            # 6) Status por confiança
            tipo_detectado = (campos_associados.get("tipo") or "").lower().strip()
            caminho_xml = (campos_associados.get("caminho_xml") or "").strip()
            xml_encontrado = bool(
                caminho_xml
                or ("xml" in tipo_detectado)
                or tipo_detectado in {"nfe", "nfce", "cte", "nfse", "cfe"}
            )
            rota = self._decidir_rota(conf_ocr=conf, campos=campos_associados, xml_encontrado=xml_encontrado)
            status_final = rota.get("status", "revisao_pendente")
            fonte_final = rota.get("fonte", "ocr/nlp")
            self._bb_decision(f"Rota final: status={status_final} fonte={fonte_final}", stage="decide")

            # 7) Persistência
            permitidas = self._campos_permitidos_documentos()
            campos_safe = self._filtrar_campos_validos(
                {k: v for k, v in campos_associados.items() if k not in ("itens_ocr", "impostos_ocr")},
                permitidas,
            )
            if campos_safe:
                self.db.atualizar_documento_campos(doc_id, **campos_safe)
            else:
                self.db.atualizar_documento_campos(
                    doc_id,
                    status="revisao_pendente",
                    motivo_rejeicao="Extração vazia/insuficiente após OCR/NLP e fallbacks."
                )
                status_final = "revisao_pendente"
                fonte_final = "fallback_incompleto"
                self._bb_decision("Persistência sem campos úteis → marcado revisão_pendente", stage="persist")

            # 8) Itens/Impostos OCR
            if itens_ocr:
                try:
                    itens_ocr = self.normalizador.normalizar_itens(itens_ocr)
                except Exception as e_norm_it:
                    log.warning("Falha ao normalizar itens (fallback) doc_id=%d: %s", doc_id, e_norm_it)

                item_id_map: Dict[int, int] = {}
                for idx, item_data in enumerate(itens_ocr):
                    try:
                        item_id = self.db.inserir_item(documento_id=doc_id, **item_data)
                        item_id_map[idx] = item_id
                    except Exception as e_item:
                        log.warning("Falha ao inserir item OCR idx=%d doc_id=%d: %s", idx, doc_id, e_item)
                if impostos_ocr:
                    for imposto_data in impostos_ocr:
                        try:
                            item_ocr_idx = imposto_data.pop("item_idx", -1)
                            if item_ocr_idx in item_id_map:
                                self.db.inserir_imposto(item_id=item_id_map[item_ocr_idx], **imposto_data)
                            else:
                                log.warning("Imposto OCR s/ item idx=%s, doc_id %d.", item_ocr_idx, doc_id)
                        except Exception as e_imp:
                            log.warning("Falha ao inserir imposto OCR doc_id=%d: %s", doc_id, e_imp)

            # 9) Validação fiscal
            try:
                self.validador.validar_documento(doc_id=doc_id, db=self.db)
                self._bb_write("validator", {"__meta__": {"executado": True}})
            except Exception as e_val:
                log.warning("Validação fiscal falhou doc_id=%d: %s", doc_id, e_val)

            # 10) Telemetria + status final
            try:
                stats = getattr(self.ocr_agent, "last_stats", {}) or {}
                avg_conf = float(stats.get("avg_confidence", conf) or conf)
                fonte_nlp = (meta_nlp or {}).get("source", "desconhecido")
                cov = float((meta_nlp or {}).get("coverage", 0.0) or 0.0)

                log.info(
                    "OCR[%s] conf=%.2f | NLP src=%s cov=%.2f | Status=%s fonte=%s",
                    nome, avg_conf, fonte_nlp, cov, status_final, fonte_final,
                )
                self._merge_meta_json(doc_id, {
                    "nlp": {"source": fonte_nlp, "coverage": cov},
                    "pipeline": {"fonte_final": fonte_final, "status_final": status_final},
                    "blackboard": {"decisions": self.blackboard.get("decisions", {}).get("log", [])}
                })

                # ciclo de feedback memória/métricas
                sucesso = (status_final in {"processado", "revisado"})
                receita = {
                    "ocr": getattr(self.ocr_agent, "last_stats", {}),
                    "nlp": {"coverage": cov, "source": fonte_nlp},
                    "decisions": self.blackboard.get("decisions", {}).get("log", []),
                }
                metricas = {
                    "coverage": cov,
                    "confidence": avg_conf,
                    "status": status_final,
                    "fonte": fonte_final,
                }
                if layout_fp:
                    self.memoria.registrar_layout_receita(
                        layout_fp=layout_fp,
                        receita=receita,
                        fonte="orchestrator",
                        sucesso=sucesso
                    )
                campos_final = campos_associados or {}
                self.memoria.registrar_execucao(
                    doc_id=doc_id,
                    layout_fp=layout_fp,
                    cnpj_emitente=campos_final.get("emitente_cnpj"),
                    cnpj_destinatario=campos_final.get("destinatario_cnpj"),
                    receita=receita,
                    metricas=metricas,
                    sucesso=sucesso,
                )

            except Exception:
                pass

            self.db.atualizar_documento_campo(doc_id, "status", status_final)
            self.db.log(
                "ingestao_midias",
                "sistema",
                f"doc_id={doc_id}|conf={conf:.2f}|status={status_final}|fonte={fonte_final}",
            )

        except Exception as e_outer:
            log.exception("Falha geral mídia '%s': %s", nome, e_outer)
            status_final = "erro"
            if doc_id > 0:
                try:
                    self.db.atualizar_documento_campos(
                        doc_id, status="erro", motivo_rejeicao=f"Falha geral: {e_outer}"
                    )
                except Exception as db_err_f:
                    log.error("Erro CRÍTICO ao marcar erro final doc_id %d: %s", doc_id, db_err_f)
        finally:
            processing_time = time.time() - t0
            if doc_id > 0:
                try:
                    self.metrics_agent.registrar_metrica(
                        db=self.db,
                        tipo_documento=tipo_doc,
                        status=status_final,
                        confianca_media=conf,
                        tempo_medio=processing_time,
                    )
                except Exception as e_m:
                    log.warning("Falha ao registrar métricas finais doc_id=%d: %s", doc_id, e_m)
        return doc_id

    # --------------------------------------------------------
    # Q&A Analítico (dash/consultas)
    # --------------------------------------------------------
    def _executar_fast_query(self, pergunta: str, catalog: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Consultas determinísticas simples usando Pandas, sem LLM."""
        log.info("Modo Seguro: FastQuery: '%s'", pergunta)
        pergunta_lower = (pergunta or "").lower()
        df_docs = catalog.get("documentos")

        texto_resposta = "Não foi possível responder a pergunta no Modo Seguro."
        tabela_resposta = None
        try:
            if "contar" in pergunta_lower and "documentos" in pergunta_lower:
                total_docs = len(df_docs) if df_docs is not None else 0
                texto_resposta = f"Total de documentos (processados/revisados): **{total_docs}**."
            elif ("valor total" in pergunta_lower or "soma" in pergunta_lower) and "documentos" in pergunta_lower:
                if df_docs is not None and "valor_total" in df_docs.columns:
                    df_docs_local = df_docs.copy()
                    df_docs_local["valor_total_num"] = pd.to_numeric(df_docs_local["valor_total"], errors="coerce").fillna(0)
                    soma_total = df_docs_local["valor_total_num"].sum()
                    texto_resposta = f"Valor total somado no escopo: **R$ {soma_total:,.2f}**."
                else:
                    texto_resposta = "A coluna 'valor_total' não está disponível."
            elif ("top 5" in pergunta_lower or "top 10" in pergunta_lower) and "valor" in pergunta_lower:
                n_top = 10 if "top 10" in pergunta_lower else 5
                if df_docs is not None and {"valor_total", "emitente_nome"}.issubset(df_docs.columns):
                    df_docs_local = df_docs.copy()
                    df_docs_local["valor_total_num"] = pd.to_numeric(df_docs_local["valor_total"], errors="coerce").fillna(0)
                    top = (
                        df_docs_local.groupby("emitente_nome")["valor_total_num"]
                        .sum()
                        .nlargest(n_top)
                        .reset_index()
                        .rename(columns={"valor_total_num": "Valor Total"})
                    )
                    texto_resposta = f"Top {n_top} Emitentes por Valor Total:"
                    tabela_resposta = top
                else:
                    texto_resposta = f"Não foi possível calcular o Top {n_top} (faltam colunas)."
        except Exception as e:
            log.error("Erro no FastQuery: %s", e)
            texto_resposta = f"Erro no FastQuery: {e}"

        return {
            "texto": texto_resposta,
            "tabela": tabela_resposta,
            "figuras": [],
            "duracao_s": 0.01,
            "code": f"# FastQuery\n# Pergunta: {pergunta}",
            "agent_name": "FastQueryAgent (Seguro)",
        }

    def responder_pergunta(
        self, pergunta: str, scope_filters: Optional[Dict[str, Any]] = None, safe_mode: bool = False
    ) -> Dict[str, Any]:
        """Delega ao AgenteAnalitico (LLM) ou ao FastQuery determinístico."""
        if not self.analitico and not safe_mode:
            log.error("Agente Analítico não inicializado.")
            return {"texto": "Erro: Agente analítico não configurado.", "tabela": None, "figuras": []}

        catalog: Dict[str, pd.DataFrame] = {}
        try:
            where_conditions = ["(status = 'processado' OR status = 'revisado')"]
            if scope_filters:
                uf_escopo = scope_filters.get("uf")
                if uf_escopo and isinstance(uf_escopo, str):
                    where_conditions.append(f"(emitente_uf = '{uf_escopo.upper()}' OR destinatario_uf = '{uf_escopo.upper()}')")
                tipo_escopo = scope_filters.get("tipo")
                if tipo_escopo and isinstance(tipo_escopo, list) and len(tipo_escopo) > 0:
                    tipos_sql = ", ".join([f"'{t}'" for t in tipo_escopo])
                    where_conditions.append(f"tipo IN ({tipos_sql})")
            where_clause = " AND ".join(where_conditions)
            catalog["documentos"] = self.db.query_table("documentos", where=where_clause)

            if not catalog["documentos"].empty:
                doc_ids = tuple(catalog["documentos"]["id"].unique().tolist())
                doc_ids_sql = ", ".join(map(str, doc_ids))
                catalog["itens"] = self.db.query_table("itens", where=f"documento_id IN ({doc_ids_sql})")
                if not catalog["itens"].empty:
                    item_ids = tuple(catalog["itens"]["id"].unique().tolist())
                    item_ids_sql = ", ".join(map(str, item_ids))
                    catalog["impostos"] = self.db.query_table("impostos", where=f"item_id IN ({item_ids_sql})")
                else:
                    catalog["impostos"] = pd.DataFrame(
                        columns=["id","item_id","tipo_imposto","cst","origem","base_calculo","aliquota","valor"]
                    )
            else:
                catalog["itens"] = pd.DataFrame(
                    columns=["id","documento_id","descricao","ncm","cest","cfop","quantidade","unidade","valor_unitario","valor_total","codigo_produto"]
                )
                catalog["impostos"] = pd.DataFrame(
                    columns=["id","item_id","tipo_imposto","cst","origem","base_calculo","aliquota","valor"]
                )
        except Exception as e:
            log.exception("Falha ao montar catálogo com filtros: %s", e)
            return {"texto": f"Erro ao carregar dados com filtros: {e}", "tabela": None, "figuras": []}

        if catalog["documentos"].empty:
            log.info("Nenhum documento válido para análise (considerando filtros).")
            return {
                "texto": "Não há documentos válidos (status 'processado' ou 'revisado') que correspondam aos filtros.",
                "tabela": None,
                "figuras": [],
            }

        if safe_mode:
            return self._executar_fast_query(pergunta, catalog)

        if not self.analitico:
            log.error("Modo Seguro desativado, mas Agente Analítico (LLM) não está configurado.")
            return {"texto": "Erro: Agente Analítico (LLM) não configurado.", "tabela": None, "figuras": []}

        log.info("AgenteAnalitico: '%s' (filtros=%s)", pergunta, scope_filters)
        return self.analitico.responder(pergunta, catalog)

    # --------------------------------------------------------
    # Revalidação & Reprocessamento
    # --------------------------------------------------------
    def revalidar_documento(self, documento_id: int) -> Dict[str, Any]:
        """Aciona a revalidação de um documento específico."""
        try:
            doc = self.db.get_documento(documento_id)
            if not doc:
                return {"ok": False, "mensagem": f"Documento {documento_id} não encontrado."}

            status_anterior = doc.get("status")
            self.validador.validar_documento(doc_id=documento_id, db=self.db, force_revalidation=True)

            doc_depois = self.db.get_documento(documento_id)
            novo_status = doc_depois.get("status") if doc_depois else "desconhecido"
            self.db.log(
                "revalidacao",
                "usuario_sistema",
                f"doc_id={documento_id}|status_anterior={status_anterior}|status_novo={novo_status}|timestamp={self.db.now()}",
            )
            return {"ok": True, "mensagem": f"Documento revalidado. Novo status: {novo_status}."}
        except Exception as e:
            log.exception("Falha ao revalidar doc_id %d: %s", documento_id, e)
            return {"ok": False, "mensagem": f"Falha ao revalidar: {e}"}

    def reprocessar_documento(self, documento_id: int) -> Dict[str, Any]:
        """Deleta um documento e seus dados associados e tenta re-ingerir o arquivo original."""
        log.info("Reprocessamento para doc_id %d...", documento_id)
        try:
            doc_original = self.db.get_documento(documento_id)
            if not doc_original:
                return {"ok": False, "mensagem": f"Documento ID {documento_id} não encontrado."}

            caminho_arquivo_str = doc_original.get("caminho_arquivo")
            if not caminho_arquivo_str:
                return {"ok": False, "mensagem": "Documento não possui caminho de arquivo original salvo."}

            caminho_arquivo = Path(caminho_arquivo_str)
            if not caminho_arquivo.exists():
                return {"ok": False, "mensagem": f"Arquivo original '{caminho_arquivo_str}' não encontrado."}

            nome_arquivo_original = doc_original.get("nome_arquivo", caminho_arquivo.name)
            origem_original = doc_original.get("origem", "reprocessamento")
            conteudo_original = caminho_arquivo.read_bytes()

            # Deleta o documento antigo
            self.db.conn.execute("DELETE FROM documentos WHERE id = ?", (documento_id,))
            self.db.conn.commit()

            # Remove o arquivo físico
            try:
                if caminho_arquivo.exists():
                    caminho_arquivo.unlink()
            except Exception as e_clean:
                log.warning("Não foi possível excluir arquivo físico '%s': %s", caminho_arquivo, e_clean)

            # Re-ingere
            novo_doc_id = self.ingestir_arquivo(nome=nome_arquivo_original, conteudo=conteudo_original, origem=origem_original)

            if novo_doc_id == documento_id:
                msg = "Reprocessamento falhou: ID antigo igual ao novo."
                return {"ok": False, "mensagem": msg}

            novo_doc_info = self.db.get_documento(novo_doc_id)
            novo_status = novo_doc_info.get("status") if novo_doc_info else "desconhecido"

            self.db.log(
                "reprocessamento",
                "usuario_sistema",
                f"doc_id_antigo={documento_id}|doc_id_novo={novo_doc_id}|status={novo_status}",
            )
            return {"ok": True, "mensagem": f"Reprocessamento concluído. Novo ID: {novo_doc_id} (Status: {novo_status}).", "novo_id": novo_doc_id}
        except Exception as e:
            log.exception("Falha ao reprocessar doc_id %d: %s", documento_id, e)
            return {"ok": False, "mensagem": f"Falha ao reprocessar: {e}"}

    # --------------------------------------------------------
    # Auto-roteamento
    # --------------------------------------------------------
    def processar_automatico(self, nome: str, conteudo: bytes, origem: str = "upload") -> int:
        """Roteia automaticamente: XML -> XMLParser; caso contrário, OCR/NLP (grafo se disponível)."""
        try:
            doc_hash = self.db.hash_bytes(conteudo)
            existing_id = self.db.find_documento_by_hash(doc_hash)
            if existing_id:
                log.info("Doc '%s' (hash %s...) já existe ID %d. Ignorando.", nome, doc_hash[:8], existing_id)
                return existing_id

            head = conteudo[:2000]
            if (
                head.strip().startswith(b"<?xml")
                or b"<NFe" in head
                or b"<CTe" in head
                or b"<MDFe" in head
                or b"<CFe" in head
                or b"NFSe" in head
                or b"Nfse" in head
            ):
                log.info("Detectado XML fiscal: %s", nome)
                return self.xml_agent.processar(nome, conteudo, origem)
            else:
                log.info("Arquivo não-XML detectado (%s), enviando para OCR/NLP...", nome)
                return self.ingestir_arquivo(nome, conteudo, origem)
        except Exception as e:
            log.exception("Falha no roteamento automático '%s': %s", nome, e)
            return self.db.inserir_documento(
                nome_arquivo=nome,
                tipo="desconhecido",
                origem=origem,
                hash=self.db.hash_bytes(conteudo),
                status="erro",
                data_upload=self.db.now(),
                motivo_rejeicao=str(e),
            )
