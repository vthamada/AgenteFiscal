# agentes/metrics_agent.py

from __future__ import annotations
import json
import logging
from typing import Optional, TYPE_CHECKING, Dict, Any, List, Tuple
import statistics as stats
import pandas as pd

if TYPE_CHECKING:
    from banco_de_dados import BancoDeDados
try:
    # LLM é opcional e fora do caminho crítico
    from langchain_core.language_models.chat_models import BaseChatModel  # type: ignore
    from langchain_core.messages import SystemMessage, HumanMessage       # type: ignore
except Exception:
    BaseChatModel = object  # type: ignore
    SystemMessage = object  # type: ignore
    HumanMessage = object   # type: ignore

log = logging.getLogger("agente_fiscal.agentes")


class MetricsAgent:
    """
    Camada de Autoavaliação Global do pipeline.
    - Determinístico no caminho crítico de registro de métricas.
    - Cognitivo opcional para síntese de recomendações e leitura de histórico.
    - Integração com o Orchestrator via 'avaliar_blackboard' e 'comparar_historico_e_sugerir'.
    """

    # Regras default (podem ser calibradas por YAML externo futuramente)
    DEGRADACAO_REL_PCT = 0.10        # 10% de queda aciona sugestão
    QUEDA_CONF_OCR = 0.10            # 10% de queda média em confiança OCR
    CUTOFF_NLP_COVERAGE = 0.60       # acionamento LLM NLP
    CUTOFF_NORMALIZER_SANITY = 0.80  # marca revisão
    CUTOFF_ASSOC_MATCH = 0.72        # associador com match baixo sugere reprocesso

    def __init__(self, llm: Optional["BaseChatModel"] = None) -> None:
        try:
            self.llm = llm if (llm is not None and isinstance(llm, BaseChatModel)) else None  # type: ignore
        except Exception:
            self.llm = llm if llm is not None else None
        log.info("MetricsAgent inicializado (LLM=%s).", "ON" if self.llm else "OFF")

    # ----------------------------------------------------------------------
    # Compatibilidade retroativa (métodos existentes)
    # ----------------------------------------------------------------------
    def registrar_metrica(
        self,
        db: "BancoDeDados",
        tipo_documento: str,
        status: str,
        confianca_media: float,
        tempo_medio: float,
    ) -> None:
        """
        Registra métrica agregada na tabela 'metricas' (uma linha por evento).
        Além das métricas básicas, agrega ICMS/IPI/PIS/COFINS médios e percentuais relativos ao valor_total
        a partir da amostra recente do mesmo tipo_documento.
        """
        try:
            def _mean_num(series: pd.Series) -> float:
                try:
                    return float(pd.to_numeric(series, errors="coerce").mean())
                except Exception:
                    return 0.0

            taxa_revisao = 1.0 if status == "revisao_pendente" else 0.0
            taxa_erro = 1.0 if status in ("erro", "quarentena") else 0.0

            try:
                df_docs = db.query_table("documentos", where=f"tipo = '{tipo_documento}'")
            except Exception as e_q:
                log.warning(f"MetricsAgent: falha ao consultar documentos para métricas: {e_q}")
                df_docs = pd.DataFrame()

            if getattr(df_docs, "empty", True):
                try:
                    db.inserir_metrica(
                        tipo_documento=tipo_documento,
                        acuracia_media=float(confianca_media or 0.0),
                        taxa_revisao=float(taxa_revisao),
                        taxa_erro=float(taxa_erro),
                        tempo_medio=float(tempo_medio or 0.0),
                    )
                except Exception as e_ins:
                    log.debug(f"MetricsAgent: inserir_metrica indisponível ({e_ins})")
                return

            total_docs = int(len(df_docs))
            media_conf = float(confianca_media or 0.0)
            media_tempo = float(tempo_medio or 0.0)
            media_valor_total = _mean_num(df_docs["valor_total"]) if "valor_total" in df_docs else 0.0

            media_icms = _mean_num(df_docs["total_icms"]) if "total_icms" in df_docs else 0.0
            media_ipi = _mean_num(df_docs["total_ipi"]) if "total_ipi" in df_docs else 0.0
            media_pis = _mean_num(df_docs["total_pis"]) if "total_pis" in df_docs else 0.0
            media_cofins = _mean_num(df_docs["total_cofins"]) if "total_cofins" in df_docs else 0.0

            def _pct(x: float, base: float) -> float:
                try:
                    return float((x / base) * 100.0) if base and base > 0 else 0.0
                except Exception:
                    return 0.0

            taxa_icms_media = _pct(media_icms, media_valor_total)
            taxa_ipi_media = _pct(media_ipi, media_valor_total)
            taxa_pis_media = _pct(media_pis, media_valor_total)
            taxa_cofins_media = _pct(media_cofins, media_valor_total)

            meta = {
                "total_documentos": total_docs,
                "media_valor_total": media_valor_total,
                "media_icms": media_icms,
                "media_ipi": media_ipi,
                "media_pis": media_pis,
                "media_cofins": media_cofins,
                "taxa_icms_media": taxa_icms_media,
                "taxa_ipi_media": taxa_ipi_media,
                "taxa_pis_media": taxa_pis_media,
                "taxa_cofins_media": taxa_cofins_media,
                "status_evento": status,
            }

            try:
                db.inserir_metrica(
                    tipo_documento=tipo_documento,
                    acuracia_media=media_conf,
                    taxa_revisao=taxa_revisao,
                    taxa_erro=taxa_erro,
                    tempo_medio=media_tempo,
                    meta_json=json.dumps(meta, ensure_ascii=False),
                )
            except Exception as e_ins2:
                log.debug(f"MetricsAgent: inserir_metrica indisponível ({e_ins2})")
        except Exception as e:
            log.error(f"Falha ao registrar métrica: {e}")
            return

    def registrar_evento(
        self,
        db: "BancoDeDados",
        agente: str,
        tipo_documento: str,
        status: str,
        detalhes: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Permite registrar eventos operacionais (ex.: falha de OCR, reprocessamento, retentativas).
        Não quebra pipeline em caso de erro.
        """
        try:
            db.inserir_evento(
                agente=agente,
                tipo_documento=tipo_documento,
                status=status,
                detalhes_json=json.dumps(detalhes or {}, ensure_ascii=False),
            )
        except Exception as e:
            log.debug(f"MetricsAgent: falha ao registrar evento ({agente}): {e}")

    # ---------------------- Camada cognitiva (opcional existente) ----------------------
    def analisar_metricas_historicas(
        self,
        db: "BancoDeDados",
        contexto: str,
        *, periodo_sql_where: Optional[str] = None
    ) -> Optional[str]:
        """
        Usa LLM (opcional) para interpretar padrões de métricas históricas e gerar um resumo explicativo.
        • Fora do caminho crítico; falhas aqui são silenciosas.
        """
        if not self.llm:
            return None
        try:
            where = f"tipo_documento = '{contexto}'"
            if periodo_sql_where:
                where = f"{where} AND ({periodo_sql_where})"
            df = db.query_table("metricas", where=where)
            if getattr(df, "empty", True):
                return None

            small = df.tail(100).copy()
            cols = [c for c in ["tipo_documento", "acuracia_media", "taxa_revisao", "taxa_erro", "tempo_medio", "meta_json"] if c in small.columns]
            small = small[cols].fillna("")
            blob = small.to_dict(orient="records")

            sys = SystemMessage(content=(
                "Você é um analista de qualidade de pipelines fiscais. "
                "Receberá métricas históricas (JSON) e deve sintetizar tendências, anomalias e recomendações objetivas. "
                "Seja conciso e específico. Não invente números; baseie-se no histórico fornecido."
            ))
            hum = HumanMessage(content=json.dumps({"contexto": contexto, "metricas": blob}, ensure_ascii=False))
            resumo = self.llm.invoke([sys, hum]).content  # type: ignore[attr-defined]
            return resumo.strip() if isinstance(resumo, str) else None
        except Exception as e:
            log.debug(f"MetricsAgent: análise cognitiva ignorada (erro: {e})")
            return None

    def registrar_insight(self, db: "BancoDeDados", contexto: str, insight_texto: str) -> None:
        """
        Persiste um insight textual (gerado ou não por LLM) na tabela 'metricas_contextuais'.
        Não quebra pipeline em caso de erro.
        """
        try:
            db.inserir_metrica_contextual(contexto=contexto, insight=insight_texto)
        except Exception as e:
            log.debug(f"MetricsAgent: falha ao registrar insight: {e}")

    # ----------------------------------------------------------------------
    # NOVO: Registro de desempenho por AGENTE (coverage/precision/recall)
    # ----------------------------------------------------------------------
    def registrar_desempenho_agente(
        self,
        db: "BancoDeDados",
        *,
        doc_id: int,
        agente: str,
        tipo_documento: str,
        coverage: float | None = None,
        precision: float | None = None,
        recall: float | None = None,
        confidence: float | None = None,
        latency_s: float | None = None,
        tp: int | None = None,
        fp: int | None = None,
        fn: int | None = None,
        extras: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Grava uma linha por execução de agente (por documento).
        Se a tabela 'metricas_agente' não existir, registra como evento.
        Esperado:
          - OCR: confidence (avg), coverage (proporção de texto útil), latency_s
          - NLP: coverage, precision/recall (quando comparável ao XML), latency_s
          - NORMALIZER: confidence/sanity (em 'confidence'), latency_s
          - VALIDATOR: precision/recall (quando aplicável), latency_s
          - ASSOCIADOR: match_score (em 'confidence'), latency_s
        """
        payload = dict(
            doc_id=int(doc_id),
            agente=str(agente),
            tipo_documento=str(tipo_documento),
            coverage=float(coverage) if coverage is not None else None,
            precision=float(precision) if precision is not None else None,
            recall=float(recall) if recall is not None else None,
            confidence=float(confidence) if confidence is not None else None,
            latency_s=float(latency_s) if latency_s is not None else None,
            tp=int(tp) if tp is not None else None,
            fp=int(fp) if fp is not None else None,
            fn=int(fn) if fn is not None else None,
            extras=extras or {},
        )
        try:
            db.inserir_metricas_agente(**payload)
        except Exception as e:
            # Fallback não quebrante
            log.debug(f"MetricsAgent: inserir_metricas_agente indisponível ({e}), registrando evento.")
            try:
                self.registrar_evento(
                    db=db,
                    agente=f"{agente}:metric",
                    tipo_documento=tipo_documento,
                    status="metric",
                    detalhes=payload,
                )
            except Exception:
                pass

    # ----------------------------------------------------------------------
    # NOVO: Comparar histórico e sugerir ajustes automáticos
    # ----------------------------------------------------------------------
    def comparar_historico_e_sugerir(
        self,
        db: "BancoDeDados",
        *,
        agente: str,
        tipo_documento: str,
        janela_recente: int = 100,
        janela_base: int = 300,
    ) -> Dict[str, Any]:
        """
        Compara janela RECENTE vs BASE em 'metricas_agente' e sugere ações.
        Retorna:
          {
            "agente": "...",
            "tipo_documento": "...",
            "sugestoes": [ ... ],
            "detalhes": { "recentes": {...}, "base": {...} }
          }
        """
        def _stats_df(df: pd.DataFrame) -> Dict[str, float]:
            def _m(c: str) -> float:
                if c in df.columns and not df[c].empty:
                    try:
                        s = pd.to_numeric(df[c], errors="coerce").dropna()
                        return float(s.mean()) if len(s) > 0 else 0.0
                    except Exception:
                        return 0.0
                return 0.0
            return {
                "coverage": _m("coverage"),
                "precision": _m("precision"),
                "recall": _m("recall"),
                "confidence": _m("confidence"),
                "latency_s": _m("latency_s"),
            }

        def _rel_queda(x_recent: float, x_base: float) -> float:
            if x_base <= 0:
                return 0.0
            return (x_base - x_recent) / x_base

        # Coleta dados
        try:
            df = db.query_table("metricas_agente", where=f"agente = '{agente}' AND tipo_documento = '{tipo_documento}'")
        except Exception as e:
            log.debug(f"MetricsAgent: sem tabela metricas_agente ({e})")
            return {"agente": agente, "tipo_documento": tipo_documento, "sugestoes": [], "detalhes": {}}

        if getattr(df, "empty", True):
            return {"agente": agente, "tipo_documento": tipo_documento, "sugestoes": [], "detalhes": {}}

        df_sorted = df.sort_index() if not hasattr(df, "sort_values") else df.sort_values(by=df.columns[0], axis=0, inplace=False)
        recent = df_sorted.tail(max(10, int(janela_recente)))
        base = df_sorted.tail(max(20, int(janela_base)))
        base = base.head(max(10, len(base) - len(recent))) if len(base) > len(recent) else base

        s_recent = _stats_df(recent)
        s_base = _stats_df(base)
        sugestoes: List[Dict[str, Any]] = []

        # Regras de ajustes
        if agente.lower().startswith("ocr"):
            queda_conf = _rel_queda(s_recent["confidence"], s_base["confidence"])
            if queda_conf >= self.QUEDA_CONF_OCR:
                sugestoes.append({
                    "acao": "reforcar_ocr",
                    "justificativa": f"Confiança média do OCR caiu {queda_conf:.0%} na janela recente.",
                    "opcoes": ["tesseract", "easyocr_pass_agressivo", "llm_correction_on"],
                })
            if s_recent["latency_s"] > s_base["latency_s"] * 1.25:
                sugestoes.append({
                    "acao": "otimizar_ocr",
                    "justificativa": "Latência do OCR aumentou significativamente.",
                    "opcoes": ["cache_hash_arquivo", "reduzir_resolucao_preprocess", "restringir_paginas_ruins"],
                })

        if agente.lower().startswith("nlp"):
            if s_recent["coverage"] < self.CUTOFF_NLP_COVERAGE:
                sugestoes.append({
                    "acao": "ativar_llm_nlp",
                    "justificativa": f"Coverage do NLP está baixo ({s_recent['coverage']:.2f}).",
                    "opcoes": ["llm_full_extractor", "ajustar_regex_tabela", "anti_monobloco_stronger"],
                })
            queda_prec = _rel_queda(s_recent["precision"], s_base["precision"])
            queda_rec = _rel_queda(s_recent["recall"], s_base["recall"])
            if queda_prec >= self.DEGRADACAO_REL_PCT or queda_rec >= self.DEGRADACAO_REL_PCT:
                sugestoes.append({
                    "acao": "revisar_extrator_nlp",
                    "justificativa": f"Precision/Recall caíram (Δprec={queda_prec:.0%}, Δrec={queda_rec:.0%}).",
                    "opcoes": ["retrain_prompts", "ajustar_fusao_heuristica_llm"],
                })

        if agente.lower().startswith("normalizador") or agente.lower().startswith("normalizer"):
            if s_recent["confidence"] < self.CUTOFF_NORMALIZER_SANITY:
                sugestoes.append({
                    "acao": "revisar_normalizador",
                    "justificativa": f"Sanidade/Confiança do normalizador está baixa ({s_recent['confidence']:.2f}).",
                    "opcoes": ["ativar_llm_refinamento_textual", "regras_cfop_uf", "nao_copiar_ie_vazia"],
                })

        if agente.lower().startswith("associador"):
            if s_recent["confidence"] < self.CUTOFF_ASSOC_MATCH:
                sugestoes.append({
                    "acao": "reforcar_associacao",
                    "justificativa": f"Match_score médio do associador está baixo ({s_recent['confidence']:.2f}).",
                    "opcoes": ["reprocessar_nlp", "usar_embeddings", "ajustar_pesos_campos"],
                })

        if agente.lower().startswith("validador"):
            queda_prec = _rel_queda(s_recent["precision"], s_base["precision"])
            if queda_prec >= self.DEGRADACAO_REL_PCT:
                sugestoes.append({
                    "acao": "revisar_regras_validador",
                    "justificativa": f"Precision do validador caiu {queda_prec:.0%}.",
                    "opcoes": ["ajustar_tolerancias_totais", "regras_cfop_uf_chave", "checar_excessos_false_positive"],
                })

        detalhes = {"recentes": s_recent, "base": s_base}
        return {"agente": agente, "tipo_documento": tipo_documento, "sugestoes": sugestoes, "detalhes": detalhes}

    # ----------------------------------------------------------------------
    # NOVO: Avaliar Blackboard e orientar o Orchestrator
    # ----------------------------------------------------------------------
    def avaliar_blackboard(self, blackboard: Dict[str, Any]) -> Dict[str, Any]:
        """
        Lê o snapshot do blackboard e devolve recomendações operacionais.
        Espera chaves: 'ocr', 'nlp', 'normalizer', 'validator', 'associador' (quando disponíveis),
        cada uma com {'data': ..., '__meta__': {...}} conforme contratos definidos.
        """
        recs: List[Dict[str, Any]] = []
        meta: Dict[str, Any] = {}

        try:
            ocr_meta = ((blackboard.get("ocr") or {}).get("__meta__") or {})
            nlp_meta = ((blackboard.get("nlp") or {}).get("__meta__") or {})
            norm_meta = ((blackboard.get("normalizer") or {}).get("__meta__") or {})
            val_meta = ((blackboard.get("validator") or {}).get("__meta__") or {})
            assoc_meta = ((blackboard.get("associador") or {}).get("__meta__") or (blackboard.get("assoc") or {}).get("__assoc_meta__") or {})

            # OCR
            ocr_conf = float(ocr_meta.get("avg_confidence") or ocr_meta.get("confidence") or 0.0)
            if ocr_conf < 0.70:
                recs.append({
                    "acao": "reprocessar_ocr",
                    "motivo": f"confiança OCR={ocr_conf:.2f} < 0.70",
                    "sugestoes": ["modo_agressivo", "llm_correction_on", "tesseract_fallback"]
                })

            # NLP
            nlp_cov = float(nlp_meta.get("coverage") or 0.0)
            if nlp_cov < self.CUTOFF_NLP_COVERAGE:
                recs.append({
                    "acao": "ativar_llm_nlp",
                    "motivo": f"coverage NLP={nlp_cov:.2f} < {self.CUTOFF_NLP_COVERAGE}",
                    "sugestoes": ["full_extractor", "reforcar_regex_tabela"]
                })

            # Normalizer (sanity/score)
            sanity = float(norm_meta.get("sanity_score") or norm_meta.get("confidence") or 0.0)
            if sanity and sanity < self.CUTOFF_NORMALIZER_SANITY:
                recs.append({
                    "acao": "revisar_normalizador",
                    "motivo": f"sanity={sanity:.2f} < {self.CUTOFF_NORMALIZER_SANITY}",
                    "sugestoes": ["llm_refinamento_textual", "regras_cfop_uf", "nao_copiar_ie_vazia"]
                })

            # Associador
            match_score = float(assoc_meta.get("match_score") or assoc_meta.get("score") or 0.0)
            if match_score and match_score < self.CUTOFF_ASSOC_MATCH:
                recs.append({
                    "acao": "reforcar_associacao",
                    "motivo": f"match_score={match_score:.2f} < {self.CUTOFF_ASSOC_MATCH}",
                    "sugestoes": ["reprocessar_nlp", "usar_embeddings", "ajustar_pesos_campos"]
                })

            # Validador: erros estruturais sugerem reprocessar
            if isinstance(val_meta.get("erros_estruturados"), list) and len(val_meta["erros_estruturados"]) > 0:
                recs.append({
                    "acao": "reprocessar_documento",
                    "motivo": "validador reportou erros estruturais",
                    "sugestoes": ["forcar_llm_ocr_nlp", "buscar_xml_oficial", "revisao_humana"]
                })

            meta["ok"] = True
        except Exception as e:
            meta["ok"] = False
            meta["erro"] = str(e)

        return {"recomendacoes": recs, "__meta__": meta}

    # ----------------------------------------------------------------------
    # NOVO: Síntese textual (LLM opcional) das recomendações
    # ----------------------------------------------------------------------
    def sintetizar_recomendacoes(self, recs: List[Dict[str, Any]]) -> Optional[str]:
        """
        Se houver LLM, gera um parágrafo curto explicando as recomendações.
        """
        if not self.llm or not recs:
            return None
        try:
            sys = SystemMessage(content=(
                "Você é um assistente de operação. Dado um conjunto de recomendações técnicas, "
                "faça um resumo objetivo (máx. 4 linhas), citando os principais gatilhos e ações sugeridas."
            ))
            hum = HumanMessage(content=json.dumps({"recomendacoes": recs}, ensure_ascii=False))
            txt = self.llm.invoke([sys, hum]).content  # type: ignore[attr-defined]
            return str(txt).strip() if txt else None
        except Exception as e:
            log.debug(f"MetricsAgent: falha ao sintetizar recomendações (ignorado): {e}")
            return None


__all__ = ["MetricsAgent"]
