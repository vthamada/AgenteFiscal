from __future__ import annotations
import json
import logging
from typing import Optional, TYPE_CHECKING, Dict, Any
import pandas as pd

if TYPE_CHECKING:
    from banco_de_dados import BancoDeDados
try:
    # LLM é opcional e fora do caminho crítico
    from langchain_core.language_models.chat_models import BaseChatModel  # type: ignore
    from langchain_core.messages import SystemMessage, HumanMessage  # type: ignore
except Exception:
    BaseChatModel = object  # type: ignore
    SystemMessage = object  # type: ignore
    HumanMessage = object  # type: ignore

log = logging.getLogger("agente_fiscal.agentes")


class MetricsAgent:
    """
    Agente responsável por calcular e persistir métricas de performance e qualidade.
    Também agrega dados fiscais e de volume de processamento.

    • Determinístico no caminho crítico (registrar_metrica / registrar_evento).
    • Camada cognitiva OPCIONAL (analisar_metricas_historicas / registrar_insight) – não bloqueante.
    """

    def __init__(self, llm: Optional["BaseChatModel"] = None) -> None:
        self.llm = llm if isinstance(llm, BaseChatModel) else None
        log.info("MetricsAgent inicializado (LLM=%s).", "ON" if self.llm else "OFF")

    # ---------------------- Caminho crítico (determinístico) ----------------------

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

            if df_docs.empty:
                db.inserir_metrica(
                    tipo_documento=tipo_documento,
                    acuracia_media=float(confianca_media or 0.0),
                    taxa_revisao=float(taxa_revisao),
                    taxa_erro=float(taxa_erro),
                    tempo_medio=float(tempo_medio or 0.0),
                )
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

            db.inserir_metrica(
                tipo_documento=tipo_documento,
                acuracia_media=media_conf,
                taxa_revisao=taxa_revisao,
                taxa_erro=taxa_erro,
                tempo_medio=media_tempo,
                meta_json=json.dumps(meta, ensure_ascii=False),
            )
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

    # ---------------------- Camada cognitiva (opcional) ----------------------

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
            if df.empty:
                return None

            # Reduz e serializa (pequeno) para prompt
            small = df.tail(100).copy()
            small = small[["tipo_documento", "acuracia_media", "taxa_revisao", "taxa_erro", "tempo_medio", "meta_json"]].fillna("")
            blob = small.to_dict(orient="records")

            sys = SystemMessage(content=(
                "Você é um analista de qualidade de pipelines de processamento fiscal. "
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


__all__ = ["MetricsAgent"]
