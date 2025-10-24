# agentes/metrics.py

from __future__ import annotations
import json
import logging
from typing import TYPE_CHECKING
import pandas as pd

if TYPE_CHECKING:
    from banco_de_dados import BancoDeDados

log = logging.getLogger(__name__)


class MetricsAgent:
    """
    Agente responsável por calcular e persistir métricas de performance e qualidade.
    Também agrega dados fiscais e de volume de processamento.
    """

    def __init__(self) -> None:
        log.info("MetricsAgent inicializado.")

    def registrar_metrica(
        self,
        db: "BancoDeDados",
        tipo_documento: str,
        status: str,
        confianca_media: float,
        tempo_medio: float,
    ) -> None:
        """
        Registra ou atualiza métricas agregadas na tabela 'metricas'.
        Além das métricas de acurácia/tempo, agrega ICMS/IPI/PIS/COFINS médios e
        percentuais relativos ao valor_total.
        """
        try:
            def _mean_num(series: pd.Series) -> float:
                try:
                    return float(pd.to_numeric(series, errors="coerce").mean())
                except Exception:
                    return 0.0

            # 1) Taxas básicas a partir do status da execução atual
            taxa_revisao = 1.0 if status == "revisao_pendente" else 0.0
            taxa_erro = 1.0 if status in ("erro", "quarentena") else 0.0

            # 2) Pega amostra recente do mesmo tipo para enriquecer agregados
            try:
                df_docs = db.query_table("documentos", where=f"tipo = '{tipo_documento}'")
            except Exception as e_q:
                log.warning(f"MetricsAgent: falha ao consultar documentos para métricas: {e_q}")
                df_docs = pd.DataFrame()

            if df_docs.empty:
                # Nenhum documento do tipo ainda — registra métrica simples
                db.inserir_metrica(
                    tipo_documento=tipo_documento,
                    acuracia_media=float(confianca_media or 0.0),
                    taxa_revisao=float(taxa_revisao),
                    taxa_erro=float(taxa_erro),
                    tempo_medio=float(tempo_medio or 0.0),
                )
                log.debug(
                    "Métrica simples registrada: tipo=%s, status=%s", tipo_documento, status
                )
                return

            # 3) Agregados principais
            total_docs = int(len(df_docs))
            media_conf = float(confianca_media or 0.0)
            media_tempo = float(tempo_medio or 0.0)
            media_valor_total = _mean_num(df_docs["valor_total"]) if "valor_total" in df_docs else 0.0

            # 4) Agregados fiscais
            media_icms = _mean_num(df_docs["total_icms"]) if "total_icms" in df_docs else 0.0
            media_ipi = _mean_num(df_docs["total_ipi"]) if "total_ipi" in df_docs else 0.0
            media_pis = _mean_num(df_docs["total_pis"]) if "total_pis" in df_docs else 0.0
            media_cofins = _mean_num(df_docs["total_cofins"]) if "total_cofins" in df_docs else 0.0

            # 5) Percentuais sobre o valor_total
            def _pct(x: float, base: float) -> float:
                try:
                    return float((x / base) * 100.0) if base and base > 0 else 0.0
                except Exception:
                    return 0.0

            taxa_icms_media = _pct(media_icms, media_valor_total)
            taxa_ipi_media = _pct(media_ipi, media_valor_total)
            taxa_pis_media = _pct(media_pis, media_valor_total)
            taxa_cofins_media = _pct(media_cofins, media_valor_total)

            # 6) Campo meta_json com KPIs detalhados
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
            }

            # 7) Persiste a linha de métrica
            db.inserir_metrica(
                tipo_documento=tipo_documento,
                acuracia_media=media_conf,
                taxa_revisao=taxa_revisao,
                taxa_erro=taxa_erro,
                tempo_medio=media_tempo,
                meta_json=json.dumps(meta, ensure_ascii=False),
            )

            log.debug(
                "Métrica registrada: tipo=%s, conf=%.2f, tempo=%.2fs, ICMS=%.2f, PIS=%.2f",
                tipo_documento,
                media_conf,
                media_tempo,
                media_icms,
                media_pis,
            )

        except Exception as e:
            log.error(f"Falha ao registrar métrica: {e}")
            # Evita quebrar o pipeline principal
            return


__all__ = ["MetricsAgent"]
