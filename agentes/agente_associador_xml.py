"""Versão enxuta (XML-only) do associador: enriquece a partir de documentos XML existentes no banco."""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Optional, Tuple, List, TYPE_CHECKING

if TYPE_CHECKING:  # somente para type-checkers; evita import em runtime
    from banco_de_dados import BancoDeDados

from .utils import _only_digits, _norm_ws

log = logging.getLogger("agente_fiscal.agentes")


class AgenteAssociadorXML:
    """
    Associador simples para enriquecer um documento a partir de registros XML já existentes no banco.
    - Sem dependência de OCR/NLP. O parâmetro 'texto_ocr' é apenas um hint opcional para extrair chave via regex.
    - Estratégias:
      1) Chave de acesso (44 dígitos) exata
      2) Heurística por valor_total ±2% e data_emissao ±1 dia (preferindo registros do tipo XML)
    - Retorna campos enriquecidos e __assoc_meta__ com 'match_score' e notas.
    """

    RE_QR_CHAVE = re.compile(r"(?:chNFe|chCTe)=([0-9]{44})", re.I)
    RE_CHAVE_SECA = re.compile(r"\b(\d{44})\b")

    CAMPOS_CORE: Tuple[str, ...] = (
        "tipo", "chave_acesso", "numero_nota", "serie", "modelo", "natureza_operacao",
        "emitente_nome", "emitente_cnpj", "emitente_cpf",
        "destinatario_nome", "destinatario_cnpj", "destinatario_cpf",
        "data_emissao", "valor_total",
        "total_produtos", "total_servicos",
        "total_icms", "total_ipi", "total_pis", "total_cofins",
        "valor_descontos", "valor_frete", "valor_seguro", "valor_outros", "valor_liquido",
        "modalidade_frete", "placa_veiculo", "uf_veiculo", "peso_bruto", "peso_liquido", "qtd_volumes",
        "forma_pagamento", "valor_pagamento", "valor_troco", "condicao_pagamento", "meio_pagamento", "bandeira_cartao",
        "caminho_xml", "versao_schema", "ambiente", "protocolo_autorizacao", "data_autorizacao", "cstat", "xmotivo",
        "responsavel_tecnico",
    )

    def __init__(self, db: "BancoDeDados", *, llm: Any = None, embedder: Any = None) -> None:
        self.db = db
        self.llm = llm
        self.embedder = embedder

    # API compatível
    def tentar_associar_pdf(
        self,
        doc_id: int,
        campos_parciais: Dict[str, Any],
        texto_ocr: str = "",
    ) -> Dict[str, Any]:
        parcial = dict(campos_parciais or {})
        try:
            chave = parcial.get("chave_acesso") or self._extrair_chave(texto_ocr)
            candidato: Optional[Dict[str, Any]] = None
            estrategia = "nao_associado"
            score = 0.0

            # 1) Por chave de acesso
            if chave and len(_only_digits(chave) or "") == 44:
                candidato = self._por_chave(_only_digits(chave) or "")
                if candidato:
                    estrategia = "chave"
                    score = 1.0

            # 2) Heurística por valor/data
            if not candidato:
                val = parcial.get("valor_total")
                data = parcial.get("data_emissao")
                if val is not None or data:
                    candidato = self._por_valor_data(val, data)
                    if candidato:
                        estrategia = "valor_data"
                        score = 0.88

            if not candidato:
                parcial.setdefault("__assoc_meta__", {
                    "status": "nao_associado",
                    "por": "nenhum",
                    "score": 0.0,
                    "match_score": 0.0,
                    "divergencias": [],
                    "reprocessar_sugerido": False,
                    "motivo": "Sem candidatos por chave/valor/data.",
                })
                return parcial

            # Mesclar campos úteis
            enrich: Dict[str, Any] = {}
            for k in self.CAMPOS_CORE:
                v = candidato.get(k)
                if v not in (None, "", []):
                    enrich[k] = v

            # Completar lacunas apenas
            for k, v in enrich.items():
                if parcial.get(k) in (None, "", []):
                    parcial[k] = v

            # Atualizar chave no alvo, se houver
            try:
                if parcial.get("chave_acesso"):
                    self.db.atualizar_documento_campos(doc_id, chave_acesso=parcial["chave_acesso"])
            except Exception:
                pass

            parcial["__assoc_meta__"] = {
                "status": "associado",
                "por": estrategia,
                "score": float(round(score, 3)),
                "match_score": float(round(score, 3)),
                "divergencias": [],
                "reprocessar_sugerido": False,
                "documento_alvo_id": int(candidato.get("id") or 0),
            }
            return parcial
        except Exception as e:
            log.warning(f"Associador enxuto: falha geral doc_id={doc_id}: {e}")
            parcial.setdefault("__assoc_meta__", {"status": "erro", "score": 0.0, "match_score": 0.0, "motivo": str(e)})
            return parcial

    # ------------- internals -------------
    def _extrair_chave(self, texto: str) -> Optional[str]:
        if not texto:
            return None
        m = self.RE_QR_CHAVE.search(texto)
        if m:
            ch = _only_digits(m.group(1)) or ""
            return ch if len(ch) == 44 else None
        for match in self.RE_CHAVE_SECA.finditer(texto):
            ch = _only_digits(match.group(1)) or ""
            if len(ch) == 44:
                return ch
        return None

    def _por_chave(self, chave: str) -> Optional[Dict[str, Any]]:
        try:
            df = self.db.query_table("documentos", where=f"chave_acesso = '{chave}'")
            if getattr(df, "empty", True):
                return None
            tipo_series = df["tipo"].astype(str) if "tipo" in getattr(df, "columns", []) else None
            if tipo_series is not None:
                mask_xml = tipo_series.str.contains("xml|nfe|nfce|cte|cfe|nfse", case=False, na=False)
                df = df[mask_xml] if hasattr(df, "__getitem__") else df
            return df.iloc[0].to_dict()
        except Exception:
            return None

    def _por_valor_data(self, valor: Optional[float], data: Optional[str]) -> Optional[Dict[str, Any]]:
        clauses: List[str] = []
        if data:
            base = str(data).strip().split("T")[0]
            clauses.append(f"(data_emissao BETWEEN date('{base}', '-1 day') AND date('{base}', '+1 day'))")
        if valor is not None:
            try:
                v = float(valor)
                clauses.append(f"(ABS(CAST(valor_total AS REAL) - {v:.2f}) <= 0.02 OR valor_total = {v:.2f})")
            except Exception:
                pass
        where = " AND ".join(clauses) if clauses else None
        try:
            df = self.db.query_table("documentos", where=where)
            if getattr(df, "empty", True):
                return None
            if "tipo" in getattr(df, "columns", []):
                df = df.sort_values(
                    by=["tipo"],
                    key=lambda s: s.astype(str).str.contains("xml|nfe|nfce|cte|cfe|nfse", case=False, na=False),
                    ascending=False,
                )
            return df.iloc[0].to_dict()
        except Exception:
            return None


__all__ = ["AgenteAssociadorXML"]
