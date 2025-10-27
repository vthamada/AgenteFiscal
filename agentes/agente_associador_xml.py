# agentes/agente_associador_xml.py

from __future__ import annotations
import logging
import re
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING, List

from .utils import _only_digits, _norm_ws

# Tolerância quando pandas não está disponível
try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    class _DFMock:
        @property
        def empty(self): return True
        def __getitem__(self, _): return self
        def astype(self, *_a, **_k): return self
        def str(self): return self
        def contains(self, *_a, **_k): return self
        def map(self, *_a, **_k): return self
        def to_dict(self): return {}
        def sort_values(self, *_, **__): return self
        def iloc(self): return self
        def __getattr__(self, _): return self
        def unique(self): return []
        def fillna(self, *_a, **_k): return self
        def sum(self): return 0.0
        def get(self, *_a, **_k): return self
        def __iter__(self): return iter([])
        def tolist(self): return []
        @property
        def columns(self): return []
    pd = type("pandas", (), {"DataFrame": _DFMock, "Series": _DFMock, "to_numeric": lambda *a, **k: 0})  # type: ignore

if TYPE_CHECKING:
    from banco_de_dados import BancoDeDados  # apenas para type hints

log = logging.getLogger("agente_fiscal.agentes")


class AgenteAssociadorXML:
    """
    Associa um PDF/imagem já ingerido (via OCR) a um XML existente no banco.

    Estratégias (em ordem):
      1) chave de acesso exata (do OCR ou detectada no texto),
      2) URL/QR com chNFe/chCTe=...,
      3) fallback por (data_emissao ±0d, valor_total ± 0,01) com desempate por nome/UF,
      4) heurística por (emitente_nome normalizado) + (valor_total).

    • Prioriza documentos do tipo XML (NFe/NFCe/CTe/CF-e/NFSe).
    • Enriquecimento agora considera também 'documentos_detalhes' (se existir).
    • Persistência de campos segue com o Orchestrator; aqui atualizamos apenas a chave.
    """

    RE_QR_CHAVE = re.compile(r"(?:chNFe|chCTe)=([0-9]{44})", re.I)
    RE_CHAVE_SECA = re.compile(r"\b(\d{44})\b")

    # Campos principais que o associador pode enriquecer diretamente do XML/DB
    _CAMPOS_CORE: List[str] = [
        # Identificação básica
        "tipo", "chave_acesso", "numero_nota", "serie", "modelo", "natureza_operacao",
        # Partes
        "emitente_nome", "emitente_cnpj", "emitente_cpf",
        "destinatario_nome", "destinatario_cnpj", "destinatario_cpf",
        # Local
        "uf", "municipio", "endereco",
        # Datas e totais
        "data_emissao", "valor_total",
        "total_produtos", "total_servicos",
        "total_icms", "total_ipi", "total_pis", "total_cofins",
        # Complementos úteis
        "valor_descontos", "valor_frete", "valor_seguro", "valor_outros", "valor_liquido",
        # Transporte/Pagamento
        "modalidade_frete", "placa_veiculo", "uf_veiculo", "peso_bruto", "peso_liquido", "qtd_volumes",
        "forma_pagamento", "valor_pagamento", "troco",
        # XML / autorização
        "caminho_xml", "versao_schema", "ambiente", "protocolo_autorizacao", "data_autorizacao", "cstat", "xmotivo",
        "responsavel_tecnico",
        # Telemetria OCR
        "ocr_tipo",
    ]

    # Campos "universais" que podem vir só em detalhes (IE/IM/endereços específicos etc.)
    _CAMPOS_SUGERIDOS_DETALHE: List[str] = [
        "emitente_ie", "emitente_im", "emitente_uf", "emitente_municipio", "emitente_endereco",
        "destinatario_ie", "destinatario_im", "destinatario_uf", "destinatario_municipio", "destinatario_endereco",
    ]

    def __init__(self, db: "BancoDeDados"):
        self.db = db

    # ---------------------------- Público ----------------------------

    def tentar_associar_pdf(
        self,
        doc_id: int,
        campos_parciais: Dict[str, Any],
        texto_ocr: str = ""
    ) -> Dict[str, Any]:
        """
        Retorna campos enriquecidos do XML encontrado e atualiza o próprio documento com a chave (se aplicável).
        Inclui metadados de associação em `__assoc_meta__`.
        """
        try:
            parcial = dict(campos_parciais or {})
            chave_ocr = parcial.get("chave_acesso") or self._extrair_chave_do_texto(texto_ocr)
            valor = parcial.get("valor_total")
            data = parcial.get("data_emissao")
            nome_emit = self._safe_name(parcial.get("emitente_nome"))
            uf = (parcial.get("uf") or "").strip().upper() or None

            # 1) Por chave
            if chave_ocr:
                cand = self._procurar_por_chave(chave_ocr)
                if cand:
                    enriquecidos, meta = self._consolidar(parcial, cand, estrategia="chave", score=1.0)
                    return self._finalizar(doc_id, enriquecidos, meta)

            # 2) Fallback por (valor, data)
            if valor is not None or data:
                cand = self._procurar_por_valor_data(valor, data, preferir_xml=True)
                if cand:
                    score = 0.85
                    score += self._bonus_nome_uf(cand, nome_emit, uf)
                    enriquecidos, meta = self._consolidar(parcial, cand, estrategia="valor_data", score=min(score, 0.97))
                    return self._finalizar(doc_id, enriquecidos, meta)

            # 3) Heurística (nome normalizado + valor)
            if nome_emit and valor is not None:
                cand = self._procurar_por_nome_valor(nome_emit, valor, uf)
                if cand:
                    score = 0.75 + self._bonus_nome_uf(cand, nome_emit, uf)
                    enriquecidos, meta = self._consolidar(parcial, cand, estrategia="nome_valor", score=min(score, 0.9))
                    return self._finalizar(doc_id, enriquecidos, meta)

            # Nada encontrado
            parcial.setdefault("__assoc_meta__", {"status": "nao_associado", "score": 0.0})
            return parcial

        except Exception as e:
            log.warning(f"Associador: falha geral na associação doc_id={doc_id}: {e}")
            out = dict(campos_parciais or {})
            out.setdefault("__assoc_meta__", {"status": "erro", "score": 0.0, "motivo": str(e)})
            return out

    # ---------------------------- Internos ----------------------------

    def _bonus_nome_uf(self, row: Dict[str, Any], nome_emit_norm: Optional[str], uf: Optional[str]) -> float:
        bonus = 0.0
        if nome_emit_norm:
            cand_nome = self._safe_name(row.get("emitente_nome"))
            if cand_nome and cand_nome == nome_emit_norm:
                bonus += 0.05
        if uf:
            cand_uf = (row.get("uf") or "").strip().upper() or None
            if cand_uf and cand_uf == uf:
                bonus += 0.03
        return bonus

    def _finalizar(self, doc_id: int, enriquecidos: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
        # Atualiza apenas a chave no documento "alvo" (o restante é persistido pelo Orchestrator)
        try:
            chave = enriquecidos.get("chave_acesso")
            if chave:
                self.db.atualizar_documento_campos(doc_id, chave_acesso=chave)
        except Exception as e:
            log.debug(f"Associador: não foi possível atualizar chave do doc_id={doc_id}: {e}")
        enriquecidos["__assoc_meta__"] = meta
        return enriquecidos

    def _consolidar(self, parcial: Dict[str, Any], encontrado: Dict[str, Any], *, estrategia: str, score: float) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Enriquecimento:
        - Prioriza valores já presentes em `parcial` quando não-nulos;
        - Puxa campos do registro 'documentos' encontrado;
        - Se existir 'documentos_detalhes', usa-o para cobrir lacunas (IE/IM/endereços específicos etc.).
        """
        base = dict(parcial or {})

        # 1) Tenta carregar detalhes do documento encontrado (quando a tabela existe)
        detalhes_map = self._coletar_detalhes(encontrado.get("id"))

        # 2) Monta pacote de enriquecimento a partir do row de documentos
        enr_core: Dict[str, Any] = {}
        for k in self._CAMPOS_CORE:
            val = encontrado.get(k)
            if val not in (None, "", []):
                enr_core[k] = val

        # 3) Acrescenta campos de detalhe sugeridos (quando houver)
        for k in self._CAMPOS_SUGERIDOS_DETALHE:
            v = detalhes_map.get(k)
            if v not in (None, "", []):
                enr_core[k] = v

        # 4) Mescla (parcial vence onde já tiver informação não-nula)
        for k, v in enr_core.items():
            if base.get(k) in (None, "", []):
                base[k] = v

        meta = {
            "status": "associado",
            "por": estrategia,
            "score": float(round(score, 3)),
            "documento_alvo_id": int(encontrado.get("id", 0) or 0),
            "usou_detalhes": bool(detalhes_map),
        }
        return base, meta

    # ------------------------ Consulta ao banco ------------------------

    def _tabela_existe(self, nome: str) -> bool:
        try:
            cur = self.db.conn.cursor()
            cur.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1", (nome,))
            return cur.fetchone() is not None
        except Exception:
            return False

    def _coletar_detalhes(self, documento_id: Optional[int]) -> Dict[str, Any]:
        if not documento_id or not self._tabela_existe("documentos_detalhes"):
            return {}
        try:
            df = self.db.query_table("documentos_detalhes", where=f"documento_id = {int(documento_id)}")
            if getattr(df, "empty", True):
                return {}
            # documentos_detalhes: key, value (ambos TEXT). Convertemos em dict simples.
            out: Dict[str, Any] = {}
            if {"chave", "valor"}.issubset(set(df.columns)):  # layout chave/valor
                for _, row in df.iterrows():
                    k = str(row.get("chave") or "").strip()
                    v = row.get("valor")
                    if k:
                        out[k] = v
            else:
                # fallback genérico: usa a primeira linha como dict
                out = df.iloc[0].to_dict()
            return out
        except Exception as e:
            log.debug(f"Associador: falha ao coletar detalhes do doc_id={documento_id}: {e}")
            return {}

    def _procurar_por_chave(self, chave: Optional[str]) -> Optional[Dict[str, Any]]:
        if not chave:
            return None
        chave = _only_digits(chave)[:44]
        if len(chave) != 44:
            return None
        try:
            df = self.db.query_table("documentos", where=f"chave_acesso = '{chave}'")
            if getattr(df, "empty", True):
                return None
            # Prioriza XML
            tipo_series = df["tipo"].astype(str)
            mask_xml = tipo_series.str.contains("xml|nfe|nfce|cte|cfe|nfse", case=False, na=False)
            df_xml = df[mask_xml] if hasattr(df, "__getitem__") else df
            row = (df_xml if not getattr(df_xml, "empty", True) else df).iloc[0].to_dict()
            return row
        except Exception as e:
            log.warning(f"Associador: falha query por chave {chave}: {e}")
            return None

    def _procurar_por_valor_data(self, valor: Optional[float], data: Optional[str], *, preferir_xml: bool = True) -> Optional[Dict[str, Any]]:
        if valor is None and not data:
            return None

        clauses = []
        if data:
            data = str(data).strip().split("T")[0]
            clauses.append(f"data_emissao = '{data}'")
        if valor is not None:
            try:
                v = float(valor)
                clauses.append(f"(ABS(CAST(valor_total AS REAL) - {v:.2f}) <= 0.01 OR valor_total = {v:.2f})")
            except Exception:
                pass
        sql = " AND ".join(clauses) if clauses else "1=1"

        try:
            df = self.db.query_table("documentos", where=sql)
            if getattr(df, "empty", True):
                return None

            if preferir_xml and "tipo" in getattr(df, "columns", []):
                df = df.sort_values(
                    by=["tipo"],
                    key=lambda s: s.astype(str).str.contains("xml|nfe|nfce|cte|cfe|nfse", case=False, na=False),
                    ascending=False,
                )

            if "valor_total" in getattr(df, "columns", []) and valor is not None:
                df["_diff_valor"] = (pd.to_numeric(df["valor_total"], errors="coerce") - float(valor)).abs()
                df = df.sort_values(by=["_diff_valor"])
            row = df.iloc[0].to_dict()
            if isinstance(row, dict):
                row.pop("_diff_valor", None)
            return row
        except Exception as e:
            log.warning(f"Associador: falha query por valor/data: {e}")
            return None

    def _procurar_por_nome_valor(self, nome_norm: str, valor: float, uf: Optional[str]) -> Optional[Dict[str, Any]]:
        try:
            df = self.db.query_table(
                "documentos",
                where=f"(ABS(CAST(valor_total AS REAL) - {float(valor):.2f}) <= 0.01 OR valor_total = {float(valor):.2f})"
            )
            if getattr(df, "empty", True):
                return None

            if "emitente_nome" in getattr(df, "columns", []):
                df["__emit_norm"] = df["emitente_nome"].astype(str).map(self._safe_name)
            else:
                df["__emit_norm"] = None

            cand = df[df["__emit_norm"].astype(str).str.contains(re.escape(nome_norm), na=False)]

            if uf:
                if "uf" in getattr(cand, "columns", []):
                    uf_series = cand["uf"].astype(str).str.upper()
                else:
                    uf_series = pd.Series(index=getattr(cand, "index", []), dtype=str)  # type: ignore
                mask_uf = (uf_series == uf)
                cand = cand[mask_uf] if hasattr(cand, "__getitem__") else cand

            if getattr(cand, "empty", True):
                cand = df  # fallback

            if "tipo" in getattr(cand, "columns", []):
                mask_xml = cand["tipo"].astype(str).str.contains("xml|nfe|nfce|cte|cfe|nfse", case=False, na=False)
                cand_xml = cand[mask_xml]
            else:
                cand_xml = cand

            row_df = (cand_xml if not getattr(cand_xml, "empty", True) else cand)
            row = row_df.iloc[0].to_dict()
            if isinstance(row, dict):
                row.pop("__emit_norm", None)
            return row
        except Exception as e:
            log.debug(f"Associador: nome/valor falhou: {e}")
            return None

    # ---------------------------- Text/Regex helpers ----------------------------

    def _extrair_chave_do_texto(self, texto: str) -> Optional[str]:
        if not texto:
            return None
        m = self.RE_QR_CHAVE.search(texto)
        if m:
            return _only_digits(m.group(1))[:44]
        m2 = self.RE_CHAVE_SECA.search(texto)
        if m2:
            return _only_digits(m2.group(1))[:44]
        return None

    @staticmethod
    def _safe_name(s: Optional[str]) -> Optional[str]:
        if not s:
            return None
        s = _norm_ws(str(s)).strip().lower()
        # remove termos corporativos comuns para melhorar matching
        s = re.sub(r"\b(ltda|me|eireli|s\/a|s\.a\.|sa|epp|mei|comercial|comércio|comercio|holding|grupo)\b", "", s)
        s = re.sub(r"[^a-z0-9\s]", "", s)
        s = re.sub(r"\s+", " ", s)
        return s or None


__all__ = ["AgenteAssociadorXML"]
