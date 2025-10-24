# agentes/associador.py

from __future__ import annotations
import logging
import re
from typing import Any, Dict, Optional 
from .utils import _only_digits

log = logging.getLogger("projeto_fiscal.agentes")

class AgenteAssociadorXML:
    """
    Associa um PDF/imagem já ingerido (via OCR) a um XML existente no banco.
    Estratégias: chave de acesso no OCR, URL QRCode com chNFe=..., e fallback por (data, valor, nome).
    """

    RE_QR_CHAVE = re.compile(r"(?:chNFe|chCTe)=([0-9]{44})")
    RE_CHAVE_SECA = re.compile(r"\b(\d{44})\b")

    def __init__(self, db: "BancoDeDados", cofre: "Cofre"):
        self.db = db
        self.cofre = None 

    def _procurar_por_chave(self, chave: Optional[str]) -> Optional[Dict[str, Any]]:
        if not chave:
            return None
        chave = _only_digits(chave)
        try:
            df = self.db.query_table("documentos", where=f"chave_acesso = '{chave}'")
            if not df.empty:
                # prioriza documentos XML
                df2 = df[df["tipo"].astype(str).str.contains("xml", case=False, na=False)]
                row = (df2 if not df2.empty else df).iloc[0].to_dict()
                return row
        except Exception as e:
            log.warning(f"Associador: falha query por chave {chave}: {e}")
        return None

    def _procurar_por_valor_data(self, valor: Optional[float], data: Optional[str]) -> Optional[Dict[str, Any]]:
        if valor is None and not data:
            return None
        where = []
        if data:
            where.append(f"data_emissao = '{data}'")
        if valor is not None:
            where.append(f"(ABS(CAST(valor_total AS REAL) - {float(valor):.2f}) <= 0.01 OR valor_total = {float(valor):.2f})")
        sql = " AND ".join(where) if where else "1=1"
        try:
            df = self.db.query_table("documentos", where=sql)
            if not df.empty:
                df_xml = df[df["tipo"].astype(str).str.contains("xml|NFe|NFCe|NFSe|CTe|CF-e", case=False, na=False)]
                row = (df_xml if not df_xml.empty else df).iloc[0].to_dict()
                return row
        except Exception as e:
            log.warning(f"Associador: falha query por valor/data: {e}")
        return None

    def _extrair_chave_do_texto(self, texto: str) -> Optional[str]:
        if not texto:
            return None
        m = self.RE_QR_CHAVE.search(texto)
        if m:
            return _only_digits(m.group(1))
        m2 = self.RE_CHAVE_SECA.search(texto)
        if m2:
            return _only_digits(m2.group(1))
        return None

    def tentar_associar_pdf(self, doc_id: int, campos_parciais: Dict[str, Any], texto_ocr: str = "") -> Dict[str, Any]:
        """
        Retorna campos enriquecidos do XML encontrado e atualiza o próprio documento com a chave se for o caso.
        """
        chave = campos_parciais.get("chave_acesso") or self._extrair_chave_do_texto(texto_ocr)
        candidato = self._procurar_por_chave(chave) if chave else None
        if not candidato:
            candidato = self._procurar_por_valor_data(campos_parciais.get("valor_total"), campos_parciais.get("data_emissao"))

        if candidato:
            try:
                if chave and not campos_parciais.get("chave_acesso"):
                    campos_parciais["chave_acesso"] = chave
                enriquecidos = {
                    "chave_acesso": candidato.get("chave_acesso") or campos_parciais.get("chave_acesso"),
                    "tipo": candidato.get("tipo"),
                    "emitente_nome": candidato.get("emitente_nome") or campos_parciais.get("emitente_nome"),
                    "destinatario_nome": candidato.get("destinatario_nome") or campos_parciais.get("destinatario_nome"),
                    "uf": candidato.get("uf") or campos_parciais.get("uf"),
                    "municipio": candidato.get("municipio") or campos_parciais.get("municipio"),
                }
                return {**campos_parciais, **{k: v for k, v in enriquecidos.items() if v}}
            except Exception as e:
                log.warning(f"Associador: falha ao enriquecer: {e}")
        return campos_parciais


__all__ = ["AgenteAssociadorXML"]
