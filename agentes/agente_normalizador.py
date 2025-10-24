# agentes/normalizador.py

from __future__ import annotations
import re
from typing import Any, Dict

from .utils import (
    _parse_date_like, _to_float_br, _only_digits, _norm_ws,
    _safe_title, _clamp, _UF_SET
)

class AgenteNormalizadorCampos:
    """
    Normaliza e consolida campos vindos de OCR/NLP/LLM/XML antes de persistir.
    """
    RE_CIDADE_UF = re.compile(r"\b([A-Za-zÀ-ÿ'`^~\-.\s]{2,})\s*/\s*([A-Za-z]{2})\b")

    def normalizar(self, campos: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(campos or {})

        # Datas
        for k in ("data_emissao", "data_saida", "data_recebimento", "competencia"):
            if out.get(k):
                out[k] = _parse_date_like(str(out.get(k)))

        # Valores
        for k in ("valor_total", "total_produtos", "total_servicos",
                  "total_icms", "total_ipi", "total_pis", "total_cofins"):
            if out.get(k) is not None:
                out[k] = _to_float_br(str(out.get(k)))

        # CNPJ/CPF (apenas dígitos aqui; criptografia é função do Orchestrator)
        for k in ("emitente_cnpj", "destinatario_cnpj", "emitente_cpf", "destinatario_cpf"):
            if out.get(k):
                out[k] = _only_digits(out.get(k))

        # UF
        uf = (out.get("uf") or "").strip().upper() or None
        if uf and uf not in _UF_SET:
            uf = None
        out["uf"] = uf

        # Município (remove "/UF" embutido e aproveita UF se necessário)
        mun = out.get("municipio")
        if mun:
            mun = _norm_ws(mun)
            m = self.RE_CIDADE_UF.search(mun)
            if m:
                mun, uf2 = _norm_ws(m.group(1)), (m.group(2) or "").upper()
                if uf is None and uf2 in _UF_SET:
                    out["uf"] = uf2
            out["municipio"] = mun

        # Nomes
        for k in ("emitente_nome", "destinatario_nome"):
            if out.get(k):
                out[k] = _safe_title(out.get(k))

        # Alíquotas (se vierem em %)
        for k in ("aliquota_icms", "aliquota_ipi", "aliquota_pis", "aliquota_cofins"):
            if out.get(k) is not None:
                out[k] = _clamp(_to_float_br(str(out.get(k))), 0, 100)

        # Chave de acesso (somente dígitos, 44)
        if out.get("chave_acesso"):
            out["chave_acesso"] = (_only_digits(out["chave_acesso"]) or "")[:44] or None

        return out

    def fundir(self, *fontes: Dict[str, Any]) -> Dict[str, Any]:
        """
            Fusão simples por prioridade: fontes mais à direita têm prioridade por campo não-nulo.
            Ex.: fundir(nlp, llm, xml_campos) => xml_campos > llm > nlp.
        """
        out: Dict[str, Any] = {}
        for src in fontes:
            if not src:
                continue
            for k, v in src.items():
                if v in (None, "", [], {}):
                    continue
                out[k] = v
        return out


__all__ = ["AgenteNormalizadorCampos"]
