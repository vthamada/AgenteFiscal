# agentes/agente_normalizador.py

from __future__ import annotations
import re
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .utils import (
    _parse_date_like, _to_float_br, _only_digits, _norm_ws,
    _safe_title, _clamp, _UF_SET
)

# LLM opcional (somente para refino de textos em casos ambíguos)
try:
    from langchain_core.language_models.chat_models import BaseChatModel  # type: ignore
except Exception:  # pragma: no cover
    BaseChatModel = object  # type: ignore

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel as _T  # type: ignore


class AgenteNormalizadorCampos:
    """
    Normaliza e consolida campos vindos de OCR/NLP/LLM/XML antes de persistir.

    - Determinístico por padrão (datas, moedas, percentuais, CNPJ/CPF, UF/município, nomes).
    - Consolidação de aliases para os **canônicos do banco**:
        * Totais de produtos/serviços/impostos: total_*
        * Agregados financeiros: valor_total, valor_iss, valor_descontos, valor_outros, valor_frete
    - Hook LLM OPCIONAL para refino de textos (corrigir caixa/acentuação e espaços)
      sem alterar semântica fiscal.
    """

    RE_CIDADE_UF = re.compile(r"\b([A-Za-zÀ-ÿ'`^~\-.\s]{2,})\s*/\s*([A-Za-z]{2})\b")

    # Pares de aliases onde a primeira chave é a **CANÔNICA** (do DB)
    _ALIASES_NUMERICOS = [
        # Agregado global
        ("valor_total", ("total_documento", "vtotal", "valor_nota")),

        # Totais do DB (canônicos total_*)
        ("total_produtos", ("valor_produtos", "valor_dos_produtos", "vprod")),
        ("total_servicos", ("valor_servicos", "valor_dos_servicos", "vserv")),
        ("total_icms", ("valor_icms", "vicms")),
        ("total_ipi", ("valor_ipi", "vipi")),
        ("total_pis", ("valor_pis", "vpis")),
        ("total_cofins", ("valor_cofins", "vcofins")),

        # Outros agregados canônicos (valor_*)
        ("valor_iss", ("total_iss", "viss")),
        ("valor_descontos", ("desconto_total", "desconto", "vdesc")),
        ("valor_outros", ("outras_despesas", "outras_desp", "voutros")),
        ("valor_frete", ("frete", "vfrete")),
    ]

    # Espelhos legados (preenchem chaves antigas sem sobrescrever)
    _ESPELHOS_LEGADOS = [
        ("valor_produtos", "total_produtos"),
        ("valor_servicos", "total_servicos"),
        ("valor_icms", "total_icms"),
        ("valor_ipi", "total_ipi"),
        ("valor_pis", "total_pis"),
        ("valor_cofins", "total_cofins"),
        ("desconto_total", "valor_descontos"),
        ("outras_despesas", "valor_outros"),
        ("frete", "valor_frete"),
        ("total_iss", "valor_iss"),
    ]

    def __init__(self, llm: Optional["BaseChatModel"] = None, *, enable_llm: bool = False):
        self.llm = llm if enable_llm and isinstance(llm, BaseChatModel) else None
        self.enable_llm = bool(self.llm is not None)

    # ---------------------------- Público ----------------------------

    def normalizar(self, campos: Dict[str, Any]) -> Dict[str, Any]:
        """
        - Consolida aliases numéricos para os canônicos do DB.
        - Normaliza datas, numéricos, UF/município, nomes e chave de acesso.
        """
        out = dict(campos or {})

        # 0) Consolida aliases NUMÉRICOS em campos canônicos (DB)
        self._consolidar_aliases_numericos(out)

        # 1) Datas
        for k in ("data_emissao", "data_saida", "data_recebimento", "competencia"):
            if out.get(k):
                out[k] = _parse_date_like(str(out.get(k)))

        # 2) Valores numéricos/financeiros (nos canônicos do DB)
        campos_valor = {
            "valor_total",
            "total_produtos", "total_servicos",
            "total_icms", "total_ipi", "total_pis", "total_cofins",
            "valor_iss", "valor_descontos", "valor_outros", "valor_frete",
        }
        for k in list(out.keys()):
            if k in campos_valor and out.get(k) is not None:
                out[k] = _to_float_br(str(out.get(k)))

        # 3) CNPJ/CPF (apenas dígitos)
        for k in ("emitente_cnpj", "destinatario_cnpj", "emitente_cpf", "destinatario_cpf"):
            if out.get(k):
                out[k] = _only_digits(out.get(k))

        # 4) UF
        uf = (out.get("uf") or "").strip().upper() or None
        if uf and uf not in _UF_SET:
            uf = None
        out["uf"] = uf

        # 5) Município (remove "/UF" embutido e aproveita UF se necessário)
        mun = out.get("municipio")
        if mun:
            mun = _norm_ws(str(mun))
            m = self.RE_CIDADE_UF.search(mun)
            if m:
                mun_norm = _norm_ws(m.group(1))
                uf2 = (m.group(2) or "").upper()
                if uf is None and uf2 in _UF_SET:
                    out["uf"] = uf2
                mun = mun_norm
            out["municipio"] = mun

        # 6) Nomes (emitente/destinatário)
        for k in ("emitente_nome", "destinatario_nome"):
            if out.get(k):
                out[k] = _safe_title(str(out.get(k)))

        # 7) Alíquotas em percentuais, se presentes (0..100)
        for k in ("aliquota_icms", "aliquota_ipi", "aliquota_pis", "aliquota_cofins"):
            if out.get(k) is not None:
                out[k] = _clamp(_to_float_br(str(out.get(k))), 0, 100)

        # 8) Chave de acesso (somente dígitos, exatamente 44)
        if out.get("chave_acesso"):
            ch = _only_digits(out["chave_acesso"]) or ""
            out["chave_acesso"] = (ch[:44] if len(ch) >= 44 else None)

        # 9) Campos textuais — refino (LLM opcional)
        if self.enable_llm:
            out = self._llm_refine_text_fields(out, fields=("emitente_nome", "destinatario_nome", "endereco", "municipio"))

        # 10) Preencher espelhos legados para consumidores antigos (sem sobrescrever)
        self._preencher_espelhos_legados(out)

        return out

    def normalizar_itens(self, itens: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Normaliza lista de itens (quando disponível): quantidades, valores unitários e totais,
        bem como NCM/CFOP/CEST/código só dígitos, unidade em caixa alta e descrição sem espaços redundantes.
        """
        norm_itens: List[Dict[str, Any]] = []
        if not itens:
            return norm_itens
        for it in itens:
            d = dict(it or {})
            d["descricao"] = _norm_ws(str(d.get("descricao") or "")).strip() or None
            d["unidade"] = (str(d.get("unidade") or "").strip().upper() or None)
            for kk in ("quantidade", "valor_unitario", "valor_total"):
                if d.get(kk) is not None:
                    d[kk] = _to_float_br(str(d.get(kk)))
            for kk in ("ncm", "cfop", "cest", "codigo_produto"):
                if d.get(kk):
                    d[kk] = _only_digits(str(d.get(kk)))
            norm_itens.append(d)
        return norm_itens

    def fundir(self, *fontes: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fusão por prioridade: fontes mais à direita têm prioridade por campo não-nulo.
        Ex.: fundir(nlp, llm, xml_campos) => xml_campos > llm > nlp.
        Após a fusão, consolida aliases numéricos para garantir **canônicos do DB**.
        """
        out: Dict[str, Any] = {}
        for src in fontes:
            if not src:
                continue
            for k, v in src.items():
                if v in (None, "", [], {}):
                    continue
                out[k] = v
        # garantir campos canônicos do DB após a fusão
        self._consolidar_aliases_numericos(out)
        # preencher espelhos legados (não interfere no DB por causa do filtro do orchestrator)
        self._preencher_espelhos_legados(out)
        return out

    # ----------------------- Consolidação de aliases -----------------------

    def _consolidar_aliases_numericos(self, out: Dict[str, Any]) -> None:
        """
        Define sempre os campos **canônicos do DB** quando existirem sinônimos.
        Regra: se o canônico já tiver valor, mantém; senão, usa o primeiro alias não-nulo.
        """
        for canonico, aliases in self._ALIASES_NUMERICOS:
            if out.get(canonico) not in (None, "", []):
                continue
            for alias in aliases:
                if out.get(alias) not in (None, "", []):
                    out[canonico] = out.get(alias)
                    break

    def _preencher_espelhos_legados(self, out: Dict[str, Any]) -> None:
        """
        Preenche chaves antigas úteis a consumidores legados, sem sobrescrever se já existem.
        (O orchestrator ignora-as na persistência por não estarem no schema.)
        """
        for legacy_key, canonical_key in self._ESPELHOS_LEGADOS:
            if out.get(legacy_key) in (None, "", []):
                if out.get(canonical_key) not in (None, "", []):
                    out[legacy_key] = out.get(canonical_key)

    # ----------------------- LLM (opcional) -----------------------

    def _llm_refine_text_fields(self, campos: Dict[str, Any], *, fields: tuple[str, ...]) -> Dict[str, Any]:
        """
        Usa LLM (quando configurada) para pequenos ajustes de formatação em campos textuais,
        preservando semântica fiscal. Não inventa dados.
        """
        if not self.llm:
            return campos

        payload = {k: campos.get(k) for k in fields if campos.get(k)}
        if not payload:
            return campos

        try:
            from langchain_core.messages import SystemMessage, HumanMessage  # type: ignore
            sys = SystemMessage(content=(
                "Você é um assistente que NORMALIZA textos curtos de documentos fiscais brasileiros.\n"
                "Regras:\n"
                "- Corrija apenas caixa/acentuação e espaços duplicados.\n"
                "- NÃO adicione, remova ou invente termos.\n"
                "- Mantenha siglas (UF, CFOP, NCM) intactas.\n"
                "- Responda em JSON com as MESMAS chaves recebidas."
            ))
            hum = HumanMessage(content=str(payload))
            resp = self.llm.invoke([sys, hum]).content  # type: ignore[attr-defined]
            import json
            data = json.loads(resp)
            if isinstance(data, dict):
                for k, v in data.items():
                    if k in fields and isinstance(v, str) and v.strip():
                        campos[k] = _norm_ws(v)
        except Exception:
            # qualquer problema: segue determinístico
            pass
        return campos


__all__ = ["AgenteNormalizadorCampos"]
