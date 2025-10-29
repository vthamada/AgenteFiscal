# agentes/agente_nlp.py
from __future__ import annotations
import json
import logging
import re
import hashlib
from typing import Any, Dict, List, Optional, Tuple

# Utilitários locais (mantidos)
from .utils import (
    _parse_date_like, _to_float_br, _only_digits, _norm_ws, textual_truncate
)

log = logging.getLogger("agente_fiscal.agentes")

# =================== Prompts/LLM (centralizados) ===================
try:
    # Wrappers corretos (modelos_llm.py) — todos retornam dict {"content","json","meta","raw"}
    from modelos_llm import (
        extract_header_json,     # (llm, schema_keys, text, extra_instructions=None, ...)
        extract_items_json,      # idem
        extract_taxes_json,      # idem
        get_llm_identity,        # identidade do modelo
    )
except Exception:  # compat quando wrappers não existem no ambiente
    extract_header_json = None
    extract_items_json = None
    extract_taxes_json = None
    get_llm_identity = None

try:
    from langchain_core.language_models.chat_models import BaseChatModel  # type: ignore
    from langchain_core.messages import SystemMessage, HumanMessage       # type: ignore
except Exception:  # pragma: no cover
    BaseChatModel = object  # type: ignore
    SystemMessage = object  # type: ignore
    HumanMessage = object   # type: ignore


# ============================== Pré-processo ==============================
_LABELS_FISCAIS = [
    r"CNPJ", r"CPF", r"IE", r"I\.E\.", r"Inscri[çc][aã]o\s+Estadual",
    r"Endere[çc]o", r"Logradouro", r"Rua", r"Avenida", r"Bairro",
    r"Munic[ií]pio", r"Cidade", r"UF", r"CEP",
    r"Data\s+de\s+Emiss[aã]o", r"Emiss[aã]o", r"Compet[êe]ncia",
    r"Chave\s+de\s+Acesso", r"chNFe", r"chCTe",
    r"Valor\s+Total", r"Total\s+da\s+Nota", r"Valor\s+L[ií]quido", r"Valor\s+a\s*Pagar",
    r"N[ºo°]\s*Nota", r"N[úu]mero\s*da\s*Nota", r"S[ée]rie", r"Modelo",
    r"Natureza\s+da\s+Opera[çc][aã]o", r"Forma\s+de\s+Pagamento",
    r"Emitente", r"Remetente", r"Prestador", r"Destinat[áa]rio", r"Tomador",
]

def preprocess_text(texto: str) -> str:
    if not texto:
        return ""
    t = texto
    t = t.replace("\u00A0", " ").replace("\u202F", " ").replace("\u2009", " ")
    t = t.replace("\u200A", " ").replace("\u200B", "").replace("\ufeff", "")
    t = t.replace("\t", " ").replace("\r", "\n")
    t = re.sub(r"\n{3,}", "\n\n", t)
    t = re.sub(r"[ \t]{2,}", " ", t)
    pattern_labels = r"\s*(" + r"|".join(_LABELS_FISCAIS) + r")\s*[:\-]"
    t = re.sub(pattern_labels, r"\n\1: ", t, flags=re.I)
    t = re.sub(r"\s*:\s*", ": ", t)
    t = re.sub(r"(?<=\D)(?=\d)", " ", t)
    t = re.sub(r"(?<=\d)(?=\D)", " ", t)
    t = re.sub(r"[ ]{2,}", " ", t)
    linhas = [ln.strip() for ln in t.split("\n")]
    deduped: List[str] = []
    prev = None
    for ln in linhas:
        if ln != prev:
            deduped.append(ln)
        prev = ln
    return "\n".join(deduped).strip()


# ============================== Agente NLP ==============================
class AgenteNLP:
    """
    Agente de Interpretação Cognitiva:
      • Heurística determinística (regex + contexto de seção)
      • LLM condicional (gating por coverage e chaves críticas)
      • Fusão cognitiva (heurística ↔ LLM) com explicabilidade por campo
      • Memória leve por layout (layout_hash)
      • __meta__ rico e consistente com o projeto
    """

    # ---------- padrões determinísticos ----------
    RE_CNPJ = re.compile(r"\b(\d{2}\.?\d{3}\.?\d{3}/?\d{4}-?\d{2}|\d{14})\b")
    RE_CPF  = re.compile(r"\b(\d{3}\.?\d{3}\.?\d{3}-?\d{2}|\d{11})\b")
    RE_IE   = re.compile(r"\b(?:IE|I\.E\.|INSC(?:RI[ÇC][ÃA]O)?\s*ESTADUAL)[:\s\-]*([A-Z0-9.\-/]{5,20})\b", re.I)
    RE_IM   = re.compile(r"\b(?:IM|INSCRI[ÇC][ÃA]O\s*MUNICIPAL)[:\s\-]*([A-Z0-9.\-/]{3,20})\b", re.I)
    RE_UF   = re.compile(r"\b(AC|AL|AP|AM|BA|CE|DF|ES|GO|MA|MT|MS|MG|PA|PB|PR|PE|PI|RJ|RN|RS|RO|RR|SC|SP|SE|TO)\b")
    RE_CEP  = re.compile(r"\b\d{5}-?\d{3}\b")

    RE_VALOR_TOTAL = re.compile(
        r"(?:VALO[R]?|TOTAL)\s*(?:DA|DO)?\s*(?:NOTA|NF|DOCUMENTO)?|VALOR\s*L[IÍ]QUIDO|VALOR\s*A\s*PAGAR",
        re.I,
    )
    RE_NUM_VALOR = re.compile(r"(?:R\$\s*)?([\d\.\s]*,\d{2})")

    RE_DATA_EMISSAO = re.compile(
        r"\b(?:DATA\s*(?:DE)?\s*EMISS[ÃA]O|EMISS[ÃA]O|EMITIDO\s*EM|COMPET[ÊE]NCIA)\s*[:\-]?\s*([0-9]{1,4}[./-][0-9]{1,2}[./-][0-9]{1,4})\b",
        re.I,
    )

    RE_NUMERO = re.compile(r"\b(?:N[ºo°]|NÚMERO|NUMERO)\s*[:\-]?\s*([0-9]{1,10})\b", re.I)
    RE_SERIE  = re.compile(r"\b(?:S[EÉ]RIE|SERIE)\s*[:\-]?\s*([A-Z0-9]{1,6})\b", re.I)
    RE_MODELO = re.compile(r"\b(?:MODELO)\s*[:\-]?\s*([0-9A-Z\-]{1,10})\b", re.I)

    RE_NATUREZA = re.compile(r"\b(?:NATUREZA\s+DA\s+OPERA[ÇC][ÃA]O|NATUREZA\s+DE\s+OPERA[ÇC][ÃA]O)\s*[:\-]?\s*(.+)", re.I)
    RE_FORMA_PGTO = re.compile(r"\b(?:FORMA\s+DE\s+PAGAMENTO|FORMA\s+PAGAMENTO|PAGAMENTO)\s*[:\-]?\s*([A-Za-zÀ-ÿ0-9\s/\-]+)", re.I)

    RE_CNPJ_LABEL = re.compile(r"(?:EMITENTE|REMETENTE|PRESTADOR|EMPRESA)[^\n]{0,50}?\bCNP[J1]\b[:\s]*([0-9.\-/]{14,18})", re.I)
    RE_CNPJ_DEST  = re.compile(r"(?:DESTINAT[ÁA]RIO|TOMADOR)[^\n]{0,50}?\bCNP[J1]\b[:\s]*([0-9.\-/]{14,18})", re.I)

    RE_SPLIT_PIPES  = re.compile(r"\s*\|\s*")
    RE_ITEM_LINHA   = re.compile(
        r"^(?:\d{1,4}\s+)?(?P<desc>.+?)\s+(?P<unid>[A-Z]{1,4})\s+(?P<qtd>[\d.,]+)\s+(?P<vun>[\d.,]+)\s+(?P<vtot>[\d.,]+)\s*$",
        re.I | re.M,
    )
    RE_NCM_ITEM     = re.compile(r"\bNCM[:\s]*([0-9]{8})\b", re.I)
    RE_CFOP_ITEM    = re.compile(r"\bCFOP[:\s]*([0-9]{4})\b", re.I)
    RE_CST_ITEM     = re.compile(r"\bCST[:\s]*([0-9]{2})\b", re.I)

    RE_CHAVE = re.compile(r"\b(\d{44})\b")
    RE_QR    = re.compile(r"(?:chNFe|chCTe)=([0-9]{44})")

    RE_MOEDA_BR = re.compile(r"(?:R\$\s*)?(\d{1,3}(?:\.\d{3})*,\d{2}|\d+,\d{2})")

    # ----------------------- memória de layouts -----------------------
    _layout_memory: Dict[str, Dict[str, Any]] = {}  # layout_hash -> stats/hints

    # ---------------------------- Init ----------------------------
    def __init__(
        self,
        llm: Optional["BaseChatModel"] = None,
        *,
        enable_llm: bool = True,
        llm_trigger_missing_keys: Optional[List[str]] = None,
        llm_max_chars_ocr: int = 6000,
        base_weight: float = 0.7,  # peso heurística (1-base_weight é o "espaço" para LLM)
    ):
        self.llm = llm if enable_llm and isinstance(llm, BaseChatModel) else None
        self.llm_max_chars_ocr = int(llm_max_chars_ocr)
        self.base_weight = float(base_weight)
        self.trigger_keys = llm_trigger_missing_keys or ["emitente_cnpj", "valor_total", "data_emissao"]

        # Schema de cabeçalho (mantém compat com DB/normalizador)
        self.schema_campos = [
            "chave_acesso","numero_nota","serie","modelo","data_emissao","data_saida","hora_emissao","hora_saida",
            "emitente_nome","emitente_cnpj","emitente_cpf","emitente_ie","emitente_im",
            "emitente_endereco","emitente_municipio","emitente_uf",
            "destinatario_nome","destinatario_cnpj","destinatario_cpf","destinatario_ie","destinatario_im",
            "destinatario_endereco","destinatario_municipio","destinatario_uf",
            "valor_total","total_produtos","total_servicos","total_icms","total_ipi","total_pis","total_cofins","valor_iss",
            "valor_descontos","valor_outros","valor_frete","valor_liquido",
            "uf","municipio","inscricao_estadual","endereco","cfop","ncm","cst","natureza_operacao",
            "forma_pagamento","cnpj_autorizado","observacoes",
        ]

        # Schema de itens/impostos para wrappers
        self.schema_itens = ["descricao","codigo_produto","ncm","cfop","unidade","quantidade","valor_unitario","valor_total"]
        self.schema_impostos = ["item_idx","tipo_imposto","cst","origem","base_calculo","aliquota","valor"]

        # caches/explicabilidade
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._explanations: Dict[str, Dict[str, Any]] = {}
        self._conf_per_field: Dict[str, float] = {}

    # ============================ Público ============================
    def extrair_campos(self, entrada: Any, ocr_meta: Optional[dict] = None) -> Dict[str, Any]:
        """
        Suporta dois formatos de entrada:
          1) str -> texto OCR
          2) dict -> {"texto_ocr": str, "ocr_meta": {...}}
        Mantém compatibilidade com versões antigas do orchestrator.
        """
        if isinstance(entrada, dict):
            texto = (entrada.get("texto_ocr") or "") if isinstance(entrada.get("texto_ocr"), str) else ""
            meta_in = entrada.get("ocr_meta") or {}
        else:
            texto = str(entrada or "")
            meta_in = ocr_meta or {}

        if not texto.strip():
            return {"__meta__": {"source": "deterministico", "coverage": 0.0, "ocr_meta_in": meta_in}}

        layout_hash = hashlib.sha1(texto[:2000].encode("utf-8", "ignore")).hexdigest()
        key = hashlib.sha256(texto.encode("utf-8", "ignore")).hexdigest()
        if key in self._cache:
            res = dict(self._cache[key])
            res_meta = res.get("__meta__", {})
            res_meta["cache_hit"] = True
            res["__meta__"] = res_meta
            return res

        # reset explicabilidade por rodada
        self._explanations = {}
        self._conf_per_field = {}

        # pré-processo agressivo anti-monobloco
        texto_pp = preprocess_text(texto)
        t_pre   = self._pre_normalize_ocr(texto_pp)
        t_norm  = _norm_ws(t_pre)

        # 1) classificar
        cls = self._cls_doc(t_pre)
        doc_tipo = cls.get("tipo", "OUTRO")
        conf_cls = float(cls.get("conf", 0.0) or 0.0)

        # 2) heurística base
        base = self._extrair_campos_deterministico(t_norm, texto_original=t_pre)
        meta_base = self._score_meta_deterministico(base)
        base = self._heuristica_intermediaria(base, t_norm)

        itens_ocr = base.get("itens_ocr") or []
        impostos_ocr = base.get("impostos_ocr") or []

        # 3) Decidir LLM
        use_llm = self.llm is not None and (self._deve_acionar_llm(base) or not itens_ocr) and len(t_norm) >= 20
        llm_provider = None; llm_model = None
        if self.llm and get_llm_identity:
            try:
                ident = get_llm_identity(self.llm)  # {'provider':..., 'model':..., 'temperature':...}
                llm_provider, llm_model = ident.get("provider"), ident.get("model")
            except Exception:
                pass

        # 4) LLM headers/itens/impostos via wrappers corretos
        llm_hdr: Dict[str, Any] = {}
        llm_itens: List[Dict[str, Any]] = []
        llm_impostos: List[Dict[str, Any]] = []
        meta_llm_hdr: Dict[str, Any] = {}

        if use_llm:
            try:
                # Cabeçalho
                if extract_header_json and callable(extract_header_json):
                    hdr_out = extract_header_json(
                        self.llm,
                        schema_keys=self.schema_campos,
                        text=textual_truncate(t_pre, self.llm_max_chars_ocr),
                        extra_instructions=f"Tipo de documento: {doc_tipo}"
                    )
                    llm_hdr = dict(hdr_out.get("json") or {})
                    meta_llm_hdr = dict(hdr_out.get("meta") or {})
                    # Marcar explicabilidade LLM do header
                    for k, v in (llm_hdr or {}).items():
                        if v not in (None, "", [], {}):
                            self._add_explanation(k, method="llm", confidence=0.75, evidence=str(v)[:120])
                else:
                    llm_hdr = self._extract_header_llm_fallback(t_pre, doc_tipo)

                # Desambiguação Emitente/Destinatário (correção de inversão)
                llm_hdr = self._role_disambiguation(base, llm_hdr, t_pre)

                # Itens
                if not itens_ocr:
                    if extract_items_json and callable(extract_items_json):
                        it_out = extract_items_json(
                            self.llm,
                            schema_keys=self.schema_itens,
                            text=textual_truncate(t_pre, self.llm_max_chars_ocr),
                            extra_instructions=f"Tipo de documento: {doc_tipo}"
                        )
                        llm_itens = list(it_out.get("json") or [])  # lista
                        if llm_itens:
                            self._add_explanation("itens", method="llm", confidence=0.7, evidence=f"{len(llm_itens)} itens")
                    else:
                        llm_itens = self._extract_itens_llm_fallback(t_pre, doc_tipo)

                # Impostos
                if not impostos_ocr:
                    if extract_taxes_json and callable(extract_taxes_json):
                        tx_out = extract_taxes_json(
                            self.llm,
                            schema_keys=self.schema_impostos,
                            text=json.dumps({
                                "tipo": doc_tipo,
                                "texto": textual_truncate(t_pre, self.llm_max_chars_ocr),
                                "itens_preview": (llm_itens or itens_ocr)[:10]
                            }, ensure_ascii=False)
                        )
                        llm_impostos = list(tx_out.get("json") or [])
                        if llm_impostos:
                            self._add_explanation("impostos", method="llm", confidence=0.65, evidence=f"{len(llm_impostos)} impostos")
                    else:
                        llm_impostos = self._extract_impostos_llm_fallback(t_pre, llm_itens or itens_ocr, doc_tipo)

            except Exception as e:
                log.warning(f"AgenteNLP LLM erro: {e}")

        # 5) Fusão cognitiva (base ↔ LLM)
        fused, meta_fusion = self._fundir_resultados(base, llm_hdr, meta_base)
        meta_fusion["doc_tipo"] = doc_tipo
        meta_fusion["conf_cls"] = conf_cls
        meta_fusion["meta_llm_header"] = meta_llm_hdr

        # 6) unificação de itens/impostos
        itens_final = itens_ocr if itens_ocr else (llm_itens or [])
        impostos_final = impostos_ocr if impostos_ocr else (llm_impostos or [])

        # 7) sanitização + checks
        fused = self._sanear_dados(fused)
        checks = self._coherence_checks(fused, itens_final, impostos_final)
        meta_fusion.update(checks or {})

        # 8) coverage & __meta__
        coverage = sum(1 for k in self.schema_campos if fused.get(k)) / float(len(self.schema_campos))
        meta_common = {
            "source": "fusion" if use_llm else "deterministico",
            "coverage": round(coverage, 3),
            "confidence": {k: round(self._conf_per_field.get(k, 0.0), 3) for k in self.schema_campos},
            "explanations": self._explanations,
            "layout_hash": layout_hash,
            "cache_hit": False,
            "llm_provider": llm_provider,
            "llm_model": llm_model,
            "ocr_meta_in": meta_in,
        }

        fused["itens_ocr"] = itens_ocr
        fused["impostos_ocr"] = impostos_ocr
        fused["itens"] = itens_final
        fused["impostos"] = impostos_final
        fused["__meta__"] = {**meta_common, **meta_fusion}

        # 9) memória de layout (autoaprendizado leve)
        self._update_layout_memory(layout_hash, doc_tipo, fused)

        self._cache[key] = dict(fused)
        return fused

    # ===================== Núcleo determinístico =====================
    def _add_explanation(self, field: str, *, method: str, confidence: float, evidence: str = "", position: Optional[int] = None, pattern: Optional[str] = None) -> None:
        e = self._explanations.setdefault(field, {})
        e.setdefault("method", method)
        if evidence:
            e.setdefault("evidence", evidence[:220])
        if position is not None:
            e.setdefault("position", position)
        if pattern:
            e.setdefault("pattern", pattern[:180])
        self._conf_per_field[field] = max(self._conf_per_field.get(field, 0.0), float(confidence))

    def _extrair_campos_deterministico(self, t_norm: str, *, texto_original: str) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        sec_emit, sec_dest = self._split_secoes(texto_original)

        # CNPJ/CPF (com explicabilidade por seção)
        emit_cnpj = self._get_cnpj_por_contexto(sec_emit)
        if emit_cnpj:
            self._add_explanation("emitente_cnpj", method="regex_section", confidence=0.95, evidence=self._peek(sec_emit, emit_cnpj), pattern="RE_CNPJ_LABEL|RE_CNPJ")
        dest_cnpj = self._get_cnpj_por_contexto(sec_dest)
        if dest_cnpj:
            self._add_explanation("destinatario_cnpj", method="regex_section", confidence=0.92, evidence=self._peek(sec_dest, dest_cnpj), pattern="RE_CNPJ_LABEL|RE_CNPJ")

        if not emit_cnpj or (emit_cnpj and len(emit_cnpj) == 11):
            cnpjs = [_only_digits(m) for m in self.RE_CNPJ.findall(t_norm)]
            cpfs  = [_only_digits(m) for m in self.RE_CPF.findall(t_norm)]
            if not emit_cnpj and cnpjs:
                emit_cnpj = cnpjs[0]
                self._add_explanation("emitente_cnpj", method="regex_global", confidence=0.85, evidence=emit_cnpj, pattern="RE_CNPJ")
            if not dest_cnpj and (len(cnpjs) > 1):
                dest_cnpj = cnpjs[1]
                self._add_explanation("destinatario_cnpj", method="regex_global", confidence=0.82, evidence=dest_cnpj, pattern="RE_CNPJ")
            elif not dest_cnpj and len(cpfs) > 0:
                dest_cnpj = cpfs[0]

        out["emitente_cnpj"] = emit_cnpj if (emit_cnpj and len(emit_cnpj) >= 14) else None
        out["emitente_cpf"]  = emit_cnpj if (emit_cnpj and len(emit_cnpj) == 11) else None
        out["destinatario_cnpj"] = dest_cnpj if (dest_cnpj and len(dest_cnpj) >= 14) else None
        out["destinatario_cpf"]  = dest_cnpj if (dest_cnpj and len(dest_cnpj) == 11) else None

        # IE/IM
        ie_m = self.RE_IE.search(sec_emit or "") or self.RE_IE.search(t_norm)
        if ie_m:
            out["emitente_ie"] = ie_m.group(1)
            self._add_explanation("emitente_ie", method="regex", confidence=0.8, evidence=ie_m.group(0), pattern="RE_IE")
        im_m = self.RE_IM.search(sec_emit or "") or self.RE_IM.search(t_norm)
        if im_m:
            out["emitente_im"] = im_m.group(1)
            self._add_explanation("emitente_im", method="regex", confidence=0.75, evidence=im_m.group(0), pattern="RE_IM")

        out["destinatario_ie"] = self._safe_group(self.RE_IE.search(sec_dest or ""), 1)
        if out["destinatario_ie"]:
            self._add_explanation("destinatario_ie", method="regex_section", confidence=0.75, evidence=out["destinatario_ie"], pattern="RE_IE")

        out["destinatario_im"] = self._safe_group(self.RE_IM.search(sec_dest or ""), 1)
        if out["destinatario_im"]:
            self._add_explanation("destinatario_im", method="regex_section", confidence=0.7, evidence=out["destinatario_im"], pattern="RE_IM")

        # Endereços/municípios/UF por seção
        e_end, e_mun, e_uf = self._extrair_endereco_bloco(sec_emit) if sec_emit else (None, None, None)
        d_end, d_mun, d_uf = self._extrair_endereco_bloco(sec_dest) if sec_dest else (None, None, None)

        out["emitente_endereco"]  = e_end
        out["emitente_municipio"] = e_mun
        out["emitente_uf"]        = e_uf
        if e_end: self._add_explanation("emitente_endereco", method="section", confidence=0.7, evidence=e_end)
        if e_mun: self._add_explanation("emitente_municipio", method="section", confidence=0.7, evidence=e_mun)
        if e_uf: self._add_explanation("emitente_uf", method="section", confidence=0.7, evidence=e_uf)

        out["destinatario_endereco"]  = d_end
        out["destinatario_municipio"] = d_mun
        out["destinatario_uf"]        = d_uf
        if d_end: self._add_explanation("destinatario_endereco", method="section", confidence=0.7, evidence=d_end)
        if d_mun: self._add_explanation("destinatario_municipio", method="section", confidence=0.7, evidence=d_mun)
        if d_uf: self._add_explanation("destinatario_uf", method="section", confidence=0.7, evidence=d_uf)

        out["endereco"] = out["emitente_endereco"] or self._capturar_bloco(t_norm, ["endereço","endereco","logradouro","rua","avenida"], max_chars=160)
        out["municipio"] = out["emitente_municipio"] or self._capturar_bloco(t_norm, ["município","municipio","cidade"], max_chars=80)
        if out["endereco"]: self._add_explanation("endereco", method="global_block", confidence=0.6, evidence=out["endereco"])
        if out["municipio"]: self._add_explanation("municipio", method="global_block", confidence=0.6, evidence=out["municipio"])

        uf_global = e_uf or self._achar_uf_proximo(out["endereco"], out["municipio"]) or self._achar_uf_em_texto(t_norm)
        out["uf"] = uf_global
        if uf_global: self._add_explanation("uf", method="regex", confidence=0.65, evidence=uf_global, pattern="RE_UF")
        if not out.get("emitente_uf"):
            out["emitente_uf"] = uf_global

        # Nomes
        out["emitente_nome"] = self._capturar_bloco(sec_emit or t_norm, ["razão social","razao social","nome/razão","emitente","prestador","empresa"], max_chars=150)
        if out["emitente_nome"]: self._add_explanation("emitente_nome", method="block", confidence=0.55, evidence=out["emitente_nome"])
        out["destinatario_nome"] = self._capturar_bloco(sec_dest or t_norm, ["destinatário","tomador","cliente","consumidor"], max_chars=150)
        if out["destinatario_nome"]: self._add_explanation("destinatario_nome", method="block", confidence=0.55, evidence=out["destinatario_nome"])

        # Totais, datas, identificação
        out["valor_total"] = self._achar_valor_total(t_norm)
        if out["valor_total"] is not None:
            self._add_explanation("valor_total", method="regex_near", confidence=0.9, evidence=str(out["valor_total"]), pattern="RE_VALOR_TOTAL/RE_NUM_VALOR")

        m_data = self.RE_DATA_EMISSAO.search(t_norm)
        out["data_emissao"] = _parse_date_like(m_data.group(1)) if m_data else None
        if out["data_emissao"]:
            self._add_explanation("data_emissao", method="regex", confidence=0.85, evidence=m_data.group(0) if m_data else "", pattern="RE_DATA_EMISSAO")

        out["numero_nota"] = self._safe_group(self.RE_NUMERO.search(t_norm), 1)
        out["serie"]       = self._safe_group(self.RE_SERIE.search(t_norm), 1)
        out["modelo"]      = self._safe_group(self.RE_MODELO.search(t_norm), 1)
        if out["numero_nota"]: self._add_explanation("numero_nota", method="regex", confidence=0.8, evidence=out["numero_nota"], pattern="RE_NUMERO")
        if out["serie"]: self._add_explanation("serie", method="regex", confidence=0.8, evidence=out["serie"], pattern="RE_SERIE")
        if out["modelo"]: self._add_explanation("modelo", method="regex", confidence=0.7, evidence=out["modelo"], pattern="RE_MODELO")

        nat = self._safe_group(self.RE_NATUREZA.search(t_norm), 1)
        if nat:
            out["natureza_operacao"] = self._limpa_linha(nat)
            self._add_explanation("natureza_operacao", method="regex", confidence=0.65, evidence=out["natureza_operacao"], pattern="RE_NATUREZA")
        forma = self._safe_group(self.RE_FORMA_PGTO.search(t_norm), 1)
        if forma:
            out["forma_pagamento"] = self._limpa_linha(forma)
            self._add_explanation("forma_pagamento", method="regex", confidence=0.65, evidence=out["forma_pagamento"], pattern="RE_FORMA_PGTO")

        # Chave de acesso
        chave = None
        mqr = self.RE_QR.search(t_norm)
        if mqr: chave = _only_digits(mqr.group(1))
        if not chave:
            mch = self.RE_CHAVE.search(t_norm)
            if mch: chave = _only_digits(mch.group(1))
        out["chave_acesso"] = chave
        if chave: self._add_explanation("chave_acesso", method="regex", confidence=0.95, evidence=chave, pattern="RE_QR|RE_CHAVE")

        # Itens + impostos (determinístico OCR)
        itens, impostos = self._extrair_itens_impostos_ocr(texto_original)
        out["itens_ocr"]    = itens
        out["impostos_ocr"] = impostos

        soma_itens = self._soma_valor_itens(itens)
        if soma_itens and not out.get("total_produtos"):
            out["total_produtos"] = soma_itens
            self._add_explanation("total_produtos", method="sum_items", confidence=0.8, evidence=str(soma_itens))

        # NCM/CFOP/CST globais (se aparecer em cabeçalhos/rodapés)
        if not out.get("ncm"):
            m_n = self.RE_NCM_ITEM.search(t_norm)
            if m_n:
                out["ncm"] = m_n.group(1)
                self._add_explanation("ncm", method="regex", confidence=0.7, evidence=m_n.group(0), pattern="RE_NCM_ITEM")
        if not out.get("cfop"):
            m_c = self.RE_CFOP_ITEM.search(t_norm)
            if m_c:
                out["cfop"] = m_c.group(1)
                self._add_explanation("cfop", method="regex", confidence=0.7, evidence=m_c.group(0), pattern="RE_CFOP_ITEM")
        if not out.get("cst"):
            m_s = self.RE_CST_ITEM.search(t_norm)
            if m_s:
                out["cst"] = m_s.group(1)
                self._add_explanation("cst", method="regex", confidence=0.65, evidence=m_s.group(0), pattern="RE_CST_ITEM")

        out["inscricao_estadual"] = out.get("emitente_ie") or out.get("destinatario_ie")
        if out.get("inscricao_estadual"):
            self._add_explanation("inscricao_estadual", method="merge", confidence=0.8, evidence=out["inscricao_estadual"])

        return out

    # ================= Heurística intermediária =================
    def _heuristica_intermediaria(self, dados: Dict[str, Any], texto: str) -> Dict[str, Any]:
        out = dict(dados)
        if out.get("valor_total") in (None, 0):
            vt = self._achar_valor_total(texto)
            if vt:
                out["valor_total"] = vt
                self._add_explanation("valor_total", method="regex_near", confidence=0.85, evidence=str(vt))
        if not out.get("data_emissao"):
            m = re.search(r"\b([0-9]{1,2}[./-][0-9]{1,2}[./-][12][0-9]{3})\b", texto)
            if m:
                out["data_emissao"] = _parse_date_like(m.group(1))
                self._add_explanation("data_emissao", method="regex_relaxed", confidence=0.6, evidence=m.group(0))
        if not out.get("uf"):
            m_uf_g = self.RE_UF.search(texto)
            if m_uf_g:
                out["uf"] = m_uf_g.group(1)
                self._add_explanation("uf", method="regex", confidence=0.6, evidence=m_uf_g.group(0))
        return out

    # =========================== Classificação ===========================
    def _cls_doc(self, texto: str) -> Dict[str, Any]:
        t = (texto or "").upper()
        heur = "OUTRO"
        if "NFE" in t or "NF-E" in t or "CHNFE" in t:
            heur = "NF-e"
        elif "NFCE" in t or "NFC-E" in t:
            heur = "NFC-e"
        elif "NFS" in t or "NFSE" in t or "NFS-E" in t:
            heur = "NFS-e"
        elif "NFA" in t:
            heur = "NFA-e"

        if self.llm is None:
            return {"tipo": heur, "conf": 0.6}

        # Fallback simples com LLM (mantido opcional)
        sys = SystemMessage(content=(
            "Classifique o tipo de documento fiscal brasileiro no texto (NF-e, NFC-e, NFS-e, NFA-e ou OUTRO). "
            "Responda apenas JSON: {\"tipo\":..., \"conf\":0..1}."
        ))
        usr = HumanMessage(content=textual_truncate(texto, self.llm_max_chars_ocr))
        try:
            resp = self.llm.invoke([sys, usr])  # type: ignore
            payload = self._safe_parse_json(getattr(resp, "content", "") or str(resp))
            tipo = str(payload.get("tipo") or heur)
            conf = float(payload.get("conf") or (0.75 if tipo == heur else 0.65))
            return {"tipo": tipo, "conf": max(0.0, min(1.0, conf))}
        except Exception:
            return {"tipo": heur, "conf": 0.6}

    # ==================== Fallbacks de LLM (opcionais) ====================
    def _extract_header_llm_fallback(self, texto: str, doc_tipo: str) -> Dict[str, Any]:
        if self.llm is None:
            return {}
        sys = SystemMessage(content=(
            "Você é um agente fiscal. Extraia APENAS os campos do cabeçalho no schema dado. "
            "Regra: EMITENTE é quem emite; DESTINATÁRIO/TOMADOR é quem recebe. "
            "Use datas YYYY-MM-DD, decimais com ponto, ausentes=null. JSON válido."
        ))
        schema = json.dumps(self.schema_campos, ensure_ascii=False)
        usr = HumanMessage(content=(
            f"Tipo: {doc_tipo}\n"
            f"Texto OCR:\n{ textual_truncate(texto, self.llm_max_chars_ocr) }\n\n"
            f"Schema chaves: {schema}"
        ))
        resp = self.llm.invoke([sys, usr])  # type: ignore
        payload = self._safe_parse_json(getattr(resp, "content", "") or str(resp))
        if isinstance(payload, dict):
            for k, v in payload.items():
                if v not in (None, "", []):
                    self._add_explanation(k, method="llm", confidence=0.75, evidence=str(v)[:120])
            return payload
        return {}

    def _extract_itens_llm_fallback(self, texto: str, doc_tipo: str) -> List[Dict[str, Any]]:
        if self.llm is None:
            return []
        sys = SystemMessage(content=(
            "Extraia a TABELA DE ITENS de um documento fiscal brasileiro, mesmo se o layout estiver solto. "
            "Responda JSON lista com chaves: "
            "[descricao, codigo_produto, ncm, cfop, unidade, quantidade, valor_unitario, valor_total]. "
            "Números com ponto; ausentes=null."
        ))
        usr = HumanMessage(content=(
            f"Tipo: {doc_tipo}\n"
            f"Texto OCR:\n{ textual_truncate(texto, self.llm_max_chars_ocr) }"
        ))
        resp = self.llm.invoke([sys, usr])  # type: ignore
        payload = self._safe_parse_json(getattr(resp, "content", "") or str(resp))
        itens: List[Dict[str, Any]] = []
        if isinstance(payload, list):
            itens = payload
        elif isinstance(payload, dict) and "itens" in payload and isinstance(payload["itens"], list):
            itens = payload["itens"]
        clean: List[Dict[str, Any]] = []
        for it in itens:
            d = dict(it or {})
            d["descricao"] = _norm_ws(str(d.get("descricao") or "")) or None
            for k in ("quantidade","valor_unitario","valor_total"):
                if d.get(k) is not None:
                    d[k] = _to_float_br(str(d[k]))
            for k in ("ncm","cfop"):
                if d.get(k):
                    d[k] = _only_digits(str(d[k]))
            if d.get("unidade"):
                u = str(d["unidade"]).strip().upper()
                d["unidade"] = u if re.fullmatch(r"[A-Z]{1,4}", u) else None
            clean.append(d)
        if clean:
            self._add_explanation("itens", method="llm", confidence=0.7, evidence=f"{len(clean)} itens")
        return clean

    def _extract_impostos_llm_fallback(self, texto: str, itens: List[Dict[str, Any]], doc_tipo: str) -> List[Dict[str, Any]]:
        if self.llm is None:
            return []
        sys = SystemMessage(content=(
            "Extraia IMPOSTOS de documento fiscal BR. Associe ao item por item_idx quando claro. "
            "Cada imposto: {item_idx?, tipo_imposto, cst?, origem?, base_calculo?, aliquota?, valor?}. "
            "Tipos: ICMS, IPI, PIS, COFINS, ISS, OUTRO. Números com ponto. JSON lista."
        ))
        usr = HumanMessage(content=json.dumps({
            "tipo": doc_tipo,
            "texto": textual_truncate(texto, self.llm_max_chars_ocr),
            "itens_preview": (itens or [])[:10]
        }, ensure_ascii=False))
        resp = self.llm.invoke([sys, usr])  # type: ignore
        payload = self._safe_parse_json(getattr(resp, "content", "") or str(resp))
        out: List[Dict[str, Any]] = []
        if isinstance(payload, list):
            out = payload
        elif isinstance(payload, dict) and isinstance(payload.get("impostos"), list):
            out = payload["impostos"]
        imp_san: List[Dict[str, Any]] = []
        for imp in out:
            dd = dict(imp or {})
            if dd.get("item_idx") is not None:
                try: dd["item_idx"] = int(dd["item_idx"])
                except Exception: dd["item_idx"] = None
            for k in ("base_calculo","aliquota","valor"):
                if dd.get(k) is not None:
                    dd[k] = _to_float_br(str(dd[k]))
            t = (dd.get("tipo_imposto") or "").upper()
            dd["tipo_imposto"] = t if t in ("ICMS","IPI","PIS","COFINS","ISS","OUTRO") else "OUTRO"
            imp_san.append(dd)
        if imp_san:
            self._add_explanation("impostos", method="llm", confidence=0.65, evidence=f"{len(imp_san)} impostos")
        return imp_san

    # ============================ Fusão ============================
    def _fundir_resultados(self, base: Dict[str, Any], llm_out: Dict[str, Any], meta_base: Dict[str, float]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        result: Dict[str, Any] = dict(base)
        meta: Dict[str, Any] = {"meta_base": meta_base}

        # Heurística de peso: preferimos base para campos numéricos/datas/identificadores
        llm_threshold = max(0.0, min(1.0, 1.0 - float(self.base_weight)))

        for k in self.schema_campos:
            base_v = result.get(k)
            llm_v  = llm_out.get(k)
            if llm_v in (None, "", []):
                continue
            if not self._valor_valido_para_chave(k, llm_v):
                continue
            if base_v in (None, "", []):
                result[k] = llm_v
                self._add_explanation(k, method="llm", confidence=0.75, evidence=str(llm_v)[:120])
                continue
            # Se já há valor, só trocar por LLM se for campo textual e houver confiança/ganho
            if isinstance(llm_v, str) and not k.startswith(("valor_","total_","data_","emitente_cnpj","destinatario_cnpj","emitente_cpf","destinatario_cpf")):
                # sem meta de confiança por campo no wrapper — usamos threshold fixo
                if llm_threshold >= 0.3 and not isinstance(base_v, (int, float)):
                    result[k] = llm_v
                    self._add_explanation(k, method="llm_fusion", confidence=max(0.75, llm_threshold), evidence=str(llm_v)[:120])

        coverage_base = meta_base.get("coverage", 0.0)
        non_empty_llm = sum(1 for k in self.schema_campos if llm_out.get(k))
        coverage_llm  = non_empty_llm / max(1, len(self.schema_campos))
        meta["coverage_base"] = coverage_base
        meta["coverage_llm"]  = coverage_llm
        meta["coverage_final"] = round((coverage_base + coverage_llm) / 2.0, 3)
        return result, meta

    # =================== Seções & auxiliares ===================
    def _split_secoes(self, texto: str) -> Tuple[str, str]:
        t = texto or ""
        emit_pat = re.compile(r"(EMITENTE|REMETENTE|PRESTADOR)\s*[:\-]?\s*", re.I)
        dest_pat = re.compile(r"(DESTINAT[ÁA]RIO|TOMADOR)\s*[:\-]?\s*", re.I)
        e_m = emit_pat.search(t)
        d_m = dest_pat.search(t)
        if e_m and d_m:
            if e_m.start() < d_m.start():
                return (t[e_m.end():d_m.start()], t[d_m.end():])
            else:
                return (t[e_m.end():], t[d_m.end():e_m.start()])
        elif e_m:
            return (t[e_m.end():], "")
        elif d_m:
            return ("", t[d_m.end():])
        else:
            return ("", "")

    def _role_disambiguation(self, base: Dict[str, Any], cand: Dict[str, Any], texto: str) -> Dict[str, Any]:
        """
        Corrige inversão Emitente/Destinatário (problema reportado).
        Regras:
          1) Seções rotuladas têm prioridade (emitente_* deve aparecer na seção 'Emitente')
          2) CNPJ duplicado → o que aparece junto a 'Emitente' fica como emitente
          3) IE/IM só são copiadas se existirem; não preencher IE/IM com outro campo
        """
        if not cand:
            return base
        out = dict(cand)

        # Não copiar IE/IM inválidos
        for k in ("emitente_ie","emitente_im","destinatario_ie","destinatario_im"):
            v = out.get(k)
            if isinstance(v, str) and not v.strip():
                out[k] = None

        # Heurística simples: se 'emitente_cnpj' não for 14+ e 'destinatario_cnpj' for, troque
        em, ds = out.get("emitente_cnpj"), out.get("destinatario_cnpj")
        em_ok = em and len(_only_digits(str(em))) >= 14
        ds_ok = ds and len(_only_digits(str(ds))) >= 14
        if not em_ok and ds_ok:
            # Troca papéis apenas do par CNPJ/CPF/nome/end/mun/uf
            swaps = [
                ("emitente_cnpj","destinatario_cnpj"),
                ("emitente_cpf","destinatario_cpf"),
                ("emitente_nome","destinatario_nome"),
                ("emitente_endereco","destinatario_endereco"),
                ("emitente_municipio","destinatario_municipio"),
                ("emitente_uf","destinatario_uf"),
            ]
            for a,b in swaps:
                out[a], out[b] = out.get(b), out.get(a)

        return out

    def _get_cnpj_por_contexto(self, texto: Optional[str]) -> Optional[str]:
        if not texto:
            return None
        m1 = self.RE_CNPJ_LABEL.search(texto)
        if m1: return _only_digits(m1.group(1))
        m2 = self.RE_CNPJ.search(texto)
        if m2: return _only_digits(m2.group(1))
        m3 = self.RE_CPF.search(texto)
        if m3: return _only_digits(m3.group(1))
        return None

    def _extrair_endereco_bloco(self, texto: Optional[str]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        if not texto:
            return None, None, None
        linhas = [self._limpa_linha(x) or "" for x in texto.splitlines()]
        endereco = None; municipio = None; uf = None
        for ln in linhas:
            if self.RE_CEP.search(ln) or self.RE_UF.search(ln):
                m_uf = self.RE_UF.search(ln)
                if m_uf: uf = m_uf.group(1)
                m_city = re.search(r"([A-Za-zÀ-ÿ\s'.-]{3,})\s*-\s*[A-Z]{2}", ln)
                if m_city: municipio = m_city.group(1).strip(" -")
                parts = re.split(r"\s*-\s*[A-Z]{2}", ln)
                if parts: endereco = parts[0].strip()
                break
        if not endereco:
            endereco = self._capturar_bloco(texto, ["endereço","endereco","logradouro","rua","avenida","bairro"], max_chars=160)
        if not municipio:
            municipio = self._capturar_bloco(texto, ["município","municipio","cidade"], max_chars=80)
        if not uf:
            uf = self._achar_uf_proximo(endereco, municipio) or self._achar_uf_em_texto(texto)
        return endereco, municipio, uf

    def _achar_uf_proximo(self, endereco: Optional[str], municipio: Optional[str]) -> Optional[str]:
        for ref in [endereco or "", municipio or ""]:
            m = self.RE_UF.search(ref)
            if m: return m.group(1)
        return None

    def _achar_uf_em_texto(self, texto: str) -> Optional[str]:
        m = self.RE_UF.search(texto or "")
        return m.group(1) if m else None

    def _capturar_bloco(self, texto: str, labels: List[str], max_chars: int = 100, max_dist: int = 80) -> Optional[str]:
        t = texto or ""
        tl = t.lower()
        best_val = None; best_pos = 10**9
        for lab in labels:
            labl = lab.lower()
            idx = tl.find(labl)
            if idx == -1: continue
            start_value_idx = -1
            end_label_idx = idx + len(labl)
            for i in range(end_label_idx, min(end_label_idx + max_dist, len(t))):
                ch = t[i:i+1]
                if ch in (":", "|", "-", ";", ">", "\n"):
                    start_value_idx = i + 1; break
                if i == end_label_idx and ch.isspace() and not t[i + 1 : i + 2].isalnum():
                    start_value_idx = i + 1; break
            if start_value_idx == -1: continue
            end_idx = t.find("\n", start_value_idx)
            if end_idx == -1: end_idx = len(t)
            value = t[start_value_idx:end_idx].strip()
            if len(value) < 8:
                nxt_end = t.find("\n", end_idx + 1)
                if nxt_end == -1: nxt_end = len(t)
                extra = t[end_idx + 1 : nxt_end].strip()
                if extra: value = (value + " " + extra).strip()
            value = self._limpa_linha(value)[:max_chars]
            if value and idx < best_pos:
                best_pos = idx; best_val = value
        return best_val

    @staticmethod
    def _limpa_linha(s: Optional[str]) -> Optional[str]:
        if not s: return s
        v = re.sub(r"\s{2,}", " ", s.strip())
        v = re.sub(r"\s*[|;].*$", "", v).strip()
        return v

    def _safe_group(self, m: Optional[re.Match], idx: int) -> Optional[str]:
        try: return m.group(idx).strip() if m else None
        except Exception: return None

    def _achar_valor_total(self, texto: str) -> Optional[float]:
        linhas = texto.splitlines()
        for i, linha in enumerate(linhas):
            if self.RE_VALOR_TOTAL.search(linha):
                nm = self.RE_NUM_VALOR.search(linha)
                if nm:
                    v = _to_float_br(nm.group(1))
                    if v and v > 0: return v
                if i + 1 < len(linhas):
                    nm2 = self.RE_NUM_VALOR.search(linhas[i + 1])
                    v2 = _to_float_br(nm2.group(1)) if nm2 else None
                    if v2 and v2 > 0: return v2
        nums = self.RE_MOEDA_BR.findall(texto)
        if nums:
            v = _to_float_br(nums[-1])
            if v and v > 0: return v
        return None

    # ================== Itens & Impostos (OCR) ==================
    def _extrair_itens_impostos_ocr(self, texto_original: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        t = texto_original or ""
        if not t.strip():
            return [], []
        linhas = [self._limpa_linha(x) or "" for x in t.splitlines()]
        inicio_itens = -1; fim_itens = len(linhas)
        header_keywords = ["DESCRIÇÃO", "DESCRICAO", "PRODUTO", "QTD", "QUANT", "UNIT", "UN", "VALOR UNIT", "VALOR UNITÁRIO", "TOTAL"]
        for i, linha in enumerate(linhas):
            lu = (linha or "").upper()
            score = sum(1 for kw in header_keywords if kw in lu)
            if score >= 2:
                inicio_itens = i + 1; break
        if inicio_itens != -1:
            total_keywords = ["TOTAL DOS PRODUTOS", "VALOR TOTAL", "SUBTOTAL", "TOTAL A PAGAR", "TOTAL DA NOTA"]
            for i in range(inicio_itens, len(linhas)):
                if any(kw in (linhas[i] or "").upper() for kw in total_keywords):
                    fim_itens = i; break
            itens, impostos = self._parse_itens_linhas(linhas, inicio_itens, fim_itens)
            if itens:
                self._add_explanation("itens_ocr", method="regex_table", confidence=0.7, evidence=f"{len(itens)} itens")
                return itens, impostos
        itens_fb, impostos_fb = self._parse_itens_fallback(linhas)
        if itens_fb:
            self._add_explanation("itens_ocr", method="regex_fallback", confidence=0.55, evidence=f"{len(itens_fb)} itens")
            return itens_fb, impostos_fb
        return [], []

    def _parse_itens_linhas(self, linhas: List[str], inicio: int, fim: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        itens: List[Dict[str, Any]] = []
        impostos: List[Dict[str, Any]] = []
        item_idx_counter = 0
        for i in range(inicio, max(inicio, fim)):
            linha = _norm_ws(linhas[i].strip())
            if not linha: continue
            if "|" in linha:
                cols = [c.strip() for c in self.RE_SPLIT_PIPES.split(linha) if c.strip()]
                if len(cols) >= 5:
                    desc = cols[0]
                    un   = cols[1] if re.fullmatch(r"[A-Z]{1,4}", cols[1]) else None
                    qtd  = _to_float_br(cols[-3])
                    vun  = _to_float_br(cols[-2])
                    vtot = _to_float_br(cols[-1])
                    if vtot or vun:
                        item = {"descricao": _norm_ws(desc), "unidade": un, "quantidade": qtd, "valor_unitario": vun, "valor_total": vtot, "ncm": None, "cfop": None, "cst": None}
                        ncm, cfop, cst = self._ncm_cfop_cst_context(linhas, i, inicio, fim)
                        item["ncm"], item["cfop"], item["cst"] = ncm, cfop, cst
                        itens.append(item)
                        for tp in ("ICMS","IPI","PIS","COFINS"):
                            ali = self._aliquota_contexto(linhas, i, inicio, fim, tp)
                            if ali is not None:
                                impostos.append({"item_idx": item_idx_counter, "tipo_imposto": tp, "aliquota": ali})
                        item_idx_counter += 1
                        continue
            m = self.RE_ITEM_LINHA.search(linha)
            if m:
                gd = m.groupdict()
                item = {"descricao": _norm_ws(gd.get("desc","")), "unidade": gd.get("unid"), "quantidade": _to_float_br(gd.get("qtd")), "valor_unitario": _to_float_br(gd.get("vun")), "valor_total": _to_float_br(gd.get("vtot")), "ncm": None, "cfop": None, "cst": None}
                ncm, cfop, cst = self._ncm_cfop_cst_context(linhas, i, inicio, fim)
                item["ncm"], item["cfop"], item["cst"] = ncm, cfop, cst
                itens.append(item)
                for tp in ("ICMS","IPI","PIS","COFINS"):
                    ali = self._aliquota_contexto(linhas, i, inicio, fim, tp)
                    if ali is not None:
                        impostos.append({"item_idx": item_idx_counter, "tipo_imposto": tp, "aliquota": ali})
                item_idx_counter += 1; continue
            cols = re.split(r"\s{2,}", linha)
            if len(cols) >= 4:
                desc = cols[0]
                qtd  = _to_float_br(cols[-3])
                vun  = _to_float_br(cols[-2])
                vtot = _to_float_br(cols[-1])
                un   = None
                if len(cols) >= 5 and re.fullmatch(r"[A-Z]{1,4}", cols[-4] or ""):
                    un = cols[-4]
                elif len(cols) >= 2 and re.fullmatch(r"[A-Z]{1,4}", cols[1] or ""):
                    un = cols[1]
                it = {"descricao": _norm_ws(desc), "unidade": un, "quantidade": qtd, "valor_unitario": vun, "valor_total": vtot, "ncm": None, "cfop": None, "cst": None}
                ncm, cfop, cst = self._ncm_cfop_cst_context(linhas, i, inicio, fim)
                it["ncm"], it["cfop"], it["cst"] = ncm, cfop, cst
                itens.append(it)
                for tp in ("ICMS","IPI","PIS","COFINS"):
                    ali = self._aliquota_contexto(linhas, i, inicio, fim, tp)
                    if ali is not None:
                        impostos.append({"item_idx": item_idx_counter, "tipo_imposto": tp, "aliquota": ali})
                item_idx_counter += 1
        return itens, impostos

    def _parse_itens_fallback(self, linhas: List[str]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        itens: List[Dict[str, Any]] = []
        impostos: List[Dict[str, Any]] = []
        item_idx_counter = 0
        for i, ln in enumerate(linhas):
            if not ln or len(ln) < 12: continue
            found = self.RE_MOEDA_BR.findall(ln)
            nxt   = self.RE_MOEDA_BR.findall(linhas[i+1]) if i+1 < len(linhas) else []
            if len(found) >= 2 or (len(found) == 1 and len(nxt) >= 1):
                desc = re.split(self.RE_MOEDA_BR.pattern, ln)[0].strip()
                un = None; qtd = None
                qtd_m = re.search(r"(\d+(?:[\.,]\d{1,3})?)\s*(?:UN|UND|UNID|PC|PÇ|CX|KG|LT|UNIDADE|UNID\.)?", ln, re.I)
                if qtd_m:
                    qtd = _to_float_br(qtd_m.group(1))
                    un_m = re.search(r"\b(UN|UND|UNID|PC|PÇ|CX|KG|LT|UNIDADE)\b", ln, re.I)
                    if un_m: un = un_m.group(1).upper()
                valores = found + nxt
                if len(valores) >= 2:
                    vun  = _to_float_br(valores[-2])
                    vtot = _to_float_br(valores[-1])
                    item = {"descricao": _norm_ws(desc) or None, "unidade": un, "quantidade": qtd, "valor_unitario": vun, "valor_total": vtot, "ncm": None, "cfop": None, "cst": None}
                    ncm, cfop, cst = self._ncm_cfop_cst_context(linhas, i, 0, len(linhas))
                    item["ncm"], item["cfop"], item["cst"] = ncm, cfop, cst
                    itens.append(item)
                    for tp in ("ICMS","IPI","PIS","COFINS"):
                        ali = self._aliquota_contexto(linhas, i, 0, len(linhas), tp)
                        if ali is not None:
                            impostos.append({"item_idx": item_idx_counter, "tipo_imposto": tp, "aliquota": ali})
                    item_idx_counter += 1
        return itens, impostos

    def _ncm_cfop_cst_context(self, linhas: List[str], i: int, inicio: int, fim: int) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        linha_anterior = linhas[i - 1].strip() if (i - 1) >= inicio else ""
        linha_seguinte = linhas[i + 1].strip() if (i + 1) < fim else ""
        ncm = (self.RE_NCM_ITEM.search(linhas[i] or "") or self.RE_NCM_ITEM.search(linha_seguinte) or self.RE_NCM_ITEM.search(linha_anterior))
        cfop = (self.RE_CFOP_ITEM.search(linhas[i] or "") or self.RE_CFOP_ITEM.search(linha_seguinte) or self.RE_CFOP_ITEM.search(linha_anterior))
        cst = (self.RE_CST_ITEM.search(linhas[i] or "") or self.RE_CST_ITEM.search(linha_seguinte) or self.RE_CST_ITEM.search(linha_anterior))
        return (ncm.group(1) if ncm else None, cfop.group(1) if cfop else None, cst.group(1) if cst else None)

    def _aliquota_contexto(self, linhas: List[str], i: int, inicio: int, fim: int, imposto: str) -> Optional[float]:
        pad = re.compile(fr"{imposto}\s*[:\-]?\s*([0-9]+,[0-9]{{2}})\s*%?", re.I)
        for ln in (linhas[i], linhas[i+1] if i+1<fim else "", linhas[i-1] if i-1>=inicio else ""):
            m = pad.search(ln or "")
            if m: return _to_float_br(m.group(1))
        return None

    # =============== Validação & Sanitização (NLP) ===============
    def _valor_valido_para_chave(self, k: str, v: Any) -> bool:
        if v in (None, "", []): return False
        try:
            if k in ("emitente_cnpj", "destinatario_cnpj", "cnpj_autorizado"):
                d = _only_digits(str(v)); return bool(d and len(d) >= 14)
            if k in ("emitente_cpf", "destinatario_cpf"):
                d = _only_digits(str(v)); return bool(d and len(d) == 11)
            if k.startswith("data_"):
                return _parse_date_like(str(v)) is not None
            if k.startswith("valor_") or k.startswith("total_"):
                if isinstance(v, (int, float)): return float(v) >= 0
                return _to_float_br(str(v)) is not None
        except Exception:
            return False
        return True

    def _soma_valor_itens(self, itens: List[Dict[str, Any]]) -> Optional[float]:
        soma = 0.0; ok = False
        for it in itens or []:
            v = it.get("valor_total")
            if isinstance(v, (int, float)):
                soma += float(v); ok = True
            elif v:
                fv = _to_float_br(str(v))
                if fv is not None:
                    soma += fv; ok = True
        return round(soma, 2) if ok else None

    def _sanear_dados(self, d: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(d)
        for k in ("emitente_cnpj", "destinatario_cnpj", "cnpj_autorizado", "emitente_cpf", "destinatario_cpf"):
            if out.get(k): out[k] = _only_digits(str(out[k])) or None
        for k in [c for c in out.keys() if c.startswith("data_")]:
            if out.get(k): out[k] = _parse_date_like(str(out[k]))
        campos_valor = [
            "valor_total","total_produtos","total_servicos","total_icms","total_ipi",
            "total_pis","total_cofins","valor_iss","valor_descontos","valor_outros",
            "valor_frete","valor_liquido",
        ]
        for k in campos_valor:
            if out.get(k) is not None:
                out[k] = float(out[k]) if isinstance(out[k], (int, float)) else _to_float_br(str(out[k]))
        for k, v in list(out.items()):
            if isinstance(v, str) and not v.strip():
                out[k] = None
        return out

    # ================== Pré-normalização OCR ==================
    def _pre_normalize_ocr(self, texto: str) -> str:
        t = texto or ""
        subs = {
            r"\bCNPF\b": "CNPJ",
            r"\bCNPJ1\b": "CNPJ",
            r"\bVALDR\b": "VALOR",
            r"\bVAL0R\b": "VALOR",
            r"\bEMISSA0\b": "EMISSAO",
            r"\bNATURE2A\b": "NATUREZA",
            r"\bCFQP\b": "CFOP",
            r"\bICNMS\b": "ICMS",
            r"\bT0TAL\b": "TOTAL",
            r"\bVALORDA\b": "VALOR DA",
        }
        for pat, rep in subs.items():
            t = re.sub(pat, rep, t, flags=re.I)
        t = re.sub(r"(VALOR\s*TOTAL|TOTAL\s*DA\s*NOTA)(R\$)", r"\1: \2", t, flags=re.I)
        t = re.sub(r"(VALOR\s*L[IÍ]QUIDO)(R\$)", r"\1: \2", t, flags=re.I)
        t = re.sub(r"(VALOR\s*A\s*PAGAR)(R\$)", r"\1: \2", t, flags=re.I)
        t = re.sub(r"V\s*A\s*L\s*O\s*R\s*T\s*O\s*T\s*A\s*L", "VALOR TOTAL", t, flags=re.I)
        t = re.sub(r"E\s*M\s*I\s*S\s*S\s*Ã?\s*O", "EMISSAO", t, flags=re.I)
        return t

    # ================== Cobertura / Trigger LLM ==================
    def _score_meta_deterministico(self, dados: Dict[str, Any]) -> Dict[str, float]:
        def s(x: Any) -> float:
            return 1.0 if x not in (None, "", [], {}) else 0.0
        meta = {
            "emitente_cnpj": s(dados.get("emitente_cnpj")),
            "valor_total": s(dados.get("valor_total")),
            "data_emissao": s(dados.get("data_emissao")),
            "chave_acesso": s(dados.get("chave_acesso")),
            "itens_ocr": 1.0 if dados.get("itens_ocr") else 0.0,
        }
        meta["coverage"] = sum(meta.values()) / (len(meta) or 1)
        return meta

    def _deve_acionar_llm(self, dados: Dict[str, Any]) -> bool:
        missing = [k for k in self.trigger_keys if not dados.get(k)]
        cobertura_baixa = self._score_meta_deterministico(dados).get("coverage", 0.0) < 0.6
        return bool(missing) or cobertura_baixa

    # =================== Checks de coerência ===================
    def _coherence_checks(self, campos: Dict[str, Any], itens: List[Dict[str, Any]], impostos: List[Dict[str, Any]]) -> Dict[str, Any]:
        checks: Dict[str, Any] = {}
        vt = campos.get("valor_total")
        soma_it = self._soma_valor_itens(itens)
        if vt and soma_it:
            checks["total_consistente_com_itens"] = abs(float(vt) - float(soma_it)) <= max(1.0, 0.25 * float(vt))  # (±25%) como “bom sinal”
        else:
            checks["total_consistente_com_itens"] = None
        checks["emitente_cnpj_ok"] = bool(_only_digits(str(campos.get("emitente_cnpj") or "")) and len(_only_digits(str(campos.get("emitente_cnpj") or ""))) >= 14)
        checks["destinatario_cnpj_ok"] = bool(_only_digits(str(campos.get("destinatario_cnpj") or "")) and len(_only_digits(str(campos.get("destinatario_cnpj") or ""))) >= 14) or bool(_only_digits(str(campos.get("destinatario_cpf") or "")) and len(_only_digits(str(campos.get("destinatario_cpf") or ""))) == 11)
        checks["data_emissao_ok"] = campos.get("data_emissao") is not None
        checks["tem_itens"] = bool(itens)
        checks["tem_impostos"] = bool(impostos)
        return checks

    # =================== Autoaprendizado (layout) ===================
    def _update_layout_memory(self, layout_hash: str, doc_tipo: str, fused: Dict[str, Any]) -> None:
        mem = self._layout_memory.setdefault(layout_hash, {"count": 0, "tipos": {}, "hints": {}})
        mem["count"] = int(mem.get("count", 0)) + 1
        tipos = mem.setdefault("tipos", {})
        tipos[doc_tipo] = tipos.get(doc_tipo, 0) + 1
        hints = mem.setdefault("hints", {})
        if fused.get("emitente_cnpj"):
            hints["emitente_cnpj_present"] = True
        if fused.get("chave_acesso"):
            hints["has_chave_acesso"] = True

    # =================== Utilidades ===================
    @staticmethod
    def _safe_parse_json(s: str) -> Any:
        try: return json.loads(s)
        except Exception: return {}

    @staticmethod
    def _peek(hay: Optional[str], needle_digits: Optional[str], window: int = 80) -> str:
        if not hay or not needle_digits: return ""
        i = hay.find(needle_digits)
        if i == -1: return hay[:window]
        start = max(0, i - 10); end = min(len(hay), i + len(needle_digits) + 10)
        return hay[start:end]


__all__ = ["AgenteNLP"]
