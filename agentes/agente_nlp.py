# agentes/agente_nlp.py

from __future__ import annotations
import json
import logging
import re
import hashlib
from typing import Any, Dict, List, Optional, Tuple

from .utils import (
    _parse_date_like, _to_float_br, _only_digits, _norm_ws, textual_truncate
)

try:
    from langchain_core.messages import SystemMessage, HumanMessage  # type: ignore
    from langchain_core.language_models.chat_models import BaseChatModel  # type: ignore
except Exception:  # pragma: no cover
    SystemMessage = object  # type: ignore
    HumanMessage = object   # type: ignore
    BaseChatModel = object  # type: ignore

# ---------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------
log = logging.getLogger("agente_fiscal.agentes")

# ---------------------------------------------------------------------
# Agente NLP (híbrido: determinístico + LLM opcional) com robustez extra
# ---------------------------------------------------------------------
class AgenteNLP:
    """
    Extrai campos e itens do texto OCR.

    Camadas:
      0) cache leve (sha256) para evitar reprocessamento idempotente
      1) pré-normalização semântica de OCR (corrige ruídos comuns)
      2) determinístico (regex + heurísticas)
      3) heurística intermediária (datas, valores, rótulos)
      4) LLM opcional e seletiva (quando cobertura baixa / campos críticos ausentes)
      5) fusão semântica com validação
      6) sanitização final (tipos/formatos coerentes)
    """

    # ----------------- Regex e padrões determinísticos -----------------
    RE_CNPJ = re.compile(r"\b(\d{2}\.?\d{3}\.?\d{3}/?\d{4}-?\d{2}|\d{14})\b")
    RE_CPF = re.compile(r"\b(\d{3}\.?\d{3}\.?\d{3}-?\d{2}|\d{11})\b")
    RE_IE = re.compile(r"\b(?:IE|I\.E\.|INSC(?:RI[ÇC][ÃA]O)?\s*ESTADUAL)[:\s\-]*([A-Z0-9.\-/]{5,20})\b", re.I)
    RE_UF = re.compile(r"\b(AC|AL|AP|AM|BA|CE|DF|ES|GO|MA|MT|MS|MG|PA|PB|PR|PE|PI|RJ|RN|RS|RO|RR|SC|SP|SE|TO)\b")

    # Totais (tolerante a OCR e rótulos variantes)
    RE_VALOR_TOTAL = re.compile(
        r"(?:VALO[R]?|TOTAL)\s*(?:DA|DO)?\s*(?:NOTA|NF|DOCUMENTO)?|VALOR\s*L[IÍ]QUIDO|VALOR\s*A\s*PAGAR",
        re.I,
    )
    RE_NUM_VALOR = re.compile(r"(?:R\$\s*)?([\d\.\s]*,\d{2})")  # número BR

    # Datas
    RE_DATA_EMISSAO = re.compile(
        r"\b(?:DATA\s*(?:DE)?\s*EMISS[ÃA]O|EMISS[ÃA]O|EMITIDO\s*EM|COMPET[ÊE]NCIA)\s*[:\-]?\s*([0-9]{1,4}[./-][0-9]{1,2}[./-][0-9]{1,4})\b",
        re.I,
    )

    RE_NUMERO = re.compile(r"\b(?:N[ºo°]|NÚMERO|NUMERO)\s*[:\-]?\s*([0-9]{1,10})\b", re.I)
    RE_SERIE = re.compile(r"\b(?:S[EÉ]RIE|SERIE)\s*[:\-]?\s*([A-Z0-9]{1,6})\b", re.I)
    RE_MODELO = re.compile(r"\b(?:MODELO)\s*[:\-]?\s*([0-9A-Z\-]{1,10})\b", re.I)
    RE_NATUREZA = re.compile(r"\b(?:NATUREZA\s+DA\s+OPERA[ÇC][ÃA]O|NATUREZA\s+DE\s+OPERA[ÇC][ÃA]O)\s*[:\-]?\s*(.+)", re.I)
    RE_FORMA_PGTO = re.compile(r"\b(?:FORMA\s+DE\s+PAGAMENTO|FORMA\s+PAGAMENTO|PAGAMENTO)\s*[:\-]?\s*([A-Za-zÀ-ÿ0-9\s/\-]+)", re.I)

    # Labels com OCR ruidoso (CNP1 ~ CNPJ)
    RE_CNPJ_LABEL = re.compile(r"(?:EMITENTE|REMETENTE|PRESTADOR|EMPRESA)[^\n]{0,50}?\bCNP[J1]\b[:\s]*([0-9.\-/]{14,18})", re.I)
    RE_CNPJ_DEST = re.compile(r"(?:DESTINAT[ÁA]RIO|TOMADOR)[^\n]{0,50}?\bCNP[J1]\b[:\s]*([0-9.\-/]{14,18})", re.I)

    # Itens (linha clássica)
    RE_ITEM_LINHA = re.compile(
        r"^(?:\d{1,4}\s+)?(?P<desc>.+?)\s+(?P<unid>[A-Z]{1,4})\s+(?P<qtd>[\d.,]+)\s+(?P<vun>[\d.,]+)\s+(?P<vtot>[\d.,]+)\s*$",
        re.I | re.M,
    )
    RE_NCM_ITEM = re.compile(r"\bNCM[:\s]*([0-9]{8})\b", re.I)
    RE_CFOP_ITEM = re.compile(r"\bCFOP[:\s]*([0-9]{4})\b", re.I)

    RE_CHAVE = re.compile(r"\b(\d{44})\b")
    RE_QR = re.compile(r"(?:chNFe|chCTe)=([0-9]{44})")

    # Fallback de itens
    RE_MOEDA_BR = re.compile(r"(?:R\$\s*)?(\d{1,3}(?:\.\d{3})*,\d{2}|\d+,\d{2})")

    # ---------------------------- Init --------------------------------
    def __init__(
        self,
        llm: Optional["BaseChatModel"] = None,
        *,
        enable_llm: bool = True,
        llm_trigger_missing_keys: Optional[List[str]] = None,
        llm_max_chars_ocr: int = 6000,
        base_weight: float = 0.7,
    ):
        self.llm = llm if enable_llm and isinstance(llm, BaseChatModel) else None
        self.llm_max_chars_ocr = int(llm_max_chars_ocr)
        self.base_weight = float(base_weight)
        self.trigger_keys = llm_trigger_missing_keys or [
            "emitente_cnpj", "valor_total", "data_emissao"
        ]

        # === SCHEMA ALINHADO AO BANCO/UNIFICAÇÃO ===
        self.schema_campos = [
            # Identificação e chave
            "chave_acesso","numero_nota","serie","modelo","data_emissao","data_saida","hora_emissao","hora_saida",
            # Emitente
            "emitente_nome","emitente_cnpj","emitente_cpf","emitente_ie","emitente_im",
            "emitente_endereco","emitente_municipio","emitente_uf",
            # Destinatário / Tomador
            "destinatario_nome","destinatario_cnpj","destinatario_cpf","destinatario_ie","destinatario_im",
            "destinatario_endereco","destinatario_municipio","destinatario_uf",
            # Totais (unificados)
            "valor_total","total_produtos","total_servicos","total_icms","total_ipi","total_pis","total_cofins","valor_iss",
            "valor_descontos","valor_outros","valor_frete","valor_liquido",
            # Gerais / rótulos úteis
            "uf","municipio","inscricao_estadual","endereco","cfop","ncm","cst","natureza_operacao",
            # Pagamento / extras
            "forma_pagamento","cnpj_autorizado","observacoes",
        ]

        self._cache: Dict[str, Dict[str, Any]] = {}

    # --------------------------- Público -------------------------------
    def extrair_campos(self, texto: str) -> Dict[str, Any]:
        texto = texto or ""
        if not texto.strip():
            return {"__meta__": {"source": "deterministico", "coverage": 0.0}}

        key = hashlib.sha256(texto.encode("utf-8", "ignore")).hexdigest()
        if key in self._cache:
            res = dict(self._cache[key])
            res["__meta__"] = {**res.get("__meta__", {}), "cache_hit": True}
            return res

        t_pre = self._pre_normalize_ocr(texto)
        t_norm = _norm_ws(t_pre)

        base = self._extrair_campos_deterministico(t_norm, texto_original=t_pre)
        meta_base = self._score_meta_deterministico(base)

        base = self._heuristica_intermediaria(base, t_norm)

        use_llm = (
            self.llm is not None
            and (self._deve_acionar_llm(base) or not base.get("itens_ocr"))
            and len(t_norm) >= 20
        )

        if not use_llm:
            final = self._sanear_dados(base)
            final["__meta__"] = {"source": "deterministico", **meta_base}
            self._cache[key] = dict(final)
            return final

        try:
            llm_out = self._extrair_campos_llm(t_pre)
        except Exception as e:
            log.warning(f"AgenteNLP LLM desabilitado por erro: {e}")
            final = self._sanear_dados(base)
            final["__meta__"] = {"source": "deterministico", **meta_base, "llm_error": str(e)}
            self._cache[key] = dict(final)
            return final

        merged, meta = self._fundir_resultados(base, llm_out, meta_base)
        merged = self._sanear_dados(merged)
        merged["__meta__"] = meta
        self._cache[key] = dict(merged)
        return merged

    # -------------------- Núcleo determinístico -----------------------
    def _extrair_campos_deterministico(self, t_norm: str, *, texto_original: str) -> Dict[str, Any]:
        out: Dict[str, Any] = {}

        # --- CNPJ/CPF por rótulos + fallback ---
        emit_cnpj = None
        dest_cnpj = None
        try:
            m_emit_label = self.RE_CNPJ_LABEL.search(texto_original)
            m_dest_label = self.RE_CNPJ_DEST.search(texto_original)
            if m_emit_label:
                emit_cnpj = _only_digits(m_emit_label.group(1))
            if m_dest_label:
                dest_cnpj = _only_digits(m_dest_label.group(1))
        except Exception:
            pass

        cnpjs = [_only_digits(m) for m in self.RE_CNPJ.findall(t_norm)]
        cpfs = [_only_digits(m) for m in self.RE_CPF.findall(t_norm)]

        if not emit_cnpj:
            emit_cnpj = cnpjs[0] if cnpjs else (cpfs[0] if cpfs else None)
        if not dest_cnpj:
            if len(cnpjs) > 1:
                dest_cnpj = cnpjs[1]
            elif cnpjs and cpfs:
                dest_cnpj = cpfs[0]
            elif len(cpfs) > 1:
                dest_cnpj = cpfs[1]

        out["emitente_cnpj"] = emit_cnpj if (emit_cnpj and len(emit_cnpj) >= 14) else None
        out["emitente_cpf"] = emit_cnpj if (emit_cnpj and len(emit_cnpj) == 11) else None
        out["destinatario_cnpj"] = dest_cnpj if (dest_cnpj and len(dest_cnpj) >= 14) else None
        out["destinatario_cpf"] = dest_cnpj if (dest_cnpj and len(dest_cnpj) == 11) else None

        # --- IE/IM/UF/Município/Endereço ---
        m_ie = self.RE_IE.search(t_norm)
        out["inscricao_estadual"] = m_ie.group(1).strip() if m_ie else None
        out["emitente_ie"] = out.get("inscricao_estadual")

        out["endereco"] = self._capturar_bloco(t_norm, ["endereço", "endereco", "logradouro", "rua", "avenida"], max_chars=160)
        out["municipio"] = self._capturar_bloco(t_norm, ["município", "municipio", "cidade"], max_chars=80)

        uf = None
        for ref in [out.get("endereco") or "", out.get("municipio") or "", t_norm[-400:]]:
            m_uf = self.RE_UF.search(ref)
            if m_uf:
                uf = m_uf.group(1)
                break
        out["uf"] = uf

        # Deriva campos específicos do emitente
        out["emitente_endereco"] = out.get("endereco")
        out["emitente_municipio"] = out.get("municipio")
        out["emitente_uf"] = out.get("uf")

        # --- Razão social (emitente_nome) ---
        out["emitente_nome"] = self._capturar_bloco(
            t_norm,
            ["razão social", "razao social", "nome/razão", "emitente", "emitida por", "prestador", "empresa"],
            max_chars=150
        )

        # --- Totais e datas ---
        out["valor_total"] = self._achar_valor_total(t_norm)
        m_data = self.RE_DATA_EMISSAO.search(t_norm)
        out["data_emissao"] = _parse_date_like(m_data.group(1)) if m_data else None

        # --- Numero, Série, Modelo, Natureza, Forma de pagamento ---
        out["numero_nota"] = self._safe_group(self.RE_NUMERO.search(t_norm), 1)
        out["serie"] = self._safe_group(self.RE_SERIE.search(t_norm), 1)
        out["modelo"] = self._safe_group(self.RE_MODELO.search(t_norm), 1)
        nat = self._safe_group(self.RE_NATUREZA.search(t_norm), 1)
        if nat:
            out["natureza_operacao"] = self._limpa_linha(nat)
        forma = self._safe_group(self.RE_FORMA_PGTO.search(t_norm), 1)
        out["forma_pagamento"] = self._limpa_linha(forma) if forma else None

        # --- Chave de acesso ---
        chave = None
        mqr = self.RE_QR.search(t_norm)
        if mqr:
            chave = _only_digits(mqr.group(1))
        if not chave:
            mch = self.RE_CHAVE.search(t_norm)
            if mch:
                chave = _only_digits(mch.group(1))
        out["chave_acesso"] = chave

        # Itens + impostos por item (determinístico com fallback)
        itens_extraidos, impostos_extraidos = self._extrair_itens_impostos_ocr(texto_original)
        out["itens_ocr"] = itens_extraidos
        out["impostos_ocr"] = impostos_extraidos

        # Se a soma dos itens fizer sentido, popular total_produtos
        soma_itens = self._soma_valor_itens(itens_extraidos)
        if soma_itens and (not out.get("total_produtos")):
            # usar como total_produtos quando houver itens (caso comum NFe/NFCe)
            out["total_produtos"] = soma_itens

        return out

    # ---------------- Heurística intermediária (reforços) --------------
    def _heuristica_intermediaria(self, dados: Dict[str, Any], texto: str) -> Dict[str, Any]:
        out = dict(dados)

        if out.get("valor_total") in (None, 0):
            vt = self._achar_valor_total(texto)
            if vt:
                out["valor_total"] = vt

        if not out.get("data_emissao"):
            m = re.search(r"\b([0-9]{1,2}[./-][0-9]{1,2}[./-][12][0-9]{3})\b", texto)
            if m:
                out["data_emissao"] = _parse_date_like(m.group(1))

        if not out.get("uf"):
            m_uf_g = self.RE_UF.search(texto)
            if m_uf_g:
                out["uf"] = m_uf_g.group(1)

        return out

    # --------------------------- Modo LLM ------------------------------
    def _extrair_campos_llm(self, texto: str) -> Dict[str, Any]:
        if self.llm is None or not isinstance(self.llm, BaseChatModel):
            return {}

        sys = SystemMessage(content=(
            "Você é um especialista em leitura fiscal brasileira. "
            "Extraia **campos estruturados** de uma nota (NFe, NFCe, NFSe ou CTe) a partir do texto OCR.\n\n"
            "Regras:\n"
            "- Responda **apenas** com um JSON válido (sem explicações).\n"
            "- Use exatamente as chaves do schema fornecido.\n"
            "- Campos ausentes => null.\n"
            "- Datas em YYYY-MM-DD; valores numéricos com ponto decimal (ex.: 1234.56) e sem 'R$'.\n"
            "- Inclua `__meta__` com confiança 0..1 por campo.\n"
            "- Não invente dados; só preencha o que estiver explicitamente no texto."
        ))

        user = HumanMessage(content=(
            f"Texto OCR (truncado):\n{ textual_truncate(texto, self.llm_max_chars_ocr) }\n\n"
            f"Schema (ordem das chaves): { json.dumps(self.schema_campos, ensure_ascii=False) }\n\n"
            "Responda APENAS com o JSON contendo os campos do schema acima e o objeto __meta__."
        ))

        resp = self.llm.invoke([sys, user])  # type: ignore
        resposta = getattr(resp, "content", None) or str(resp)

        payload = self._safe_parse_json(resposta)
        if not isinstance(payload, dict):
            return {}

        out = {k: payload.get(k, None) for k in self.schema_campos}
        meta = payload.get("__meta__", {})
        out["__meta__"] = meta if isinstance(meta, dict) else {}
        return out

    @staticmethod
    def _safe_parse_json(texto: str) -> Any:
        try:
            m = re.search(r"\{.*\}", texto, re.S)
            if not m:
                return {}
            return json.loads(m.group(0))
        except Exception:
            return {}

    # --------------------------- Fusão --------------------------------
    def _fundir_resultados(
        self,
        base: Dict[str, Any],
        llm_out: Dict[str, Any],
        meta_base: Dict[str, float],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        result: Dict[str, Any] = dict(base)
        meta: Dict[str, Any] = {"source": "fusion", "meta_base": meta_base}

        meta_llm = llm_out.get("__meta__") if isinstance(llm_out, dict) else {}
        if not isinstance(meta_llm, dict):
            meta_llm = {}
        meta["meta_llm"] = meta_llm

        llm_threshold = max(0.0, min(1.0, 1.0 - float(self.base_weight)))

        for k in self.schema_campos:
            base_v = result.get(k)
            llm_v = llm_out.get(k)

            if llm_v in (None, "", []):
                continue

            if not self._valor_valido_para_chave(k, llm_v):
                continue

            if base_v in (None, "", []):
                result[k] = llm_v
                continue

            try:
                conf_llm = float(meta_llm.get(k, 0.0))
            except Exception:
                conf_llm = 0.0

            if conf_llm >= llm_threshold:
                result[k] = llm_v

        coverage_base = meta_base.get("coverage", 0.0)
        non_empty_llm = sum(1 for k in self.schema_campos if llm_out.get(k))
        coverage_llm = non_empty_llm / max(1, len(self.schema_campos))
        meta["coverage_base"] = coverage_base
        meta["coverage_llm"] = coverage_llm

        itens = result.get("itens_ocr") or []
        soma_itens = self._soma_valor_itens(itens)
        vt = result.get("valor_total")
        if soma_itens and vt and abs(soma_itens - float(vt)) / max(1.0, float(vt)) < 0.25:
            meta["total_consistente_com_itens"] = True
        else:
            meta["total_consistente_com_itens"] = False

        meta["coverage_final"] = round((coverage_base + coverage_llm) / 2.0, 3)
        return result, meta

    # ---------------------- Heurísticas auxiliares ---------------------
    def _capturar_bloco(self, texto: str, labels: List[str], max_chars: int = 100, max_dist: int = 80) -> Optional[str]:
        t = texto
        tl = t.lower()
        best_val = None
        best_pos = 10**9

        for lab in labels:
            labl = lab.lower()
            idx = tl.find(labl)
            if idx == -1:
                continue

            start_value_idx = -1
            end_label_idx = idx + len(labl)
            for i in range(end_label_idx, min(end_label_idx + max_dist, len(t))):
                ch = t[i:i+1]
                if ch in (":", "|", "-", ";", ">", "\n"):
                    start_value_idx = i + 1
                    break
                if i == end_label_idx and ch.isspace() and not t[i + 1 : i + 2].isalnum():
                    start_value_idx = i + 1
                    break

            if start_value_idx == -1:
                continue

            end_idx = t.find("\n", start_value_idx)
            if end_idx == -1:
                end_idx = len(t)
            value = t[start_value_idx:end_idx].strip()

            if len(value) < 8:
                nxt_end = t.find("\n", end_idx + 1)
                if nxt_end == -1:
                    nxt_end = len(t)
                extra = t[end_idx + 1 : nxt_end].strip()
                if extra:
                    value = (value + " " + extra).strip()

            value = self._limpa_linha(value)[:max_chars]
            if value and idx < best_pos:
                best_pos = idx
                best_val = value

        return best_val

    @staticmethod
    def _limpa_linha(s: Optional[str]) -> Optional[str]:
        if not s:
            return s
        v = re.sub(r"\s{2,}", " ", s.strip())
        v = re.sub(r"\s*[|;].*$", "", v).strip()
        return v

    def _safe_group(self, m: Optional[re.Match], idx: int) -> Optional[str]:
        try:
            return m.group(idx).strip() if m else None
        except Exception:
            return None

    def _achar_valor_total(self, texto: str) -> Optional[float]:
        linhas = texto.splitlines()
        for i, linha in enumerate(linhas):
            if self.RE_VALOR_TOTAL.search(linha):
                nm = self.RE_NUM_VALOR.search(linha)
                if nm:
                    v = _to_float_br(nm.group(1))
                    if v and v > 0:
                        return v
                if i + 1 < len(linhas):
                    nm2 = self.RE_NUM_VALOR.search(linhas[i + 1])
                    v2 = _to_float_br(nm2.group(1)) if nm2 else None
                    if v2 and v2 > 0:
                        return v2
        nums = self.RE_MOEDA_BR.findall(texto)
        if nums:
            v = _to_float_br(nums[-1])
            if v and v > 0:
                return v
        return None

    # ----------------- Itens & Impostos (OCR) -------------------------
    def _extrair_itens_impostos_ocr(self, texto_original: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        t = texto_original or ""
        if not t.strip():
            return [], []

        linhas = [self._limpa_linha(x) or "" for x in t.splitlines()]
        inicio_itens = -1
        fim_itens = len(linhas)

        header_keywords = ["DESCRIÇÃO", "DESCRICAO", "QTD", "QUANT", "UNIT", "UN", "TOTAL", "PRODUTO", "VALOR UNIT", "VALOR UNITÁRIO"]
        for i, linha in enumerate(linhas):
            lu = (linha or "").upper()
            score = sum(1 for kw in header_keywords if kw in lu)
            if score >= 2:
                inicio_itens = i + 1
                break

        if inicio_itens != -1:
            total_keywords = ["TOTAL DOS PRODUTOS", "VALOR TOTAL", "SUBTOTAL", "TOTAL A PAGAR", "TOTAL DA NOTA"]
            for i in range(inicio_itens, len(linhas)):
                if any(kw in (linhas[i] or "").upper() for kw in total_keywords):
                    fim_itens = i
                    break

            itens, impostos = self._parse_itens_linhas(linhas, inicio_itens, fim_itens)
            if itens:
                return itens, impostos

        itens_fb: List[Dict[str, Any]] = []
        impostos_fb: List[Dict[str, Any]] = []
        item_idx_counter = 0

        for i, linha in enumerate(linhas):
            if not linha or len(linha) < 12:
                continue
            cols = re.split(r"\s{2,}", linha)
            if len(cols) >= 4:
                desc = cols[0]
                tail = cols[1:]
                if len(tail) >= 3:
                    qtd = _to_float_br(tail[-3])
                    vun = _to_float_br(tail[-2])
                    vtot = _to_float_br(tail[-1])
                    un = None
                    if re.fullmatch(r"[A-Z]{1,4}", tail[0] or ""):
                        un = tail[0]
                    item = {
                        "descricao": _norm_ws(desc),
                        "unidade": un,
                        "quantidade": qtd,
                        "valor_unitario": vun,
                        "valor_total": vtot,
                        "ncm": None,
                        "cfop": None,
                    }
                    linha_anterior = linhas[i - 1] if i - 1 >= 0 else ""
                    linha_seguinte = linhas[i + 1] if i + 1 < len(linhas) else ""
                    ncm_match = self.RE_NCM_ITEM.search(linha) or self.RE_NCM_ITEM.search(linha_seguinte) or self.RE_NCM_ITEM.search(linha_anterior)
                    cfop_match = self.RE_CFOP_ITEM.search(linha) or self.RE_CFOP_ITEM.search(linha_seguinte) or self.RE_CFOP_ITEM.search(linha_anterior)
                    if ncm_match:
                        item["ncm"] = ncm_match.group(1)
                    if cfop_match:
                        item["cfop"] = cfop_match.group(1)
                    icms_match = re.search(r"ICMS.*?(\d+,\d{2})\s*%", linha_seguinte, re.I) or re.search(r"ICMS.*?(\d+,\d{2})\s*%", linha_anterior, re.I)
                    if icms_match:
                        impostos_fb.append({"item_idx": item_idx_counter, "tipo_imposto": "ICMS", "aliquota": _to_float_br(icms_match.group(1))})
                    itens_fb.append(item)
                    item_idx_counter += 1

        if itens_fb:
            return itens_fb, impostos_fb

        itens_c: List[Dict[str, Any]] = []
        i = 0
        while i < len(linhas):
            ln = linhas[i]
            found = self.RE_MOEDA_BR.findall(ln)
            nxt_found = self.RE_MOEDA_BR.findall(linhas[i + 1]) if i + 1 < len(linhas) else []
            if len(found) >= 2 or (len(found) == 1 and len(nxt_found) >= 1):
                desc = re.split(self.RE_MOEDA_BR.pattern, ln)[0].strip()
                qtd = None
                un = None
                qtd_m = re.search(r"(\d+(?:[\.,]\d{1,3})?)\s*(?:UN|UND|UNID|PC|PÇ|CX|KG|LT|UNIDADE|UNID\.)?", ln, re.I)
                if qtd_m:
                    qtd = _to_float_br(qtd_m.group(1))
                    un_m = re.search(r"\b(UN|UND|UNID|PC|PÇ|CX|KG|LT|UNIDADE)\b", ln, re.I)
                    if un_m:
                        un = un_m.group(1).upper()

                valores = found + nxt_found
                if len(valores) >= 2:
                    vun = _to_float_br(valores[-2])
                    vtot = _to_float_br(valores[-1])
                    item = {
                        "descricao": _norm_ws(desc) or None,
                        "unidade": un,
                        "quantidade": qtd,
                        "valor_unitario": vun,
                        "valor_total": vtot,
                        "ncm": None,
                        "cfop": None,
                    }
                    itens_c.append(item)
                    i += 2
                    continue
            i += 1

        return itens_c, []

    def _parse_itens_linhas(self, linhas: List[str], inicio: int, fim: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        itens: List[Dict[str, Any]] = []
        impostos: List[Dict[str, Any]] = []
        item_idx_counter = 0

        for i in range(inicio, max(inicio, fim)):
            linha = _norm_ws(linhas[i].strip())
            if not linha:
                continue

            match = self.RE_ITEM_LINHA.search(linha)
            if match:
                item_data = match.groupdict()
                item = {
                    "descricao": _norm_ws(item_data.get("desc", "")),
                    "unidade": item_data.get("unid"),
                    "quantidade": _to_float_br(item_data.get("qtd")),
                    "valor_unitario": _to_float_br(item_data.get("vun")),
                    "valor_total": _to_float_br(item_data.get("vtot")),
                    "ncm": None,
                    "cfop": None,
                }

                linha_anterior = linhas[i - 1].strip() if (i - 1) >= inicio else ""
                linha_seguinte = linhas[i + 1].strip() if (i + 1) < fim else ""
                ncm_match = self.RE_NCM_ITEM.search(linha) or self.RE_NCM_ITEM.search(linha_seguinte) or self.RE_NCM_ITEM.search(linha_anterior)
                cfop_match = self.RE_CFOP_ITEM.search(linha) or self.RE_CFOP_ITEM.search(linha_seguinte) or self.RE_CFOP_ITEM.search(linha_anterior)
                if ncm_match:
                    item["ncm"] = ncm_match.group(1)
                if cfop_match:
                    item["cfop"] = cfop_match.group(1)

                itens.append(item)

                icms_match = re.search(r"ICMS.*?(\d+,\d{2})\s*%", linha_seguinte, re.I) or re.search(r"ICMS.*?(\d+,\d{2})\s*%", linha_anterior, re.I)
                if icms_match:
                    impostos.append({"item_idx": item_idx_counter, "tipo_imposto": "ICMS", "aliquota": _to_float_br(icms_match.group(1))})

                item_idx_counter += 1
                continue

            cols = re.split(r"\s{2,}", linha)
            if len(cols) >= 4:
                desc = cols[0]
                qtd = _to_float_br(cols[-3])
                vun = _to_float_br(cols[-2])
                vtot = _to_float_br(cols[-1])
                un = None
                if re.fullmatch(r"[A-Z]{1,4}", cols[1] or ""):
                    un = cols[1]

                it = {
                    "descricao": _norm_ws(desc),
                    "unidade": un,
                    "quantidade": qtd,
                    "valor_unitario": vun,
                    "valor_total": vtot,
                    "ncm": None,
                    "cfop": None,
                }
                itens.append(it)
                item_idx_counter += 1

        log.info(f"AgenteNLP: Extraídos {len(itens)} itens via OCR (intervalo).")
        return itens, impostos

    # ---------------------- Validações & Sanitização -------------------
    def _valor_valido_para_chave(self, k: str, v: Any) -> bool:
        if v in (None, "", []):
            return False
        try:
            if k in ("emitente_cnpj", "destinatario_cnpj", "cnpj_autorizado"):
                d = _only_digits(str(v))
                return bool(d and len(d) >= 14)
            if k in ("emitente_cpf", "destinatario_cpf"):
                d = _only_digits(str(v))
                return bool(d and len(d) == 11)
            if k.startswith("data_"):
                return _parse_date_like(str(v)) is not None
            if k.startswith("valor_") or k.startswith("total_"):
                if isinstance(v, (int, float)):
                    return float(v) >= 0
                return _to_float_br(str(v)) is not None
        except Exception:
            return False
        return True

    def _soma_valor_itens(self, itens: List[Dict[str, Any]]) -> Optional[float]:
        soma = 0.0
        ok = False
        for it in itens or []:
            v = it.get("valor_total")
            if isinstance(v, (int, float)):
                soma += float(v)
                ok = True
            elif v:
                fv = _to_float_br(str(v))
                if fv is not None:
                    soma += fv
                    ok = True
        return round(soma, 2) if ok else None

    def _sanear_dados(self, d: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(d)

        for k in ("emitente_cnpj", "destinatario_cnpj", "cnpj_autorizado"):
            if out.get(k):
                out[k] = _only_digits(str(out[k])) or None
        for k in ("emitente_cpf", "destinatario_cpf"):
            if out.get(k):
                out[k] = _only_digits(str(out[k])) or None

        for k in [c for c in out.keys() if c.startswith("data_")]:
            if out.get(k):
                out[k] = _parse_date_like(str(out[k]))

        campos_valor = [
            "valor_total","total_produtos","total_servicos","total_icms","total_ipi","total_pis","total_cofins","valor_iss",
            "valor_descontos","valor_outros","valor_frete","valor_liquido",
        ]
        for k in campos_valor:
            if out.get(k) is not None:
                if isinstance(out[k], (int, float)):
                    out[k] = float(out[k])
                else:
                    out[k] = _to_float_br(str(out[k]))

        for k, v in list(out.items()):
            if isinstance(v, str) and not v.strip():
                out[k] = None

        return out

    # --------------------- Pré-normalização de OCR ---------------------
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

    # --------------------- Cobertura / Trigger LLM ---------------------
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


__all__ = ["AgenteNLP"]
