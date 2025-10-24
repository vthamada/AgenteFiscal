# agentes/nlp.py

from __future__ import annotations
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from .utils import (
    _parse_date_like, _to_float_br, _only_digits, _norm_ws
)

# ---------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------
log = logging.getLogger("projeto_fiscal.agentes")

# ---------------------------------------------------------------------
# Agente NLP
# ---------------------------------------------------------------------
class AgenteNLP:
    """Heurísticas regex para extrair campos e itens de texto OCR."""

    RE_CNPJ = re.compile(r"\b(\d{2}\.?\d{3}\.?\d{3}/?\d{4}-?\d{2}|\d{14})\b")
    RE_CPF = re.compile(r"\b(\d{3}\.?\d{3}\.?\d{3}-?\d{2}|\d{11})\b")
    RE_IE = re.compile(r"\b(?:IE|I\.E\.|INSC(?:RI[ÇC][ÃA]O)?\sESTADUAL)[:\s\-]*([A-Z0-9.\-/]{5,20})\b", re.I)
    RE_UF = re.compile(r"\b(AC|AL|AP|AM|BA|CE|DF|ES|GO|MA|MT|MS|MG|PA|PB|PR|PE|PI|RJ|RN|RS|RO|RR|SC|SP|SE|TO)\b")
    RE_VALOR_TOTAL = re.compile(
        r"\b(?:VALOR\s+TOTAL\s+DA\s+NOTA|VALOR\s+TOTAL\s+DA\s+NF|VALOR\s+TOTAL|TOTAL\s+DA\s+NOTA|VALOR\s+L[IÍ]QUIDO|VALOR\s+A\s+PAGAR)\s*[:\-]?\s*R?\$\s*([\d.,]+)\b",
        re.I,
    )
    RE_DATA_EMISSAO = re.compile(
        r"\b(?:DATA\s+(?:DE\s+)?EMISS[ÃA]O|EMISS[ÃA]O|EMITIDO\s+EM|COMPET[ÊE]NCIA)\s*[:\-]?\s*(\d{2,4}[-/]\d{2}[-/]\d{2,4})\b",
        re.I,
    )
    RE_ITEM_LINHA = re.compile(
        r"^(?:\d+\s+)?(?P<desc>.+?)\s+(?P<unid>[A-Z]{1,3})\s+(?P<qtd>[\d.,]+)\s+(?P<vun>[\d.,]+)\s+(?P<vtot>[\d.,]+)$",
        re.I | re.M,
    )
    RE_NCM_ITEM = re.compile(r"NCM[:\s]*([0-9]{8})", re.I)
    RE_CFOP_ITEM = re.compile(r"CFOP[:\s]*([0-9]{4})", re.I)
    RE_CHAVE = re.compile(r"\b(\d{44})\b")
    RE_QR = re.compile(r"(?:chNFe|chCTe)=([0-9]{44})")

    def extrair_campos(self, texto: str) -> Dict[str, Any]:
        t_norm = _norm_ws(texto)

        cnpjs = [_only_digits(m) for m in self.RE_CNPJ.findall(t_norm)]
        cpfs = [_only_digits(m) for m in self.RE_CPF.findall(t_norm)]

        emit_cnpj_cpf = (cnpjs[0] if cnpjs else None) or (cpfs[0] if cpfs else None)
        dest_cnpj_cpf = None
        if len(cnpjs) > 1:
            dest_cnpj_cpf = cnpjs[1]
        elif cnpjs and cpfs:
            dest_cnpj_cpf = cpfs[0]
        elif len(cpfs) > 1:
            dest_cnpj_cpf = cpfs[1]

        m_ie = self.RE_IE.search(t_norm)
        ie = m_ie.group(1).strip() if m_ie else None

        endereco_match = self._match_after(t_norm, ["endereço", "endereco", "logradouro", "rua"], max_len=150)
        municipio_match = self._match_after(t_norm, ["município", "municipio", "cidade"], max_len=80)

        uf = None
        uf_match = self.RE_UF.search(endereco_match[-10:]) if endereco_match else None
        if not uf_match and municipio_match:
            uf_match = self.RE_UF.search(municipio_match[-10:])
        if not uf_match:
            uf_match = self.RE_UF.search(t_norm[-200:])
        uf = uf_match.group(1) if uf_match else None

        # razão social / emitente
        razao = self._match_after(
            t_norm,
            ["razão social", "razao social", "nome/razão", "emitente", "emitida por", "prestador", "empresa"],
            max_len=120,
        )

        m_valor = self.RE_VALOR_TOTAL.search(t_norm)
        valor_total = _to_float_br(m_valor.group(1)) if m_valor else None

        m_data = self.RE_DATA_EMISSAO.search(t_norm)
        data_emissao = _parse_date_like(m_data.group(1)) if m_data else None

        # Chave de acesso possível
        chave = None
        mqr = self.RE_QR.search(t_norm) or self.RE_CHAVE.search(t_norm)
        if mqr:
            chave = _only_digits(mqr.group(1))

        itens_extraidos, impostos_extraidos = self._extrair_itens_impostos_ocr(texto)

        return {
            "emitente_cnpj": _only_digits(emit_cnpj_cpf) if len(emit_cnpj_cpf or "") >= 14 else None,
            "emitente_cpf": _only_digits(emit_cnpj_cpf) if len(emit_cnpj_cpf or "") == 11 else None,
            "destinatario_cnpj": _only_digits(dest_cnpj_cpf) if len(dest_cnpj_cpf or "") >= 14 else None,
            "destinatario_cpf": _only_digits(dest_cnpj_cpf) if len(dest_cnpj_cpf or "") == 11 else None,
            "inscricao_estadual": ie,
            "emitente_nome": razao,
            "endereco": endereco_match,
            "uf": uf,
            "municipio": municipio_match,
            "valor_total": valor_total,
            "data_emissao": data_emissao,
            "chave_acesso": chave,
            "itens_ocr": itens_extraidos,
            "impostos_ocr": impostos_extraidos,
        }

    def _match_after(self, texto: str, labels: List[str], max_len: int = 80, max_dist: int = 50) -> Optional[str]:
        texto_lower = texto.lower()
        best_match = None
        min_pos = float("inf")

        for lab in labels:
            lab_lower = lab.lower()
            idx = texto_lower.find(lab_lower)
            if idx == -1:
                continue

            start_value_idx = -1
            end_label_idx = idx + len(lab_lower)
            for i in range(end_label_idx, min(end_label_idx + max_dist, len(texto))):
                if texto[i] in ":|-;>\n" or (i == end_label_idx and texto[i].isspace() and not texto[i + 1 : i + 2].isalnum()):
                    start_value_idx = i + 1
                    break

            if start_value_idx == -1:
                continue

            end_idx = texto.find("\n", start_value_idx)
            if end_idx == -1:
                end_idx = len(texto)

            for next_lab in labels:
                next_idx = texto.find(next_lab, start_value_idx, end_idx)
                if next_idx != -1:
                    end_idx = next_idx

            value = texto[start_value_idx:end_idx].strip()
            value = re.sub(r"\s*[|;/-].*$", "", value).strip()
            if value and idx < min_pos:
                min_pos = idx
                best_match = value[:max_len]

        return best_match

    def _extrair_itens_impostos_ocr(self, texto_original: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        itens: List[Dict[str, Any]] = []
        impostos: List[Dict[str, Any]] = []

        linhas = texto_original.splitlines()
        inicio_itens = -1
        fim_itens = len(linhas)

        header_keywords = ["DESCRIÇÃO", "DESCRICAO", "QTD", "QUANT", "UNIT", "UN", "TOTAL", "PRODUTO", "VALOR UNIT"]
        for i, linha in enumerate(linhas):
            linha_upper = linha.upper()
            score = sum(1 for kw in header_keywords if kw in linha_upper)
            if score >= 2:  # precisa de pelo menos duas pistas
                inicio_itens = i + 1
                break

        if inicio_itens == -1:
            log.info("Início da seção de itens OCR não encontrado.")
            return [], []

        total_keywords = ["TOTAL DOS PRODUTOS", "VALOR TOTAL", "SUBTOTAL", "TOTAL A PAGAR"]
        for i in range(inicio_itens, len(linhas)):
            if any(kw in linhas[i].upper() for kw in total_keywords):
                fim_itens = i
                break

        item_idx_counter = 0
        for i in range(inicio_itens, fim_itens):
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

                linha_anterior = linhas[i - 1].strip() if (i - 1) >= inicio_itens else ""
                linha_seguinte = linhas[i + 1].strip() if (i + 1) < fim_itens else ""
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

        log.info(f"AgenteNLP: Extraídos {len(itens)} itens via OCR.")
        return itens, impostos


__all__ = ["AgenteNLP"]
