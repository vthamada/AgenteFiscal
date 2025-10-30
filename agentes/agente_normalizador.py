# agentes/agente_normalizador.py

from __future__ import annotations
import re
from typing import Any, Dict, List, Optional, Tuple

from .utils import (
    _parse_date_like, _to_float_br, _only_digits, _norm_ws,
    _safe_title, _clamp, _UF_SET
)

# LLM opcional (somente para refino/contexto quando útil)
try:
    from langchain_core.language_models.chat_models import BaseChatModel  # type: ignore
    from langchain_core.messages import SystemMessage, HumanMessage  # type: ignore
except Exception:  # pragma: no cover
    BaseChatModel = object  # type: ignore
    SystemMessage = object  # type: ignore
    HumanMessage = object  # type: ignore

# >>> Integrações com modelos_llm (wrappers canônicos) <<<
try:
    from modelos_llm import (
        normalize_text_fields,      # normalizador leve de nomes/endereços
        invoke_with_context,        # invocação genérica (System+User) com meta
        validation_explanations,    # gera explicações curtas e técnicas em PT-BR
    )
except Exception:
    # Fallbacks no caso de ausência do módulo — mantêm operação determinística
    normalize_text_fields = None     # type: ignore
    invoke_with_context = None       # type: ignore
    validation_explanations = None   # type: ignore

# Nota: Evitamos imports de tipos opcionais em tempo de análise para reduzir alertas do linter.


class AgenteNormalizadorCampos:
    """
    Agente de Contextualização Fiscal:
    - Normaliza e consolida campos vindos de XML (e outras fontes estruturadas) antes de persistir.
    - Adiciona raciocínio contextual para adaptar regras conforme o TIPO de documento.
    - Gera explicações das normalizações e um score de sanidade (0–1).
    - Sugere reprocessamento quando detectar inconsistências relevantes.

    Compatibilidade com DB:
    - remove campos genéricos (uf/municipio/endereco/inscricao_estadual) se drop_general_fields=True
      e guarda cópia em __context__ se keep_context_copy=True.
    """

    # Cidade/UF padrões
    RE_CIDADE_UF_SLASH = re.compile(r"\b([A-Za-zÀ-ÿ'`^~\-.\s]{2,})\s*/\s*([A-Za-z]{2})\b")
    RE_CIDADE_UF_HIFEN = re.compile(r"\b([A-Za-zÀ-ÿ'`^~\-.\s]{2,})\s*-\s*([A-Za-z]{2})\b")
    RE_CIDADE_UF_VIRG  = re.compile(r"\b([A-Za-zÀ-ÿ'`^~\-.\s]{2,}),\s*([A-Za-z]{2})\b")
    RE_UF               = re.compile(r"\b(AC|AL|AP|AM|BA|CE|DF|ES|GO|MA|MT|MS|MG|PA|PB|PR|PE|PI|RJ|RN|RS|RO|RR|SC|SP|SE|TO)\b")

    # Chave de acesso (44 dígitos em qualquer lugar)
    RE_CHAVE_44 = re.compile(r"(\d{44})")

    # Identificação
    RE_NUMERO = re.compile(r"^\d{1,10}$")
    RE_SERIE  = re.compile(r"^[A-Z0-9\-]{1,6}$", re.I)
    RE_MODELO = re.compile(r"^[0-9A-Z\-]{1,10}$", re.I)

    # CFOP/NCM/CEST
    RE_CFOP = re.compile(r"^\d{4}$")
    RE_NCM  = re.compile(r"^\d{8}$")
    RE_CEST = re.compile(r"^\d{7}$")

    # IE válida: somente dígitos (>=3) ou "ISENTO"
    RE_IE_OK = re.compile(r"^(?:\d{3,}|ISENTO)$", re.IGNORECASE)

    # Indicadores de tipo documental (heurística)
    _HINTS_NFE  = (r"\bNF[\- ]?E\b", r"\bNFE\b", r"\bMODELO\s*55\b", r"\bchNFe\b")
    _HINTS_NFCE = (r"\bNFC[\- ]?E\b", r"\bNFCE\b", r"\bMODELO\s*65\b", r"CONSUMIDOR FINAL", r"QR[- ]?CODE")
    _HINTS_NFSE = (r"\bNF[\- ]?S[EÉ]\b", r"\bNFS[EÉ]\b", r"\bISS\b", r"\bSERVI[ÇC]O[S]?\b")
    _HINTS_CTE  = (r"\bCT[\- ]?E\b", r"\bCTE\b", r"\bConhecimento de Transporte\b")

    # Bandeiras/cartões conhecidas (normalização)
    _BANDEIRAS = {
        "VISA", "MASTERCARD", "ELO", "AMEX", "AMERICAN EXPRESS", "HIPERCARD",
        "DINERS", "ALELO", "SOROCRED", "CABAL", "BANESCARD"
    }

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

    # Palavras que indicam PJ (para heurística de inversão)
    _TOKENS_PJ = (
        "LTDA", "EIRELI", "MEI", "EPP", "ME", "S/A", "S.A.", "INDÚSTRIA", "INDUSTRIA",
        "COMÉRCIO", "COMERCIO", "COMERCIAL", "IGREJA", "MINISTÉRIO", "MINISTERIO",
        "PREFEITURA", "ASSOCIAÇÃO", "ASSOCIACAO", "FUNDACAO", "FUNDAÇÃO", "COOPERATIVA",
        "SERVIÇOS", "SERVICOS", "TRANSPORTES", "TRANSPORTE"
    )

    def __init__(
        self,
        llm: Optional[Any] = None,
        *,
        enable_llm: bool = False,
        drop_general_fields: bool = True,
        keep_context_copy: bool = True,
        preserve_siglas: bool = True,
    ):
        self.llm = llm if enable_llm and isinstance(llm, BaseChatModel) else None
        self.enable_llm = bool(self.llm is not None)
        self.drop_general_fields = bool(drop_general_fields)
        self.keep_context_copy = bool(keep_context_copy)
        self.preserve_siglas = bool(preserve_siglas)

    # ---------------------------------------------------------------------
    # Público
    # ---------------------------------------------------------------------
    def normalizar(
        self,
        campos: Dict[str, Any],
        *,
        drop_general_fields: Optional[bool] = None,
        keep_context_copy: bool = False
    ) -> Dict[str, Any]:
        """
        - Consolida aliases, normaliza datas/valores/identificação/UF/município/IE/IM/pagamento/chave.
        - Valida CFOP/NCM/CEST (cabeçalho) e faz heurística de inversão emitente↔destinatário.
        - Adiciona __meta__['normalizador'] com:
            {
              "tipo_documento": "NFE|NFCE|NFSE|CTE|CFE|DESCONHECIDO",
              "sanity_score": 0..1,
              "explicacoes": [ ... ],
              "sugerir_reprocessamento": bool
            }
        """
        # Flags com fallback seguro
        use_drop = bool(drop_general_fields if drop_general_fields is not None else getattr(self, "drop_general_fields", False))
        use_ctx  = bool(keep_context_copy or getattr(self, "keep_context_copy", False))

        src = dict(campos or {})
        out = dict(src)
        explicacoes: List[str] = []

        # 0) Snapshot de campos genéricos (se for remover depois)
        context_snapshot: Dict[str, Any] = {}
        if use_ctx:
            for k in ("uf", "municipio", "endereco", "inscricao_estadual"):
                if k in out:
                    context_snapshot[k] = out.get(k)

        # 1) Consolidar aliases numéricos (fail-safe)
        try:
            self._consolidar_aliases_numericos(out)
        except Exception:
            pass

        # 2) Datas
        for k in ("data_emissao", "data_saida", "data_recebimento", "competencia", "data_autorizacao"):
            if out.get(k):
                before = out.get(k)
                out[k] = _parse_date_like(str(out.get(k)))
                if before and out[k] != before:
                    explicacoes.append(f"Normalizei '{k}' para formato ISO (antes='{before}', depois='{out[k]}').")

        # 3) Valores numéricos/financeiros
        campos_valor = {
            "valor_total",
            "total_produtos", "total_servicos",
            "total_icms", "total_ipi", "total_pis", "total_cofins",
            "valor_iss", "valor_descontos", "valor_outros", "valor_frete",
            "valor_seguro", "valor_despesas_acessorias", "valor_troco"
        }
        for k in list(out.keys()):
            if k in campos_valor and out.get(k) is not None:
                before = out.get(k)
                out[k] = _to_float_br(str(out.get(k)))
                if before != out[k]:
                    explicacoes.append(f"Converto '{k}' para número decimal (antes='{before}', depois='{out[k]}').")

        # 4) CNPJ/CPF somente dígitos
        for k in ("emitente_cnpj", "destinatario_cnpj", "emitente_cpf", "destinatario_cpf", "cnpj_autorizado"):
            if out.get(k):
                before = out.get(k)
                out[k] = _only_digits(out.get(k))
                if before != out[k]:
                    explicacoes.append(f"Sanitizo '{k}' para dígitos (antes='{before}', depois='{out[k]}').")

        # 5) IE/IM sanitização conservadora
        def _sanitize_ie(val: Any) -> Optional[str]:
            if not val:
                return None
            v = str(val).strip().upper()
            v = re.sub(r"[^0-9A-Z./\-]", "", v)
            v = v.replace("BAIRRO", "").replace("POSTO", "").replace("RUA", "")
            v = _norm_ws(v)
            return v if self.RE_IE_OK.match(v or "") else None

        for k in ("emitente_ie", "destinatario_ie", "inscricao_estadual"):
            before = out.get(k)
            out[k] = _sanitize_ie(out.get(k))
            if before and before != out.get(k):
                explicacoes.append(f"Sanitizo '{k}' (antes='{before}', depois='{out.get(k)}').")

        for k in ("emitente_im", "destinatario_im"):
            if out.get(k):
                before = out.get(k)
                v = str(out.get(k)).strip().upper()
                v = re.sub(r"[^0-9A-Z./\-]", "", v)
                out[k] = _norm_ws(v) or None
                if before != out[k]:
                    explicacoes.append(f"Sanitizo '{k}' (antes='{before}', depois='{out[k]}').")

        # 6) Identificação (número/serie/modelo)
        out["numero_nota"] = self._norm_numero_nota(out.get("numero_nota"))
        out["serie"]       = self._norm_serie(out.get("serie"))
        out["modelo"]      = self._norm_modelo(out.get("modelo"))

        # 7) UF/Município
        try:
            self._normalize_uf_municipio(out)
        except Exception:
            # Fallback mínimo (nunca quebra)
            uf_keys = ["emitente_uf", "destinatario_uf", "uf"]
            for k in uf_keys:
                if out.get(k):
                    s = str(out[k]).strip().upper()
                    s = re.sub(r"[^A-Z]", "", s)
                    out[k] = s[:2] if len(s) >= 2 else None
            for k in ("emitente_municipio", "destinatario_municipio", "municipio"):
                if out.get(k):
                    s = str(out.get(k)).strip()
                    s = re.sub(r"\s*-\s*[A-Z]{2}\b.*$", "", s)
                    s = re.sub(r"\bCEP[:\s]*\d{5}-?\d{3}.*$", "", s, flags=re.I)
                    out[k] = s.strip() or None

        # 8) Nomes (title case com siglas preservadas)
        for k in ("emitente_nome", "destinatario_nome"):
            if out.get(k):
                before = out.get(k)
                out[k] = self._safe_title_preserve_siglas(str(out.get(k)))
                if before and out[k] != before:
                    explicacoes.append(f"Padronizo '{k}' para Title Case com siglas preservadas.")

        # 8.1) Refino opcional via LLM (curto e seguro)
        if self.enable_llm and self.llm and normalize_text_fields:
            try:
                refined = self._llm_refine_text_fields(
                    out, fields=("emitente_nome", "destinatario_nome", "endereco", "municipio")
                )
                if refined is not out:
                    out = refined
            except Exception:
                pass

        # 9) Chave de acesso (44 dígitos)
        if out.get("chave_acesso"):
            before = out.get("chave_acesso")
            ca = self._find_chave_44(str(out.get("chave_acesso")))
            out["chave_acesso"] = ca or None
            if before and out["chave_acesso"] != before:
                explicacoes.append("Ajusto 'chave_acesso' para sequência válida de 44 dígitos.")

        # 10) Pagamento
        try:
            self._normalize_pagamento(out)
        except Exception:
            # Fallback: não quebra se a função for custom e lançar erro
            if out.get("forma_pagamento"):
                out["forma_pagamento"] = _norm_ws(str(out.get("forma_pagamento")).upper())

        # 11) CFOP/NCM/CEST (cabeçalho)
        for head_key, pattern in (("cfop", self.RE_CFOP), ("ncm", self.RE_NCM), ("cest", self.RE_CEST)):
            if out.get(head_key):
                before = out.get(head_key)
                v = _only_digits(str(out.get(head_key)))
                out[head_key] = v if pattern.match(v or "") else None
                if before and out[head_key] != before:
                    explicacoes.append(f"Sanitizo '{head_key}' (antes='{before}', depois='{out[head_key]}').")

        # 12) Heurística de inversão emitente↔destinatário
        swap_done = False
        try:
            swap_done = self._fix_possible_swap_emit_dest(out)
        except Exception:
            swap_done = False
        if swap_done:
            explicacoes.append("Detecto inversão Emitente↔Destinatário e corrijo de forma segura.")

        # 13) Inferência do tipo de documento (heurística + LLM desempate)
        try:
            tipo_doc = self._infer_document_type(out)
        except Exception:
            tipo_doc = "DESCONHECIDO"

        if tipo_doc == "DESCONHECIDO" and self.enable_llm and self.llm and invoke_with_context:
            texto_ref = " ".join(
                str(x) for x in [
                    out.get("modelo"), out.get("natureza_operacao"), out.get("observacoes"),
                    out.get("emitente_nome"), out.get("destinatario_nome")
                ] if x
            ).upper()
            try:
                r = invoke_with_context(  # type: ignore[misc]
                    self.llm,
                    system_prompt=(
                        "Classifique o tipo de documento fiscal brasileiro. "
                        "Escolha UMA opção entre: NFE, NFCE, NFSE, CTE, DESCONHECIDO. "
                        "Responda apenas a palavra."
                    ),
                    user_prompt=texto_ref[:2000],
                    task_tag="doc_type_classifier",
                    temperature=0.0,
                    json_expected=False,
                )
                guess = (r.get("content") or "").strip().upper()
                if guess in {"NFE", "NFCE", "NFSE", "CTE"}:
                    tipo_doc = guess
                    explicacoes.append(f"Tipo documental inferido via LLM: {tipo_doc}.")
            except Exception:
                pass

        # 14) Score de sanidade + verificações semânticas
        def _avaliar_coerencia_semantica_local(d: Dict[str, Any]) -> Tuple[float, List[str], List[str]]:
            inconsist: List[str] = []
            dicas: List[str] = []
            crit_ok = 0
            total_crit = 3  # emitente_cnpj, valor_total, data_emissao

            if _only_digits(str(d.get("emitente_cnpj") or "")).isdigit() and len(_only_digits(str(d.get("emitente_cnpj") or ""))) >= 14:
                crit_ok += 1
            else:
                inconsist.append("CNPJ do emitente ausente/inválido.")

            if isinstance(d.get("valor_total"), (int, float)) and (d.get("valor_total") or 0) > 0:
                crit_ok += 1
            else:
                inconsist.append("Valor total ausente/inválido.")

            if d.get("data_emissao"):
                crit_ok += 1
            else:
                inconsist.append("Data de emissão ausente/inválida.")

            # Consistência básica com itens (quando existir)
            try:
                if "itens" in d and isinstance(d["itens"], list) and d.get("valor_total"):
                    soma = 0.0
                    ok = False
                    for it in d["itens"]:
                        v = it.get("valor_total")
                        if isinstance(v, (int, float)):
                            soma += float(v); ok = True
                    if ok:
                        # aceita divergência de até 25% ou R$ 1,00
                        if abs(float(d["valor_total"]) - soma) > max(1.0, 0.25 * float(d["valor_total"])):
                            inconsist.append("Soma dos itens distante do valor total (tolerância 25% ou R$ 1,00).")
            except Exception:
                pass

            sanity = max(0.0, min(1.0, crit_ok / float(total_crit)))
            if sanity < 0.65:
                dicas.append("Reprocessar com fonte mais rica ou revisar manualmente; campos críticos estão ausentes.")
            return sanity, inconsist, dicas

        try:
            sanity_score, inconsistencias, dicas = self._avaliar_coerencia_semantica(out, tipo_doc)
        except Exception:
            sanity_score, inconsistencias, dicas = _avaliar_coerencia_semantica_local(out)

        # Explicações determinísticas
        explicacoes.extend(inconsistencias)
        sugerir_reproc = sanity_score < 0.65 or any(
            any(tok in s.lower() for tok in ("distante", "ausente", "inválido", "invalido"))
            for s in inconsistencias
        )
        if sugerir_reproc and dicas:
            explicacoes.extend(dicas)

        # …e versões curtas/claras via LLM (opcional)
        if self.enable_llm and self.llm and validation_explanations and (inconsistencias or dicas):
            try:
                issues_payload = {"inconsistencias": inconsistencias, "dicas": dicas, "tipo_documento": tipo_doc}
                ve = validation_explanations(self.llm, issues_payload=issues_payload, temperature=0.0)  # type: ignore[misc]
                msgs = ve.get("json")
                if isinstance(msgs, list) and msgs:
                    for m in msgs:
                        sm = str(m).strip()
                        if sm and sm not in explicacoes:
                            explicacoes.append(sm)
            except Exception:
                pass

        # 15) Compat DB: remover genéricos (salvando cópia em __context__ se for o caso)
        if use_ctx:
            ctx = dict(out.get("__context__", {}))
            if context_snapshot:
                ctx.update({"general_fields": context_snapshot})
            if ctx:
                out["__context__"] = ctx

        if use_drop:
            for k in ("uf", "municipio", "endereco", "inscricao_estadual"):
                out.pop(k, None)

        # 16) __meta__.normalizador (sem quebrar meta já existente)
        meta_norm = {
            "tipo_documento": str(tipo_doc or "DESCONHECIDO").upper(),
            "sanity_score": round(float(sanity_score), 3),
            "explicacoes": explicacoes,
            "sugerir_reprocessamento": bool(sugerir_reproc)
        }
        meta = dict(out.get("__meta__", {}))
        meta["normalizador"] = meta_norm
        out["__meta__"] = meta

        # 17) Espelhos legados
        try:
            self._preencher_espelhos_legados(out)
        except Exception:
            pass

        return out

    def normalizar_itens(self, itens: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Normaliza itens: descrição/unidade, quantidades/valores, NCM/CFOP/CEST/código,
        alíquotas (0..100) e valida CFOP/NCM/CEST por comprimento.
        """
        norm_itens: List[Dict[str, Any]] = []
        if not itens:
            return norm_itens

        for it in itens:
            d = dict(it or {})

            # Descrição / unidade
            d["descricao"] = _norm_ws(str(d.get("descricao") or "")).strip() or None
            d["unidade"]   = (str(d.get("unidade") or "").strip().upper() or None)

            # Quantitativos e valores
            for kk in ("quantidade", "valor_unitario", "valor_total",
                       "valor_icms", "valor_ipi", "valor_pis", "valor_cofins"):
                if d.get(kk) is not None:
                    d[kk] = _to_float_br(str(d.get(kk)))

            # Alíquotas (0..100)
            for kk in ("aliquota_icms", "aliquota_ipi", "aliquota_pis", "aliquota_cofins"):
                if d.get(kk) is not None:
                    d[kk] = _clamp(_to_float_br(str(d.get(kk))), 0, 100)

            # Códigos fiscais
            for kk, patt in (("ncm", self.RE_NCM), ("cfop", self.RE_CFOP), ("cest", self.RE_CEST)):
                if d.get(kk):
                    v = _only_digits(str(d.get(kk)))
                    d[kk] = v if patt.match(v or "") else None

            # Código do produto
            if d.get("codigo_produto"):
                vraw = str(d.get("codigo_produto")).strip()
                vdigits = _only_digits(vraw)
                d["codigo_produto"] = vdigits if vdigits else _norm_ws(vraw)

            norm_itens.append(d)

        return norm_itens

    def fundir(self, *fontes: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fusão por prioridade: fontes mais à direita têm prioridade por campo não-nulo.
        Após a fusão, consolida aliases e preenche espelhos legados.
        """
        out: Dict[str, Any] = {}
        for src in fontes:
            if not src:
                continue
            for k, v in src.items():
                if v in (None, "", [], {}):
                    continue
                out[k] = v

        self._consolidar_aliases_numericos(out)
        self._preencher_espelhos_legados(out)
        return out

    # ---------------------------------------------------------------------
    # Consolidação / Espelhos
    # ---------------------------------------------------------------------
    def _consolidar_aliases_numericos(self, out: Dict[str, Any]) -> None:
        for canonico, aliases in self._ALIASES_NUMERICOS:
            if out.get(canonico) not in (None, "", []):
                continue
            for alias in aliases:
                if out.get(alias) not in (None, "", []):
                    out[canonico] = out.get(alias)
                    break

    def _preencher_espelhos_legados(self, out: Dict[str, Any]) -> None:
        for legacy_key, canonical_key in self._ESPELHOS_LEGADOS:
            if out.get(legacy_key) in (None, "", []):
                if out.get(canonical_key) not in (None, "", []):
                    out[legacy_key] = out[canonical_key]

    # ---------------------------------------------------------------------
    # Identificação
    # ---------------------------------------------------------------------
    def _norm_numero_nota(self, v: Any) -> Optional[str]:
        if v is None:
            return None
        s = _only_digits(str(v))
        if not s:
            return None
        s = s.lstrip("0") or "0"
        return s if self.RE_NUMERO.match(s) else None

    def _norm_serie(self, v: Any) -> Optional[str]:
        if v is None:
            return None
        s = str(v).strip().upper().replace(" ", "")
        return s if self.RE_SERIE.match(s) else None

    def _norm_modelo(self, v: Any) -> Optional[str]:
        if v is None:
            return None
        s = str(v).strip().upper().replace(" ", "")
        return s if self.RE_MODELO.match(s) else None

    # ---------------------------------------------------------------------
    # UF / Município
    # ---------------------------------------------------------------------
    def _normalize_uf_municipio(self, out: Dict[str, Any]) -> None:
        uf = (out.get("uf") or "").strip().upper() or None
        mun = _norm_ws(str(out.get("municipio"))) if out.get("municipio") else None

        def _apply_city_uf_patterns(text: str) -> Tuple[Optional[str], Optional[str]]:
            textn = _norm_ws(text)
            for regex in (self.RE_CIDADE_UF_SLASH, self.RE_CIDADE_UF_HIFEN, self.RE_CIDADE_UF_VIRG):
                m = regex.search(textn)
                if m:
                    cidade = _norm_ws(m.group(1))
                    ufm = (m.group(2) or "").upper()
                    if ufm in _UF_SET:
                        return cidade, ufm
            m2 = re.search(r"([A-Za-zÀ-ÿ'`^~\-.\s]{2,})\s+([A-Za-z]{2})\b", textn)
            if m2:
                c2 = _norm_ws(m2.group(1))
                uf2 = (m2.group(2) or "").upper()
                if uf2 in _UF_SET:
                    return c2, uf2
            return None, None

        if mun:
            c, u = _apply_city_uf_patterns(mun)
            if c:
                mun = c
            if not uf and u:
                uf = u

        if not out.get("emitente_uf") and uf:
            out["emitente_uf"] = uf

        if uf:
            out["uf"] = uf
        if mun:
            out["municipio"] = mun

        if out.get("endereco"):
            out["endereco"] = _norm_ws(str(out.get("endereco")))

    # ---------------------------------------------------------------------
    # Pagamento
    # ---------------------------------------------------------------------
    def _normalize_pagamento(self, out: Dict[str, Any]) -> None:
        if out.get("condicao_pagamento"):
            s = _norm_ws(str(out.get("condicao_pagamento"))).lower()
            if any(x in s for x in ("avista", "à vista", "a vista", "cash", "imediat")):
                out["condicao_pagamento"] = "A_VISTA"
            elif any(x in s for x in ("parcel", "a prazo", "crediario")):
                out["condicao_pagamento"] = "PARCELADO"
            else:
                out["condicao_pagamento"] = s.upper()

        if out.get("meio_pagamento"):
            s = _norm_ws(str(out.get("meio_pagamento"))).lower()
            mp = "OUTRO"
            if any(x in s for x in ("pix", "qr", "instantaneo", "instantâneo")):
                mp = "PIX"
            elif any(x in s for x in ("credito", "crédito")):
                mp = "CARTAO_CREDITO"
            elif any(x in s for x in ("debito", "débito")):
                mp = "CARTAO_DEBITO"
            elif any(x in s for x in ("dinheiro", "cash")):
                mp = "DINHEIRO"
            elif any(x in s for x in ("boleto",)):
                mp = "BOLETO"
            elif any(x in s for x in ("cheque", "check")):
                mp = "CHEQUE"
            elif any(x in s for x in ("transfer", "ted", "doc")):
                mp = "TRANSFERENCIA"
            out["meio_pagamento"] = mp

        if out.get("bandeira_cartao"):
            s = _norm_ws(str(out.get("bandeira_cartao"))).upper()
            if "AMERICAN" in s:
                s = "AMERICAN EXPRESS"
            elif "MASTER" in s:
                s = "MASTERCARD"
            out["bandeira_cartao"] = s

        if out.get("valor_troco") is not None:
            out["valor_troco"] = _to_float_br(str(out.get("valor_troco")))

    # ---------------------------------------------------------------------
    # Chave de Acesso
    # ---------------------------------------------------------------------
    def _find_chave_44(self, text: str) -> Optional[str]:
        if not text:
            return None
        m = self.RE_CHAVE_44.search(_only_digits(text))
        return m.group(1) if m else None

    # ---------------------------------------------------------------------
    # Inversão Emitente↔Destinatário
    # ---------------------------------------------------------------------
    @staticmethod
    def _looks_pj(name: Optional[str]) -> bool:
        if not name:
            return False
        n = _norm_ws(name).upper()
        return any(tok in n for tok in AgenteNormalizadorCampos._TOKENS_PJ)

    @staticmethod
    def _looks_pf(name: Optional[str]) -> bool:
        if not name:
            return False
        n = _norm_ws(name)
        words = [w for w in re.split(r"\s+", n.strip()) if w]
        if len(words) < 2 or len(words) > 4:
            return False
        n_up = n.upper()
        if any(tok in n_up for tok in AgenteNormalizadorCampos._TOKENS_PJ):
            return False
        return True

    def _fix_possible_swap_emit_dest(self, out: Dict[str, Any]) -> bool:
        en, dn = out.get("emitente_nome"), out.get("destinatario_nome")
        if not en and not dn:
            return False

        emit_is_pj = self._looks_pj(en)
        dest_is_pj = self._looks_pj(dn)
        emit_is_pf = self._looks_pf(en)
        dest_is_pf = self._looks_pf(dn)

        emit_cnpj = bool(_only_digits(out.get("emitente_cnpj") or ""))
        dest_cnpj = bool(_only_digits(out.get("destinatario_cnpj") or ""))

        should_swap = False
        if emit_is_pf and dest_is_pj:
            should_swap = True
        elif (not emit_cnpj) and dest_cnpj and dest_is_pj:
            should_swap = True

        if not should_swap:
            return False

        groups = ("nome", "cnpj", "cpf", "ie", "im", "uf", "municipio", "endereco")
        for g in groups:
            e_key = f"emitente_{g}"
            d_key = f"destinatario_{g}"
            out[e_key], out[d_key] = out.get(d_key), out.get(e_key)
        return True

    # ---------------------------------------------------------------------
    # Tipo de Documento (heurística)
    # ---------------------------------------------------------------------
    def _infer_document_type(self, d: Dict[str, Any]) -> str:
        texto_ref = " ".join(
            str(x) for x in [
                d.get("modelo"), d.get("natureza_operacao"), d.get("observacoes"),
                d.get("emitente_nome"), d.get("destinatario_nome")
            ] if x
        ).upper()

        def _has_any(patterns: Tuple[str, ...]) -> bool:
            return any(re.search(p, texto_ref, re.I) for p in patterns)

        if (d.get("modelo") or "").strip() in ("55", "055"):
            return "NFE"
        if (d.get("modelo") or "").strip() in ("65", "065"):
            return "NFCE"

        if _has_any(self._HINTS_NFSE):
            return "NFSE"
        if _has_any(self._HINTS_NFCE):
            return "NFCE"
        if _has_any(self._HINTS_NFE):
            return "NFE"
        if _has_any(self._HINTS_CTE):
            return "CTE"
        return "DESCONHECIDO"

    # ---------------------------------------------------------------------
    # Coerência Semântica / Sanity Score
    # ---------------------------------------------------------------------
    def _avaliar_coerencia_semantica(self, d: Dict[str, Any], tipo_doc: str) -> Tuple[float, List[str], List[str]]:
        """
        Retorna (sanity_score, inconsistencias[], dicas[]).
        - Compara valor_total com decomposição (produtos/serviços + impostos + frete + outros - descontos).
        - Verifica CFOP x UF (regras simples de intra vs interestadual).
        - Checa emitente/destinatário básicos.
        """
        inconsistencias: List[str] = []
        dicas: List[str] = []

        # 1) Comparação de totais
        vt = self._as_float(d.get("valor_total"))
        tp = self._as_float(d.get("total_produtos"))
        ts = self._as_float(d.get("total_servicos"))
        icms = self._as_float(d.get("total_icms"))
        ipi  = self._as_float(d.get("total_ipi"))
        pis  = self._as_float(d.get("total_pis"))
        cof  = self._as_float(d.get("total_cofins"))
        iss  = self._as_float(d.get("valor_iss"))
        desc = self._as_float(d.get("valor_descontos"))
        out_ = self._as_float(d.get("valor_outros"))
        fre  = self._as_float(d.get("valor_frete"))

        comp = 0.0
        for v in (tp, ts, icms, ipi, pis, cof, iss, out_, fre):
            if v is not None:
                comp += v
        if desc is not None:
            comp -= desc

        rel_ok = None
        if vt is not None and comp > 0:
            rel = abs(comp - vt) / max(1.0, vt)
            rel_ok = rel < 0.25  # tolerância de 25%
            if not rel_ok:
                inconsistencias.append(
                    f"Soma de componentes ({round(comp,2)}) inconsistente com valor_total ({round(vt,2)})."
                )
                dicas.append("Revise itens/impostos/frete/descontos; verifique colunas ou origem dos dados.")

        # 2) CFOP x UF (regra simples)
        cfop = (d.get("cfop") or "").strip()
        uf_e = (d.get("emitente_uf") or "").strip().upper()
        uf_d = (d.get("destinatario_uf") or "").strip().upper()
        if cfop and uf_e and uf_d and cfop.isdigit() and len(cfop) == 4:
            p = cfop[0]
            if p == "6" and uf_e == uf_d:
                inconsistencias.append(f"CFOP {cfop} indica operação interestadual, mas emitente_uf={uf_e} = destinatario_uf={uf_d}.")
                dicas.append("Verifique CFOP (6xxx = interestadual). Se operação for interna, use 5xxx/1xxx conforme o caso.")
            if p == "5" and uf_e != uf_d:
                inconsistencias.append(f"CFOP {cfop} indica operação interna, mas UF diferentes (emitente={uf_e}, destinatario={uf_d}).")
                dicas.append("Para interestadual, CFOP típico inicia com 6xxx.")

        # 3) Emissor/destinatário mínimos
        if not d.get("emitente_cnpj") and not d.get("emitente_cpf"):
            inconsistencias.append("Emitente sem CNPJ/CPF.")
        if not d.get("destinatario_cnpj") and not d.get("destinatario_cpf"):
            inconsistencias.append("Destinatário sem CNPJ/CPF.")
        if not d.get("data_emissao"):
            inconsistencias.append("Data de emissão ausente ou inválida.")
        if not d.get("numero_nota"):
            inconsistencias.append("Número da nota ausente ou inválida.")

        # 4) Ajuste do score
        score = 1.0
        penal = 0.0

        # penalizações por inconsistências
        for inc in inconsistencias:
            if "inconsistente" in inc.lower():
                penal += 0.25
            elif "sem" in inc.lower() or "ausente" in inc.lower():
                penal += 0.15
            else:
                penal += 0.10

        if rel_ok is False:
            penal += 0.2

        if tipo_doc == "DESCONHECIDO":
            penal += 0.05

        score = max(0.0, min(1.0, 1.0 - penal))
        return score, inconsistencias, dicas

    @staticmethod
    def _as_float(v: Any) -> Optional[float]:
        if v is None or v == "":
            return None
        try:
            if isinstance(v, (int, float)):
                return float(v)
            return _to_float_br(str(v))
        except Exception:
            return None

    # ---------------------------------------------------------------------
    # Nomes / Siglas
    # ---------------------------------------------------------------------
    def _safe_title_preserve_siglas(self, s: str) -> str:
        t = _safe_title(s)
        if not self.preserve_siglas:
            return _norm_ws(t)

        def up(word: str) -> str:
            return re.sub(rf"\b{re.escape(word)}\b", word.upper(), t, flags=re.I)

        siglas = [
            "LTDA", "ME", "EPP", "EIRELI", "S/A", "SA", "MEI", "SPE", "SIMPLES",
            "SIMPLES NACIONAL", "SCP", "S.A."
        ]
        for sgl in siglas:
            t = up(sgl)
        t = t.replace("S. A.", "S.A.").replace("S / A", "S/A").replace("S. / A.", "S/A")
        return _norm_ws(t)

    # ---------------------------------------------------------------------
    # LLM (opcional) — refino leve de textos curtos (via modelos_llm.normalize_text_fields)
    # ---------------------------------------------------------------------
    def _llm_refine_text_fields(self, campos: Dict[str, Any], *, fields: Tuple[str, ...]) -> Dict[str, Any]:
        if not (self.llm and normalize_text_fields):
            return campos

        payload = {k: campos.get(k) for k in fields if campos.get(k)}
        if not payload:
            return campos

        try:
            resp = normalize_text_fields(self.llm, fields_payload=payload, temperature=0.0)  # type: ignore[misc]
            data = resp.get("json")
            if isinstance(data, dict):
                for k, v in data.items():
                    if k in fields and isinstance(v, str) and v.strip():
                        campos[k] = _norm_ws(v)
        except Exception:
            pass
        return campos


__all__ = ["AgenteNormalizadorCampos"]
