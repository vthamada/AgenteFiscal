# validacao.py

from __future__ import annotations
from typing import Optional, Dict, Any, TYPE_CHECKING, Set, List, Tuple
import re
import yaml
from pathlib import Path
import logging
from functools import lru_cache
import os
import json

if TYPE_CHECKING:
    from banco_de_dados import BancoDeDados
    try:
        from langchain_core.language_models.chat_models import BaseChatModel  # type: ignore
        from langchain_core.messages import SystemMessage, HumanMessage       # type: ignore
    except Exception:  # pragma: no cover
        BaseChatModel = object  # type: ignore
        SystemMessage = object  # type: ignore
        HumanMessage = object   # type: ignore

# --------------------------------------------------------------------
# Imports tolerantes (mock sob falha)
# --------------------------------------------------------------------
try:
    from banco_de_dados import BancoDeDados
    import pandas as pd
except Exception as e:
    logging.error(f"Erro ao importar BancoDeDados/pandas: {e}")

    class _DBMock(object):
        def get_documento(self, _id): return {}
        def query_table(self, *_args, **_kw):
            try:
                import pandas as _pd
                return _pd.DataFrame()
            except Exception:
                return pd.DataFrame()  # type: ignore
        def atualizar_documento_campo(self, *_args, **_kw): return None
        def log(self, *_args, **_kw): return None

    try:
        import pandas as _pd
        pd = _pd  # type: ignore
    except Exception:
        class _DFMock:
            @property
            def empty(self): return True
            @property
            def columns(self): return []
            def fillna(self, _): return self
            def sum(self): return 0.0
            def iterrows(self): return iter([])
            def __getitem__(self, _): return self
            def unique(self): return []
            def tolist(self): return []
            def sort_values(self, *_, **__): return self
            def astype(self, *_a, **_k): return self
        pd = type("pandas", (), {"DataFrame": _DFMock, "Series": _DFMock, "to_numeric": lambda *a, **k: 0})  # type: ignore

    BancoDeDados = _DBMock  # type: ignore

# --------------------------------------------------------------------
# Import LLM tolerante (p/ _analisar_anomalias_llm e enriquecimento)
# --------------------------------------------------------------------
try:
    from langchain_core.language_models.chat_models import BaseChatModel  # type: ignore
    from langchain_core.messages import SystemMessage, HumanMessage       # type: ignore
except Exception:  # pragma: no cover
    BaseChatModel = object  # type: ignore
    class _Msg:
        def __init__(self, content: str): self.content = content
    SystemMessage = _Msg  # type: ignore
    HumanMessage = _Msg   # type: ignore

log = logging.getLogger("agente_fiscal.validacao")
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    log.addHandler(_h)
    log.setLevel(logging.INFO)

# --------------------------------------------------------------------
# Auxiliares determinísticos
# --------------------------------------------------------------------
def _only_digits(s: Optional[str]) -> str:
    return re.sub(r"\D+", "", s or "")

def _valida_cnpj(cnpj: Optional[str]) -> bool:
    if not cnpj:
        return False
    c = _only_digits(cnpj)

    if os.getenv("ALLOW_DEV_CNPJ") in ("1", "true", "True"):
        if c.startswith(("000", "111", "222", "333", "444", "555", "666", "777", "888", "999", "123")):
            return True

    if len(c) != 14 or len(set(c)) == 1:
        return False
    try:
        pesos1 = [5,4,3,2,9,8,7,6,5,4,3,2]
        soma1 = sum(int(c[i]) * pesos1[i] for i in range(12))
        dv1 = 0 if (soma1 % 11) < 2 else 11 - (soma1 % 11)
        pesos2 = [6] + pesos1
        soma2 = sum(int(c[i]) * pesos2[i] for i in range(13))
        dv2 = 0 if (soma2 % 11) < 2 else 11 - (soma2 % 11)
        return c[-2:] == f"{dv1}{dv2}"
    except Exception:
        return False

def _valida_cpf(cpf: Optional[str]) -> bool:
    c = _only_digits(cpf)
    if len(c) != 11 or len(set(c)) == 1:
        return False
    try:
        for i in range(9, 11):
            soma = sum(int(c[num]) * ((i + 1) - num) for num in range(0, i))
            digito = ((soma * 10) % 11) % 10
            if int(c[i]) != digito:
                return False
        return True
    except Exception:
        return False

def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        try:
            s = str(x).strip()
            if s.count(",") == 1:
                s = s.replace(".", "").replace(",", ".")
            return float(s)
        except Exception:
            return 0.0

# --------------------------------------------------------------------
# Regras: código da UF a partir do cUF (2 dígitos) da chave 44
# --------------------------------------------------------------------
_COD_UF_TO_UF = {
    "11":"RO","12":"AC","13":"AM","14":"RR","15":"PA","16":"AP","17":"TO",
    "21":"MA","22":"PI","23":"CE","24":"RN","25":"PB","26":"PE","27":"AL","28":"SE","29":"BA",
    "31":"MG","32":"ES","33":"RJ","35":"SP",
    "41":"PR","42":"SC","43":"RS",
    "50":"MS","51":"MT","52":"GO","53":"DF"
}
def _uf_from_chave44(chave: str) -> Optional[str]:
    c = _only_digits(chave)
    if len(c) != 44:
        return None
    return _COD_UF_TO_UF.get(c[:2])

# --------------------------------------------------------------------
# Regras fiscais (YAML) com cache
# --------------------------------------------------------------------
REGRAS_FISCAIS_PATH = Path("regras_fiscais.yaml")

@lru_cache(maxsize=1)
def _carregar_regras_fiscais(path: Path = REGRAS_FISCAIS_PATH) -> Dict[str, Any]:
    if not path.exists():
        log.warning(f"Regras fiscais '{path}' não encontradas. Usando defaults.")
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            return data if isinstance(data, dict) else {}
    except Exception as e:
        log.error(f"Falha ao carregar regras '{path}': {e}. Usando defaults.")
        return {}

# --------------------------------------------------------------------
# Validador híbrido → Agente de Consistência Explicativa
# --------------------------------------------------------------------
class ValidadorFiscal:
    """
    - Núcleo determinístico + explicabilidade estruturada
    - Enriquecimento automático (opcional) ANTES de marcar revisão
    - Camada cognitiva (LLM) para resumo das anomalias
    - Feedback para Orchestrator: reprocessar_sugerido + motivos
    """

    UFS_VALIDAS: Set[str] = {
        "AC","AL","AP","AM","BA","CE","DF","ES","GO","MA","MT","MS","MG","PA","PB","PR","PE",
        "PI","RJ","RN","RS","RO","RR","SC","SP","SE","TO"
    }

    # Mínimo obrigatório padrão (usado como base; pode variar por tipo)
    CAMPOS_OBRIGATORIOS: Tuple[str, ...] = (
        "emitente_cnpj", "valor_total", "data_emissao", "chave_acesso"
    )

    # Campos que tentamos enriquecer automaticamente
    CAMPOS_ENRIQUECIVEIS: Tuple[str, ...] = (
        "emitente_cnpj","destinatario_cnpj","emitente_cpf","destinatario_cpf",
        "valor_total","data_emissao","chave_acesso","numero_nota","serie","modelo","uf"
    )

    # Regex para extrair chave do texto (quando vier “suja”)
    RE_QR_CHAVE = re.compile(r"(?:chNFe|chCTe)=([0-9]{44})", re.I)
    RE_CHAVE_SECA = re.compile(r"\b(\d{44})\b")

    def __init__(self,
                 regras_path: Path = REGRAS_FISCAIS_PATH,
                 llm: Optional[Any] = None,
                 *,
                 require_general_location: bool = False  # ← compat: não exige uf/municipio/endereco legados
                 ):
        self.regras = _carregar_regras_fiscais(regras_path)
        try:
            self.llm = llm if (llm is not None and isinstance(llm, BaseChatModel)) else None  # type: ignore
        except Exception:
            self.llm = llm if llm is not None else None

        tol_cfg = self.regras.get("tolerancias", {}) if isinstance(self.regras.get("tolerancias", {}), dict) else {}
        # tolerâncias default conservadoras (podem ser sobrescritas via YAML)
        self.tolerancia_abs = float(tol_cfg.get("total_documento", 0.05) or 0.05)
        self.tolerancia_pct = float(tol_cfg.get("total_documento_pct", 0.25) or 0.25)  # 25% default

        self.cfops_validos: Set[str] = set((self.regras.get("cfop") or {}).keys())
        self.cst_icms_validos: Set[str] = set((self.regras.get("cst_icms") or {}).keys())
        self.csosn_icms_validos: Set[str] = set((self.regras.get("csosn_icms") or {}).keys())
        # Mantemos, mas não reprovamos por ausência no catálogo:
        self.ncm_validos: Set[str] = set((self.regras.get("ncm") or {}).keys())

        self.require_general_location = bool(require_general_location)

    # --------------------------- API pública ---------------------------

    def validar_documento(
        self,
        *,
        doc_id: int | None = None,
        doc: Dict[str, Any] | None = None,
    db: Any,
        force_revalidation: bool = False,
        usar_llm: bool = True,
        agente_llm: Any | None = None,
        _ja_enriquecido: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Executa validações determinísticas; tenta ENRIQUECER (se faltar obrigatório)
        e só então marca revisão. Retorna explicações estruturadas.
        """
        if doc_id is None and doc is None:
            raise ValueError("Informe doc_id ou doc para validação.")

        if doc is None:
            doc = db.get_documento(int(doc_id))
            if not doc:
                log.error(f"Documento {doc_id} não encontrado para validação.")
                return {"status": "invalido",
                        "confianca_deterministica": 0.0,
                        "erros": ["Documento não encontrado"],
                        "erros_estruturados": [],
                        "reprocessar_sugerido": True,
                        "motivos_reprocessamento": ["dados_ausentes"],
                        "resumo_llm": None}

        current_doc_id = int(doc.get("id", doc_id or 0))
        if not current_doc_id:
            log.error("ID do documento inválido durante a validação.")
            return {"status": "invalido",
                    "confianca_deterministica": 0.0,
                    "erros": ["ID do documento inválido"],
                    "erros_estruturados": [],
                    "reprocessar_sugerido": True,
                    "motivos_reprocessamento": ["dados_ausentes"],
                    "resumo_llm": None}

        status_atual = str(doc.get("status") or "")
        if not force_revalidation and status_atual not in ("", "processando", "quarentena", "revisao_pendente"):
            log.debug(f"Validação pulada (status={status_atual}, force=False) doc_id={current_doc_id}")
            return {"status": status_atual if status_atual else "parcial",
                    "confianca_deterministica": 1.0,
                    "erros": [],
                    "erros_estruturados": [],
                    "reprocessar_sugerido": False,
                    "motivos_reprocessamento": [],
                    "resumo_llm": None}

        log.info(f"Iniciando validação doc_id={current_doc_id} (status atual: {status_atual})")

        # ---- Determinístico
        erros_str: List[str] = []                         # compat retro
        erros_estruturados: List[Dict[str, Any]] = []     # novo formato
        avisos: List[str] = []
        checks_total = 0
        checks_ok = 0
        motivos_reprocessamento: List[str] = []

        def _emit_erro(campo: str, erro: str, gravidade: str = "media", sugestao: Optional[str] = None, *, estrutural: bool = False):
            nonlocal erros_str, erros_estruturados, motivos_reprocessamento
            msg = f"[{campo}] {erro}"
            erros_str.append(msg)
            payload = {"campo": campo, "erro": erro, "gravidade": gravidade}
            if sugestao:
                payload["sugestao"] = sugestao
            erros_estruturados.append(payload)
            if estrutural:
                if "estrutural" not in motivos_reprocessamento:
                    motivos_reprocessamento.append("estrutural")

        def _check(ok: bool, campo: str, erro: str | None = None, *, gravidade: str = "media", sugestao: str | None = None, aviso: bool = False, estrutural: bool = False):
            nonlocal checks_total, checks_ok, avisos
            checks_total += 1
            if ok:
                checks_ok += 1
            else:
                if aviso and erro:
                    avisos.append(f"[{campo}] {erro}")
                elif erro:
                    _emit_erro(campo, erro, gravidade, sugestao, estrutural=estrutural)

        # 0) Normalização mínima de chave de acesso
        doc = self._ensure_chave_44(doc)

        # 1) Identificadores
        cnpj_emit = _only_digits(doc.get("emitente_cnpj") or "")
        cnpj_dest = _only_digits(doc.get("destinatario_cnpj") or "")
        cpf_emit = _only_digits(doc.get("emitente_cpf") or "")
        cpf_dest = _only_digits(doc.get("destinatario_cpf") or "")

        if cnpj_emit:
            _check(_valida_cnpj(cnpj_emit), "emitente_cnpj", "CNPJ do emitente inválido.", gravidade="alta")
        if cnpj_dest and len(cnpj_dest) == 14:
            _check(_valida_cnpj(cnpj_dest), "destinatario_cnpj", "CNPJ do destinatário inválido.", gravidade="alta")
        if cpf_emit:
            _check(_valida_cpf(cpf_emit), "emitente_cpf", "CPF do emitente inválido.", gravidade="media")
        if cpf_dest:
            _check(_valida_cpf(cpf_dest), "destinatario_cpf", "CPF do destinatário inválido.", gravidade="media")

        # 2) Localidade (compat com remoção dos campos gerais)
        uf = (str(doc.get("emitente_uf") or doc.get("destinatario_uf") or doc.get("uf") or "").strip().upper() or None)
        municipio = (str(doc.get("emitente_municipio") or doc.get("destinatario_municipio") or doc.get("municipio") or "").strip() or None)
        endereco = (str(doc.get("emitente_endereco") or doc.get("destinatario_endereco") or doc.get("endereco") or "").strip() or None)

        if uf:
            _check(uf in self.UFS_VALIDAS, "uf", f"UF '{uf}' inválida.", gravidade="alta")
        else:
            _check(False, "uf", "UF não informada.", gravidade="media", aviso=(not self.require_general_location), estrutural=bool(self.require_general_location))

        _check(bool(municipio), "municipio", "Município não informado.", gravidade="baixa", aviso=(not self.require_general_location))
        _check(bool(endereco), "endereco", "Endereço não informado.", gravidade="baixa", aviso=(not self.require_general_location))

        # 2.1) Consistência UF do emitente ↔ cUF da chave 44
        chave = _only_digits(doc.get("chave_acesso") or "")
        uf_chave = _uf_from_chave44(chave) if chave else None
        if uf and uf_chave:
            if uf != uf_chave:
                _check(False, "uf", f"UF do emitente ({uf}) difere da UF emissora na chave ({uf_chave}).",
                       gravidade="alta", sugestao=f"Verifique chave/emitente_uf. Esperado UF={uf_chave} para esta chave.",
                       estrutural=True)

        # 3) Carrega itens/impostos
        try:
            itens_df = db.query_table("itens", where=f"documento_id = {current_doc_id}")
            if getattr(itens_df, "empty", True):
                impostos_df = pd.DataFrame()
            else:
                ids = getattr(itens_df["id"], "unique", lambda: [])()
                ids_list = list(ids) if ids is not None else []
                if ids_list:
                    ids_sql = ", ".join(map(str, ids_list))
                    impostos_df = db.query_table("impostos", where=f"item_id IN ({ids_sql})")
                else:
                    impostos_df = pd.DataFrame()
        except Exception as e_load:
            log.error(f"Erro ao carregar itens/impostos doc_id={current_doc_id}: {e_load}")
            itens_df, impostos_df = pd.DataFrame(), pd.DataFrame()
            _emit_erro("itens", f"Falha ao carregar itens/impostos ({type(e_load).__name__}).", gravidade="alta", estrutural=True)

        # 4) Totais: soma itens ≈ valor_total
        total_doc = _safe_float(doc.get("valor_total"))
        if not getattr(itens_df, "empty", True) and ("valor_total" in getattr(itens_df, "columns", [])):
            try:
                soma_itens = float(itens_df["valor_total"].fillna(0).sum())
                diff_abs_ok = abs(soma_itens - total_doc) <= self.tolerancia_abs
                diff_pct_ok = True
                if self.tolerancia_pct > 0 and total_doc > 0:
                    diff_pct_ok = abs(soma_itens - total_doc) / total_doc <= self.tolerancia_pct
                _check(diff_abs_ok and diff_pct_ok,
                       "valor_total",
                       f"Inconsistência de totais: Soma Itens={soma_itens:.2f} vs Total Doc={total_doc:.2f}.",
                       gravidade="alta",
                       sugestao="Revise extração de itens/colunas (OCR) ou valor_total.",
                       estrutural=not (diff_abs_ok or diff_pct_ok))
            except Exception as e_tot:
                log.error(f"Erro validar totais doc_id={current_doc_id}: {e_tot}")
                _emit_erro("valor_total", f"Erro interno ao verificar totais ({type(e_tot).__name__}).", gravidade="media")

        # 5) Totais fiscais agregados (ICMS/IPI/PIS/COFINS)
        if not getattr(impostos_df, "empty", True) and all(c in getattr(impostos_df, "columns", []) for c in ("tipo_imposto", "valor")):
            try:
                def soma_tipo(t: str) -> float:
                    try:
                        sel = impostos_df[impostos_df["tipo_imposto"] == t]
                        return float(sel["valor"].fillna(0).sum())
                    except Exception:
                        return 0.0
                for tipo, campo in (("ICMS", "total_icms"), ("IPI", "total_ipi"), ("PIS", "total_pis"), ("COFINS", "total_cofins")):
                    valor_doc = _safe_float(doc.get(campo))
                    valor_calc = soma_tipo(tipo)
                    ok_abs = abs(valor_calc - valor_doc) <= self.tolerancia_abs
                    ok_pct = True
                    base = valor_doc if valor_doc > 0 else (total_doc if total_doc > 0 else None)
                    if self.tolerancia_pct > 0 and base:
                        ok_pct = abs(valor_calc - valor_doc) / float(base) <= self.tolerancia_pct
                    _check(ok_abs and ok_pct,
                           campo,
                           f"Inconsistência de {tipo}: itens={valor_calc:.2f} vs total={valor_doc:.2f}.",
                           gravidade="media",
                           sugestao=f"Reveja {tipo} por item e o agregado {campo}.")
            except Exception as e_fisc:
                log.error(f"Erro validar totais fiscais doc_id={current_doc_id}: {e_fisc}")
                _emit_erro("impostos", f"Erro interno ao validar totais fiscais ({type(e_fisc).__name__}).", gravidade="media")

        # 6) Códigos (CFOP/NCM/CST) — regras do YAML ou fallback de formato
        if not getattr(itens_df, "empty", True):
            # CFOP (item)
            if "cfop" in getattr(itens_df, "columns", []):
                for idx, item in itens_df.iterrows():
                    cfop = str(item.get("cfop", "")).strip() if pd.notna(item.get("cfop")) else ""
                    if cfop:
                        if self.cfops_validos:
                            ok = cfop in self.cfops_validos
                        else:
                            ok = bool(re.fullmatch(r"\d{4}", cfop))
                        if not ok:
                            _emit_erro("cfop", f"Item {item.get('id','?')} (linha {idx+1}): CFOP '{cfop}' inválido.", gravidade="media")

            # NCM (item) — **agora com aviso se faltar no catálogo**
            if "ncm" in getattr(itens_df, "columns", []):
                for idx, item in itens_df.iterrows():
                    ncm = str(item.get("ncm", "")).strip() if pd.notna(item.get("ncm")) else ""
                    if ncm:
                        ncm_res = self._validar_ncm(ncm, db)
                        if not ncm_res.get("ok"):
                            _emit_erro("ncm", ncm_res.get("erro", f"NCM '{ncm}' inválido."), gravidade=ncm_res.get("gravidade", "media"))
                        elif "aviso" in ncm_res:
                            avisos.append(f"[ncm] Item {idx+1}: {ncm_res['aviso']}")

        if not getattr(impostos_df, "empty", True) and \
           "tipo_imposto" in getattr(impostos_df, "columns", []) and \
           "cst" in getattr(impostos_df, "columns", []):
            icms_df = impostos_df[impostos_df["tipo_imposto"] == "ICMS"] if not getattr(impostos_df, "empty", True) else pd.DataFrame()
            for idx, imp in icms_df.iterrows():
                cst = str(imp.get("cst", "")).strip() if pd.notna(imp.get("cst")) else ""
                if cst:
                    if self.cst_icms_validos or self.csosn_icms_validos:
                        valido = (cst in self.cst_icms_validos) or (cst in self.csosn_icms_validos)
                        if not valido:
                            _emit_erro("cst_icms", f"Imposto {imp.get('id','?')} (Item {imp.get('item_id','?')}): CST/CSOSN '{cst}' inválido.", gravidade="media")
                    else:
                        if not bool(re.fullmatch(r"\d{2,3}", cst)):
                            _emit_erro("cst_icms", f"Imposto {imp.get('id','?')} (Item {imp.get('item_id','?')}): CST/CSOSN '{cst}' inválido (2–3 dígitos).", gravidade="media")

        # 7) CFOP x UF (cabeçalho)
        cfop_head = (str(doc.get("cfop") or "").strip())
        emit_uf = (str(doc.get("emitente_uf") or "").strip().upper() or "")
        dest_uf = (str(doc.get("destinatario_uf") or "").strip().upper() or "")
        if cfop_head and len(cfop_head) == 4 and emit_uf and dest_uf:
            p = cfop_head[0]
            if p == "6" and emit_uf == dest_uf:
                _emit_erro("cfop", f"CFOP {cfop_head} indica operação interestadual, mas emitente_uf={emit_uf} = destinatario_uf={dest_uf}.",
                           gravidade="alta", sugestao="Usar CFOP 5xxx/1xxx para operação interna.", estrutural=True)
            if p == "5" and emit_uf != dest_uf:
                _emit_erro("cfop", f"CFOP {cfop_head} indica operação interna, mas UFs distintas (emitente={emit_uf}, destinatario={dest_uf}).",
                           gravidade="alta", sugestao="Para interestadual, CFOP típico inicia com 6xxx.", estrutural=True)

        # ---- Score & status (provisório)
        confianca = self._score_confianca(checks_ok, checks_total)
        obrigatorios_faltando = self._listar_obrigatorios_faltando(doc)
        if obrigatorios_faltando:
            _emit_erro("obrigatorios", f"Campos obrigatórios ausentes: {obrigatorios_faltando}.", gravidade="alta", estrutural=True)

        if erros_str or obrigatorios_faltando:
            status_provisorio = "parcial" if confianca >= 0.6 else "invalido"
        else:
            status_provisorio = "valido"

        # ===================== ENRIQUECIMENTO AUTOMÁTICO ======================
        if (usar_llm or agente_llm) and not _ja_enriquecido:
            if obrigatorios_faltando:
                log.info(f"Campos obrigatórios ausentes: {obrigatorios_faltando} → tentando enriquecimento automático (doc_id={current_doc_id})")
                doc_enriq = self._tentar_enriquecimento(doc, agente_llm=agente_llm)
                if doc_enriq is not None and doc_enriq != doc:
                    try:
                        for k in self.CAMPOS_ENRIQUECIVEIS:
                            if doc_enriq.get(k) and doc_enriq.get(k) != doc.get(k):
                                db.atualizar_documento_campo(current_doc_id, k, doc_enriq.get(k))
                    except Exception as e:
                        log.debug(f"Falha ao persistir enriquecimentos (ignorado): {e}")

                    return self.validar_documento(
                        doc_id=current_doc_id,
                        doc=doc_enriq,
                        db=db,
                        force_revalidation=True,
                        usar_llm=usar_llm,
                        agente_llm=None,
                        _ja_enriquecido=True,
                    )
                else:
                    log.info("Enriquecimento não alterou o documento ou falhou. Prosseguindo com status atual.")
        # =====================================================================

        # ---- Status final
        if erros_str or obrigatorios_faltando:
            status_final = "parcial" if confianca >= 0.6 else "invalido"
        else:
            status_final = "valido"

        # ---- Decisão cognitiva para o Orchestrator
        reprocessar_sugerido = bool(motivos_reprocessamento) or (status_final != "valido" and confianca < 0.65)

        # ---- Camada cognitiva: resumo
        resumo_llm: Optional[str] = None
        if usar_llm and self.llm and (erros_str or status_final != "valido"):
            try:
                resumo_llm = self._analisar_anomalias_llm(doc, erros_str + ([f"Avisos: {avisos}"] if avisos else []))
            except Exception as e_llm:
                log.debug(f"LLM análise de anomalias falhou (ignorado): {e_llm}")

        # ---- Persistência
        self._persistir_status(
            db=db,
            doc_id=current_doc_id,
            status_atual=status_atual,
            status_final=status_final,
            erros=erros_str if erros_str else ([f"Obrigatórios ausentes: {obrigatorios_faltando}"] if obrigatorios_faltando else []),
        )

        return {
            "status": status_final,
            "confianca_deterministica": float(round(confianca, 4)),
            "erros": erros_str,
            "erros_estruturados": erros_estruturados,
            "avisos": avisos,
            "reprocessar_sugerido": reprocessar_sugerido,
            "motivos_reprocessamento": list(set(motivos_reprocessamento)),
            "resumo_llm": resumo_llm,
        }

    # --------------------------- Internos ----------------------------

    @staticmethod
    def _score_confianca(ok: int, total: int) -> float:
        if total <= 0:
            return 0.0
        base = ok / total
        bonus = 0.05 if total >= 12 and base > 0.7 else 0.0
        return max(0.0, min(1.0, base + bonus))

    def _persistir_status(self, *, db: Any, doc_id: int, status_atual: str,
                          status_final: str, erros: List[str]) -> None:
        if status_final in ("invalido", "parcial"):
            novo_status = "revisao_pendente"
            motivo = "; ".join([e for e in erros if e])[:255] if erros else "Inconsistências detectadas."
            try:
                db.atualizar_documento_campo(doc_id, "status", novo_status)
                db.atualizar_documento_campo(doc_id, "motivo_rejeicao", motivo)
                db.log("validacao_inconsistente", "sistema",
                       f"doc_id={doc_id}|status_final={status_final}|erros={len(erros)}|resumo={(motivo[:120] if motivo else '')}")
                log.warning(f"Documento {doc_id} marcado para revisão. ({status_final})")
            except Exception as e:
                log.error(f"Falha ao persistir status de revisão doc_id={doc_id}: {e}")
        else:
            try:
                if status_atual in ("", "processando", "quarentena", "revisao_pendente"):
                    db.atualizar_documento_campo(doc_id, "status", "processado")
                    db.atualizar_documento_campo(doc_id, "motivo_rejeicao", None)
                    db.log("validacao_ok", "sistema",
                           f"doc_id={doc_id}|status_anterior={status_atual}|novo_status=processado")
                    log.info(f"Documento {doc_id} validado com sucesso.")
                else:
                    db.log("revalidacao_ok", "sistema", f"doc_id={doc_id}|status={status_atual}")
            except Exception as e:
                log.error(f"Falha ao persistir status OK doc_id={doc_id}: {e}")

    # -------------------- Enriquecimento automático -------------------

    def _campos_obrigatorios_ok(self, doc: Dict[str, Any]) -> bool:
        return not self._listar_obrigatorios_faltando(doc)

    def _listar_obrigatorios_faltando(self, doc: Dict[str, Any]) -> List[str]:
        """Define obrigatórios por tipo de documento.
        - NFe/NFCe/CTe/MDF-e/CF-e: exige chave_acesso (44) e emitente_cnpj/emitente_cpf
        - NFSe: NÃO exige chave_acesso; exige (emitente_cnpj OU emitente_cpf), data_emissao, valor_total
        Outras variações caem no conjunto padrão conservador.
        """
        tipo = str(doc.get("tipo") or "").strip().lower()
        cnpj_emit = (doc.get("emitente_cnpj") or "").strip()
        cpf_emit = (doc.get("emitente_cpf") or "").strip()

        base: List[str] = ["data_emissao"]
        # Valor: aceitar alias total_servicos quando valor_total vier vazio (ex.: NFSe)
        valor_total = doc.get("valor_total")
        if valor_total in (None, "", [], {}) and doc.get("total_servicos") not in (None, "", [], {}):
            # Trata como presente
            pass
        else:
            base.append("valor_total")

        if tipo in {"nfse", "nfs-e", "nfs-e", "nf s-e"}:
            # Em NFSe não há chave de acesso 44 dígitos; há 'Código de Verificação' e afins
            # Exigir que ao menos um identificador do emitente exista (CNPJ ou CPF)
            if not (cnpj_emit or cpf_emit):
                base.append("emitente_cnpj")  # mantemos mensagem compatível
        else:
            # Demais documentos: exigir chave 44 e CNPJ/CPF do emitente
            base.extend(["chave_acesso"])
            if not (cnpj_emit or cpf_emit):
                base.append("emitente_cnpj")

        faltando: List[str] = []
        for k in base:
            v = doc.get(k)
            if v in (None, "", [], {}):
                faltando.append(k)
        return faltando

    def _ensure_chave_44(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """Garante que 'chave_acesso' tenha 44 dígitos; tenta extrair do texto se necessário."""
        chave = _only_digits(doc.get("chave_acesso") or "")
        if len(chave) == 44:
            return doc
        raw = (doc.get("texto_ocr") or doc.get("xml_conteudo") or "")
        if raw:
            m = self.RE_QR_CHAVE.search(raw) or self.RE_CHAVE_SECA.search(raw)
            if m:
                chave2 = _only_digits(m.group(1))
                if len(chave2) == 44:
                    doc = dict(doc)
                    doc["chave_acesso"] = chave2
        return doc

    def _tentar_enriquecimento(self, doc: Dict[str, Any], *, agente_llm: Any | None) -> Optional[Dict[str, Any]]:
        if agente_llm and hasattr(agente_llm, "enriquecer"):
            try:
                novo = agente_llm.enriquecer(doc)
                if isinstance(novo, dict) and novo:
                    return self._merge_enriquecimento(self._ensure_chave_44(doc), novo)
            except Exception as e:
                log.debug(f"agente_llm.enriquecer falhou: {e}")

        if not self.llm:
            return None

        texto_base = (doc.get("texto_ocr") or doc.get("xml_conteudo") or "")[:8000]
        if not str(texto_base).strip():
            return None

        try:
            sys = SystemMessage(content=(
                "Você é um assistente de extração fiscal conservador. "
                "Extraia SOMENTE valores explicitamente presentes no texto (sem inferir). "
                "Responda APENAS com JSON válido. Campos ausentes => null."
            ))  # type: ignore

            schema = list(self.CAMPOS_ENRIQUECIVEIS)
            user = HumanMessage(content=(
                f"Texto (parcial):\n{texto_base}\n\n"
                f"Schema (apenas estas chaves): {json.dumps(schema, ensure_ascii=False)}\n"
                "Devolva somente o JSON com essas chaves."
            ))  # type: ignore

            resp = self.llm.invoke([sys, user])  # type: ignore[attr-defined]
            txt = getattr(resp, "content", None) or str(resp)
            m = re.search(r"\{.*\}", txt, re.S)
            payload = json.loads(m.group(0)) if m else {}
            if isinstance(payload, dict) and payload:
                return self._merge_enriquecimento(self._ensure_chave_44(doc), payload)
        except Exception as e:
            log.debug(f"Enriquecimento LLM interno falhou: {e}")
        return None

    def _merge_enriquecimento(self, doc: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
        novo = dict(doc)
        for k in self.CAMPOS_ENRIQUECIVEIS:
            val_atual = novo.get(k)
            val_novo = extra.get(k)
            if (val_atual in (None, "", [], {})) and (val_novo not in (None, "", [], {})):
                if k.endswith("_cnpj") or k.endswith("_cpf") or k == "cnpj_autorizado":
                    novo[k] = _only_digits(str(val_novo))
                elif k in ("valor_total",):
                    novo[k] = _safe_float(val_novo)
                elif k.startswith("data_"):
                    novo[k] = str(val_novo)
                else:
                    novo[k] = val_novo
        return novo

    # -------------------- NCM (formato + aviso de catálogo) --------------------

    def _validar_ncm(self, ncm: Optional[str], db) -> dict:
        """
        • ausente -> erro média
        • formato inválido (não numérico ou !=8) -> erro alta
        • formato ok, catálogo local ausente ou sem o código -> OK com AVISO
        """
        n = (ncm or "").strip()
        if not n:
            return {"ok": False, "gravidade": "media", "erro": "NCM ausente", "sugestao": None}
        if not n.isdigit() or len(n) != 8:
            return {"ok": False, "gravidade": "alta", "erro": f"NCM '{n}' inválido (esperado 8 dígitos).", "sugestao": None}
        # Catálogo opcional
        try:
            if hasattr(db, "query_table"):
                df = db.query_table("ncm_catalogo", where=f"codigo = '{n}'")
                if df is not None and not getattr(df, "empty", True):
                    return {"ok": True}
                else:
                    return {"ok": True, "aviso": f"NCM '{n}' não encontrado no catálogo local; recomendado conferir."}
        except Exception:
            pass
        return {"ok": True, "aviso": f"NCM '{n}' não verificado (catálogo indisponível)."}

    # -------------------- LLM: resumo de anomalias --------------------

    def _analisar_anomalias_llm(self, doc: Dict[str, Any], erros: List[str]) -> Optional[str]:
        if not self.llm or not erros:
            return None
        try:
            sys = SystemMessage(content=(
                "Você é um auditor fiscal eletrônico. Analise inconsistências detectadas por validações determinísticas.\n"
                "- Não invente valores. Dê hipóteses e próximos passos práticos.\n"
                "- Seja breve (máx. 4 linhas)."
            ))  # type: ignore

            fields = {
                "valor_total": doc.get("valor_total"),
                "emitente_uf": doc.get("emitente_uf"),
                "destinatario_uf": doc.get("destinatario_uf"),
                "chave_acesso": doc.get("chave_acesso"),
                "data_emissao": doc.get("data_emissao"),
                "emitente_cnpj": doc.get("emitente_cnpj"),
                "destinatario_cnpj": doc.get("destinatario_cnpj"),
            }
            hum = HumanMessage(content=(
                f"Inconsistências: {erros[:8]}\n"
                f"Contexto resumido: {fields}\n"
                "Explique sucintamente o que pode ter ocorrido e qual ação recomenda (ex.: reprocessar OCR, buscar XML, revisão humana)."
            ))  # type: ignore

            resp = self.llm.invoke([sys, hum])  # type: ignore[attr-defined]
            resumo = getattr(resp, "content", None) or str(resp)
            resumo = str(resumo).strip()
            resumo = re.sub(r"\s+", " ", resumo)
            return resumo[:500]
        except Exception:
            return None


__all__ = ["ValidadorFiscal"]
