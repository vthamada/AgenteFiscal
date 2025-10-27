# validacao.py

from __future__ import annotations
from typing import Optional, Dict, Any, TYPE_CHECKING, Set, List, Tuple
import re
import yaml
from pathlib import Path
import logging
from functools import lru_cache
import os

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
# Imports tolerantes (mock sob falha) ─ mantém retrocompatibilidade
# --------------------------------------------------------------------
try:
    from banco_de_dados import BancoDeDados
    import pandas as pd
except Exception as e:
    logging.error(f"Erro ao importar BancoDeDados/pandas: {e}")

    class _DBMock(object):
        def get_documento(self, _id): return {}
        def query_table(self, *_args, **_kw): return pd.DataFrame()  # type: ignore
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
        pd = type("pandas", (), {"DataFrame": _DFMock})  # type: ignore

    BancoDeDados = _DBMock  # type: ignore

# --------------------------------------------------------------------
# Import LLM tolerante em tempo de execução (necessário para _analisar_anomalias_llm)
# --------------------------------------------------------------------
try:
    from langchain_core.language_models.chat_models import BaseChatModel  # type: ignore
    from langchain_core.messages import SystemMessage, HumanMessage       # type: ignore
except Exception:  # pragma: no cover
    BaseChatModel = object  # type: ignore

    class _Msg:
        def __init__(self, content: str):
            self.content = content
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

    # Em ambientes de DEV é possível permitir sequências artificiais ativando a flag:
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
    except Exception as e:
        log.error(f"Erro DV CNPJ '{cnpj}': {e}")
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
            # tenta converter strings BR "1.234,56"
            s = str(x).strip()
            if s.count(",") == 1:
                s = s.replace(".", "").replace(",", ".")
            return float(s)
        except Exception:
            return 0.0

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
# Validador híbrido (determinístico + camada cognitiva opcional)
# --------------------------------------------------------------------
class ValidadorFiscal:
    """
    - Núcleo determinístico: valida identificadores, totais e códigos.
    - Camada cognitiva OPCIONAL (LLM): analisa anomalias e sugere ações.
    - Retrocompatível: continua atualizando status no DB.
    - Retorno padronizado: dict com status, confianca_deterministica, erros, resumo_llm.
    """

    UFS_VALIDAS: Set[str] = {
        "AC","AL","AP","AM","BA","CE","DF","ES","GO","MA","MT","MS","MG","PA","PB","PR","PE",
        "PI","RJ","RN","RS","RO","RR","SC","SP","SE","TO"
    }

    def __init__(self,
                 regras_path: Path = REGRAS_FISCAIS_PATH,
                 llm: Optional["BaseChatModel"] = None):
        self.regras = _carregar_regras_fiscais(regras_path)
        # Corrigido: não usar isinstance(..., object)
        try:
            self.llm = llm if (llm is not None and isinstance(llm, BaseChatModel)) else None  # type: ignore
        except Exception:
            self.llm = llm if llm is not None else None

        tol_cfg = self.regras.get("tolerancias", {}) if isinstance(self.regras.get("tolerancias", {}), dict) else {}
        # tolerância ABSOLUTA (ex.: 0.05)
        try:
            self.tolerancia_abs = float(tol_cfg.get("total_documento", 0.05))
        except Exception:
            self.tolerancia_abs = 0.05
        # tolerância PERCENTUAL (ex.: 0.01 = 1%)
        try:
            self.tolerancia_pct = float(tol_cfg.get("total_documento_pct", 0.0))
        except Exception:
            self.tolerancia_pct = 0.0

        # códigos válidos
        self.cfops_validos: Set[str] = set((self.regras.get("cfop") or {}).keys())
        self.cst_icms_validos: Set[str] = set((self.regras.get("cst_icms") or {}).keys())
        self.csosn_icms_validos: Set[str] = set((self.regras.get("csosn_icms") or {}).keys())
        self.ncm_validos: Set[str] = set((self.regras.get("ncm") or {}).keys())

    # --------------------------- API pública ---------------------------

    def validar_documento(
        self,
        *,
        doc_id: int | None = None,
        doc: Dict[str, Any] | None = None,
        db: "BancoDeDados",
        force_revalidation: bool = False,
        usar_llm: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Executa validações determinísticas (+ análise LLM opcional) e
        **continua** atualizando o status no DB (retrocompatibilidade).

        Retorna:
        {
          "status": "valido"|"parcial"|"invalido",
          "confianca_deterministica": float(0..1),
          "erros": [str, ...],
          "resumo_llm": Optional[str]
        }
        """
        if doc_id is None and doc is None:
            raise ValueError("Informe doc_id ou doc para validação.")

        if doc is None:
            doc = db.get_documento(int(doc_id))
            if not doc:
                log.error(f"Documento {doc_id} não encontrado para validação.")
                return {"status": "invalido", "confianca_deterministica": 0.0, "erros": ["Documento não encontrado"], "resumo_llm": None}

        current_doc_id = int(doc.get("id", doc_id or 0))
        if not current_doc_id:
            log.error("ID do documento inválido durante a validação.")
            return {"status": "invalido", "confianca_deterministica": 0.0, "erros": ["ID do documento inválido"], "resumo_llm": None}

        status_atual = str(doc.get("status") or "")
        if not force_revalidation and status_atual not in ("", "processando", "quarentena", "revisao_pendente"):
            log.debug(f"Validação pulada (status={status_atual}, force=False) doc_id={current_doc_id}")
            return {"status": status_atual if status_atual else "parcial", "confianca_deterministica": 1.0, "erros": [], "resumo_llm": None}

        log.info(f"Iniciando validação doc_id={current_doc_id} (status atual: {status_atual})")

        # ---- Determinístico
        erros: List[str] = []
        checks_total = 0
        checks_ok = 0

        # 1) Identificadores
        cnpj_emit = _only_digits(doc.get("emitente_cnpj") or "")
        cnpj_dest = _only_digits(doc.get("destinatario_cnpj") or "")
        cpf_emit = _only_digits(doc.get("emitente_cpf") or "")
        cpf_dest = _only_digits(doc.get("destinatario_cpf") or "")

        def _check(ok: bool, msg: str | None = None):
            nonlocal checks_total, checks_ok
            checks_total += 1
            if ok:
                checks_ok += 1
            elif msg:
                erros.append(msg)

        if cnpj_emit:
            _check(_valida_cnpj(cnpj_emit), "CNPJ do emitente inválido.")
        if cnpj_dest and len(cnpj_dest) == 14:
            _check(_valida_cnpj(cnpj_dest), "CNPJ do destinatário inválido.")
        if cpf_emit:
            _check(_valida_cpf(cpf_emit), "CPF do emitente inválido.")
        if cpf_dest:
            _check(_valida_cpf(cpf_dest), "CPF do destinatário inválido.")

        # 2) Localidade
        uf = str(doc.get("uf") or "").strip().upper()
        municipio = str(doc.get("municipio") or "").strip()
        endereco = str(doc.get("endereco") or "").strip()

        if uf:
            _check(uf in self.UFS_VALIDAS, f"UF '{uf}' inválida.")
        else:
            _check(False, "UF não informada.")

        _check(bool(municipio), "Município não informado.")
        _check(bool(endereco), "Endereço do emitente ausente.")

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
            erros.append(f"Falha interna ao carregar itens/impostos para validação ({type(e_load).__name__}).")

        # 4) Totais
        total_doc = _safe_float(doc.get("valor_total"))
        if not getattr(itens_df, "empty", True) and ("valor_total" in getattr(itens_df, "columns", [])):
            try:
                soma_itens = float(itens_df["valor_total"].fillna(0).sum())
                # tolerância absoluta
                diff_abs_ok = abs(soma_itens - total_doc) <= self.tolerancia_abs
                # tolerância percentual (se definida)
                diff_pct_ok = True
                if self.tolerancia_pct > 0 and total_doc > 0:
                    diff_pct_ok = abs(soma_itens - total_doc) / total_doc <= self.tolerancia_pct

                _check(diff_abs_ok and diff_pct_ok,
                       f"Inconsistência de totais: Soma Itens={soma_itens:.2f} vs Total Doc={total_doc:.2f}.")
            except Exception as e_tot:
                log.error(f"Erro validar totais doc_id={current_doc_id}: {e_tot}")
                _check(False, f"Erro interno ao verificar totais do documento ({type(e_tot).__name__}).")
        else:
            # sem itens, não penaliza (documentos de serviço/CTe etc.)
            _check(True)

        # 5) Totais fiscais agregados
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
                    # aplica mesmas tolerâncias
                    ok_abs = abs(valor_calc - valor_doc) <= self.tolerancia_abs
                    ok_pct = True
                    base = valor_doc if valor_doc > 0 else (total_doc if total_doc > 0 else None)
                    if self.tolerancia_pct > 0 and base:
                        ok_pct = abs(valor_calc - valor_doc) / float(base) <= self.tolerancia_pct
                    _check(ok_abs and ok_pct,
                           f"Inconsistência de {tipo}: itens={valor_calc:.2f} vs total={valor_doc:.2f}.")
            except Exception as e_fisc:
                log.error(f"Erro validar totais fiscais doc_id={current_doc_id}: {e_fisc}")
                _check(False, f"Erro interno ao validar totais fiscais ({type(e_fisc).__name__}).")
        else:
            # se não há impostos, não penaliza (NFS-e sem desdobramento, p.ex.)
            _check(True)

        # 6) Códigos: CFOP, NCM, CST/CSOSN (ICMS)
        if not getattr(itens_df, "empty", True):
            # CFOP
            if self.cfops_validos and "cfop" in getattr(itens_df, "columns", []):
                for idx, item in itens_df.iterrows():
                    cfop = str(item.get("cfop", "")).strip() if pd.notna(item.get("cfop")) else ""
                    if cfop:
                        _check(cfop in self.cfops_validos,
                               f"Item {item.get('id','?')} (linha {idx+1}): CFOP '{cfop}' inválido.")
            # NCM
            if self.ncm_validos and "ncm" in getattr(itens_df, "columns", []):
                for idx, item in itens_df.iterrows():
                    ncm = str(item.get("ncm", "")).strip() if pd.notna(item.get("ncm")) else ""
                    if ncm:
                        _check(ncm in self.ncm_validos,
                               f"Item {item.get('id','?')} (linha {idx+1}): NCM '{ncm}' inválido.")

        if not getattr(impostos_df, "empty", True) and "tipo_imposto" in getattr(impostos_df, "columns", []) and "cst" in getattr(impostos_df, "columns", []):
            icms_df = impostos_df[impostos_df["tipo_imposto"] == "ICMS"] if not getattr(impostos_df, "empty", True) else pd.DataFrame()
            for idx, imp in icms_df.iterrows():
                cst = str(imp.get("cst", "")).strip() if pd.notna(imp.get("cst")) else ""
                if cst:
                    valido = (cst in self.cst_icms_validos) or (cst in self.csosn_icms_validos)
                    _check(valido,
                           f"Imposto {imp.get('id','?')} (Item {imp.get('item_id','?')}): CST/CSOSN '{cst}' inválido.")

        # ---- Score de confiança determinística
        confianca = self._score_confianca(checks_ok, checks_total)

        # ---- Status final determinístico
        if erros:
            status_final = "parcial" if confianca >= 0.6 else "invalido"
        else:
            status_final = "valido"

        # ---- Camada cognitiva (opcional)
        resumo_llm: Optional[str] = None
        if usar_llm and self.llm and erros:
            try:
                resumo_llm = self._analisar_anomalias_llm(doc, erros)
            except Exception as e_llm:
                log.debug(f"LLM análise de anomalias falhou (ignorado): {e_llm}")

        # ---- Persistência (retrocompatível)
        self._persistir_status(db=db,
                               doc_id=current_doc_id,
                               status_atual=status_atual,
                               status_final=status_final,
                               erros=erros)

        return {
            "status": status_final,
            "confianca_deterministica": float(round(confianca, 4)),
            "erros": erros,
            "resumo_llm": resumo_llm,
        }

    # --------------------------- Internos ----------------------------

    @staticmethod
    def _score_confianca(ok: int, total: int) -> float:
        if total <= 0:
            return 0.0
        # curva suave: privilegia muitos checks passando
        base = ok / total
        bonus = 0.05 if total >= 12 and base > 0.7 else 0.0
        return max(0.0, min(1.0, base + bonus))

    def _persistir_status(self, *, db: "BancoDeDados", doc_id: int, status_atual: str,
                          status_final: str, erros: List[str]) -> None:
        if status_final in ("invalido", "parcial"):
            novo_status = "revisao_pendente"
            motivo = "; ".join(erros)[:255] if erros else "Inconsistências detectadas."
            try:
                db.atualizar_documento_campo(doc_id, "status", novo_status)
                db.atualizar_documento_campo(doc_id, "motivo_rejeicao", motivo)
                db.log("validacao_inconsistente", "sistema",
                       f"doc_id={doc_id}|status_final={status_final}|erros={len(erros)}|resumo={(motivo[:120] if motivo else '')}")
                log.warning(f"Documento {doc_id} marcado para revisão. ({status_final})")
            except Exception as e:
                log.error(f"Falha ao persistir status de revisão doc_id={doc_id}: {e}")
        else:
            # válido
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

    def _analisar_anomalias_llm(self, doc: Dict[str, Any], erros: List[str]) -> Optional[str]:
        """Analisa o conjunto de erros e sugere hipóteses/ações (LLM opcional)."""
        if not self.llm or not erros:
            return None
        try:
            # A mensagem de sistema define o papel do modelo (sem inventar dados)
            sys = SystemMessage(content=(
                "Você é um auditor fiscal eletrônico. Analise inconsistências detectadas por validações determinísticas.\n"
                "- Não invente valores. Dê hipóteses e próximos passos práticos.\n"
                "- Seja breve (máx. 4 linhas)."
            ))  # type: ignore

            # Incluímos apenas um subconjunto seguro do documento para contexto
            fields = {
                "tipo": doc.get("tipo"),
                "valor_total": doc.get("valor_total"),
                "uf": doc.get("uf"),
                "emitente_cnpj": doc.get("emitente_cnpj"),
                "destinatario_cnpj": doc.get("destinatario_cnpj"),
                "data_emissao": doc.get("data_emissao"),
                "chave_acesso": doc.get("chave_acesso"),
            }
            hum = HumanMessage(content=(
                f"Inconsistências: {erros[:8]}\n"
                f"Contexto resumido: {fields}\n"
                "Explique sucintamente o que pode ter ocorrido e qual ação recomenda (ex.: reprocessar OCR, buscar XML, revisão humana)."
            ))  # type: ignore

            resp = self.llm.invoke([sys, hum])  # type: ignore[attr-defined]
            resumo = getattr(resp, "content", None) or str(resp)
            resumo = str(resumo).strip()
            # Sanitiza (1 parágrafo curto)
            resumo = re.sub(r"\s+", " ", resumo)
            return resumo[:500]
        except Exception as e:
            log.debug(f"LLM.analisar_anomalias falhou: {e}")
            return None


__all__ = ["ValidadorFiscal"]
