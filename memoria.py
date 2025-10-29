# memoria.py

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from textwrap import shorten
import hashlib
import json
import re
import unicodedata

from banco_de_dados import BancoDeDados


class MemoriaSessao:
    """
    Memória cognitiva persistente.
    - Compatível com a versão simples (salvar/resumo).
    - Adições para aprendizado incremental por layout e por entidade (CNPJ).
    - Telemetria de execuções para o Orquestrador/metrics.

    Tabelas usadas (criadas em banco_de_dados._criar_schema):
      memoria             -> histórico de perguntas/respostas (compat)
      memoria_layout      -> 'receitas' por fingerprint de layout
      memoria_entidade    -> preferências por CNPJ (emitente/destinatário/fornecedor)
      memoria_execucao    -> telemetria por documento (decisões & métricas)
    """

    def __init__(self, db: BancoDeDados):
        self.db = db

    # ---------------------------------------------------------------------
    # Compat: histórico textual simples
    # ---------------------------------------------------------------------
    def salvar(self, pergunta: str, resposta_resumo: str, duracao_s: float = 0.0) -> None:
        self.db.conn.execute(
            "INSERT INTO memoria (pergunta, resposta_resumo, duracao_s) VALUES (?, ?, ?)",
            (pergunta, resposta_resumo, float(duracao_s)),
        )
        self.db.conn.commit()

    def resumo(self, limite: int = 6) -> str:
        cur = self.db.conn.cursor()
        cur.execute("SELECT pergunta, resposta_resumo FROM memoria ORDER BY id DESC LIMIT ?", (int(limite),))
        rows = cur.fetchall()
        partes: List[str] = []
        for r in rows:
            p = shorten(r["pergunta"] or "", width=80, placeholder="…")
            a = shorten(r["resposta_resumo"] or "", width=100, placeholder="…")
            partes.append(f"- Q: {p}\n  A: {a}")
        return "\n".join(reversed(partes)) or "(sem histórico)"

    # ---------------------------------------------------------------------
    # Aprendizado por LAYOUT (fingerprint → melhor 'receita' de processamento)
    # ---------------------------------------------------------------------
    @staticmethod
    def layout_fingerprint(texto_ocr: str, colspec: Optional[str] = None) -> str:
        """
        Cria um fingerprint estável do layout a partir do texto OCR normalizado
        e (opcionalmente) de uma assinatura de colunas/linhas detectadas.
        Não armazena conteúdo sensível; apenas hash.
        """
        norm = MemoriaSessao._normalize_text_for_fp(texto_ocr or "")
        payload = (colspec or "") + "||" + norm
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]  # curto e suficiente

    def registrar_layout_receita(self, *, layout_fp: str, receita: Dict[str, Any], fonte: str = "orchestrator",
                                 sucesso: Optional[bool] = None, alpha: float = 0.2) -> int:
        """
        Upsert em memoria_layout + ajuste de sucesso_pct (EMA) quando houver feedback.
        Ex.: receita={"ocr":"easyocr_agressivo","nlp_llm":True,"regex_tabela":"pipes"}
        """
        return self.db.upsert_memoria_layout(
            layout_fp=layout_fp, fonte=fonte, melhor_receita=receita, sucesso=sucesso, alpha=alpha
        )

    def receita_sugerida_por_layout(self, layout_fp: str, *, limiar_sucesso: float = 0.7, min_amostras: int = 3) -> Optional[Dict[str, Any]]:
        """
        Retorna a melhor receita conhecida para o layout quando o histórico é confiável.
        """
        row = self.db.get_memoria_layout(layout_fp)
        if not row:
            return None
        amostras = int(row.get("amostras") or 0)
        sucesso_pct = row.get("sucesso_pct")
        if amostras < min_amostras or (sucesso_pct is not None and float(sucesso_pct) < float(limiar_sucesso)):
            return None
        try:
            return json.loads(row.get("melhor_receita_json") or "{}")
        except Exception:
            return None

    # ---------------------------------------------------------------------
    # Aprendizado por ENTIDADE (CNPJ emitente/destinatário/fornecedor)
    # ---------------------------------------------------------------------
    def atualizar_perfil_entidade(
        self, *, cnpj: str, tipo: str, preferencia: Dict[str, Any], sucesso: Optional[bool] = None, alpha: float = 0.2
    ) -> int:
        """
        Upsert em memoria_entidade com merge de preferências (UF, CFOPs comuns, gates preferidos, etc.)
        tipo ∈ {"emitente","destinatario","fornecedor"}
        """
        cnpj = MemoriaSessao._only_digits(cnpj)
        if not cnpj:
            raise ValueError("CNPJ inválido para atualizar_perfil_entidade.")
        return self.db.upsert_memoria_entidade(
            cnpj=cnpj, tipo=tipo, preferencia=preferencia, sucesso=sucesso, alpha=alpha
        )

    def obter_perfil_entidade(self, cnpj: str, tipo: str) -> Optional[Dict[str, Any]]:
        """
        Retorna dict com chaves: {id, cnpj, tipo, preferencia_json, score, amostras, updated_at}
        """
        cnpj = MemoriaSessao._only_digits(cnpj)
        if not cnpj:
            return None
        row = self.db.get_memoria_entidade(cnpj, tipo)
        if not row:
            return None
        # Normaliza preferencia_json em dict
        try:
            row["preferencia"] = json.loads(row.get("preferencia_json") or "{}")
        except Exception:
            row["preferencia"] = {}
        return row

    # ---------------------------------------------------------------------
    # Telemetria de execução (para aprendizado do Orchestrador)
    # ---------------------------------------------------------------------
    def registrar_execucao(
        self,
        *,
        doc_id: Optional[int],
        layout_fp: Optional[str],
        cnpj_emitente: Optional[str],
        cnpj_destinatario: Optional[str],
        receita: Dict[str, Any],
        metricas: Dict[str, Any],
        sucesso: bool,
    ) -> int:
        """
        Registra a decisão aplicada e métricas (coverage, confidence, sanity, match_score, latências, etc.)
        em memoria_execucao. Use após o término do processamento do documento.
        """
        return self.db.inserir_memoria_execucao(
            doc_id=doc_id,
            layout_fp=layout_fp,
            cnpj_emitente=self._maybe_digits(cnpj_emitente),
            cnpj_destinatario=self._maybe_digits(cnpj_destinatario),
            receita_aplicada=receita,
            metricas=metricas,
            sucesso=sucesso,
        )

    # ---------------------------------------------------------------------
    # Helpers de normalização
    # ---------------------------------------------------------------------
    @staticmethod
    def _normalize_text_for_fp(texto: str) -> str:
        """Normalização leve para fingerprint: lower, sem acentos, collapses brancos/pontuação."""
        t = (texto or "").lower().strip()
        # remove acentos
        t = "".join(ch for ch in unicodedata.normalize("NFD", t) if unicodedata.category(ch) != "Mn")
        # substitui pontuação por espaço
        t = re.sub(r"[^a-z0-9\s|:;,.%-]", " ", t)
        # compacta espaços
        t = re.sub(r"\s+", " ", t).strip()
        # limita tamanho para hashing estável (evita fingerprints gigantes por textos enormes)
        if len(t) > 50000:  # 50k chars é mais que suficiente p/ layout
            t = t[:50000]
        return t

    @staticmethod
    def _only_digits(s: Optional[str]) -> Optional[str]:
        if s is None:
            return None
        s2 = re.sub(r"\D+", "", s)
        return s2 or None

    @staticmethod
    def _maybe_digits(s: Optional[str]) -> Optional[str]:
        return MemoriaSessao._only_digits(s) if s else None
