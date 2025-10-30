# orchestrator_clean.py — Orquestrador cognitivo (XML-first, clean)
from __future__ import annotations

import os
import json
import time
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from banco_de_dados import BancoDeDados
from validacao import ValidadorFiscal
from memoria import MemoriaSessao
from agentes import (
    AgenteXMLParser,
    AgenteNormalizadorCampos,
    MetricsAgent,
)

# Agente analítico é opcional (exportado apenas quando disponível)
try:
    from agentes import AgenteAnalitico  # type: ignore
except Exception:
    AgenteAnalitico = None  # type: ignore

log = logging.getLogger(__name__)


class Orchestrator:
    """
    Pipeline cognitivo e enxuto, 100% focado em XML.
    - Somente .xml é processado; demais formatos vão para quarentena.
    - Após extração via AgenteXMLParser, normaliza campos, avalia qualidade
      (completeness/critical/sanity), registra decisões (blackboard) e ajusta status.
    - Reprocessamento/revalidação utilitários.
    - Q&A com LLM quando disponível; caso contrário, respostas determinísticas seguras.
    """

    # Modo de operação
    ONLY_XML: bool = os.getenv("ONLY_XML", "1").strip().lower() in {"1", "true", "yes", "on"}

    # Sinais de qualidade
    CRITICAL_FIELDS: Tuple[str, ...] = ("emitente_cnpj", "valor_total", "data_emissao")
    COMPLETENESS_MIN: float = float(os.getenv("COMPLETENESS_MIN", "0.65"))
    CRITICAL_MIN_HITS: int = int(os.getenv("CRITICAL_MIN_HITS", "2"))
    SANITY_MIN: float = float(os.getenv("SANITY_MIN", "0.75"))

    def __init__(
        self,
        db: BancoDeDados,
        validador: Optional[ValidadorFiscal] = None,
        memoria: Optional[MemoriaSessao] = None,
        llm: Optional[Any] = None,
    ) -> None:
        self.db = db
        self.validador = validador or ValidadorFiscal()
        self.memoria = memoria or MemoriaSessao(self.db)
        self.llm = llm

        self.metrics_agent = MetricsAgent(llm=self.llm)
        self.xml_agent = AgenteXMLParser(self.db, self.validador, self.metrics_agent)
        self.normalizador = AgenteNormalizadorCampos(
            llm=self.llm,
            enable_llm=True if self.llm else False,
            drop_general_fields=False,
            keep_context_copy=True,
        )
        self.analitico = (AgenteAnalitico(self.llm, self.memoria) if (self.llm and AgenteAnalitico) else None)  # type: ignore

        if self.ONLY_XML:
            log.info("Orchestrator: Modo Somente XML ativo.")
        else:
            log.info("Orchestrator: XML preferencial (sem OCR/NLP).")

    # --------------------------- Ingestão ---------------------------
    def processar_automatico(self, nome: str, conteudo: bytes, origem: str = "upload_ui") -> int:
        ext = Path(nome).suffix.lower()
        if ext == ".xml":
            doc_id = int(self.xml_agent.processar(nome, conteudo, origem=origem))
            # Normalização pós-extração
            try:
                self._normalizar_documento(doc_id)
            except Exception as e:
                log.debug(f"Normalização falhou doc_id={doc_id}: {e}")
            # Camada cognitiva pós-extração
            try:
                self._avaliar_e_anotar(doc_id)
            except Exception as e:
                log.debug(f"Avaliacao cognitiva falhou doc_id={doc_id}: {e}")
            return doc_id

        # Quarentena para não-XML
        try:
            caminho = str(self.db.save_upload(nome, conteudo))
        except Exception:
            caminho = None
        motivo = (
            "Modo Somente XML: envie o XML fiscal correspondente." if self.ONLY_XML else "Formato não suportado."
        )
        try:
            doc_id = self.db.inserir_documento(
                nome_arquivo=nome,
                tipo=ext.lstrip(".") or "desconhecido",
                origem=origem,
                hash=self.db.hash_bytes(conteudo),
                status="quarentena",
                data_upload=self.db.now(),
                caminho_arquivo=caminho,
                motivo_rejeicao=motivo,
            )
            # Registra decisão para transparência
            self._merge_meta_json(doc_id, {
                "blackboard": {"decisions": [{"ts": time.time(), "msg": "Arquivo não-XML enviado à quarentena."}]}
            })
            return int(doc_id)
        except Exception as e:
            log.error(f"Falha ao colocar em quarentena '{nome}': {e}")
            try:
                return int(self.db.inserir_documento(
                    nome_arquivo=nome,
                    tipo=ext.lstrip(".") or "desconhecido",
                    origem=origem,
                    hash=self.db.hash_bytes(conteudo),
                    status="erro",
                    data_upload=self.db.now(),
                    motivo_rejeicao=f"Falha na ingestão: {e}",
                ))
            except Exception:
                return -1

    # --------------------------- Reprocessamento ---------------------------
    def reprocessar_documento(self, documento_id: int) -> Dict[str, Any]:
        doc = self.db.get_documento(int(documento_id)) or {}
        nome = doc.get("nome_arquivo") or f"doc_{int(documento_id)}.xml"
        caminho = doc.get("caminho_xml") or doc.get("caminho_arquivo")
        if not caminho or not os.path.exists(caminho):
            return {"ok": False, "mensagem": "Caminho do arquivo não encontrado para reprocessar."}
        try:
            with open(caminho, "rb") as f:
                conteudo = f.read()
            if Path(nome).suffix.lower() != ".xml" and Path(caminho).suffix.lower() == ".xml":
                nome = Path(nome).with_suffix(".xml").name
            novo_id = self.xml_agent.processar(nome, conteudo, origem="reprocessamento")
            try:
                self._normalizar_documento(int(novo_id))
                self._avaliar_e_anotar(int(novo_id))
            except Exception:
                pass
            return {"ok": True, "mensagem": f"Documento reprocessado (novo id {int(novo_id)}).", "novo_id": int(novo_id)}
        except Exception as e:
            return {"ok": False, "mensagem": f"Falha no reprocessamento: {e}"}

    # --------------------------- Revalidação ---------------------------
    def revalidar_documento(self, documento_id: int) -> Dict[str, Any]:
        try:
            self.validador.validar_documento(doc_id:=int(documento_id), db=self.db)  # type: ignore
            return {"ok": True, "mensagem": f"Documento {int(documento_id)} revalidado."}
        except Exception as e:
            return {"ok": False, "mensagem": f"Falha na revalidação: {e}"}

    # --------------------------- Q&A / Análises ---------------------------
    def responder_pergunta(
        self,
        pergunta: str,
        scope_filters: Optional[Dict[str, Any]] = None,
        safe_mode: bool = False,
    ) -> Dict[str, Any]:
        t0 = time.time()
        scope_filters = scope_filters or {}

        if not pergunta:
            return {"texto": "Pergunta vazia.", "duracao_s": 0.0, "agent_name": "SafeQuery"}

        # Preferir LLM sempre que disponível; respostas determinísticas apenas como fallback
        if not safe_mode and self.analitico and self.llm:
            try:
                catalog = self._build_catalog(scope_filters)
                out = self.analitico.responder(pergunta, catalog)
                out["duracao_s"] = float(time.time() - t0)
                out["agent_name"] = out.get("agent_name", "AgenteAnalitico")
                return out
            except Exception as e:
                log.warning(f"LLM/Analítico falhou; caindo para modo seguro: {e}")

        # Modo seguro: respostas determinísticas simples
        code, df, text = self._safe_answer(pergunta, scope_filters)
        return {
            "texto": text,
            "tabela": df,
            "code": code,
            "duracao_s": float(time.time() - t0),
            "agent_name": "SafeQuery",
        }

    def _build_catalog(self, scope: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Constrói o catálogo de DataFrames para o Agente Analítico a partir do SQLite,
        aplicando filtros de escopo quando fornecidos.
        Tabelas incluídas: documentos, itens, impostos, documentos_detalhes.
        """
        scope = scope or {}
        where_parts: list[str] = []
        uf = (scope.get("uf") or "").strip().upper()
        if uf:
            where_parts.append(f"(emitente_uf = '{uf}' OR destinatario_uf = '{uf}')")
        tipos = scope.get("tipo") or []
        if isinstance(tipos, (list, tuple)) and tipos:
            tipos_sql = ", ".join([f"'{str(t).strip()}'" for t in tipos])
            where_parts.append(f"tipo IN ({tipos_sql})")
        doc_where = " AND ".join(where_parts) if where_parts else None

        # Documentos
        df_docs = self.db.query_table("documentos", where=doc_where)

        # Itens (filtra por documentos, quando houver filtro)
        itens_where = None
        if doc_where:
            itens_where = f"documento_id IN (SELECT id FROM documentos WHERE {doc_where})"
        df_itens = self.db.query_table("itens", where=itens_where)

        # Impostos (filtra por itens -> documentos quando houver filtro)
        impostos_where = None
        if doc_where:
            impostos_where = (
                "item_id IN (SELECT id FROM itens WHERE documento_id IN (SELECT id FROM documentos WHERE "
                + doc_where + "))"
            )
        df_impostos = self.db.query_table("impostos", where=impostos_where)

        # Detalhes (key-value por documento)
        detalhes_where = None
        if doc_where:
            detalhes_where = f"documento_id IN (SELECT id FROM documentos WHERE {doc_where})"
        df_det = self.db.query_table("documentos_detalhes", where=detalhes_where)

        return {
            "documentos": df_docs,
            "itens": df_itens,
            "impostos": df_impostos,
            "documentos_detalhes": df_det,
        }

    # --------------------------- Camada cognitiva ---------------------------
    def _avaliar_e_anotar(self, doc_id: int) -> None:
        doc = self.db.get_documento(doc_id) or {}
        meta = self._safe_json_loads(doc.get("meta_json"))
        bblog: list[Dict[str, Any]] = []

        # 1) Sinais extraídos
        meta_root = meta.get("__meta__") or {}
        completeness = float(meta_root.get("field_completeness") or 0.0)
        sanity = None
        try:
            sanity = float(((meta.get("__meta__") or {}).get("normalizador") or {}).get("sanity_score"))
        except Exception:
            sanity = None
        crit_hits = 0
        for f in self.CRITICAL_FIELDS:
            val = doc.get(f)
            if val not in (None, "", [], {}):
                crit_hits += 1

        bblog.append({"ts": time.time(), "msg": f"Completeness={completeness:.2f}; critical_hits={crit_hits}; sanity={sanity if sanity is not None else '—'}"})

        # 2) Heurísticas de qualidade → status
        status = doc.get("status") or "processado"
        needs_review = False
        if completeness < self.COMPLETENESS_MIN:
            needs_review = True
            bblog.append({"ts": time.time(), "msg": f"Completeness abaixo do mínimo ({self.COMPLETENESS_MIN})."})
        if crit_hits < self.CRITICAL_MIN_HITS:
            needs_review = True
            bblog.append({"ts": time.time(), "msg": f"Campos críticos insuficientes (min={self.CRITICAL_MIN_HITS})."})
        if sanity is not None and sanity < self.SANITY_MIN:
            needs_review = True
            bblog.append({"ts": time.time(), "msg": f"Sanidade do normalizador abaixo do mínimo ({self.SANITY_MIN})."})

        if needs_review and status not in ("revisado",):
            self.db.atualizar_documento_campos(doc_id, status="revisao_pendente")
            bblog.append({"ts": time.time(), "msg": "Status ajustado para revisao_pendente."})
        else:
            bblog.append({"ts": time.time(), "msg": f"Status mantido: {status}."})

        # 3) Persistir decisões no meta_json
        patch = meta.copy()
        patch.setdefault("blackboard", {})
        patch["blackboard"].setdefault("decisions", [])
        patch["blackboard"]["decisions"].extend(bblog)
        patch.setdefault("quality", {"completeness": completeness, "critical_hits": crit_hits, "sanity": sanity})
        self._merge_meta_json(doc_id, patch)

    # --------------------------- Helpers ---------------------------
    def _safe_json_loads(self, text: Optional[str]) -> Dict[str, Any]:
        if not text:
            return {}
        try:
            data = json.loads(text)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _merge_meta_json(self, doc_id: int, patch: Dict[str, Any]) -> None:
        try:
            atual = self.db.get_documento(doc_id) or {}
            base = self._safe_json_loads(atual.get("meta_json"))

            def _deep(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
                out = dict(a or {})
                for k, v in (b or {}).items():
                    if isinstance(v, dict) and isinstance(out.get(k), dict):
                        out[k] = _deep(out[k], v)
                    else:
                        out[k] = v
                return out

            novo = _deep(base, patch or {})
            self.db.atualizar_documento_campos(doc_id, meta_json=json.dumps(novo, ensure_ascii=False))
        except Exception as e:
            log.debug(f"Falha ao atualizar meta_json doc_id={doc_id}: {e}")

    def _normalizar_documento(self, doc_id: int) -> None:
        """Aplica normalização contextual e persiste campos e meta do normalizador."""
        doc = self.db.get_documento(doc_id) or {}
        if not doc:
            return
        try:
            out = self.normalizador.normalizar(doc, keep_context_copy=True)
        except Exception:
            return
        if not isinstance(out, dict) or not out:
            return

        # Se vier __meta__.normalizador, anexa ao meta_json
        meta_patch: Dict[str, Any] = {}
        nm = ((out.get("__meta__") or {}).get("normalizador") or None)
        if nm and isinstance(nm, dict):
            meta_patch.setdefault("__meta__", {})
            meta_patch["__meta__"]["normalizador"] = nm

        # Atualiza campos canônicos presentes
        allowed = set(self.db._colunas_tabela("documentos")) if hasattr(self.db, "_colunas_tabela") else set()
        update_fields: Dict[str, Any] = {}
        for k, v in out.items():
            if k.startswith("__"):
                continue
            if allowed and k not in allowed:
                continue
            if v is not None and v != doc.get(k):
                update_fields[k] = v

        if update_fields:
            try:
                self.db.atualizar_documento_campos(doc_id, **update_fields)
            except Exception:
                pass
        if meta_patch:
            self._merge_meta_json(doc_id, meta_patch)

    def _safe_answer(self, pergunta: str, scope: Dict[str, Any]) -> tuple[str, Any, str]:
        # Importa pandas sob demanda para evitar hard-dependency no import do módulo
        import pandas as pd  # type: ignore

        where = []
        if scope.get("uf"):
            uf = str(scope["uf"]).strip().upper()
            if uf:
                where.append(f"(emitente_uf = '{uf}' OR destinatario_uf = '{uf}')")
        tipos = scope.get("tipo") or []
        if tipos:
            tipos_sql = ", ".join([f"'{str(t).strip()}'" for t in tipos])
            where.append(f"tipo IN ({tipos_sql})")
        where_clause = " AND ".join(where) if where else None

        q = pergunta.lower()
        if "top" in q and ("emitente" in q or "fornecedor" in q):
            sql = "SELECT emitente_nome, SUM(COALESCE(valor_total,0)) AS total FROM documentos"
            if where_clause:
                sql += f" WHERE {where_clause}"
            sql += " GROUP BY emitente_nome ORDER BY total DESC LIMIT 5"
            df = pd.read_sql_query(sql, self.db.conn)
            text = "Top emitentes por valor total."
            return sql, df, text

        if "uf" in q and ("valor" in q or "faturamento" in q):
            sql = "SELECT COALESCE(emitente_uf, destinatario_uf) AS uf, SUM(COALESCE(valor_total,0)) AS total FROM documentos"
            if where_clause:
                sql += f" WHERE {where_clause}"
            sql += " GROUP BY uf ORDER BY total DESC"
            df = pd.read_sql_query(sql, self.db.conn)
            text = "Resumo por UF."
            return sql, df, text

        sql = "SELECT tipo, COUNT(1) AS qtd, SUM(COALESCE(valor_total,0)) AS total FROM documentos"
        if where_clause:
            sql += f" WHERE {where_clause}"
        sql += " GROUP BY tipo ORDER BY total DESC"
        df = pd.read_sql_query(sql, self.db.conn)
        text = "Resumo por tipo de documento."
        return sql, df, text
