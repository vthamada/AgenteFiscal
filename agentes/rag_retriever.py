# agentes/rag_retriever.py
from __future__ import annotations

import json
import math
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd

try:
    from langchain_openai import OpenAIEmbeddings  # type: ignore
except Exception:
    OpenAIEmbeddings = None  # type: ignore

try:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings  # type: ignore
except Exception:
    GoogleGenerativeAIEmbeddings = None  # type: ignore


def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    s = sum(x*y for x, y in zip(a, b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(y*y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return float(s / (na * nb))


class SimpleRAG:
    """
    RAG mínimo baseado em embeddings (OpenAI ou Gemini) persistidos em SQLite.
    - Tabela: rag_chunks(id, documento_id, source, chunk, embedding_json)
    - Coleta texto de documentos_detalhes, itens (descricao, ncm, cfop) e impostos
    - Similaridade por cosseno em memória (ok para bases pequenas/médias)
    """

    def __init__(self, db, provider: Optional[str], api_key: Optional[str]):
        self.db = db
        self.provider = (provider or '').lower().strip() if provider else ''
        self.api_key = api_key
        self._ensure_schema()
        self.emb = self._make_embeddings()

    def _ensure_schema(self) -> None:
        try:
            self.db.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS rag_chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    documento_id INTEGER NOT NULL,
                    source TEXT,
                    chunk TEXT,
                    embedding_json TEXT,
                    criado_em TEXT DEFAULT (datetime('now')),
                    FOREIGN KEY(documento_id) REFERENCES documentos(id) ON DELETE CASCADE
                )
                """
            )
            self.db.conn.execute("CREATE INDEX IF NOT EXISTS ix_rag_doc ON rag_chunks(documento_id)")
            self.db.conn.commit()
        except Exception:
            pass

    def _make_embeddings(self):
        if self.provider == 'openai' and OpenAIEmbeddings is not None:
            return OpenAIEmbeddings()
        if self.provider in {'gemini','google','google-genai'} and GoogleGenerativeAIEmbeddings is not None:
            return GoogleGenerativeAIEmbeddings(model="text-embedding-004")
        # Fallback: no-embed -> None (RAG desativado)
        return None

    def _chunk_text(self, text: str, max_chars: int = 800) -> List[str]:
        text = (text or '').strip()
        if not text:
            return []
        chunks: List[str] = []
        step = max_chars
        for i in range(0, len(text), step):
            chunks.append(text[i:i+step])
        return chunks

    def _doc_text_blocks(self, documento_id: int) -> List[Tuple[str,str]]:
        blocks: List[Tuple[str,str]] = []
        try:
            # Detalhes K/V
            det = self.db.query_table('documentos_detalhes', where=f"documento_id = {int(documento_id)}")
            if det is not None and not det.empty:
                det['par'] = det.apply(lambda r: f"{r.get('chave')}: {r.get('valor')}", axis=1)
                blocks.append(("detalhes", "\n".join(det['par'].astype(str).tolist())))
        except Exception:
            pass
        try:
            it = self.db.query_table('itens', where=f"documento_id = {int(documento_id)}")
            if it is not None and not it.empty:
                cols = [c for c in ['numero_item','descricao','codigo_produto','ncm','cfop','quantidade','valor_unitario','valor_total'] if c in it.columns]
                it2 = it[cols].fillna('').astype(str)
                lines = [" | ".join(row) for row in it2.values.tolist()]
                blocks.append(("itens","\n".join(lines)))
        except Exception:
            pass
        try:
            imp = self.db.query_table('impostos', where=f"item_id IN (SELECT id FROM itens WHERE documento_id = {int(documento_id)})")
            if imp is not None and not imp.empty:
                imp2 = imp.fillna('').astype(str)
                lines = [" | ".join(map(str, row)) for row in imp2.values.tolist()]
                blocks.append(("impostos","\n".join(lines)))
        except Exception:
            pass
        return blocks

    def index_documents(self, doc_ids: List[int]) -> int:
        if not self.emb:
            return 0
        if not doc_ids:
            return 0
        added = 0
        for did in doc_ids:
            try:
                # skip if already indexed
                cur = self.db.conn.execute("SELECT COUNT(1) FROM rag_chunks WHERE documento_id = ?", (int(did),))
                if int(cur.fetchone()[0]) > 0:
                    continue
                blocks = self._doc_text_blocks(int(did))
                for source, txt in blocks:
                    for ch in self._chunk_text(txt, max_chars=800):
                        vec = self.emb.embed_query(ch) if self.emb else []
                        self.db.conn.execute(
                            "INSERT INTO rag_chunks(documento_id, source, chunk, embedding_json) VALUES (?, ?, ?, ?)",
                            (int(did), source, ch, json.dumps(vec))
                        )
                        added += 1
                self.db.conn.commit()
            except Exception:
                pass
        return added

    def query(self, question: str, scope_doc_ids: Optional[List[int]] = None, top_k: int = 5) -> pd.DataFrame:
        if not self.emb:
            return pd.DataFrame(columns=["documento_id","chunk","score"])
        vec_q = self.emb.embed_query(question)
        where = ""
        params: Tuple[Any, ...] = tuple()
        if scope_doc_ids:
            if len(scope_doc_ids) == 1:
                where = "WHERE documento_id = ?"
                params = (int(scope_doc_ids[0]),)
            else:
                ids_sql = ",".join(map(str, [int(x) for x in scope_doc_ids]))
                where = f"WHERE documento_id IN ({ids_sql})"
        rows = self.db.conn.execute(f"SELECT documento_id, chunk, embedding_json FROM rag_chunks {where}", params).fetchall()
        scored: List[Tuple[int, str, float]] = []
        for r in rows:
            try:
                emb = json.loads(r[2]) if r[2] else []
                s = _cosine(vec_q, emb)
                if s > 0:
                    scored.append((int(r[0]), str(r[1]), float(s)))
            except Exception:
                continue
        scored.sort(key=lambda x: x[2], reverse=True)
        top = scored[:max(1, top_k)]
        return pd.DataFrame(top, columns=["documento_id","chunk","score"])