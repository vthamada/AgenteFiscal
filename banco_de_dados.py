# banco_de_dados.py

from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Optional, Dict, Any, Iterable
from pathlib import Path
import os
import sqlite3
import hashlib
import datetime as dt
import json

import pandas as pd


DATA_DIR = Path("data")
UPLOADS_DIR = DATA_DIR / "uploads"
DB_PATH = DATA_DIR / "base_dados.sqlite"


def _ensure_dirs() -> None:
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


def _utcnow_iso() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


class BancoDeDados:
    """
    Camada de persistência (SQLite) com schema e helpers usados por agentes.py.
    """

    def __init__(self, db_path: Path = DB_PATH):
        _ensure_dirs()
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._criar_schema()

    # ------------------------- Infra & utilidades -------------------------
    def now(self) -> str:
        return _utcnow_iso()

    def hash_bytes(self, content: bytes) -> str:
        h = hashlib.sha256()
        h.update(content)
        return h.hexdigest()

    def save_upload(self, nome_arquivo: str, conteudo: bytes) -> Path:
        """
        Salva o arquivo original em data/uploads/<YYYYMMDD>/<hash>__<nome>
        """
        h = self.hash_bytes(conteudo)[:16]
        sub = dt.datetime.utcnow().strftime("%Y%m%d")
        folder = UPLOADS_DIR / sub
        folder.mkdir(parents=True, exist_ok=True)
        safe_name = nome_arquivo.replace("/", "_").replace("\\", "_")
        destino = folder / f"{h}__{safe_name}"
        with open(destino, "wb") as f:
            f.write(conteudo)
        return destino

    # ------------------------- Schema -------------------------
    def _criar_schema(self) -> None:
        cur = self.conn.cursor()

        cur.execute("""
        CREATE TABLE IF NOT EXISTS documentos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nome_arquivo TEXT NOT NULL,
            tipo TEXT,
            origem TEXT,
            hash TEXT UNIQUE,
            chave_acesso TEXT,
            status TEXT,
            data_upload TEXT,
            data_emissao TEXT,
            emitente_cnpj TEXT,
            emitente_nome TEXT,
            destinatario_cnpj TEXT,
            destinatario_nome TEXT,
            valor_total REAL,
            caminho_arquivo TEXT,
            motivo_rejeicao TEXT
        )
        """)

        cur.execute("""
        CREATE TABLE IF NOT EXISTS itens (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            documento_id INTEGER NOT NULL,
            descricao TEXT,
            ncm TEXT,
            cest TEXT,
            cfop TEXT,
            quantidade REAL,
            unidade TEXT,
            valor_unitario REAL,
            valor_total REAL,
            codigo_produto TEXT,
            FOREIGN KEY(documento_id) REFERENCES documentos(id) ON DELETE CASCADE
        )
        """)

        cur.execute("""
        CREATE TABLE IF NOT EXISTS impostos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            item_id INTEGER NOT NULL,
            tipo_imposto TEXT,
            cst TEXT,
            origem TEXT,
            base_calculo REAL,
            aliquota REAL,
            valor REAL,
            FOREIGN KEY(item_id) REFERENCES itens(id) ON DELETE CASCADE
        )
        """)

        cur.execute("""
        CREATE TABLE IF NOT EXISTS extracoes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            documento_id INTEGER NOT NULL,
            agente TEXT,
            confianca_media REAL,
            texto_extraido TEXT,
            linguagem TEXT,
            tempo_processamento REAL,
            criado_em TEXT DEFAULT (datetime('now')),
            FOREIGN KEY(documento_id) REFERENCES documentos(id) ON DELETE CASCADE
        )
        """)

        cur.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            evento TEXT,
            usuario TEXT,
            detalhes TEXT,
            criado_em TEXT DEFAULT (datetime('now'))
        )
        """)

        cur.execute("""
        CREATE TABLE IF NOT EXISTS memoria (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pergunta TEXT,
            resposta_resumo TEXT,
            duracao_s REAL,
            criado_em TEXT DEFAULT (datetime('now'))
        )
        """)

        self.conn.commit()

    # ------------------------- Operações de escrita -------------------------
    def inserir_documento(self, **campos) -> int:
        """
        Insere na tabela documentos. Retorna ID.
        """
        columns = ", ".join(campos.keys())
        placeholders = ", ".join("?" for _ in campos)
        sql = f"INSERT INTO documentos ({columns}) VALUES ({placeholders})"
        cur = self.conn.cursor()
        cur.execute(sql, list(campos.values()))
        self.conn.commit()
        return int(cur.lastrowid)

    def atualizar_documento_campo(self, documento_id: int, campo: str, valor: Any) -> None:
        sql = f"UPDATE documentos SET {campo} = ? WHERE id = ?"
        self.conn.execute(sql, (valor, documento_id))
        self.conn.commit()

    def atualizar_documento_campos(self, documento_id: int, **campos) -> None:
        if not campos:
            return
        sets = ", ".join(f"{k} = ?" for k in campos.keys())
        vals = list(campos.values()) + [documento_id]
        sql = f"UPDATE documentos SET {sets} WHERE id = ?"
        self.conn.execute(sql, vals)
        self.conn.commit()

    def inserir_item(self, **campos) -> int:
        columns = ", ".join(campos.keys())
        placeholders = ", ".join("?" for _ in campos)
        sql = f"INSERT INTO itens ({columns}) VALUES ({placeholders})"
        cur = self.conn.cursor()
        cur.execute(sql, list(campos.values()))
        self.conn.commit()
        return int(cur.lastrowid)

    def inserir_imposto(self, **campos) -> int:
        columns = ", ".join(campos.keys())
        placeholders = ", ".join("?" for _ in campos)
        sql = f"INSERT INTO impostos ({columns}) VALUES ({placeholders})"
        cur = self.conn.cursor()
        cur.execute(sql, list(campos.values()))
        self.conn.commit()
        return int(cur.lastrowid)

    def inserir_extracao(self, **campos) -> int:
        columns = ", ".join(campos.keys())
        placeholders = ", ".join("?" for _ in campos)
        sql = f"INSERT INTO extracoes ({columns}) VALUES ({placeholders})"
        cur = self.conn.cursor()
        cur.execute(sql, list(campos.values()))
        self.conn.commit()
        return int(cur.lastrowid)

    def log(self, evento: str, usuario: str, detalhes: str) -> None:
        self.conn.execute(
            "INSERT INTO logs (evento, usuario, detalhes) VALUES (?, ?, ?)",
            (evento, usuario, detalhes),
        )
        self.conn.commit()

    # ------------------------- Operações de leitura -------------------------
    def find_documento_by_hash(self, doc_hash: str) -> Optional[int]:
        cur = self.conn.cursor()
        cur.execute("SELECT id FROM documentos WHERE hash = ?", (doc_hash,))
        row = cur.fetchone()
        return int(row["id"]) if row else None

    def get_documento(self, documento_id: int) -> Optional[Dict[str, Any]]:
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM documentos WHERE id = ?", (documento_id,))
        row = cur.fetchone()
        return dict(row) if row else None

    def query_table(self, table: str, where: Optional[str] = None) -> pd.DataFrame:
        if table not in {"documentos", "itens", "impostos", "extracoes", "logs", "memoria"}:
            raise ValueError(f"Tabela não suportada: {table}")
        sql = f"SELECT * FROM {table}"
        if where:
            sql += f" WHERE {where}"
        return pd.read_sql_query(sql, self.conn)

    # ------------------------- Fechamento -------------------------
    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass
