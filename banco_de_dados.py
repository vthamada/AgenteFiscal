# banco_de_dados.py

from __future__ import annotations
from typing import Optional, Dict, Any
from pathlib import Path
import sqlite3
import hashlib
import datetime as dt

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
        # check_same_thread=False para permitir uso básico multi-thread no app
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        # Garante chave estrangeira e algumas otimizações leves
        try:
            self.conn.execute("PRAGMA foreign_keys = ON")
            self.conn.execute("PRAGMA journal_mode = WAL")
            self.conn.execute("PRAGMA synchronous = NORMAL")
            self.conn.execute("PRAGMA temp_store = MEMORY")
        except sqlite3.DatabaseError:
            pass
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

        # DOCUMENTOS
        cur.execute("""
        CREATE TABLE IF NOT EXISTS documentos (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            nome_arquivo        TEXT NOT NULL,
            tipo                TEXT,                 -- NFe, NFCe, CTe, CF-e, pdf, jpg...
            origem              TEXT,                 -- upload, web, reprocessamento...
            hash                TEXT UNIQUE,          -- sha256 do arquivo
            chave_acesso        TEXT,                 -- para NFe/CTe quando existir
            status              TEXT,                 -- processando, processado, revisao_pendente, revisado, erro, quarentena
            data_upload         TEXT,                 -- ISO UTC
            data_emissao        TEXT,                 -- YYYY-MM-DD

            -- Identificação do emitente/destinatário (podem estar criptografados)
            emitente_cnpj       TEXT,
            emitente_cpf        TEXT,
            emitente_nome       TEXT,
            destinatario_cnpj   TEXT,
            destinatario_cpf    TEXT,
            destinatario_nome   TEXT,

            -- Metadados básicos para filtros rápidos
            inscricao_estadual  TEXT,
            uf                  TEXT,
            municipio           TEXT,

            -- Totais de nota (quando disponíveis)
            valor_total         REAL,
            total_produtos      REAL,
            total_servicos      REAL,

            -- Totais de impostos agregados (opcionais; úteis para relatórios)
            total_icms          REAL,
            total_ipi           REAL,
            total_pis           REAL,
            total_cofins        REAL,

            caminho_arquivo     TEXT,
            motivo_rejeicao     TEXT,
            meta_json           TEXT                  -- campo livre para metadados variados
        )
        """)

        # ITENS
        cur.execute("""
        CREATE TABLE IF NOT EXISTS itens (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            documento_id    INTEGER NOT NULL,
            numero_item     INTEGER,     -- nItem da NFe quando existir
            descricao       TEXT,
            ean             TEXT,
            ncm             TEXT,
            cest            TEXT,
            cfop            TEXT,
            quantidade      REAL,
            unidade         TEXT,
            valor_unitario  REAL,
            valor_total     REAL,
            desconto        REAL,
            outras_despesas REAL,
            codigo_produto  TEXT,
            FOREIGN KEY(documento_id) REFERENCES documentos(id) ON DELETE CASCADE
        )
        """)

        # IMPOSTOS POR ITEM
        cur.execute("""
        CREATE TABLE IF NOT EXISTS impostos (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            item_id       INTEGER NOT NULL,
            tipo_imposto  TEXT,     -- ICMS, IPI, PIS, COFINS, ISS etc.
            cst           TEXT,     -- CST/CSOSN
            origem        TEXT,     -- 'orig' do ICMS (0..8)
            base_calculo  REAL,
            aliquota      REAL,
            valor         REAL,
            FOREIGN KEY(item_id) REFERENCES itens(id) ON DELETE CASCADE
        )
        """)

        # EXTRACOES (OCR, XML parsing, etc)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS extracoes (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            documento_id        INTEGER NOT NULL,
            agente              TEXT,     -- OCRAgent, XMLParser, etc
            confianca_media     REAL,
            texto_extraido      TEXT,
            linguagem           TEXT,
            tempo_processamento REAL,
            criado_em           TEXT DEFAULT (datetime('now')),
            FOREIGN KEY(documento_id) REFERENCES documentos(id) ON DELETE CASCADE
        )
        """)

        # REVISOES manuais
        cur.execute("""
        CREATE TABLE IF NOT EXISTS revisoes (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            documento_id     INTEGER NOT NULL,
            campo            TEXT,    -- nome do campo revisado
            valor_anterior   TEXT,
            valor_corrigido  TEXT,
            usuario          TEXT,
            data_revisao     TEXT DEFAULT (datetime('now')),
            FOREIGN KEY(documento_id) REFERENCES documentos(id) ON DELETE CASCADE
        )
        """)

        # USUARIOS
        cur.execute("""
        CREATE TABLE IF NOT EXISTS usuarios (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            nome        TEXT,
            email       TEXT UNIQUE,
            perfil      TEXT,     -- admin, auditor, operador, etc
            senha_hash  TEXT
        )
        """)

        # METRICAS
        cur.execute("""
        CREATE TABLE IF NOT EXISTS metricas (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            tipo_documento  TEXT,
            acuracia_media  REAL,
            taxa_revisao    REAL,
            tempo_medio     REAL,
            taxa_erro       REAL,
            registrado_em   TEXT DEFAULT (datetime('now'))
        )
        """)

        # LOGS
        cur.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            evento     TEXT,
            usuario    TEXT,
            detalhes   TEXT,
            criado_em  TEXT DEFAULT (datetime('now'))
        )
        """)

        # MEMORIA (para o Agente Analítico)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS memoria (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            pergunta        TEXT,
            resposta_resumo TEXT,
            duracao_s       REAL,
            criado_em       TEXT DEFAULT (datetime('now'))
        )
        """)

        # ------------------- Índices úteis para performance -------------------
        cur.execute("CREATE INDEX IF NOT EXISTS ix_documentos_status ON documentos(status)")
        cur.execute("CREATE INDEX IF NOT EXISTS ix_documentos_tipo ON documentos(tipo)")
        cur.execute("CREATE INDEX IF NOT EXISTS ix_documentos_uf ON documentos(uf)")
        cur.execute("CREATE INDEX IF NOT EXISTS ix_documentos_data_emissao ON documentos(data_emissao)")
        cur.execute("CREATE INDEX IF NOT EXISTS ix_itens_doc ON itens(documento_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS ix_impostos_item ON impostos(item_id)")

        self.conn.commit()

    # ------------------------- Operações de escrita -------------------------
    def inserir_documento(self, **campos) -> int:
        """
        Insere na tabela documentos. Retorna ID.
        """
        if not campos:
            raise ValueError("Nenhum campo fornecido para inserir_documento.")
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
        if not campos:
            raise ValueError("Nenhum campo fornecido para inserir_item.")
        columns = ", ".join(campos.keys())
        placeholders = ", ".join("?" for _ in campos)
        sql = f"INSERT INTO itens ({columns}) VALUES ({placeholders})"
        cur = self.conn.cursor()
        cur.execute(sql, list(campos.values()))
        self.conn.commit()
        return int(cur.lastrowid)

    def inserir_imposto(self, **campos) -> int:
        if not campos:
            raise ValueError("Nenhum campo fornecido para inserir_imposto.")
        columns = ", ".join(campos.keys())
        placeholders = ", ".join("?" for _ in campos)
        sql = f"INSERT INTO impostos ({columns}) VALUES ({placeholders})"
        cur = self.conn.cursor()
        cur.execute(sql, list(campos.values()))
        self.conn.commit()
        return int(cur.lastrowid)

    def inserir_extracao(self, **campos) -> int:
        if not campos:
            raise ValueError("Nenhum campo fornecido para inserir_extracao.")
        columns = ", ".join(campos.keys())
        placeholders = ", ".join("?" for _ in campos)
        sql = f"INSERT INTO extracoes ({columns}) VALUES ({placeholders})"
        cur = self.conn.cursor()
        cur.execute(sql, list(campos.values()))
        self.conn.commit()
        return int(cur.lastrowid)

    # --- CRUDs auxiliares ---
    def inserir_revisao(self, **campos) -> int:
        if not campos:
            raise ValueError("Nenhum campo fornecido para inserir_revisao.")
        columns = ", ".join(campos.keys())
        placeholders = ", ".join("?" for _ in campos)
        sql = f"INSERT INTO revisoes ({columns}) VALUES ({placeholders})"
        cur = self.conn.cursor()
        cur.execute(sql, list(campos.values()))
        self.conn.commit()
        return int(cur.lastrowid)

    def inserir_metrica(self, **campos) -> int:
        if not campos:
            raise ValueError("Nenhum campo fornecido para inserir_metrica.")
        columns = ", ".join(campos.keys())
        placeholders = ", ".join("?" for _ in campos)
        sql = f"INSERT INTO metricas ({columns}) VALUES ({placeholders})"
        cur = self.conn.cursor()
        cur.execute(sql, list(campos.values()))
        self.conn.commit()
        return int(cur.lastrowid)

    def inserir_usuario(self, **campos) -> int:
        if not campos:
            raise ValueError("Nenhum campo fornecido para inserir_usuario.")
        columns = ", ".join(campos.keys())
        placeholders = ", ".join("?" for _ in campos)
        sql = f"INSERT INTO usuarios ({columns}) VALUES ({placeholders})"
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
        # Tabelas permitidas
        allowed_tables = {
            "documentos", "itens", "impostos", "extracoes",
            "logs", "memoria", "revisoes", "usuarios", "metricas"
        }
        if table not in allowed_tables:
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
