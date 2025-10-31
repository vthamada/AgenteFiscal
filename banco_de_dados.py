# banco_de_dados.py

from __future__ import annotations
from typing import Optional, Dict, Any, List
from pathlib import Path
import sqlite3
import hashlib
import datetime as dt
import pandas as pd
import json
import logging

log = logging.getLogger(__name__)

DATA_DIR = Path("data")
UPLOADS_DIR = DATA_DIR / "uploads"
DB_PATH = DATA_DIR / "base_dados.sqlite"


def _ensure_dirs() -> None:
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


def _utcnow_iso() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


class BancoDeDados:
    """
    Camada de persistência (SQLite) com schema completo e índices.
    Sem migrações automáticas nesta fase de desenvolvimento.
    Compatível com agentes.py, validacao.py, orchestrator e testes integrados.
    """

    def __init__(self, db_path: Path = DB_PATH):
        _ensure_dirs()
        self.db_path = Path(db_path)
        # check_same_thread=False para uso básico multithread (ex.: app web)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        # Pragmas importantes
        try:
            self.conn.execute("PRAGMA foreign_keys = ON")
            self.conn.execute("PRAGMA journal_mode = WAL")
            self.conn.execute("PRAGMA synchronous = NORMAL")
            self.conn.execute("PRAGMA temp_store = MEMORY")
        except sqlite3.DatabaseError:
            pass
        # Cria o schema (não há migração nesta versão)
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

    # ------------------------- Schema (criação base) -------------------------
    def _criar_schema(self) -> None:
        cur = self.conn.cursor()

        # DOCUMENTOS (campos universais + agregados p/ filtros e dashboards)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS documentos (
            id                      INTEGER PRIMARY KEY AUTOINCREMENT,
            nome_arquivo            TEXT NOT NULL,
            tipo                    TEXT,                 -- NFe, NFCe, CTe, NFSe, pdf, jpg...
            origem                  TEXT,                 -- upload, web, reprocessamento...
            hash                    TEXT UNIQUE,          -- sha256 do arquivo
            chave_acesso            TEXT,                 -- para NFe/CTe quando existir

            status                  TEXT,                 -- processando, processado, revisao_pendente, revisado, erro, quarentena
            data_upload             TEXT,                 -- ISO UTC
            data_emissao            TEXT,                 -- YYYY-MM-DD

            -- Identificação do emitente/destinatário (podem estar criptografados)
            emitente_cnpj           TEXT,
            emitente_cpf            TEXT,
            emitente_nome           TEXT,
            destinatario_cnpj       TEXT,
            destinatario_cpf        TEXT,
            destinatario_nome       TEXT,

            -- Detalhes do emitente
            emitente_ie             TEXT,
            emitente_im             TEXT,
            emitente_uf             TEXT,
            emitente_municipio      TEXT,
            emitente_endereco       TEXT,

            -- Detalhes do destinatário
            destinatario_ie         TEXT,
            destinatario_im         TEXT,
            destinatario_uf         TEXT,
            destinatario_municipio  TEXT,
            destinatario_endereco   TEXT,

            -- Endereços detalhados (novos campos)
            emitente_logradouro     TEXT,
            emitente_numero         TEXT,
            emitente_complemento    TEXT,
            emitente_bairro         TEXT,
            emitente_cep            TEXT,
            destinatario_logradouro TEXT,
            destinatario_numero     TEXT,
            destinatario_complemento TEXT,
            destinatario_bairro     TEXT,
            destinatario_cep        TEXT,

            -- Identificação fiscal adicional
            numero_nota             TEXT,
            serie                   TEXT,
            modelo                  TEXT,

            -- Totais de nota (quando disponíveis)
            valor_total             REAL,
            total_produtos          REAL,
            total_servicos          REAL,

            -- Totais complementares (XML/NLP)
            valor_descontos         REAL,
            valor_frete             REAL,
            valor_seguro            REAL,
            valor_outros            REAL,
            valor_liquido           REAL,

            -- Totais de impostos agregados
            total_icms              REAL,
            total_ipi               REAL,
            total_pis               REAL,
            total_cofins            REAL,

            -- Transporte (universais)
            modalidade_frete        TEXT,
            placa_veiculo           TEXT,
            uf_veiculo              TEXT,
            peso_bruto              REAL,
            peso_liquido            REAL,
            qtd_volumes             REAL,

            -- Pagamento (universais)
            forma_pagamento         TEXT,
            valor_pagamento         REAL,
            troco                   REAL,

            -- Dados específicos de XML e autorização
            caminho_arquivo         TEXT,                 -- caminho do arquivo original (PDF/imagem)
            caminho_xml             TEXT,                 -- caminho do XML (se houver)
            versao_schema           TEXT,                 -- ex.: 4.00
            ambiente                TEXT,                 -- 1=produção, 2=homologação
            protocolo_autorizacao   TEXT,                 -- nProt
            data_autorizacao        TEXT,                 -- dhRecbto
            cstat                   TEXT,                 -- código de status
            xmotivo                 TEXT,                 -- descrição do status
            responsavel_tecnico     TEXT,

            -- Telemetria de OCR
            ocr_tipo                TEXT,                 -- 'nativo' | 'ocr' | null

            motivo_rejeicao         TEXT,
            meta_json               TEXT
        )
        """)

        # DETALHES DINÂMICOS (key-value)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS documentos_detalhes (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            documento_id    INTEGER NOT NULL,
            chave           TEXT NOT NULL,   -- ex.: 'emit/enderEmit/xLgr', 'dest/enderDest/CEP', 'vFCPST'
            valor           TEXT,            -- armazenado como TEXT; conversão na aplicação
            origem          TEXT,            -- xml_parser, ocr_nlp, revisao, integracao, etc.
            criado_em       TEXT DEFAULT (datetime('now')),
            UNIQUE(documento_id, chave),
            FOREIGN KEY(documento_id) REFERENCES documentos(id) ON DELETE CASCADE
        )
        """)

        # ITENS
        cur.execute("""
        CREATE TABLE IF NOT EXISTS itens (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            documento_id    INTEGER NOT NULL,
            numero_item     INTEGER,
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

        # METRICAS (inclui meta_json)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS metricas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tipo_documento TEXT,
            acuracia_media REAL,
            taxa_revisao REAL,
            taxa_erro REAL,
            tempo_medio REAL,
            meta_json TEXT,
            registrado_em TEXT DEFAULT (datetime('now'))
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

        # CONFIGURAÇÕES (chave/valor) para preferências do usuário e integrações
        cur.execute("""
        CREATE TABLE IF NOT EXISTS config (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            chave    TEXT NOT NULL,
            valor    TEXT,
            usuario  TEXT,
            UNIQUE(chave, usuario)
        )
        """)

        # CATÁLOGO NCM
        cur.execute("""
        CREATE TABLE IF NOT EXISTS ncm_catalogo (
            codigo TEXT PRIMARY KEY,
            descricao TEXT
        )
        """)

        # ------------------- Índices úteis para performance -------------------
        cur.execute("CREATE INDEX IF NOT EXISTS ix_documentos_status ON documentos(status)")
        cur.execute("CREATE INDEX IF NOT EXISTS ix_documentos_tipo ON documentos(tipo)")
        cur.execute("CREATE INDEX IF NOT EXISTS ix_documentos_data_emissao ON documentos(data_emissao)")
        cur.execute("CREATE INDEX IF NOT EXISTS ix_documentos_num_serie ON documentos(numero_nota, serie)")
        cur.execute("CREATE INDEX IF NOT EXISTS ix_documentos_chave ON documentos(chave_acesso)")
        cur.execute("CREATE INDEX IF NOT EXISTS ix_documentos_emit_cnpj ON documentos(emitente_cnpj)")
        cur.execute("CREATE INDEX IF NOT EXISTS ix_documentos_dest_cnpj ON documentos(destinatario_cnpj)")
        cur.execute("CREATE INDEX IF NOT EXISTS ix_documentos_emit_nome ON documentos(emitente_nome)")
        cur.execute("CREATE INDEX IF NOT EXISTS ix_itens_doc ON itens(documento_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS ix_itens_ncm ON itens(ncm)")
        cur.execute("CREATE INDEX IF NOT EXISTS ix_itens_cfop ON itens(cfop)")
        cur.execute("CREATE INDEX IF NOT EXISTS ix_impostos_item ON impostos(item_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS ix_impostos_cst ON impostos(cst)")
        cur.execute("CREATE INDEX IF NOT EXISTS ix_docdet_doc ON documentos_detalhes(documento_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS ix_docdet_chave ON documentos_detalhes(chave)")
        cur.execute("CREATE INDEX IF NOT EXISTS ix_config_chave_usuario ON config(chave, usuario)")

        self.conn.commit()

        # Garante colunas novas via ALTER TABLE quando base já existe
        try:
            self._ensure_column("documentos", "emitente_logradouro TEXT")
            self._ensure_column("documentos", "emitente_numero TEXT")
            self._ensure_column("documentos", "emitente_complemento TEXT")
            self._ensure_column("documentos", "emitente_bairro TEXT")
            self._ensure_column("documentos", "emitente_cep TEXT")
            self._ensure_column("documentos", "destinatario_logradouro TEXT")
            self._ensure_column("documentos", "destinatario_numero TEXT")
            self._ensure_column("documentos", "destinatario_complemento TEXT")
            self._ensure_column("documentos", "destinatario_bairro TEXT")
            self._ensure_column("documentos", "destinatario_cep TEXT")
            # Tabela RAG para chunks de texto e embeddings
            self.conn.execute(
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
            self.conn.execute("CREATE INDEX IF NOT EXISTS ix_rag_doc ON rag_chunks(documento_id)")
            self.conn.commit()
        except Exception:
            pass

    # ------------------------- Helpers internos -------------------------
    def _colunas_tabela(self, tabela: str) -> set[str]:
        cur = self.conn.cursor()
        cur.execute(f"PRAGMA table_info({tabela})")
        return {row[1] for row in cur.fetchall()}

    def _filtrar_campos_validos(self, tabela: str, campos: dict) -> dict:
        """Remove chaves inexistentes e converte dict/list em JSON."""
        if not campos:
            return {}
        colunas_validas = self._colunas_tabela(tabela)
        filtrados = {}
        for k, v in campos.items():
            if not isinstance(k, str):
                k = str(k)
            if k.startswith("__"):
                continue
            if k not in colunas_validas:
                continue
            if isinstance(v, (dict, list)):
                try:
                    v = json.dumps(v, ensure_ascii=False)
                except Exception:
                    v = str(v)
            filtrados[k] = v
        if len(filtrados) < len(campos):
            diff = set(campos.keys()) - set(filtrados.keys())
            if diff:
                log.debug(f"Campos ignorados para '{tabela}': {diff}")
        return filtrados

    def _ensure_column(self, table: str, column_def: str) -> None:
        """Adiciona coluna se não existir (uso interno em migrações leves)."""
        try:
            col_name = column_def.split()[0].strip()
            cur = self.conn.cursor()
            cur.execute(f"PRAGMA table_info({table})")
            existing = {row[1] for row in cur.fetchall()}
            if col_name not in existing:
                self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {column_def}")
                self.conn.commit()
        except Exception:
            pass

    # ------------------------- Operações de escrita -------------------------
    def inserir_documento(self, **campos) -> int:
        """Insere na tabela documentos. Retorna ID."""
        if not campos:
            raise ValueError("Nenhum campo fornecido para inserir_documento.")
        campos = self._filtrar_campos_validos("documentos", campos)
        if not campos:
            raise ValueError("Nenhum campo válido para inserir em 'documentos'.")
        columns = ", ".join(campos.keys())
        placeholders = ", ".join("?" for _ in campos)
        sql = f"INSERT INTO documentos ({columns}) VALUES ({placeholders})"
        cur = self.conn.cursor
        cur = self.conn.cursor()
        cur.execute(sql, list(campos.values()))
        self.conn.commit()
        return int(cur.lastrowid)

    def atualizar_documento_campo(self, documento_id: int, campo: str, valor: Any) -> None:
        sql = f"UPDATE documentos SET {campo} = ? WHERE id = ?"
        self.conn.execute(sql, (valor, documento_id))
        self.conn.commit()

    def atualizar_documento_campos(self, documento_id: int, **campos) -> None:
        """Atualiza apenas colunas existentes na tabela 'documentos'."""
        if not campos:
            return
        campos = self._filtrar_campos_validos("documentos", campos)
        if not campos:
            log.warning(f"[DB] Nenhum campo válido para atualizar (id={documento_id}).")
            return
        sets = ", ".join(f"{k} = ?" for k in campos.keys())
        vals = list(campos.values()) + [documento_id]
        sql = f"UPDATE documentos SET {sets} WHERE id = ?"
        self.conn.execute(sql, vals)
        self.conn.commit()

    def inserir_item(self, **campos) -> int:
        if not campos:
            raise ValueError("Nenhum campo fornecido para inserir_item.")
        campos = self._filtrar_campos_validos("itens", campos)
        if not campos:
            raise ValueError("Nenhum campo válido para inserir em 'itens'.")
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
        campos = self._filtrar_campos_validos("impostos", campos)
        if not campos:
            raise ValueError("Nenhum campo válido para inserir em 'impostos'.")
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
        campos = self._filtrar_campos_validos("extracoes", campos)
        if not campos:
            raise ValueError("Nenhum campo válido para inserir em 'extracoes'.")
        columns = ", ".join(campos.keys())
        placeholders = ", ".join("?" for _ in campos)
        sql = f"INSERT INTO extracoes ({columns}) VALUES ({placeholders})"
        cur = self.conn.cursor()
        cur.execute(sql, list(campos.values()))
        self.conn.commit()
        return int(cur.lastrowid)

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

    # ------------------------- Operações de detalhes (key-value) -------------------------
    def upsert_detalhe(self, documento_id: int, chave: str, valor: Any, origem: Optional[str] = None) -> int:
        """
        Insere/atualiza um par (documento_id, chave) em documentos_detalhes.
        valor é salvo como TEXT (use json.dumps se precisar preservar estrutura).
        Retorna id do registro.
        """
        if not chave:
            raise ValueError("chave do detalhe não pode ser vazia.")
        val = json.dumps(valor, ensure_ascii=False) if isinstance(valor, (dict, list)) else valor
        cur = self.conn.cursor()
        cur.execute("""
            UPDATE documentos_detalhes
               SET valor = ?, origem = COALESCE(?, origem)
             WHERE documento_id = ? AND chave = ?
        """, (val, origem, documento_id, chave))
        if cur.rowcount == 0:
            cur.execute("""
                INSERT INTO documentos_detalhes (documento_id, chave, valor, origem, criado_em)
                VALUES (?, ?, ?, ?, ?)
            """, (documento_id, chave, val, origem, self.now()))
        self.conn.commit()
        cur.execute("SELECT id FROM documentos_detalhes WHERE documento_id = ? AND chave = ?", (documento_id, chave))
        row = cur.fetchone()
        return int(row["id"]) if row else 0

    def upsert_detalhes_bulk(self, documento_id: int, pares: Dict[str, Any], origem: Optional[str] = None) -> None:
        """Upsert em lote para (documento_id, chave->valor)."""
        if not pares:
            return
        cur = self.conn.cursor()
        for chave, valor in pares.items():
            if not chave:
                continue
            val = json.dumps(valor, ensure_ascii=False) if isinstance(valor, (dict, list)) else valor
            cur.execute("""
                UPDATE documentos_detalhes
                   SET valor = ?, origem = COALESCE(?, origem)
                 WHERE documento_id = ? AND chave = ?
            """, (val, origem, documento_id, chave))
            if cur.rowcount == 0:
                cur.execute("""
                    INSERT INTO documentos_detalhes (documento_id, chave, valor, origem, criado_em)
                    VALUES (?, ?, ?, ?, ?)
                """, (documento_id, chave, val, origem, self.now()))
        self.conn.commit()

    def inserir_detalhe(self, documento_id: int, chave: str, valor: Any, origem: Optional[str] = None) -> int:
        """Insere detalhe (falha se já existir a mesma chave). Prefira upsert_detalhe."""
        if not chave:
            raise ValueError("chave do detalhe não pode ser vazia.")
        val = json.dumps(valor, ensure_ascii=False) if isinstance(valor, (dict, list)) else valor
        cur = self.conn.cursor()
        cur.execute("""
            INSERT OR FAIL INTO documentos_detalhes (documento_id, chave, valor, origem, criado_em)
            VALUES (?, ?, ?, ?, ?)
        """, (documento_id, chave, val, origem, self.now()))
        self.conn.commit()
        return int(cur.lastrowid)

    def listar_detalhes(self, documento_id: int) -> Dict[str, str]:
        """Retorna todos os detalhes (key->value TEXT) de um documento."""
        cur = self.conn.cursor()
        cur.execute("SELECT chave, valor FROM documentos_detalhes WHERE documento_id = ?", (documento_id,))
        out: Dict[str, str] = {}
        for row in cur.fetchall():
            out[str(row["chave"])] = row["valor"]
        return out

    def buscar_detalhe(self, documento_id: int, chave: str) -> Optional[str]:
        """Retorna o valor TEXT de uma chave específica de um documento (ou None)."""
        cur = self.conn.cursor()
        cur.execute("SELECT valor FROM documentos_detalhes WHERE documento_id = ? AND chave = ?", (documento_id, chave))
        row = cur.fetchone()
        return row["valor"] if row else None

    def detalhes_dataframe(self, documento_id: Optional[int] = None, chave_prefix: Optional[str] = None) -> pd.DataFrame:
        """Retorna DataFrame de documentos_detalhes, filtrando por documento e/ou prefixo da chave."""
        where: List[str] = []
        params: List[Any] = []
        if documento_id is not None:
            where.append("documento_id = ?")
            params.append(documento_id)
        if chave_prefix:
            where.append("chave LIKE ?")
            params.append(f"{chave_prefix}%")
        sql = "SELECT * FROM documentos_detalhes"
        if where:
            sql += " WHERE " + " AND ".join(where)
        return pd.read_sql_query(sql, self.conn, params=params)

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

    def query_table(self, table: str, where: Optional[str] = None,
                    order_by: Optional[str] = None, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Consulta tabelas conhecidas. `where` deve ser uma expressão SQL segura (montada internamente no projeto).
        Parâmetros adicionais `order_by` e `limit` ajudam em telas e análises.
        """
        allowed_tables = {
            "documentos", "itens", "impostos", "extracoes",
            "logs", "memoria", "revisoes", "usuarios", "metricas", "documentos_detalhes", "config",
            "rag_chunks"
        }
        if table not in allowed_tables:
            raise ValueError(f"Tabela não suportada: {table}")
        sql = f"SELECT * FROM {table}"
        if where:
            sql += f" WHERE {where}"
        if order_by:
            sql += f" ORDER BY {order_by}"
        if limit is not None and isinstance(limit, int) and limit > 0:
            sql += f" LIMIT {limit}"
        return pd.read_sql_query(sql, self.conn)

    # ------------------------- Config (key/value) -------------------------
    def set_config(self, chave: str, valor: Any, usuario: Optional[str] = None) -> None:
        """Upsert em config; valor é convertido para JSON quando possível."""
        if not chave:
            return
        if isinstance(valor, (dict, list)):
            try:
                valor = json.dumps(valor, ensure_ascii=False)
            except Exception:
                valor = str(valor)
        cur = self.conn.cursor()
        cur.execute(
            """
            UPDATE config SET valor = ? WHERE chave = ? AND IFNULL(usuario,'') = IFNULL(?, '')
            """,
            (valor, chave, usuario),
        )
        if cur.rowcount == 0:
            cur.execute(
                """
                INSERT INTO config (chave, valor, usuario) VALUES (?, ?, ?)
                """,
                (chave, valor, usuario),
            )
        self.conn.commit()

    def get_config(self, chave: str, usuario: Optional[str] = None, default: Any = None) -> Any:
        cur = self.conn.cursor()
        cur.execute(
            "SELECT valor FROM config WHERE chave = ? AND IFNULL(usuario,'') = IFNULL(?, '')",
            (chave, usuario),
        )
        row = cur.fetchone()
        if not row:
            return default
        val = row[0]
        # tenta desserializar JSON
        try:
            return json.loads(val)
        except Exception:
            return val

    # ------------------------- Fechamento -------------------------
    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass

    # ------------------------- Manutenção do banco -------------------------
    def analyze(self) -> None:
        """Executa ANALYZE para atualizar estatísticas do SQLite."""
        try:
            self.conn.execute("ANALYZE")
            self.conn.commit()
        except Exception:
            pass

    def vacuum(self) -> None:
        """Executa VACUUM para compactação de arquivo e limpeza."""
        try:
            self.conn.execute("VACUUM")
        except Exception:
            pass

    def db_file_path(self) -> Path:
        """Retorna o caminho do arquivo físico do banco SQLite."""
        return self.db_path
