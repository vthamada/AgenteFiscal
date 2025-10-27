# banco_de_dados.py

from __future__ import annotations
from typing import Optional, Dict, Any, Iterable, List, Tuple
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
    Camada de persistência (SQLite) com schema completo, índices e migração automática.
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
        # Cria (se necessário) e migra o schema para a versão atual
        self._criar_schema()
        self._migrar_schema_se_preciso()

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

            -- Detalhes universais adicionais (fixos) de emitente/destinatário
            emitente_ie             TEXT,
            emitente_im             TEXT,
            emitente_uf             TEXT,
            emitente_municipio      TEXT,
            emitente_endereco       TEXT,

            destinatario_ie         TEXT,
            destinatario_im         TEXT,
            destinatario_uf         TEXT,
            destinatario_municipio  TEXT,
            destinatario_endereco   TEXT,

            -- Metadados básicos para filtros rápidos (documento)
            inscricao_estadual      TEXT,
            uf                      TEXT,
            municipio               TEXT,
            endereco                TEXT,

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

        # NOVA: DETALHES DINÂMICOS (key-value) – garante evolução sem migração de schema
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

        # ------------------- Índices úteis para performance -------------------
        cur.execute("CREATE INDEX IF NOT EXISTS ix_documentos_status ON documentos(status)")
        cur.execute("CREATE INDEX IF NOT EXISTS ix_documentos_tipo ON documentos(tipo)")
        cur.execute("CREATE INDEX IF NOT EXISTS ix_documentos_uf ON documentos(uf)")
        cur.execute("CREATE INDEX IF NOT EXISTS ix_documentos_data_emissao ON documentos(data_emissao)")
        cur.execute("CREATE INDEX IF NOT EXISTS ix_documentos_num_serie ON documentos(numero_nota, serie)")
        cur.execute("CREATE INDEX IF NOT EXISTS ix_documentos_chave ON documentos(chave_acesso)")
        cur.execute("CREATE INDEX IF NOT EXISTS ix_documentos_emit_cnpj ON documentos(emitente_cnpj)")
        cur.execute("CREATE INDEX IF NOT EXISTS ix_documentos_dest_cnpj ON documentos(destinatario_cnpj)")
        cur.execute("CREATE INDEX IF NOT EXISTS ix_itens_doc ON itens(documento_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS ix_impostos_item ON impostos(item_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS ix_docdet_doc ON documentos_detalhes(documento_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS ix_docdet_chave ON documentos_detalhes(chave)")

        self.conn.commit()

    # ------------------------- Migração/Auto-fix do schema -------------------------
    def _migrar_schema_se_preciso(self) -> None:
        """
        Verifica e adiciona colunas/tabelas que podem faltar em bancos antigos.
        Não remove/renomeia automaticamente (operações destrutivas).
        """
        def _colunas(tabela: str) -> set[str]:
            cur = self.conn.cursor()
            cur.execute(f"PRAGMA table_info({tabela})")
            return {row[1] for row in cur.fetchall()}

        def _tabelas() -> set[str]:
            cur = self.conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
            return {r[0] for r in cur.fetchall()}

        def _add_col_if_missing(tabela: str, coluna: str, decl: str) -> None:
            existentes = _colunas(tabela)
            if coluna not in existentes:
                self.conn.execute(f"ALTER TABLE {tabela} ADD COLUMN {decl}")
                self.conn.commit()

        tabs = _tabelas()

        # Garante documentos_detalhes em bases antigas
        if "documentos_detalhes" not in tabs:
            self.conn.execute("""
            CREATE TABLE documentos_detalhes (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                documento_id    INTEGER NOT NULL,
                chave           TEXT NOT NULL,
                valor           TEXT,
                origem          TEXT,
                criado_em       TEXT DEFAULT (datetime('now')),
                UNIQUE(documento_id, chave),
                FOREIGN KEY(documento_id) REFERENCES documentos(id) ON DELETE CASCADE
            )
            """)
            self.conn.execute("CREATE INDEX IF NOT EXISTS ix_docdet_doc ON documentos_detalhes(documento_id)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS ix_docdet_chave ON documentos_detalhes(chave)")
            self.conn.commit()

        # documentos: garantir campos universais usados por OCR/NLP/Validação + XML
        _add_col_if_missing("documentos", "endereco", "endereco TEXT")
        _add_col_if_missing("documentos", "meta_json", "meta_json TEXT")
        _add_col_if_missing("documentos", "chave_acesso", "chave_acesso TEXT")
        _add_col_if_missing("documentos", "motivo_rejeicao", "motivo_rejeicao TEXT")

        # totais impostos e produtos/serviços
        _add_col_if_missing("documentos", "total_icms", "total_icms REAL")
        _add_col_if_missing("documentos", "total_ipi", "total_ipi REAL")
        _add_col_if_missing("documentos", "total_pis", "total_pis REAL")
        _add_col_if_missing("documentos", "total_cofins", "total_cofins REAL")
        _add_col_if_missing("documentos", "total_produtos", "total_produtos REAL")
        _add_col_if_missing("documentos", "total_servicos", "total_servicos REAL")

        # novos campos fiscal/identificação
        _add_col_if_missing("documentos", "numero_nota", "numero_nota TEXT")
        _add_col_if_missing("documentos", "serie", "serie TEXT")
        _add_col_if_missing("documentos", "modelo", "modelo TEXT")

        # detalhamento emitente/destinatário
        _add_col_if_missing("documentos", "emitente_ie", "emitente_ie TEXT")
        _add_col_if_missing("documentos", "emitente_im", "emitente_im TEXT")
        _add_col_if_missing("documentos", "emitente_uf", "emitente_uf TEXT")
        _add_col_if_missing("documentos", "emitente_municipio", "emitente_municipio TEXT")
        _add_col_if_missing("documentos", "emitente_endereco", "emitente_endereco TEXT")
        _add_col_if_missing("documentos", "destinatario_ie", "destinatario_ie TEXT")
        _add_col_if_missing("documentos", "destinatario_im", "destinatario_im TEXT")
        _add_col_if_missing("documentos", "destinatario_uf", "destinatario_uf TEXT")
        _add_col_if_missing("documentos", "destinatario_municipio", "destinatario_municipio TEXT")
        _add_col_if_missing("documentos", "destinatario_endereco", "destinatario_endereco TEXT")

        # totais complementares
        _add_col_if_missing("documentos", "valor_descontos", "valor_descontos REAL")
        _add_col_if_missing("documentos", "valor_frete", "valor_frete REAL")
        _add_col_if_missing("documentos", "valor_seguro", "valor_seguro REAL")
        _add_col_if_missing("documentos", "valor_outros", "valor_outros REAL")
        _add_col_if_missing("documentos", "valor_liquido", "valor_liquido REAL")

        # transporte universais
        _add_col_if_missing("documentos", "modalidade_frete", "modalidade_frete TEXT")
        _add_col_if_missing("documentos", "placa_veiculo", "placa_veiculo TEXT")
        _add_col_if_missing("documentos", "uf_veiculo", "uf_veiculo TEXT")
        _add_col_if_missing("documentos", "peso_bruto", "peso_bruto REAL")
        _add_col_if_missing("documentos", "peso_liquido", "peso_liquido REAL")
        _add_col_if_missing("documentos", "qtd_volumes", "qtd_volumes REAL")

        # pagamento universais
        _add_col_if_missing("documentos", "forma_pagamento", "forma_pagamento TEXT")
        _add_col_if_missing("documentos", "valor_pagamento", "valor_pagamento REAL")
        _add_col_if_missing("documentos", "troco", "troco REAL")

        # metadados XML/autorização
        _add_col_if_missing("documentos", "caminho_xml", "caminho_xml TEXT")
        _add_col_if_missing("documentos", "versao_schema", "versao_schema TEXT")
        _add_col_if_missing("documentos", "ambiente", "ambiente TEXT")
        _add_col_if_missing("documentos", "protocolo_autorizacao", "protocolo_autorizacao TEXT")
        _add_col_if_missing("documentos", "data_autorizacao", "data_autorizacao TEXT")
        _add_col_if_missing("documentos", "cstat", "cstat TEXT")
        _add_col_if_missing("documentos", "xmotivo", "xmotivo TEXT")
        _add_col_if_missing("documentos", "responsavel_tecnico", "responsavel_tecnico TEXT")

        # OCR
        _add_col_if_missing("documentos", "ocr_tipo", "ocr_tipo TEXT")

        # itens: campos adicionais usados pelo XML/NLP
        _add_col_if_missing("itens", "numero_item", "numero_item INTEGER")
        _add_col_if_missing("itens", "ean", "ean TEXT")
        _add_col_if_missing("itens", "cest", "cest TEXT")
        _add_col_if_missing("itens", "codigo_produto", "codigo_produto TEXT")
        _add_col_if_missing("itens", "desconto", "desconto REAL")
        _add_col_if_missing("itens", "outras_despesas", "outras_despesas REAL")

        # impostos: reforço
        _add_col_if_missing("impostos", "origem", "origem TEXT")
        _add_col_if_missing("impostos", "base_calculo", "base_calculo REAL")
        _add_col_if_missing("impostos", "aliquota", "aliquota REAL")
        _add_col_if_missing("impostos", "cst", "cst TEXT")

        # extracoes: garantir colunas
        _add_col_if_missing("extracoes", "agente", "agente TEXT")
        _add_col_if_missing("extracoes", "confianca_media", "confianca_media REAL")
        _add_col_if_missing("extracoes", "texto_extraido", "texto_extraido TEXT")
        _add_col_if_missing("extracoes", "linguagem", "linguagem TEXT")
        _add_col_if_missing("extracoes", "tempo_processamento", "tempo_processamento REAL")
        _add_col_if_missing("extracoes", "criado_em", "criado_em TEXT")

        # metricas: meta_json é crítico
        _add_col_if_missing("metricas", "meta_json", "meta_json TEXT")

        # logs/memoria: garantem created_at se necessário
        _add_col_if_missing("logs", "criado_em", "criado_em TEXT")
        _add_col_if_missing("memoria", "criado_em", "criado_em TEXT")

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
        if chave is None or chave == "":
            raise ValueError("chave do detalhe não pode ser vazia.")
        val = valor
        if isinstance(val, (dict, list)):
            try:
                val = json.dumps(val, ensure_ascii=False)
            except Exception:
                val = str(val)
        cur = self.conn.cursor()
        # Tenta update; se não afetar linhas, faz insert
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
        # Retorna id
        cur.execute("SELECT id FROM documentos_detalhes WHERE documento_id = ? AND chave = ?", (documento_id, chave))
        row = cur.fetchone()
        return int(row["id"]) if row else 0

    def upsert_detalhes_bulk(self, documento_id: int, pares: Dict[str, Any], origem: Optional[str] = None) -> None:
        """
        Upsert em lote para (documento_id, chave->valor).
        """
        if not pares:
            return
        cur = self.conn.cursor()
        for chave, valor in pares.items():
            if not isinstance(chave, str) or not chave:
                continue
            val = valor
            if isinstance(val, (dict, list)):
                try:
                    val = json.dumps(val, ensure_ascii=False)
                except Exception:
                    val = str(val)
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
        """
        Insere detalhe (falha se já existir a mesma chave). Prefira upsert_detalhe.
        """
        if not chave:
            raise ValueError("chave do detalhe não pode ser vazia.")
        val = valor
        if isinstance(val, (dict, list)):
            try:
                val = json.dumps(val, ensure_ascii=False)
            except Exception:
                val = str(val)
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
        """
        Retorna DataFrame de documentos_detalhes, filtrando por documento e/ou prefixo da chave.
        """
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
            "logs", "memoria", "revisoes", "usuarios", "metricas", "documentos_detalhes"
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

    # ------------------------- Fechamento -------------------------
    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass
