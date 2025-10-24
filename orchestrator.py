# agentes/orchestrator.py

from __future__ import annotations

import json
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd

log = logging.getLogger(__name__)

try:
    from seguranca import Cofre, carregar_chave_do_env, CRYPTO_OK
    from validacao import ValidadorFiscal
    from agentes import (  
        AgenteLLMMapper,
        AgenteXMLParser,
        AgenteOCR,
        AgenteNLP,
        AgenteAnalitico,
        AgenteNormalizadorCampos,
        AgenteAssociadorXML,
        AgenteConfiancaRouter,
        MetricsAgent,
        CORE_MODULES_AVAILABLE,
    )
    from memoria import MemoriaSessao
    from banco_de_dados import BancoDeDados
    from langchain_core.language_models.chat_models import BaseChatModel
except Exception:  
    from seguranca import Cofre, carregar_chave_do_env, CRYPTO_OK  
    from validacao import ValidadorFiscal  
    from agentes import (  
        AgenteLLMMapper,
        AgenteXMLParser,
        AgenteOCR,
        AgenteNLP,
        AgenteAnalitico,
        AgenteNormalizadorCampos,
        AgenteAssociadorXML,
        AgenteConfiancaRouter,
        MetricsAgent,
        CORE_MODULES_AVAILABLE,
    )
    from memoria import MemoriaSessao 
    from banco_de_dados import BancoDeDados  
    try:
        from langchain_core.language_models.chat_models import BaseChatModel 
    except Exception:
        BaseChatModel = object  


class Orchestrator:
    """Coordena o pipeline de processamento e a camada analítica."""

    db: "BancoDeDados"
    validador: "ValidadorFiscal"
    memoria: "MemoriaSessao"
    llm: Optional["BaseChatModel"] = None
    cofre: Optional["Cofre"] = None
    metrics_agent: Optional["MetricsAgent"] = None

    def __init__(
        self,
        db: "BancoDeDados",
        validador: Optional["ValidadorFiscal"] = None,
        memoria: Optional["MemoriaSessao"] = None,
        llm: Optional["BaseChatModel"] = None,
        cofre: Optional["Cofre"] = None,
    ):
        self.db = db
        self.validador = validador
        self.memoria = memoria
        self.llm = llm
        self.cofre = cofre
        self.metrics_agent = None
        self.__post_init__()

    # --- Lista branca base (fallback) de colunas válidas da tabela 'documentos'
    _DOC_COLS_BASE = {
        "id",
        "status",
        "motivo_rejeicao",
        "chave_acesso",
        "data_emissao",
        "data_saida",
        "hora_emissao",
        "hora_saida",
        "emitente_cnpj",
        "emitente_cpf",
        "emitente_nome",
        "emitente_ie",
        "emitente_im",
        "emitente_endereco",
        "emitente_municipio",
        "emitente_uf",
        "destinatario_cnpj",
        "destinatario_cpf",
        "destinatario_nome",
        "destinatario_ie",
        "destinatario_im",
        "destinatario_endereco",
        "destinatario_municipio",
        "destinatario_uf",
        "inscricao_estadual",
        "uf",
        "municipio",
        "endereco",
        "valor_total",
        "valor_produtos",
        "valor_servicos",
        "valor_icms",
        "valor_ipi",
        "valor_pis",
        "valor_cofins",
        "valor_iss",
        "desconto_total",
        "outras_despesas",
        "frete",
        "total_icms",
        "total_ipi",
        "total_pis",
        "total_cofins",
        "caminho_arquivo",
        "nome_arquivo",
        "origem",
        "tipo",
        "serie",
        "numero_nota",
        "modelo",
        "ambiente",
        "cfop",
        "ncm",
        "cst",
        "natureza_operacao",
        "forma_pagamento",
        "cnpj_autorizado",
        "observacoes",
    }

    # ----------------------- Utils internos -----------------------
    def _campos_permitidos_documentos(self) -> set:
        """Obtém dinamicamente as colunas reais da tabela 'documentos' via PRAGMA, com fallback."""
        try:
            cur = self.db.conn.execute("PRAGMA table_info(documentos)")
            cols = {row[1] for row in cur.fetchall()}
            if cols:
                return cols
        except Exception:
            pass
        return set(self._DOC_COLS_BASE)

    @staticmethod
    def _filtrar_campos_validos(d: Dict[str, Any], permitidas: set) -> Dict[str, Any]:
        """Remove chaves internas (__meta__ etc.) e ignora campos fora do schema."""
        if not d:
            return {}
        limpo: Dict[str, Any] = {}
        for k, v in d.items():
            if not k:
                continue
            kstr = str(k)
            if kstr.startswith("__"):
                continue
            if kstr in permitidas:
                limpo[kstr] = v
        return limpo

    # ----------------------- Init -----------------------
    def __post_init__(self):
        """Inicializa agentes, Cofre e Métricas."""
        if not CORE_MODULES_AVAILABLE:
            log.error("Orchestrator: Módulos CORE ausentes.")
            if self.cofre is None:
                self.cofre = Cofre(key=None)
            if getattr(self, "validador", None) is None:
                self.validador = ValidadorFiscal(cofre=self.cofre)
        else:
            if self.cofre is None:
                chave_criptografia = carregar_chave_do_env("APP_SECRET_KEY")
                self.cofre = Cofre(key=chave_criptografia)
            if getattr(self.cofre, "available", False):
                log.info("Criptografia ATIVA.")
            else:
                log.warning("Criptografia INATIVA.")
                if not CRYPTO_OK:
                    log.warning("-> Lib 'cryptography' ausente.")

            if getattr(self, "validador", None) is None:
                self.validador = ValidadorFiscal(cofre=self.cofre)

        if self.metrics_agent is None:
            self.metrics_agent = MetricsAgent()

        # Agentes principais
        self.xml_agent = AgenteXMLParser(self.db, self.validador, self.cofre, self.metrics_agent)
        self.ocr_agent = AgenteOCR()
        self.nlp_agent = AgenteNLP()
        self.analitico = AgenteAnalitico(self.llm, self.memoria) if self.llm else None
        self.normalizador = AgenteNormalizadorCampos()
        self.associador = AgenteAssociadorXML(self.db, self.cofre)
        self.router = AgenteConfiancaRouter()
        self.llm_mapper = AgenteLLMMapper(self.llm) if self.llm else AgenteLLMMapper(None)

        if self.llm:
            log.info("Agente Analítico INICIALIZADO.")
        else:
            log.warning("Agente Analítico NÃO inicializado (LLM ausente).")

    # ----------------------- Ingestão -----------------------
    def ingestir_arquivo(self, nome: str, conteudo: bytes, origem: str = "web") -> int:
        """Processa um arquivo, retornando o ID do documento."""
        t_start = time.time()
        doc_id = -1
        status = "erro"
        motivo = "?"
        doc_hash = self.db.hash_bytes(conteudo)
        ext = Path(nome).suffix.lower()
        tipo_doc = ext.strip(".") or "binario"
        try:
            existing_id = self.db.find_documento_by_hash(doc_hash)
            if existing_id:
                log.info("Doc '%s' (hash %s...) já existe ID %d. Ignorando.", nome, doc_hash[:8], existing_id)
                return existing_id

            if ext == ".xml":
                tipo_doc = "xml"  # Será refinado pelo parser
                doc_id = self.xml_agent.processar(nome, conteudo, origem)
            elif ext in {".pdf", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}:
                doc_id = self._processar_midias(nome, conteudo, origem)
                tipo_doc = Path(nome).suffix.lower().strip(".")
            else:
                motivo = f"Extensão '{ext}' não suportada."
                status = "quarentena"
                log.warning("Arquivo '%s' rejeitado: %s", nome, motivo)
                tipo_doc = "desconhecido"
                doc_id = self.db.inserir_documento(
                    nome_arquivo=nome,
                    tipo=tipo_doc,
                    origem=origem,
                    hash=doc_hash,
                    status=status,
                    data_upload=self.db.now(),
                    motivo_rejeicao=motivo,
                )
                # Métrica para arquivo não suportado
                self.metrics_agent.registrar_metrica(
                    db=self.db,
                    tipo_documento=tipo_doc,
                    status=status,
                    confianca_media=0.0,
                    tempo_medio=(time.time() - t_start),
                )

            if doc_id > 0:
                doc_info = self.db.get_documento(doc_id)
                if doc_info:
                    status = doc_info.get("status", status)

        except Exception as e:
            log.exception("Falha ingestão '%s': %s", nome, e)
            motivo = f"Erro: {str(e)}"
            status = "erro"
            try:
                existing_id_on_error = self.db.find_documento_by_hash(doc_hash)
                if existing_id_on_error:
                    doc_id = existing_id_on_error
                    self.db.atualizar_documento_campos(doc_id, status="erro", motivo_rejeicao=motivo)
                elif doc_id > 0:
                    self.db.atualizar_documento_campos(doc_id, status="erro", motivo_rejeicao=motivo)
                else:
                    doc_id = self.db.inserir_documento(
                        nome_arquivo=nome,
                        tipo=tipo_doc,
                        origem=origem,
                        hash=doc_hash,
                        status="erro",
                        data_upload=self.db.now(),
                        motivo_rejeicao=motivo,
                    )
                self.metrics_agent.registrar_metrica(
                    db=self.db,
                    tipo_documento=tipo_doc,
                    status="erro",
                    confianca_media=0.0,
                    tempo_medio=(time.time() - t_start),
                )
            except Exception as db_err:
                log.error("Erro CRÍTICO ao registrar falha '%s': %s", nome, db_err)
                return -1
        finally:
            log.info("Ingestão '%s' (ID: %d, Status: %s) em %.2fs", nome, doc_id, status, time.time() - t_start)
        return doc_id

    def _processar_midias(self, nome: str, conteudo: bytes, origem: str) -> int:
        """Processa PDF/Imagem via OCR/NLP + normalização, associação, roteamento e métricas."""
        doc_id = -1
        t_start_proc = time.time()
        status_final = "erro"
        conf = 0.0
        tipo_doc = Path(nome).suffix.lower().strip(".")
        fonte_final = "desconhecida"
        try:
            doc_id = self.db.inserir_documento(
                nome_arquivo=nome,
                tipo=tipo_doc,
                origem=origem,
                hash=self.db.hash_bytes(conteudo),
                status="processando",
                data_upload=self.db.now(),
                caminho_arquivo=str(self.db.save_upload(nome, conteudo)),
            )
            log.info("Processando mídia '%s' (doc_id %d)", nome, doc_id)

            # ---------- OCR ----------
            texto = ""
            t_start_ocr = time.time()
            try:
                texto, conf = self.ocr_agent.reconhecer(nome, conteudo)
                ocr_time = time.time() - t_start_ocr
                log.info("OCR doc_id %d: conf=%.2f, time=%.2fs.", doc_id, conf, ocr_time)
                self.db.inserir_extracao(
                    documento_id=doc_id,
                    agente="OCRAgent",
                    confianca_media=float(conf),
                    texto_extraido=texto[:50000] + ("..." if len(texto) > 50000 else ""),
                    linguagem="pt",
                    tempo_processamento=round(ocr_time, 3),
                )
            except Exception as e_ocr:
                log.error("Falha OCR doc_id %d: %s", doc_id, e_ocr)
                status_final = "erro"
                self.db.atualizar_documento_campos(doc_id, status=status_final, motivo_rejeicao=f"Falha OCR: {e_ocr}")
                self.db.log("ocr_erro", "sistema", f"doc_id={doc_id}|erro={e_ocr}")
                raise

            # ---------- NLP (heurística) ----------
            campos_nlp: Dict[str, Any] = {}
            itens_ocr = []
            impostos_ocr = []
            if texto:
                try:
                    t_start_nlp = time.time()
                    log.info("NLP doc_id %d...", doc_id)
                    campos_nlp = self.nlp_agent.extrair_campos(texto)
                    nlp_time = time.time() - t_start_nlp
                    log.info("NLP doc_id %d em %.2fs.", doc_id, nlp_time)
                    itens_ocr = campos_nlp.pop("itens_ocr", []) or []
                    impostos_ocr = campos_nlp.pop("impostos_ocr", []) or []
                except Exception as e_nlp:
                    log.warning("NLP falhou doc_id %d: %s", doc_id, e_nlp)
                    campos_nlp = {}

            # ---------- LLM Mapper (se necessário) ----------
            precisa_llm = (
                conf < 0.75
                or not campos_nlp
                or not campos_nlp.get("emitente_cnpj")
                or not campos_nlp.get("valor_total")
                or not campos_nlp.get("data_emissao")
            )

            campos_llm: Dict[str, Any] = {}
            if precisa_llm:
                try:
                    campos_llm = self.llm_mapper.mapear(texto, nome_arquivo=nome) or {}
                    if campos_llm:
                        meta_llm = campos_llm.pop("__meta__", {})
                        conf_llm = float(sum(meta_llm.values()) / len(meta_llm)) if meta_llm else None
                        try:
                            self.db.inserir_extracao(
                                documento_id=doc_id,
                                agente="LLMMapper",
                                confianca_media=conf_llm,
                                texto_extraido=json.dumps(meta_llm, ensure_ascii=False) if meta_llm else None,
                                linguagem="pt",
                                tempo_processamento=0.0,
                            )
                            log.info("LLMMapper executado doc_id=%d conf_media=%s", doc_id, conf_llm)
                        except Exception as e_ext:
                            log.warning("Falha ao registrar extração LLMMapper doc_id=%d: %s", doc_id, e_ext)
                except Exception as e_map:
                    log.warning("LLMMapper falhou doc_id %d: %s", doc_id, e_map)

            # ---------- Fusão & Normalização ----------
            campos_fusao = self.normalizador.fundir(campos_nlp, campos_llm)
            campos_norm = self.normalizador.normalizar(campos_fusao)

            # ---------- Associação a XML existente ----------
            campos_associados = self.associador.tentar_associar_pdf(doc_id, campos_norm, texto_ocr=texto)
            xml_encontrado = bool(
                campos_associados.get("tipo") and str(campos_associados.get("tipo")).lower().startswith("xml")
            )

            # ---------- Roteamento por confiança ----------
            rota = self.router.decidir(conf_ocr=conf, campos=campos_associados, xml_encontrado=xml_encontrado)
            status_final = rota.get("status", "revisao_pendente")
            fonte_final = rota.get("fonte", "ocr/nlp")

            # --- FILTRA CAMPOS VÁLIDOS (evita '__meta__') ---
            permitidas = self._campos_permitidos_documentos()
            campos_safe = self._filtrar_campos_validos(
                {k: v for k, v in campos_associados.items() if k not in ("itens_ocr", "impostos_ocr")}, permitidas
            )

            # Persistência
            if campos_safe:
                self.db.atualizar_documento_campos(doc_id, **campos_safe)

            # Itens/Impostos OCR (se houver)
            if itens_ocr:
                log.info("Salvando %d itens OCR doc_id %d.", len(itens_ocr), doc_id)
                item_id_map: Dict[int, int] = {}
                for idx, item_data in enumerate(itens_ocr):
                    try:
                        item_id = self.db.inserir_item(documento_id=doc_id, **item_data)
                        item_id_map[idx] = item_id
                    except Exception as e_item:
                        log.warning("Falha ao inserir item OCR idx=%d doc_id=%d: %s", idx, doc_id, e_item)
                if impostos_ocr:
                    log.info("Salvando %d impostos OCR doc_id %d.", len(impostos_ocr), doc_id)
                    for imposto_data in impostos_ocr:
                        try:
                            item_ocr_idx = imposto_data.pop("item_idx", -1)
                            if item_ocr_idx in item_id_map:
                                self.db.inserir_imposto(item_id=item_id_map[item_ocr_idx], **imposto_data)
                            else:
                                log.warning("Imposto OCR s/ item idx=%s, doc_id %d.", item_ocr_idx, doc_id)
                        except Exception as e_imp:
                            log.warning("Falha ao inserir imposto OCR doc_id=%d: %s", doc_id, e_imp)

            # Validação fiscal
            try:
                self.validador.validar_documento(doc_id=doc_id, db=self.db)
            except Exception as e_val:
                log.warning("Validação fiscal falhou doc_id=%d: %s", doc_id, e_val)

            # Atualiza status final
            self.db.atualizar_documento_campo(doc_id, "status", status_final)
            self.db.log(
                "ingestao_midias",
                "sistema",
                f"doc_id={doc_id}|conf={conf:.2f}|status={status_final}|fonte={fonte_final}",
            )

        except Exception as e_outer:
            log.exception("Falha geral mídia '%s': %s", nome, e_outer)
            status_final = "erro"
            if doc_id > 0:
                try:
                    self.db.atualizar_documento_campos(
                        doc_id, status="erro", motivo_rejeicao=f"Falha geral: {e_outer}"
                    )
                except Exception as db_err_f:
                    log.error("Erro CRÍTICO ao marcar erro final doc_id %d: %s", doc_id, db_err_f)
        finally:
            # Métricas
            processing_time = time.time() - t_start_proc
            if doc_id > 0:
                self.metrics_agent.registrar_metrica(
                    db=self.db,
                    tipo_documento=tipo_doc,
                    status=status_final,
                    confianca_media=conf,
                    tempo_medio=processing_time,
                )
        return doc_id

    # ----------------------- Q&A Analítico -----------------------
    def _executar_fast_query(self, pergunta: str, catalog: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Consultas determinísticas simples usando Pandas, sem LLM."""
        log.info("Modo Seguro: Tentando responder com FastQuery: '%s'", pergunta)
        pergunta_lower = pergunta.lower()
        df_docs = catalog.get("documentos")

        texto_resposta = "Não foi possível responder a esta pergunta com o 'Modo Seguro' (FastQueryAgent)."
        tabela_resposta = None
        try:
            if "contar" in pergunta_lower and "documentos" in pergunta_lower:
                total_docs = len(df_docs) if df_docs is not None else 0
                texto_resposta = (
                    f"Total de documentos (processados/revisados) no escopo atual: **{total_docs}**."
                )
            elif ("valor total" in pergunta_lower or "soma" in pergunta_lower) and "documentos" in pergunta_lower:
                if df_docs is not None and "valor_total" in df_docs.columns:
                    df_docs_local = df_docs.copy()
                    df_docs_local["valor_total_num"] = pd.to_numeric(
                        df_docs_local["valor_total"], errors="coerce"
                    ).fillna(0)
                    soma_total = df_docs_local["valor_total_num"].sum()
                    texto_resposta = (
                        f"O valor total somado dos documentos no escopo é: **R$ {soma_total:,.2f}**."
                    )
                else:
                    texto_resposta = "A coluna 'valor_total' não está disponível nos documentos para soma."
            elif ("top 5" in pergunta_lower or "top 10" in pergunta_lower) and "valor" in pergunta_lower:
                n_top = 10 if "top 10" in pergunta_lower else 5
                if (
                    df_docs is not None
                    and "valor_total" in df_docs.columns
                    and "emitente_nome" in df_docs.columns
                ):
                    df_docs_local = df_docs.copy()
                    df_docs_local["valor_total_num"] = pd.to_numeric(
                        df_docs_local["valor_total"], errors="coerce"
                    ).fillna(0)
                    top_fornecedores = (
                        df_docs_local.groupby("emitente_nome")["valor_total_num"]
                        .sum()
                        .nlargest(n_top)
                        .reset_index()
                        .rename(columns={"valor_total_num": "Valor Total"})
                    )
                    texto_resposta = f"Top {n_top} Emitentes por Valor Total:"
                    tabela_resposta = top_fornecedores
                else:
                    texto_resposta = (
                        f"Não foi possível calcular o Top {n_top} (colunas 'emitente_nome' ou 'valor_total' ausentes)."
                    )
        except Exception as e:
            log.error("Erro no FastQuery: %s", e)
            texto_resposta = f"Ocorreu um erro ao tentar executar a consulta rápida: {e}"

        return {
            "texto": texto_resposta,
            "tabela": tabela_resposta,
            "figuras": [],
            "duracao_s": 0.01,
            "code": f"# FastQuery (Determinístico)\n# Pergunta: {pergunta}",
            "agent_name": "FastQueryAgent (Modo Seguro)",
        }

    def responder_pergunta(
        self, pergunta: str, scope_filters: Optional[Dict[str, Any]] = None, safe_mode: bool = False
    ) -> Dict[str, Any]:
        """Delega a pergunta analítica para o AgenteAnalitico ou FastQueryAgent (com filtros)."""
        if not self.analitico and not safe_mode:
            log.error("Agente Analítico não inicializado.")
            return {"texto": "Erro: Agente analítico não configurado.", "tabela": None, "figuras": []}

        catalog: Dict[str, pd.DataFrame] = {}
        try:
            where_conditions = ["(status = 'processado' OR status = 'revisado')"]
            if scope_filters:
                uf_escopo = scope_filters.get("uf")
                if uf_escopo and isinstance(uf_escopo, str):
                    where_conditions.append(f"uf = '{uf_escopo.upper()}'")
                tipo_escopo = scope_filters.get("tipo")
                if tipo_escopo and isinstance(tipo_escopo, list) and len(tipo_escopo) > 0:
                    tipos_sql = ", ".join([f"'{t}'" for t in tipo_escopo])
                    where_conditions.append(f"tipo IN ({tipos_sql})")
            where_clause = " AND ".join(where_conditions)
            log.info("Carregando catálogo para LLM com filtro: %s", where_clause)

            catalog["documentos"] = self.db.query_table("documentos", where=where_clause)
            if not catalog["documentos"].empty:
                doc_ids = tuple(catalog["documentos"]["id"].unique().tolist())
                doc_ids_sql = ", ".join(map(str, doc_ids))
                catalog["itens"] = self.db.query_table("itens", where=f"documento_id IN ({doc_ids_sql})")
                if not catalog["itens"].empty:
                    item_ids = tuple(catalog["itens"]["id"].unique().tolist())
                    item_ids_sql = ", ".join(map(str, item_ids))
                    catalog["impostos"] = self.db.query_table(
                        "impostos", where=f"item_id IN ({item_ids_sql})"
                    )
                else:
                    catalog["impostos"] = pd.DataFrame(
                        columns=[
                            "id",
                            "item_id",
                            "tipo_imposto",
                            "cst",
                            "origem",
                            "base_calculo",
                            "aliquota",
                            "valor",
                        ]
                    )
            else:
                catalog["itens"] = pd.DataFrame(
                    columns=[
                        "id",
                        "documento_id",
                        "descricao",
                        "ncm",
                        "cest",
                        "cfop",
                        "quantidade",
                        "unidade",
                        "valor_unitario",
                        "valor_total",
                        "codigo_produto",
                    ]
                )
                catalog["impostos"] = pd.DataFrame(
                    columns=[
                        "id",
                        "item_id",
                        "tipo_imposto",
                        "cst",
                        "origem",
                        "base_calculo",
                        "aliquota",
                        "valor",
                    ]
                )
        except Exception as e:
            log.exception("Falha ao montar catálogo com filtros: %s", e)
            return {"texto": f"Erro ao carregar dados com filtros: {e}", "tabela": None, "figuras": []}

        if catalog["documentos"].empty:
            log.info("Nenhum documento válido para análise (considerando filtros).")
            return {
                "texto": "Não há documentos válidos (status 'processado' ou 'revisado') que correspondam aos filtros selecionados para análise.",
                "tabela": None,
                "figuras": [],
            }

        if safe_mode:
            return self._executar_fast_query(pergunta, catalog)

        if not self.analitico:
            log.error("Modo Seguro desativado, mas Agente Analítico (LLM) não está configurado.")
            return {
                "texto": "Erro: Modo Seguro desativado, mas o Agente Analítico (LLM) não está configurado.",
                "tabela": None,
                "figuras": [],
            }

        log.info("Iniciando AgenteAnalitico (Filtros: %s): '%.100s'", scope_filters, pergunta)
        return self.analitico.responder(pergunta, catalog)

    # ----------------------- Revalidação & Reprocessamento -----------------------
    def revalidar_documento(self, documento_id: int) -> Dict[str, Any]:
        """Aciona a revalidação de um documento específico."""
        try:
            doc = self.db.get_documento(documento_id)
            if not doc:
                log.warning("Revalidar: Doc ID %d não encontrado.", documento_id)
                return {"ok": False, "mensagem": f"Documento ID {documento_id} não encontrado."}
            status_anterior = doc.get("status")
            log.info("Iniciando revalidação doc_id %d (status: %s)", documento_id, status_anterior)

            self.validador.validar_documento(doc_id=documento_id, db=self.db, force_revalidation=True)

            doc_depois = self.db.get_documento(documento_id)
            novo_status = doc_depois.get("status") if doc_depois else "desconhecido"
            self.db.log(
                "revalidacao",
                "usuario_sistema",
                f"doc_id={documento_id}|status_anterior={status_anterior}|status_novo={novo_status}|timestamp={self.db.now()}",
            )
            log.info("Revalidação doc_id %d concluída. Novo status: %s", documento_id, novo_status)
            return {"ok": True, "mensagem": f"Documento revalidado. Novo status: {novo_status}."}
        except Exception as e:
            log.exception("Falha ao revalidar doc_id %d: %s", documento_id, e)
            return {"ok": False, "mensagem": f"Falha ao revalidar: {e}"}

    def reprocessar_documento(self, documento_id: int) -> Dict[str, Any]:
        """Deleta um documento e seus dados associados, e tenta re-ingerir o arquivo original."""
        log.info("Iniciando reprocessamento para doc_id %d...", documento_id)
        try:
            doc_original = self.db.get_documento(documento_id)
            if not doc_original:
                return {"ok": False, "mensagem": f"Documento ID {documento_id} não encontrado."}

            caminho_arquivo_str = doc_original.get("caminho_arquivo")
            if not caminho_arquivo_str:
                return {
                    "ok": False,
                    "mensagem": f"Documento ID {documento_id} não possui caminho de arquivo original salvo.",
                }

            caminho_arquivo = Path(caminho_arquivo_str)
            if not caminho_arquivo.exists():
                return {"ok": False, "mensagem": f"Arquivo original '{caminho_arquivo_str}' não encontrado no disco."}

            nome_arquivo_original = doc_original.get("nome_arquivo", caminho_arquivo.name)
            origem_original = doc_original.get("origem", "reprocessamento")
            conteudo_original = caminho_arquivo.read_bytes()

            # Deleta o documento antigo
            self.db.conn.execute("DELETE FROM documentos WHERE id = ?", (documento_id,))
            self.db.conn.commit()
            log.info("Documento ID %d e dados associados deletados do banco.", documento_id)

            # Remove o arquivo físico antigo para evitar lixo
            try:
                if caminho_arquivo.exists():
                    caminho_arquivo.unlink()
                    log.info("Arquivo físico '%s' removido durante reprocessamento.", caminho_arquivo)
            except Exception as e_clean:
                log.warning("Não foi possível excluir arquivo físico '%s': %s", caminho_arquivo, e_clean)

            # Re-ingere o arquivo
            log.info("Re-ingerindo arquivo '%s'...", nome_arquivo_original)
            novo_doc_id = self.ingestir_arquivo(
                nome=nome_arquivo_original, conteudo=conteudo_original, origem=origem_original
            )

            if novo_doc_id == documento_id:
                msg = (
                    f"Reprocessamento falhou. O documento ID {documento_id} não pôde ser deletado e re-inserido."
                )
                log.error(msg)
                return {"ok": False, "mensagem": msg}

            novo_doc_info = self.db.get_documento(novo_doc_id)
            novo_status = novo_doc_info.get("status") if novo_doc_info else "desconhecido"

            msg = (
                f"Reprocessamento concluído. ID antigo: {documento_id}. Novo ID: {novo_doc_id} (Status: {novo_status})."
            )
            log.info(msg)
            self.db.log(
                "reprocessamento",
                "usuario_sistema",
                f"doc_id_antigo={documento_id}|doc_id_novo={novo_doc_id}|status={novo_status}",
            )
            return {"ok": True, "mensagem": msg, "novo_id": novo_doc_id}

        except Exception as e:
            log.exception("Falha ao reprocessar doc_id %d: %s", documento_id, e)
            return {"ok": False, "mensagem": f"Falha ao reprocessar: {e}"}

    # ----------------------- Auto-roteamento -----------------------
    def processar_automatico(self, nome: str, conteudo: bytes, origem: str = "upload") -> int:
        """Roteia automaticamente: XML -> AgenteXMLParser; caso contrário, OCR/NLP."""
        try:
            doc_hash = self.db.hash_bytes(conteudo)
            existing_id = self.db.find_documento_by_hash(doc_hash)
            if existing_id:
                log.info("Doc '%s' (hash %s...) já existe ID %d. Ignorando.", nome, doc_hash[:8], existing_id)
                return existing_id

            head = conteudo[:2000]
            if (
                head.strip().startswith(b"<?xml")
                or b"<NFe" in head
                or b"<CTe" in head
                or b"<MDFe" in head
                or b"<CFe" in head
                or b"NFSe" in head
                or b"Nfse" in head
            ):
                log.info("Detectado XML fiscal: %s", nome)
                return self.xml_agent.processar(nome, conteudo, origem)
            else:
                log.info("Arquivo não XML detectado (%s), enviando para OCR/NLP...", nome)
                return self._processar_midias(nome, conteudo, origem)
        except Exception as e:
            log.exception("Falha no roteamento automático '%s': %s", nome, e)
            return self.db.inserir_documento(
                nome_arquivo=nome,
                tipo="desconhecido",
                origem=origem,
                hash=self.db.hash_bytes(conteudo),
                status="erro",
                data_upload=self.db.now(),
                motivo_rejeicao=str(e),
            )
