# agentes/agente_xml_parser.py

from __future__ import annotations

import logging
import re
import time
import json
import unicodedata
from typing import Any, Dict, Iterable, List, Optional, Tuple
from xml.etree import ElementTree as ET

from .utils import (
    _parse_date_like, _only_digits, _norm_ws
)

log = logging.getLogger("agente_fiscal.agentes")

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from banco_de_dados import BancoDeDados
    from validacao import ValidadorFiscal
    from metrics_agent import MetricsAgent


class AgenteXMLParser:
    """
    Interpreta XMLs fiscais (NFe, NFCe, CTe/CTeOS, MDF-e, CF-e, NFSe),
    normaliza dados conforme o schema unificado do projeto e
    (opcionalmente) persiste no banco, registra métricas e valida.
    """

    def __init__(self, db: "BancoDeDados", validador: "ValidadorFiscal", metrics_agent: "MetricsAgent"):
        self.db = db
        self.validador = validador
        self.metrics_agent = metrics_agent

    # ==================== API 1: Processamento com persistência ====================
    def processar(self, nome: str, conteudo: bytes, origem: str = "upload") -> int:
        """
        Processa um arquivo XML, extrai dados, aplica normalizações,
        insere no banco, extrai itens/impostos e dispara validação.

        Retorna o documento_id no banco (ou -1 em caso de falha grave).
        """
        t_start = time.time()
        doc_id = -1
        status = "erro"
        tipo = "xml/desconhecido"
        motivo_rejeicao = "Falha desconhecida no processamento XML"
        confianca_media = 1.0  # XML estruturado => alta confiança
        caminho_salvo: Optional[str] = None
        field_completeness: float = 0.0

        # 1) Parse seguro com tolerância
        try:
            try:
                root = ET.fromstring(conteudo)
            except ET.ParseError:
                root = ET.fromstring(conteudo.decode("latin-1", errors="ignore").encode("utf-8"))
        except Exception as e:
            log.warning("Falha ao parsear XML '%s': %s", nome, e)
            return self._registrar_xml_invalido(nome, conteudo, origem, f"XML mal formado: {e}", t_start)

        # 2) Detecta tipo (NF-e, NFC-e etc.)
        try:
            tipo = self._detectar_tipo(root)
        except Exception as e:
            log.warning("Detecção de tipo falhou: %s", e)

        # 3) Extrai campos principais + extras
        try:
            campos, extras = self._extrair_campos_por_tipo(root, tipo)

            # 3.1) Completeness de campos canônicos (para métricas/decisão)
            canons = [
                "emitente_cnpj","emitente_nome","emitente_uf","emitente_municipio",
                "destinatario_cnpj","destinatario_nome","destinatario_uf","destinatario_municipio",
                "valor_total","data_emissao","numero_nota","serie","modelo",
                "total_produtos","total_servicos","total_icms","total_pis","total_cofins","total_ipi"
            ]
            filled = sum(1 for k in canons if self._is_filled(campos.get(k)))
            field_completeness = round(filled / max(len(canons), 1), 3)
            extras = extras or {}
            extras.setdefault("__diagnostics__", {})
            extras["__diagnostics__"]["field_completeness"] = field_completeness

            # 3.2) Tokens normalizados para o associador (barateia similaridade OCR↔XML)
            extras["emitente_nome_tokens"] = self._norm_tokens(campos.get("emitente_nome"))
            extras["destinatario_nome_tokens"] = self._norm_tokens(campos.get("destinatario_nome"))
            extras["emitente_endereco_tokens"] = self._norm_tokens(campos.get("emitente_endereco"))
            extras["destinatario_endereco_tokens"] = self._norm_tokens(campos.get("destinatario_endereco"))

            # 4) Salvar arquivo físico (mantemos como caminho_xml e caminho_arquivo)
            caminho_salvo = str(self.db.save_upload(nome, conteudo))

            # 5) Inserção do documento (campos fixos)
            doc_id = self.db.inserir_documento(
                nome_arquivo=nome,
                tipo=tipo,
                origem=origem,
                hash=self.db.hash_bytes(conteudo),

                # Identificação
                chave_acesso=campos.get("chave_acesso"),
                numero_nota=campos.get("numero_nota"),
                serie=campos.get("serie"),
                modelo=campos.get("modelo"),
                natureza_operacao=campos.get("natureza_operacao"),

                # Status & datas
                status="processando",
                data_upload=self.db.now(),
                data_emissao=campos.get("data_emissao"),

                # Emitente
                emitente_cnpj=campos.get("emitente_cnpj"),
                emitente_cpf=campos.get("emitente_cpf"),
                emitente_nome=campos.get("emitente_nome"),
                emitente_ie=campos.get("emitente_ie"),
                emitente_im=campos.get("emitente_im"),
                emitente_uf=campos.get("emitente_uf"),
                emitente_municipio=campos.get("emitente_municipio"),
                emitente_endereco=campos.get("emitente_endereco"),

                # Destinatário
                destinatario_cnpj=campos.get("destinatario_cnpj"),
                destinatario_cpf=campos.get("destinatario_cpf"),
                destinatario_nome=campos.get("destinatario_nome"),
                destinatario_ie=campos.get("destinatario_ie"),
                destinatario_im=campos.get("destinatario_im"),
                destinatario_uf=campos.get("destinatario_uf"),
                destinatario_municipio=campos.get("destinatario_municipio"),
                destinatario_endereco=campos.get("destinatario_endereco"),

                # Totais principais
                valor_total=campos.get("valor_total"),
                total_produtos=campos.get("total_produtos"),
                total_servicos=campos.get("total_servicos"),

                # Totais complementares
                valor_descontos=campos.get("valor_descontos"),
                valor_frete=campos.get("valor_frete"),
                valor_seguro=campos.get("valor_seguro"),
                valor_outros=campos.get("valor_outros"),
                valor_liquido=campos.get("valor_liquido"),

                # Totais de impostos agregados
                total_icms=campos.get("total_icms"),
                total_ipi=campos.get("total_ipi"),
                total_pis=campos.get("total_pis"),
                total_cofins=campos.get("total_cofins"),

                # Transporte
                modalidade_frete=campos.get("modalidade_frete"),
                placa_veiculo=campos.get("placa_veiculo"),
                uf_veiculo=campos.get("uf_veiculo"),
                peso_bruto=campos.get("peso_bruto"),
                peso_liquido=campos.get("peso_liquido"),
                qtd_volumes=campos.get("qtd_volumes"),

                # Pagamento
                forma_pagamento=campos.get("forma_pagamento"),
                valor_pagamento=campos.get("valor_pagamento"),
                troco=campos.get("troco"),

                # Autorização / XML meta
                versao_schema=campos.get("versao_schema"),
                ambiente=campos.get("ambiente"),
                protocolo_autorizacao=campos.get("protocolo_autorizacao"),
                data_autorizacao=campos.get("data_autorizacao"),
                cstat=campos.get("cstat"),
                xmotivo=campos.get("xmotivo"),
                responsavel_tecnico=campos.get("responsavel_tecnico"),

                # Caminhos
                caminho_arquivo=caminho_salvo,
                caminho_xml=caminho_salvo,

                # Meta genérica (inclui extras não mapeados em colunas fixas + __meta__)
                meta_json=json.dumps({
                    "detalhes": extras,
                    "__meta__": {
                        "tipo": tipo,
                        "field_completeness": field_completeness
                    }
                }, ensure_ascii=False),
            )

            # 6) Itens & Impostos
            self._extrair_itens_impostos(root, doc_id, tipo)

            # 7) Validação fiscal
            try:
                self.validador.validar_documento(doc_id=doc_id, db=self.db)
            except Exception as e_val:
                log.warning("Validação fiscal falhou doc_id=%s: %s", doc_id, e_val)

            final_doc_info = self.db.get_documento(doc_id) or {}
            status = final_doc_info.get("status") or ("processado" if campos.get("valor_total") is not None else "revisao_pendente")

        except Exception as e_proc:
            log.exception("Erro durante o processamento do XML (doc_id=%s): %s", doc_id, e_proc)
            status = "erro"
            motivo_rejeicao = f"Erro no processamento: {e_proc}"
            if doc_id > 0:
                self.db.atualizar_documento_campos(doc_id, status=status, motivo_rejeicao=motivo_rejeicao)

        finally:
            # 8) Registro de extração + métricas
            processing_time = time.time() - t_start
            if doc_id > 0:
                try:
                    self.db.inserir_extracao(
                        documento_id=doc_id,
                        agente="XMLParser",
                        confianca_media=confianca_media,
                        texto_extraido=None,
                        linguagem="pt",
                        tempo_processamento=round(processing_time, 3),
                    )
                    self.db.log("ingestao_xml", usuario="sistema", detalhes=f"doc_id={doc_id}|tipo={tipo}|status={status}")
                finally:
                    # Métricas agregadas (retrocompatível)
                    self.metrics_agent.registrar_metrica(
                        db=self.db,
                        tipo_documento=tipo,
                        status=status,
                        confianca_media=confianca_media,
                        tempo_medio=processing_time,
                    )
                    # NOVO: desempenho por agente p/ aprendizado do pipeline
                    try:
                        self.metrics_agent.registrar_desempenho_agente(
                            db=self.db,
                            doc_id=doc_id,
                            agente="XMLParser",
                            tipo_documento=tipo,
                            coverage=field_completeness,
                            confidence=1.0,
                            latency_s=processing_time,
                            extras={"has_itens": True, "schema": campos.get("versao_schema")}
                        )
                    except Exception as e_m:
                        log.debug(f"registrar_desempenho_agente falhou (ignorado): {e_m}")

        return doc_id

    # ==================== API 2: Extração pura (sem DB) ====================
    def extrair_campos_dict(self, conteudo: bytes) -> Dict[str, Any]:
        """
        Extrai e retorna um dicionário alinhado ao schema unificado,
        incluindo __meta__ com coverage aproximado e 'extras' (detalhes).
        """
        try:
            try:
                root = ET.fromstring(conteudo)
            except ET.ParseError:
                root = ET.fromstring(conteudo.decode("latin-1", errors="ignore").encode("utf-8"))
        except Exception as e:
            log.warning("extrair_campos_dict: XML inválido: %s", e)
            return {"__meta__": {"source": "xml", "coverage": 0.0, "erro": "xml_invalido"}}

        tipo = self._detectar_tipo(root)
        campos, extras = self._extrair_campos_por_tipo(root, tipo)

        # Cobertura expandida (mais fiel ao pipeline)
        principais = [
            "emitente_cnpj","emitente_nome","emitente_uf","emitente_municipio",
            "destinatario_cnpj","destinatario_nome","destinatario_uf","destinatario_municipio",
            "valor_total","data_emissao","numero_nota","serie","modelo",
            "total_produtos","total_servicos","total_icms","total_pis","total_cofins","total_ipi",
            "chave_acesso"
        ]
        cov = sum(1 for k in principais if self._is_filled(campos.get(k))) / max(len(principais), 1)

        campos["__meta__"] = {
            "source": "xml",
            "coverage": round(float(cov), 3),
            "tipo": tipo
        }
        campos["extras"] = extras or {}
        return campos

    # ==================== Núcleo de extração por tipo ====================
    def _extrair_campos_por_tipo(self, root: ET.Element, tipo: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if tipo in ("NFe", "NFCe"):
            campos, extras = self._extrair_campos_nfe(root)
        elif tipo in ("CTe", "CTeOS"):
            campos, extras = self._extrair_campos_cte(root)
        elif tipo == "MDF-e":
            campos, extras = self._extrair_campos_mdfe(root)
        elif tipo == "CF-e":
            campos, extras = self._extrair_campos_cfe(root)
        elif tipo == "NFSe":
            campos, extras = self._extrair_campos_nfse(root)
        else:
            campos, extras = self._extrair_campos_generico(root)

        # Chave de acesso (multi-rotas)
        chave = (
            self._get_attr(root, ".//{*}infNFe", "Id")
            or self._get_attr(root, ".//{*}infCTe", "Id")
            or self._get_text(root, ".//{*}chNFe")
            or self._get_text(root, ".//{*}chCTe")
            or self._get_text(root, ".//infCFe")
        )
        if chave:
            chave = _only_digits(chave)
        campos.setdefault("chave_acesso", chave)

        # Datas consolidadas
        campos["data_emissao"] = campos.get("data_emissao") or self._primeiro_valido(
            _parse_date_like(self._get_text(root, ".//{*}ide/{*}dhEmi")),
            _parse_date_like(self._get_text(root, ".//{*}ide/{*}dEmi")),
            _parse_date_like(self._get_text(root, ".//{*}ide/{*}dIniViagem")),
            _parse_date_like(self._get_text(root, ".//{*}DataEmissao")),
            _parse_date_like(self._get_text(root, ".//{*}dataEmissao")),
            _parse_date_like(self._get_text(root, ".//{*}Competencia")),
        )

        # Normaliza nomes (espaços)
        for k in ("emitente_nome", "destinatario_nome", "natureza_operacao"):
            if campos.get(k):
                campos[k] = _norm_ws(campos[k])

        # Total consolidado (fallback)
        if campos.get("valor_total") is None:
            campos["valor_total"] = self._coalesce_total(root, tipo)

        # Normaliza CNPJ/CPF
        for chave_id in ("emitente_cnpj", "emitente_cpf", "destinatario_cnpj", "destinatario_cpf"):
            if campos.get(chave_id):
                campos[chave_id] = _only_digits(campos[chave_id])

        # Metadados do XML/autorização (se existirem no documento)
        meta = self._extrair_meta_autorizacao(root)
        campos.update(meta)

        return campos, extras

    # -------------------- Detecção de Tipo --------------------
    def _detectar_tipo(self, root: ET.Element) -> str:
        mod_node = self._find(root, ".//{*}ide")
        mod = None
        if mod_node is not None:
            for ch in mod_node.iter():
                if ch.tag.split("}", 1)[-1].lower() == "mod" and ch.text:
                    mod = ch.text.strip()
                    break
        mapa = {"55": "NFe", "65": "NFCe", "57": "CTe", "67": "CTeOS", "58": "MDF-e", "59": "CF-e"}
        if mod in mapa:
            return mapa[mod]

        root_tag = root.tag.split("}", 1)[-1].lower()
        if "nfse" in root_tag or "servico" in root_tag or "serviço" in root_tag:
            return "NFSe"

        for el in root.iter():
            lname = el.tag.split("}", 1)[-1].lower()
            if "nfse" in lname or "servico" in lname or "serviço" in lname:
                return "NFSe"

        if "nfe" in root_tag:
            return "NFe"
        if "cte" in root_tag:
            return "CTe"
        if "mdfe" in root_tag:
            return "MDF-e"
        if "cfe" in root_tag or "sat" in root_tag:
            return "CF-e"
        return "xml/desconhecido"

    # -------------------- Extrações por Tipo --------------------
    def _extrair_campos_nfe(self, root: ET.Element) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        emit = self._find(root, ".//{*}emit")
        dest = self._find(root, ".//{*}dest")
        ide = self._find(root, ".//{*}ide")
        tot = self._find(root, ".//{*}total/{*}ICMSTot")
        end_emit = self._find(root, ".//{*}emit/{*}enderEmit")
        end_dest = self._find(root, ".//{*}dest/{*}enderDest")

        # Totais complementares
        descontos = self._parse_number(self._text(tot, "vDesc"), decimals=2)
        frete = self._parse_number(self._text(tot, "vFrete"), decimals=2)
        seguro = self._parse_number(self._text(tot, "vSeg"), decimals=2)
        outros = self._parse_number(self._text(tot, "vOutro"), decimals=2)
        vprod = self._parse_number(self._text(tot, "vProd"), decimals=2)
        vserv = self._parse_number(self._text(tot, "vServ"), decimals=2)

        # Impostos agregados
        v_icms = self._parse_number(self._text(tot, "vICMS"), decimals=2)
        v_ipi = self._parse_number(self._text(tot, "vIPI"), decimals=2)
        v_pis = self._parse_number(self._text(tot, "vPIS"), decimals=2)
        v_cofins = self._parse_number(self._text(tot, "vCOFINS"), decimals=2)

        # Pagamento (procura em ambos padrões: pag e detPag)
        forma_pag, valor_pag, troco = self._extrair_pagamento(root)

        # Transporte
        modal_frete, placa, uf_placa, peso_b, peso_l, q_vol = self._extrair_transporte(root)

        campos = {
            # Identificação
            "numero_nota": self._text(ide, "nNF"),
            "serie": self._text(ide, "serie"),
            "modelo": self._text(ide, "mod"),
            "natureza_operacao": self._text(ide, "natOp"),

            # Emitente (completos)
            "emitente_nome": self._text(emit, "xNome"),
            "emitente_cnpj": self._text(emit, "CNPJ"),
            "emitente_cpf": self._text(emit, "CPF"),
            "emitente_ie": self._text(emit, "IE"),
            "emitente_im": self._text(emit, "IM"),
            "emitente_uf": self._text(end_emit, "UF"),
            "emitente_municipio": self._text(end_emit, "xMun"),
            "emitente_endereco": self._build_address(end_emit),

            # Destinatário (completos)
            "destinatario_nome": self._text(dest, "xNome"),
            "destinatario_cnpj": self._text(dest, "CNPJ"),
            "destinatario_cpf": self._text(dest, "CPF"),
            "destinatario_ie": self._text(dest, "IE"),
            "destinatario_im": self._text(dest, "IM"),
            "destinatario_uf": self._text(end_dest, "UF"),
            "destinatario_municipio": self._text(end_dest, "xMun"),
            "destinatario_endereco": self._build_address(end_dest),

            # Datas / Totais
            "data_emissao": _parse_date_like(self._text_any(ide, ("dhEmi", "dEmi"))),
            "valor_total": self._parse_number(self._text(tot, "vNF"), decimals=2),
            "total_produtos": vprod,
            "total_servicos": vserv,

            # Totais complementares
            "valor_descontos": descontos,
            "valor_frete": frete,
            "valor_seguro": seguro,
            "valor_outros": outros,
            "valor_liquido": None,  # pode ser calculado depois

            # Impostos agregados
            "total_icms": v_icms,
            "total_ipi": v_ipi,
            "total_pis": v_pis,
            "total_cofins": v_cofins,

            # Transporte
            "modalidade_frete": modal_frete,
            "placa_veiculo": placa,
            "uf_veiculo": uf_placa,
            "peso_bruto": peso_b,
            "peso_liquido": peso_l,
            "qtd_volumes": q_vol,

            # Pagamento
            "forma_pagamento": forma_pag,
            "valor_pagamento": valor_pag,
            "troco": troco,

            # Chave
            "chave_acesso": (_only_digits(self._get_attr(root, ".//{*}infNFe", "Id")) or None),
        }

        extras: Dict[str, Any] = {}
        cfop_first = self._text(self._find(root, ".//{*}det/{*}prod"), "CFOP")
        if cfop_first:
            extras["det/0/CFOP"] = cfop_first

        return campos, extras

    def _extrair_campos_cte(self, root: ET.Element) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        emit = self._find(root, ".//{*}emit")
        rem = self._find(root, ".//{*}rem")
        dest = self._find(root, ".//{*}dest") or rem
        ide = self._find(root, ".//{*}ide")
        vprest = self._find(root, ".//{*}vPrest")

        end_emit = self._find(emit, ".//{*}enderEmit")
        end_dest = self._find(dest, ".//{*}enderDest")
        end_rem = self._find(rem, ".//{*}enderReme")

        # Totais / carga
        v_total = self._parse_number(self._text(vprest, "vTPrest"), decimals=2)
        v_carga = self._parse_number(self._first_text_by_local_name(root, "vCarga"), decimals=2)

        # Quantidade de volumes (infQ cUnid="03" => UN)
        qtd_vol = None
        for infq in self._findall(root, ".//{*}infQ"):
            cUnid = None
            qCarga = None
            for child in infq:
                lname = child.tag.split("}", 1)[-1]
                if lname == "cUnid" and child.text:
                    cUnid = child.text.strip()
                if lname == "qCarga" and child.text:
                    qCarga = child.text.strip()
            if cUnid == "03" and qCarga:
                qtd_vol = self._parse_number(qCarga, decimals=0)
                break

        # Placa / UF (rodoviário)
        rodo = self._find(root, ".//{*}rodo")
        veic = self._find(rodo, ".//{*}veic")
        placa = self._text(veic, "placa") if veic is not None else None
        uf_placa = self._text(veic, "UF") if veic is not None else None

        campos = {
            "numero_nota": self._text(ide, "nCT"),
            "serie": self._text(ide, "serie"),
            "modelo": self._text(ide, "mod"),

            "emitente_nome": self._text(emit, "xNome"),
            "emitente_cnpj": self._text(emit, "CNPJ"),
            "emitente_ie": self._text(emit, "IE"),
            "emitente_uf": self._text(end_emit, "UF"),
            "emitente_municipio": self._text(end_emit, "xMun"),

            "destinatario_nome": self._text(dest, "xNome"),
            "destinatario_cnpj": self._text(dest, "CNPJ"),
            "destinatario_ie": self._text(dest, "IE"),
            "destinatario_uf": self._text(end_dest, "UF") or self._text(end_rem, "UF"),
            "destinatario_municipio": self._text(end_dest, "xMun") or self._text(end_rem, "xMun"),

            # Endereços completos
            "emitente_endereco": self._build_address(end_emit),
            "destinatario_endereco": self._build_address(end_dest) or self._build_address(end_rem),

            # Datas / Totais
            "data_emissao": _parse_date_like(self._text_any(ide, ("dhEmi", "dEmi"))),
            "valor_total": v_total,

            "total_produtos": None,
            "total_servicos": None,
            "valor_descontos": None,
            "valor_frete": None,
            "valor_seguro": None,
            "valor_outros": None,
            "valor_liquido": None,

            "total_icms": None,
            "total_ipi": None,
            "total_pis": None,
            "total_cofins": None,

            # Transporte (CT-e)
            "modalidade_frete": None,
            "placa_veiculo": placa,
            "uf_veiculo": uf_placa,
            "peso_bruto": None,
            "peso_liquido": None,
            "qtd_volumes": qtd_vol,

            # Pagamento
            "forma_pagamento": None,
            "valor_pagamento": None,
            "troco": None,

            "chave_acesso": (_only_digits(self._get_attr(root, ".//{*}infCTe", "Id")) or None),
        }

        extras = {}
        if v_carga is not None:
            extras["cte/infCarga/vCarga"] = v_carga
        return campos, extras

    def _extrair_campos_mdfe(self, root: ET.Element) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        emit = self._find(root, ".//{*}emit")
        ide = self._find(root, ".//{*}ide")
        tot = self._find(root, ".//{*}tot")
        end_emit = self._find(emit, ".//{*}enderEmit")

        # Placa / UF (rodoviário tração)
        veic_tr = self._find(root, ".//{*}infModal/{*}rodo/{*}veicTracao")
        placa = self._text(veic_tr, "placa") if veic_tr is not None else None
        uf_placa = self._text(veic_tr, "UF") if veic_tr is not None else None

        campos = {
            "numero_nota": self._text(ide, "nMDF"),
            "serie": self._text(ide, "serie"),
            "modelo": self._text(ide, "mod"),

            "emitente_nome": self._text(emit, "xNome"),
            "emitente_cnpj": self._text(emit, "CNPJ"),
            "emitente_ie": self._text(emit, "IE"),
            "emitente_uf": self._text(end_emit, "UF"),
            "emitente_municipio": self._text(end_emit, "xMun"),
            "emitente_endereco": self._build_address(end_emit),

            "destinatario_nome": None,
            "destinatario_cnpj": None,
            "destinatario_ie": None,
            "destinatario_uf": None,
            "destinatario_municipio": None,
            "destinatario_endereco": None,

            "data_emissao": _parse_date_like(self._text_any(ide, ("dhEmi", "dEmi")) or self._text(ide, "dIniViagem")),
            "valor_total": self._parse_number(self._text(tot, "vCarga"), decimals=2),

            "total_produtos": None,
            "total_servicos": None,
            "valor_descontos": None,
            "valor_frete": None,
            "valor_seguro": None,
            "valor_outros": None,
            "valor_liquido": None,

            "total_icms": None,
            "total_ipi": None,
            "total_pis": None,
            "total_cofins": None,

            # Transporte (MDF-e)
            "modalidade_frete": None,
            "placa_veiculo": placa,
            "uf_veiculo": uf_placa,
            "peso_bruto": None,
            "peso_liquido": None,
            "qtd_volumes": None,

            "forma_pagamento": None,
            "valor_pagamento": None,
            "troco": None,
        }
        extras = {}
        qcte = self._first_text_by_local_name(root, "qCTe")
        if qcte:
            extras["mdfe/tot/qCTe"] = qcte
        return campos, extras

    def _extrair_campos_cfe(self, root: ET.Element) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        emit = self._find(root, ".//emit") or self._find(root, ".//{*}emit")
        dest = self._find(root, ".//dest") or self._find(root, ".//{*}dest")
        total = self._find(root, ".//total") or self._find(root, ".//{*}total")
        ide = self._find(root, ".//ide") or self._find(root, ".//{*}ide")
        pgto = self._find(root, ".//pgto") or self._find(root, ".//{*}pgto")

        end_emit = self._find(emit, ".//{*}enderEmit") or self._find(emit, ".//enderEmit")
        end_dest = self._find(dest, ".//{*}enderDest") or self._find(dest, ".//enderDest")

        # Pagamento (SAT): MP/cMP, MP/vMP e vTroco
        forma_pag = None
        valor_pag = None
        if pgto is not None:
            vtotal = 0.0
            first_forma = None
            for mp in self._findall(pgto, ".//{*}MP"):
                cod = self._text(mp, "cMP")
                val = self._parse_number(self._text(mp, "vMP"), decimals=2)
                if first_forma is None and cod:
                    first_forma = cod
                if val is not None:
                    vtotal += val
            if first_forma:
                forma_pag = first_forma
            if vtotal > 0:
                valor_pag = round(vtotal, 2)
        troco = self._parse_number(self._first_text_by_local_name(pgto, "vTroco"), decimals=2) if pgto is not None else None

        campos = {
            "numero_nota": self._text(ide, "nCFe") or self._text(ide, "nNF"),
            "serie": self._text(ide, "serie"),
            "modelo": self._text(ide, "mod"),

            "emitente_nome": self._text(emit, "xNome"),
            "emitente_cnpj": self._text(emit, "CNPJ"),
            "emitente_ie": self._text(emit, "IE"),
            "emitente_uf": self._text(end_emit, "UF"),
            "emitente_municipio": self._text(end_emit, "xMun"),
            "emitente_endereco": self._build_address(end_emit),

            "destinatario_nome": self._text(dest, "xNome"),
            "destinatario_cnpj": self._text(dest, "CNPJ"),
            "destinatario_ie": self._text(dest, "IE"),
            "destinatario_uf": self._text(end_dest, "UF"),
            "destinatario_municipio": self._text(end_dest, "xMun"),
            "destinatario_endereco": self._build_address(end_dest),

            "data_emissao": _parse_date_like(self._text_any(ide, ("dhEmi", "dEmi"))),
            "valor_total": self._parse_number(
                self._text(total, "vCFe") or self._text(total, "vCFeLei12741") or self._text(total, "vNF"), decimals=2
            ),

            "total_produtos": None,
            "total_servicos": None,
            "valor_descontos": None,
            "valor_frete": None,
            "valor_seguro": None,
            "valor_outros": None,
            "valor_liquido": None,

            "total_icms": None,
            "total_ipi": None,
            "total_pis": None,
            "total_cofins": None,

            "modalidade_frete": None,
            "placa_veiculo": None,
            "uf_veiculo": None,
            "peso_bruto": None,
            "peso_liquido": None,
            "qtd_volumes": None,

            "forma_pagamento": forma_pag,
            "valor_pagamento": valor_pag,
            "troco": troco,
        }
        return campos, {}

    def _extrair_campos_nfse(self, root: ET.Element) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        prest = self._find(root, ".//{*}PrestadorServico") or self._find(root, ".//{*}Prestador")
        toma = self._find(root, ".//{*}TomadorServico") or self._find(root, ".//{*}Tomador")

        emit_nome = self._get_text(prest, ".//{*}RazaoSocial") or self._get_text(prest, ".//{*}NomeFantasia") or self._get_text(prest, ".//{*}xNome")
        dest_nome = self._get_text(toma, ".//{*}RazaoSocial") or self._get_text(toma, ".//{*}xNome") or self._get_text(toma, ".//{*}Nome")

        emit_cnpj = self._get_text(prest, ".//{*}Cnpj") or self._get_text(prest, ".//{*}CNPJ")
        emit_cpf = self._get_text(prest, ".//{*}Cpf") or self._get_text(prest, ".//{*}CPF")
        dest_cnpj = self._get_text(toma, ".//{*}Cnpj") or self._get_text(toma, ".//{*}CNPJ")
        dest_cpf = self._get_text(toma, ".//{*}Cpf") or self._get_text(toma, ".//{*}CPF")

        end_prest = self._find(prest, ".//{*}Endereco")
        end_toma = self._find(toma, ".//{*}Endereco")
        emit_endereco = self._build_address_nfse(end_prest) if end_prest is not None else None
        dest_endereco = self._build_address_nfse(end_toma) if end_toma is not None else None

        # UF/Mun: tenta por endereço; se não houver, usa códigos de município
        emit_uf = self._get_text(end_prest, ".//{*}Estado") or self._get_text(end_prest, ".//{*}UF")
        dest_uf = self._get_text(end_toma, ".//{*}Estado") or self._get_text(end_toma, ".//{*}UF")
        emit_mun = self._get_text(end_prest, ".//{*}Municipio") or self._get_text(end_prest, ".//{*}xMun")
        dest_mun = self._get_text(end_toma, ".//{*}Municipio") or self._get_text(end_toma, ".//{*}xMun")

        if not emit_uf:
            cod_mun_prest = self._get_text(prest, ".//{*}CodigoMunicipioPrestador") or self._get_text(root, ".//{*}CodigoMunicipio")
            uf_from_code = self._uf_from_cod_municipio(cod_mun_prest)
            if uf_from_code:
                emit_uf = uf_from_code
        if not dest_uf:
            cod_mun_toma = self._get_text(toma, ".//{*}CodigoMunicipioTomador") or self._get_text(toma, ".//{*}CodigoMunicipio")
            uf_from_code = self._uf_from_cod_municipio(cod_mun_toma)
            if uf_from_code:
                dest_uf = uf_from_code

        valor_total = self._coalesce(
            self._parse_number(self._get_text(root, ".//{*}ValorServicos"), decimals=2),
            self._parse_number(self._get_text(root, ".//{*}vServ"), decimals=2),
            self._parse_number(self._get_text(root, ".//{*}ValorLiquidoNfse"), decimals=2),
        )

        data_emissao = _parse_date_like(
            self._coalesce(
                self._get_text(root, ".//{*}DataEmissao"),
                self._get_text(root, ".//{*}dtEmissao"),
                self._get_text(root, ".//{*}dhEmi"),
                self._get_text(root, ".//{*}Competencia"),
            )
        )

        campos = {
            # Emitente/Destinatário + endereços próprios
            "emitente_nome": emit_nome,
            "emitente_cnpj": emit_cnpj,
            "emitente_cpf": emit_cpf,
            "emitente_endereco": emit_endereco,

            "destinatario_nome": dest_nome,
            "destinatario_cnpj": dest_cnpj,
            "destinatario_cpf": dest_cpf,
            "destinatario_endereco": dest_endereco,

            "emitente_uf": emit_uf,
            "emitente_municipio": emit_mun,
            "destinatario_uf": dest_uf,
            "destinatario_municipio": dest_mun,

            "data_emissao": data_emissao,
            "valor_total": valor_total,
            "total_servicos": valor_total,

            "total_produtos": None,
            "valor_descontos": None,
            "valor_frete": None,
            "valor_seguro": None,
            "valor_outros": None,
            "valor_liquido": None,

            "total_icms": None,
            "total_ipi": None,
            "total_pis": None,
            "total_cofins": None,

            "modalidade_frete": None,
            "placa_veiculo": None,
            "uf_veiculo": None,
            "peso_bruto": None,
            "peso_liquido": None,
            "qtd_volumes": None,

            "forma_pagamento": None,
            "valor_pagamento": None,
            "troco": None,
        }

        extras: Dict[str, Any] = {}
        # Guardar códigos de serviço, alíquota ISS, deduções, CNAE, etc., quando existirem
        cod_serv = self._get_text(root, ".//{*}ItemListaServico") or self._get_text(root, ".//{*}CodigoServico")
        if cod_serv:
            extras["servico/Codigo"] = cod_serv

        aliq_iss = (
            self._get_text(root, ".//{*}Aliquota")
            or self._get_text(root, ".//{*}pISS")
            or self._get_text(root, ".//{*}Servico/{*}Valores/{*}Aliquota")
        )
        if aliq_iss:
            extras["iss/Aliquota"] = aliq_iss

        return campos, extras

    def _extrair_campos_generico(self, root: ET.Element) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        # Fallback extremamente conservador: nunca preenche UF/Mun/End genéricos
        emit = self._find(root, ".//{*}emit") or self._find(root, ".//emit")
        end_emit = self._find(root, ".//{*}enderEmit") or self._find(root, ".//enderEmit")
        ide = self._find(root, ".//{*}ide") or self._find(root, ".//ide")
        total = self._find(root, ".//{*}total") or self._find(root, ".//total")

        campos = {
            "numero_nota": self._text(ide, "nNF") or self._text(ide, "nCT") or self._text(ide, "nCFe"),
            "serie": self._text(ide, "serie"),
            "modelo": self._text(ide, "mod"),

            "emitente_nome": self._text(emit, "xNome"),
            "emitente_cnpj": self._text(emit, "CNPJ"),
            "emitente_cpf": self._text(emit, "CPF"),
            "emitente_ie": self._text(emit, "IE"),
            "emitente_uf": self._text(end_emit, "UF"),
            "emitente_municipio": self._text(end_emit, "xMun"),
            "emitente_endereco": self._build_address(end_emit),

            # Nenhum campo genérico de UF/Mun/End no documento
            "data_emissao": _parse_date_like(self._text_any(ide, ("dhEmi", "dEmi"))),
            "valor_total": self._parse_number(
                self._text(total, "vNF") or self._text(total, "vCFe") or self._text(total, "vTPrest"), decimals=2
            ),

            "total_produtos": None,
            "total_servicos": None,
            "valor_descontos": None,
            "valor_frete": None,
            "valor_seguro": None,
            "valor_outros": None,
            "valor_liquido": None,

            "total_icms": None,
            "total_ipi": None,
            "total_pis": None,
            "total_cofins": None,

            "modalidade_frete": None,
            "placa_veiculo": None,
            "uf_veiculo": None,
            "peso_bruto": None,
            "peso_liquido": None,
            "qtd_volumes": None,

            "forma_pagamento": None,
            "valor_pagamento": None,
            "troco": None,
        }
        return campos, {}

    # -------------------- Itens & Impostos --------------------
    def _extrair_itens_impostos(self, root: ET.Element, doc_id: int, tipo: str) -> None:
        """
        Extrai itens/impostos com tolerância.
        Para CTe/MDF-e e NFSe ignoramos o detalhamento de itens (sem padronização).
        """
        try:
            if tipo in ("CTe", "CTeOS", "MDF-e", "NFSe"):
                return

            for det in self._findall(root, ".//{*}det"):
                prod = self._find(det, ".//{*}prod")
                imposto = self._find(det, ".//{*}imposto")
                if prod is None:
                    continue

                desc = self._text(prod, "xProd")
                ncm = self._text(prod, "NCM")
                cfop = self._text(prod, "CFOP")
                qnt = self._parse_number(self._text(prod, "qCom"), decimals=3)
                vun = self._parse_number(self._text(prod, "vUnCom"), decimals=4)
                vtot = self._parse_number(self._text(prod, "vProd"), decimals=2)
                unid = self._text(prod, "uCom")
                cprod = self._text(prod, "cProd")
                cest = self._text(prod, "CEST")
                n_item = self._parse_int_safe(self._get_attr(det, ".//{*}det", "nItem")) or None

                item_id = self.db.inserir_item(
                    documento_id=doc_id,
                    numero_item=n_item,
                    descricao=desc,
                    ncm=ncm,
                    cest=cest,
                    cfop=cfop,
                    quantidade=qnt,
                    unidade=unid,
                    valor_unitario=vun,
                    valor_total=vtot,
                    codigo_produto=cprod,
                )

                if imposto is not None:
                    # ICMS / ICMSUFDest + FCP
                    icms_node = self._find(imposto, ".//{*}ICMS") or self._find(imposto, ".//{*}ICMSUFDest")
                    if icms_node is not None:
                        icms_detalhe = next(iter(list(icms_node)), None)
                        if icms_detalhe is not None:
                            cst = self._text_any(icms_detalhe, ("CST", "CSOSN"))
                            orig = self._text(icms_detalhe, "orig")
                            bc = self._text_any(icms_detalhe, ("vBC", "vBCST", "vBCSTRet", "vBCUFDest"))
                            aliq = self._text_any(
                                icms_detalhe, ("pICMS", "pICMSST", "pICMSSTRet", "pICMSUFDest", "pICMSInter", "pICMSInterPart")
                            )
                            val = self._text_any(
                                icms_detalhe, ("vICMS", "vICMSST", "vICMSSTRet", "vICMSUFDest", "vICMSPartDest", "vICMSPartRemet")
                            )
                            if val is not None:
                                self.db.inserir_imposto(
                                    item_id=item_id,
                                    tipo_imposto="ICMS",
                                    cst=cst,
                                    origem=orig,
                                    base_calculo=self._parse_number(bc, decimals=2),
                                    aliquota=self._parse_number(aliq, decimals=4),
                                    valor=self._parse_number(val, decimals=2),
                                )
                            # FCP
                            vFCP = self._text(icms_detalhe, "vFCP")
                            pFCP = self._text(icms_detalhe, "pFCP")
                            if vFCP:
                                self.db.inserir_imposto(
                                    item_id=item_id,
                                    tipo_imposto="FCP",
                                    cst=None,
                                    origem=None,
                                    base_calculo=None,
                                    aliquota=self._parse_number(pFCP, decimals=4),
                                    valor=self._parse_number(vFCP, decimals=2),
                                )

                    # IPI
                    ipi_node = self._find(imposto, ".//{*}IPI")
                    ipi_trib_node = self._find(ipi_node, ".//{*}IPITrib") if ipi_node is not None else None
                    if ipi_trib_node is not None:
                        cst = self._text(ipi_trib_node, "CST")
                        bc = self._text(ipi_trib_node, "vBC")
                        aliq = self._text(ipi_trib_node, "pIPI")
                        val = self._text(ipi_trib_node, "vIPI")
                        if val is not None:
                            self.db.inserir_imposto(
                                item_id=item_id,
                                tipo_imposto="IPI",
                                cst=cst,
                                origem=None,
                                base_calculo=self._parse_number(bc, decimals=2),
                                aliquota=self._parse_number(aliq, decimals=4),
                                valor=self._parse_number(val, decimals=2),
                            )

                    # PIS
                    pis_node = self._find(imposto, ".//{*}PIS")
                    pis_aliq = self._find(pis_node, ".//{*}PISAliq") if pis_node is not None else None
                    if pis_aliq is not None:
                        cst = self._text(pis_aliq, "CST")
                        bc = self._text(pis_aliq, "vBC")
                        aliq = self._text(pis_aliq, "pPIS")
                        val = self._text(pis_aliq, "vPIS")
                        if val is not None:
                            self.db.inserir_imposto(
                                item_id=item_id,
                                tipo_imposto="PIS",
                                cst=cst,
                                origem=None,
                                base_calculo=self._parse_number(bc, decimals=2),
                                aliquota=self._parse_number(aliq, decimals=4),
                                valor=self._parse_number(val, decimals=2),
                            )

                    # COFINS
                    cofins_node = self._find(imposto, ".//{*}COFINS")
                    cofins_aliq = self._find(cofins_node, ".//{*}COFINSAliq") if cofins_node is not None else None
                    if cofins_aliq is not None:
                        cst = self._text(cofins_aliq, "CST")
                        bc = self._text(cofins_aliq, "vBC")
                        aliq = self._text(cofins_aliq, "pCOFINS")
                        val = self._text(cofins_aliq, "vCOFINS")
                        if val is not None:
                            self.db.inserir_imposto(
                                item_id=item_id,
                                tipo_imposto="COFINS",
                                cst=cst,
                                origem=None,
                                base_calculo=self._parse_number(bc, decimals=2),
                                aliquota=self._parse_number(aliq, decimals=4),
                                valor=self._parse_number(val, decimals=2),
                            )
        except Exception as e:
            log.warning("Falha ao extrair itens/impostos (doc_id=%s, tipo=%s): %s", doc_id, tipo, e)

    # -------------------- Autorização / Metadados do XML --------------------
    def _extrair_meta_autorizacao(self, root: ET.Element) -> Dict[str, Any]:
        """
        Extrai versão do schema, ambiente, protocolo (nProt), data de autorização (dhRecbto),
        cStat e xMotivo quando presentes (ex.: protNFe, protCTe, retConsReciNFe etc.)
        """
        meta: Dict[str, Any] = {}

        versao = root.attrib.get("versao") if hasattr(root, "attrib") else None
        if versao:
            meta["versao_schema"] = versao

        tpAmb = self._first_text_by_local_name(root, "tpAmb")
        if tpAmb:
            meta["ambiente"] = tpAmb

        nProt = self._first_text_by_local_name(root, "nProt")
        if nProt:
            meta["protocolo_autorizacao"] = nProt

        dhRecbto = self._first_text_by_local_name(root, "dhRecbto")
        if dhRecbto:
            meta["data_autorizacao"] = _parse_date_like(dhRecbto)

        cStat = self._first_text_by_local_name(root, "cStat")
        if cStat:
            meta["cstat"] = cStat

        xMotivo = self._first_text_by_local_name(root, "xMotivo")
        if xMotivo:
            meta["xmotivo"] = xMotivo

        respTec = self._find(root, ".//{*}respTec")
        if respTec is not None:
            meta["responsavel_tecnico"] = (
                self._text(respTec, "xContato") or self._text(respTec, "CNPJ") or self._text(respTec, "email")
            )

        return meta

    # -------------------- Pagamento & Transporte helpers --------------------
    def _extrair_pagamento(self, root: ET.Element) -> Tuple[Optional[str], Optional[float], Optional[float]]:
        """
        Retorna (forma_pagamento, valor_pagamento, troco)
        Busca em <pag><detPag> (layout 4.00+) e em variantes anteriores.
        """
        det = self._find(root, ".//{*}pag/{*}detPag")
        if det is not None:
            forma = self._text(det, "tPag") or self._text(det, "indPag")
            valor = self._parse_number(self._text(det, "vPag"), decimals=2)
            troco = self._parse_number(self._first_text_by_local_name(root, "vTroco"), decimals=2)
            return forma, valor, troco

        pag = self._find(root, ".//{*}pag")
        if pag is not None:
            forma = self._text(pag, "tPag") or self._text(pag, "indPag")
            valor = self._parse_number(self._text(pag, "vPag"), decimals=2)
            troco = self._parse_number(self._first_text_by_local_name(root, "vTroco"), decimals=2)
            return forma, valor, troco

        return None, None, None

    def _extrair_transporte(self, root: ET.Element) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[float], Optional[float], Optional[float]]:
        """
        Retorna (modalidade_frete, placa, uf_placa, peso_bruto, peso_liquido, qtd_volumes)
        """
        transp = self._find(root, ".//{*}transp")
        if transp is None:
            return None, None, None, None, None, None

        mod_frete = self._text(transp, "modFrete")
        veic = self._find(transp, ".//{*}veicTransp") or self._find(transp, ".//{*}veic")
        placa = self._text(veic, "placa") if veic is not None else None
        uf_placa = self._text(veic, "UF") if veic is not None else None

        vol = self._find(transp, ".//{*}vol")
        qvol = self._parse_number(self._text(vol, "qVol"), decimals=0) if vol is not None else None
        peso_b = self._parse_number(self._text(vol, "pesoB"), decimals=3) if vol is not None else None
        peso_l = self._parse_number(self._text(vol, "pesoL"), decimals=3) if vol is not None else None

        return mod_frete, placa, uf_placa, peso_b, peso_l, qvol

    # -------------------- Fall-backs & Utils --------------------
    def _coalesce_total(self, root: ET.Element, tipo: str) -> Optional[float]:
        names: List[str] = []
        if tipo in ("NFe", "NFCe"):
            names += ["vNF"]
        if tipo in ("CTe", "CTeOS"):
            names += ["vTPrest"]
        if tipo == "CF-e":
            names += ["vCFe", "vCFeLei12741", "vNF"]
        if tipo == "NFSe":
            names += ["ValorServicos", "vServ", "ValorLiquidoNfse"]
        names += ["vNF", "vCFe", "vTPrest", "ValorServicos"]
        for nm in names:
            txt = self._first_text_by_local_name(root, nm)
            val = self._parse_number(txt, decimals=2)
            if val is not None:
                return val
        return None

    def _first_text_by_local_name(self, node: ET.Element, local_name: str) -> Optional[str]:
        lname = local_name.lower()
        for el in node.iter():
            if el.tag.split("}", 1)[-1].lower() == lname and el.text:
                return el.text.strip()
        return None

    def _build_address(self, end_node: Optional[ET.Element]) -> Optional[str]:
        if end_node is None:
            return None
        parts = [
            self._text(end_node, "xLgr"),
            self._text(end_node, "nro"),
            self._text(end_node, "xCpl"),
            self._text(end_node, "xBairro"),
            self._text(end_node, "xMun"),
            self._text(end_node, "UF"),
            self._text(end_node, "CEP"),
        ]
        parts = [p for p in parts if p]
        return _norm_ws(", ".join(parts)) if parts else None

    def _build_address_nfse(self, end_node: Optional[ET.Element]) -> Optional[str]:
        if end_node is None:
            return None
        parts = [
            self._get_text_local(end_node, "Endereco")
            or self._get_text_local(end_node, "xLgr")
            or self._get_text_local(end_node, "Logradouro"),
            self._get_text_local(end_node, "Numero") or self._get_text_local(end_node, "nro"),
            self._get_text_local(end_node, "Complemento") or self._get_text_local(end_node, "xCpl"),
            self._get_text_local(end_node, "Bairro") or self._get_text_local(end_node, "xBairro"),
            self._get_text_local(end_node, "xMun") or self._get_text_local(end_node, "Municipio"),
            self._get_text_local(end_node, "UF") or self._get_text_local(end_node, "Estado"),
            self._get_text_local(end_node, "CEP") or self._get_text_local(end_node, "Cep"),
        ]
        parts = [p for p in parts if p]
        return _norm_ws(", ".join(parts)) if parts else None

    # ---------- XML helpers tolerantes ----------
    def _iter_local(self, node: ET.Element, local: str) -> Iterable[ET.Element]:
        lname = local.lower()
        for el in node.iter():
            if el.tag.split("}", 1)[-1].lower() == lname:
                yield el

    def _find(self, node: ET.Element, xpath: str) -> Optional[ET.Element]:
        m = re.fullmatch(r"\.//(?:\{\*\})?([A-Za-z0-9_:-]+)", (xpath or "").strip())
        if m:
            target = m.group(1)
            for el in self._iter_local(node, target):
                return el
            return None
        try:
            return node.find(xpath)
        except Exception:
            return None

    def _findall(self, node: ET.Element, xpath: str) -> List[ET.Element]:
        m = re.fullmatch(r"\.//(?:\{\*\})?([A-Za-z0-9_:-]+)", (xpath or "").strip())
        if m:
            target = m.group(1)
            return list(self._iter_local(node, target))
        try:
            return node.findall(xpath) or []
        except Exception:
            return []

    def _get_text(self, node: Optional[ET.Element], xpath: str) -> Optional[str]:
        if node is None:
            return None
        try:
            el = node.find(xpath)
            if el is not None and el.text:
                return el.text.strip()
        except Exception:
            return None
        return None

    def _get_text_local(self, node: Optional[ET.Element], local: str) -> Optional[str]:
        if node is None:
            return None
        for el in self._iter_local(node, local):
            if el.text:
                return el.text.strip()
        return None

    def _get_attr(self, node: ET.Element, xpath: str, attr: str) -> Optional[str]:
        try:
            el = node.find(xpath)
            if el is not None:
                val = el.get(attr)
                return val.strip() if isinstance(val, str) else None
        except Exception:
            return None
        return None

    def _text(self, node: Optional[ET.Element], tag_name: str) -> Optional[str]:
        if node is None:
            return None
        for child in node:
            tag = child.tag.split("}", 1)[-1]
            if tag.lower() == tag_name.lower():
                return (child.text or "").strip() if child.text else None
        return None

    def _text_any(self, node: Optional[ET.Element], tag_names: Iterable[str]) -> Optional[str]:
        if node is None:
            return None
        for t in tag_names:
            v = self._text(node, t)
            if v:
                return v
        return None

    def _coalesce(self, *values):
        for v in values:
            if v is not None and v != "":
                return v
        return None

    def _primeiro_valido(self, *values):
        for v in values:
            if v:
                return v
        return None

    # -------------------- Numérico robusto --------------------
    def _parse_number(self, s: Optional[str], *, decimals: Optional[int] = None) -> Optional[float]:
        if s is None:
            return None
        txt = (s or "").strip()
        if not txt:
            return None
        txt = re.sub(r"\s+", "", txt)
        if "," in txt:
            txt_norm = txt.replace(".", "").replace(",", ".")
        else:
            txt_norm = txt
        try:
            val = float(txt_norm)
        except Exception:
            return None
        if decimals is not None:
            return round(val, decimals)
        return val

    def _parse_int_safe(self, s: Optional[str]) -> Optional[int]:
        if not s:
            return None
        try:
            return int(re.sub(r"\D+", "", s))
        except Exception:
            return None

    # -------------------- Falhas --------------------
    def _registrar_xml_invalido(self, nome: str, conteudo: bytes, origem: str, motivo: str, t_start: float) -> int:
        tipo = "xml/invalido"
        status = "quarentena"
        doc_id = -1
        try:
            caminho = str(self.db.save_upload(nome, conteudo))
            doc_id = self.db.inserir_documento(
                nome_arquivo=nome,
                tipo=tipo,
                origem=origem,
                hash=self.db.hash_bytes(conteudo),
                status=status,
                data_upload=self.db.now(),
                motivo_rejeicao=motivo,
                caminho_arquivo=caminho,
                caminho_xml=caminho,  # reutiliza o mesmo caminho; evita duplicidade
            )
        finally:
            processing_time = time.time() - t_start
            if doc_id > 0:
                self.db.inserir_extracao(
                    documento_id=doc_id,
                    agente="XMLParser",
                    confianca_media=0.0,
                    texto_extraido=None,
                    linguagem="pt",
                    tempo_processamento=round(processing_time, 3),
                )
                self.db.log("ingestao_xml", usuario="sistema", detalhes=f"doc_id={doc_id}|tipo={tipo}|status={status}")
                self.metrics_agent.registrar_metrica(
                    db=self.db,
                    tipo_documento=tipo,
                    status=status,
                    confianca_media=0.0,
                    tempo_medio=processing_time,
                )
                try:
                    self.metrics_agent.registrar_desempenho_agente(
                        db=self.db,
                        doc_id=doc_id,
                        agente="XMLParser",
                        tipo_documento=tipo,
                        coverage=0.0,
                        confidence=0.0,
                        latency_s=processing_time,
                        extras={"erro": "xml_invalido", "motivo": motivo},
                    )
                except Exception:
                    pass
        return doc_id

    # -------------------- Enriquecedores auxiliares --------------------
    def _is_filled(self, v: Any) -> bool:
        if v is None:
            return False
        if isinstance(v, str):
            return bool(v.strip())
        return True

    def _norm_tokens(self, s: Optional[str]) -> Optional[str]:
        """Normaliza para tokens simples: lower, sem acentos, sem pontuação múltipla."""
        if not s:
            return None
        t = s.lower().strip()
        t = "".join(ch for ch in unicodedata.normalize("NFD", t) if unicodedata.category(ch) != "Mn")
        t = re.sub(r"[^a-z0-9\s]", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
        return t or None

    def _uf_from_cod_municipio(self, codigo: Optional[str]) -> Optional[str]:
        """
        Mapeia os 2 primeiros dígitos do código de município IBGE para UF.
        Aceita strings com/sem zeros à esquerda. Retorna sigla UF ou None.
        """
        if not codigo:
            return None
        cod = re.sub(r"\D+", "", codigo)
        if len(cod) < 2:
            return None
        d2 = cod[:2]
        mapa = {
            "11":"RO","12":"AC","13":"AM","14":"RR","15":"PA","16":"AP","17":"TO",
            "21":"MA","22":"PI","23":"CE","24":"RN","25":"PB","26":"PE","27":"AL","28":"SE","29":"BA",
            "31":"MG","32":"ES","33":"RJ","35":"SP",
            "41":"PR","42":"SC","43":"RS",
            "50":"MS","51":"MT","52":"GO","53":"DF",
        }
        return mapa.get(d2)

__all__ = ["AgenteXMLParser"]
