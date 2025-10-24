# agentes/xml_parser.py

from __future__ import annotations
import logging
import re
import time
from typing import Any, Dict, Iterable, List, Optional
from xml.etree import ElementTree as ET

from .utils import (
    _parse_date_like, _to_float_br, _only_digits, _norm_ws
)

# Logger do módulo
log = logging.getLogger("projeto_fiscal.agentes")

# ---------------------------------------------------------------------------
# Imports apenas para type hints (evita ciclos em tempo de execução)
# ---------------------------------------------------------------------------
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from banco_de_dados import BancoDeDados
    from validacao import ValidadorFiscal
    from seguranca import Cofre
    from metrics_agent import MetricsAgent  

# ---------------------------------------------------------------------------
# Classe principal
# ---------------------------------------------------------------------------

class AgenteXMLParser:
    """Interpreta XMLs fiscais (NFe, NFCe, CTe/CTeOS, MDF-e, CF-e, NFSe),
    criptografa dados sensíveis e popula o banco."""

    # -------------------- Inicialização --------------------
    def __init__(self, db: "BancoDeDados", validador: "ValidadorFiscal", cofre: "Cofre", metrics_agent: "MetricsAgent"):
        self.db = db
        self.validador = validador
        self.cofre = cofre
        self.metrics_agent = metrics_agent

    # -------------------- Público --------------------
    def processar(self, nome: str, conteudo: bytes, origem: str = "upload") -> int:
        """Processa um arquivo XML, extrai dados, criptografa e valida (com máxima resiliência)."""
        t_start = time.time()
        doc_id = -1
        status = "erro"
        tipo = "xml/desconhecido"
        motivo_rejeicao = "Falha desconhecida no processamento XML"
        confianca_media = 1.0  # XML é estruturado

        # 1) Parse seguro (UTF-8 -> Latin-1) + tolerância a BOM
        try:
            try:
                root = ET.fromstring(conteudo)
            except ET.ParseError:
                # Fallback para Latin-1 ignorando caracteres inválidos
                root = ET.fromstring(conteudo.decode("latin-1", errors="ignore").encode("utf-8"))
        except Exception as e:
            log.warning("Falha ao parsear XML '%s': %s", nome, e)
            return self._registrar_xml_invalido(nome, conteudo, origem, f"XML mal formado: {e}", t_start)

        # 2) Detecção de tipo (multi-sinal) + normalização
        try:
            tipo = self._detectar_tipo(root)
        except Exception as e:
            log.warning("Detecção de tipo falhou: %s", e)

        # 3) Extração principal (com fallbacks por tipo)
        try:
            campos: Dict[str, Any] = {}
            if tipo in ("NFe", "NFCe"):
                campos = self._extrair_campos_nfe(root)
            elif tipo in ("CTe", "CTeOS"):
                campos = self._extrair_campos_cte(root)
            elif tipo == "MDF-e":
                campos = self._extrair_campos_mdfe(root)
            elif tipo == "CF-e":
                campos = self._extrair_campos_cfe(root)
            elif tipo == "NFSe":
                campos = self._extrair_campos_nfse(root)
            else:
                campos = self._extrair_campos_generico(root)

            # Chave de acesso (multi-rotas)
            chave = (
                self._get_attr(root, ".//{*}infNFe", "Id")
                or self._get_attr(root, ".//{*}infCTe", "Id")
                or self._get_text(root, ".//{*}chNFe")
                or self._get_text(root, ".//{*}chCTe")
                or self._get_text(root, ".//infCFe",)
            )
            if chave:
                chave = _only_digits(chave)
            campos.setdefault("chave_acesso", chave)

            # Datas (consolidação robusta)
            campos["data_emissao"] = campos.get("data_emissao") or self._primeiro_valido(
                _parse_date_like(self._get_text(root, ".//{*}ide/{*}dhEmi")),
                _parse_date_like(self._get_text(root, ".//{*}ide/{*}dEmi")),
                _parse_date_like(self._get_text(root, ".//{*}ide/{*}dIniViagem")),  # MDF-e
                _parse_date_like(self._get_text(root, ".//{*}DataEmissao")),        # NFSe (algumas variações)
                _parse_date_like(self._get_text(root, ".//{*}dataEmissao")),
                _parse_date_like(self._get_text(root, ".//{*}Competencia")),        # NFSe: competência
            )

            # Normaliza nomes (tira espaços bizarros)
            for k in ("emitente_nome", "destinatario_nome", "municipio"):
                if campos.get(k):
                    campos[k] = _norm_ws(campos[k])

            # Endereço: se não veio pronto, tenta montar (logradouro+nro+compl+bairro+mun+UF+CEP)
            if not campos.get("endereco"):
                end_emit = self._find(root, ".//{*}emit/{*}enderEmit")
                endereco = self._build_address(end_emit) if end_emit is not None else None
                if not endereco:
                    # Algumas prefeituras (NFSe) usam outros nomes
                    end_nfse = self._find(root, ".//{*}PrestadorServico/{*}Endereco") or self._find(
                        root, ".//{*}Prestador/{*}Endereco"
                    )
                    endereco = self._build_address_nfse(end_nfse) if end_nfse is not None else None
                campos["endereco"] = endereco

            # Totais consolidados se faltarem (ex.: CF-e/NFSe variam)
            if campos.get("valor_total") is None:
                campos["valor_total"] = self._coalesce_total(root, tipo)

            # Criptografia de CNPJ/CPF (somente dígitos)
            for chave_id in ("emitente_cnpj", "emitente_cpf", "destinatario_cnpj", "destinatario_cpf"):
                if campos.get(chave_id):
                    dig = _only_digits(campos[chave_id])
                    campos[chave_id] = self.cofre.encrypt_text(dig) if dig else None

            # 4) Inserção do documento
            doc_id = self.db.inserir_documento(
                nome_arquivo=nome,
                tipo=tipo,
                origem=origem,
                hash=self.db.hash_bytes(conteudo),
                chave_acesso=campos.get("chave_acesso"),
                status="processando",
                data_upload=self.db.now(),
                data_emissao=campos.get("data_emissao"),
                emitente_cnpj=campos.get("emitente_cnpj"),
                emitente_cpf=campos.get("emitente_cpf"),
                emitente_nome=campos.get("emitente_nome"),
                destinatario_cnpj=campos.get("destinatario_cnpj"),
                destinatario_cpf=campos.get("destinatario_cpf"),
                destinatario_nome=campos.get("destinatario_nome"),
                inscricao_estadual=campos.get("inscricao_estadual"),
                uf=campos.get("uf"),
                municipio=campos.get("municipio"),
                endereco=campos.get("endereco"),
                valor_total=campos.get("valor_total"),
                total_produtos=campos.get("total_produtos"),
                total_servicos=campos.get("total_servicos"),
                total_icms=campos.get("total_icms"),
                total_ipi=campos.get("total_ipi"),
                total_pis=campos.get("total_pis"),
                total_cofins=campos.get("total_cofins"),
                caminho_arquivo=str(self.db.save_upload(nome, conteudo)),
            )

            # 5) Itens & Impostos (com tolerância por tipo)
            self._extrair_itens_impostos(root, doc_id, tipo)

            # 6) Validação fiscal
            try:
                self.validador.validar_documento(doc_id=doc_id, db=self.db)
            except Exception as e_val:
                log.warning("Validação fiscal falhou doc_id=%s: %s", doc_id, e_val)

            # Status final (pode ter sido atualizado pelo validador)
            final_doc_info = self.db.get_documento(doc_id) or {}
            status = final_doc_info.get("status") or ("processado" if campos.get("valor_total") is not None else "revisao_pendente")

        except Exception as e_proc:
            log.exception("Erro durante o processamento do XML (doc_id=%s): %s", doc_id, e_proc)
            status = "erro"
            motivo_rejeicao = f"Erro no processamento: {e_proc}"
            if doc_id > 0:
                self.db.atualizar_documento_campos(doc_id, status=status, motivo_rejeicao=motivo_rejeicao)

        finally:
            # 7) Registro de extração + métricas
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
                    self.db.log(
                        "ingestao_xml",
                        usuario="sistema",
                        detalhes=f"doc_id={doc_id}|tipo={tipo}|status={status}|crypto={'on' if getattr(self.cofre,'available',False) else 'off'}",
                    )
                finally:
                    self.metrics_agent.registrar_metrica(
                        db=self.db,
                        tipo_documento=tipo,
                        status=status,
                        confianca_media=confianca_media,
                        tempo_medio=processing_time,
                    )

        return doc_id

    # -------------------- Detecção de Tipo --------------------
    def _detectar_tipo(self, root: ET.Element) -> str:
        """
        Detecta o tipo por:
        1) ide/mod (modelos 55,65,57,67,58,59), 2) tag raiz, 3) padrões NFSe.
        """
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

        has_nfse_like = False
        for el in root.iter():
            lname = el.tag.split("}", 1)[-1].lower()
            if "nfse" in lname or "servico" in lname or "serviço" in lname:
                has_nfse_like = True
                break
        if has_nfse_like:
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
    def _extrair_campos_nfe(self, root: ET.Element) -> Dict[str, Any]:
        emit = self._find(root, ".//{*}emit")
        dest = self._find(root, ".//{*}dest")
        ide = self._find(root, ".//{*}ide")
        tot = self._find(root, ".//{*}total/{*}ICMSTot")
        end_emit = self._find(root, ".//{*}emit/{*}enderEmit")

        endereco = self._build_address(end_emit) if end_emit is not None else None
        # Extras úteis
        natOp = self._text(ide, "natOp")
        tpNF = self._text(ide, "tpNF")
        cUF = self._text(ide, "cUF")
        cMunFG = self._text(ide, "cMunFG")

        return {
            "emitente_nome": self._text(emit, "xNome"),
            "emitente_cnpj": self._text(emit, "CNPJ"),
            "emitente_cpf": self._text(emit, "CPF"),
            "destinatario_nome": self._text(dest, "xNome"),
            "destinatario_cnpj": self._text(dest, "CNPJ"),
            "destinatario_cpf": self._text(dest, "CPF"),
            "inscricao_estadual": self._text(emit, "IE"),
            "uf": self._text(end_emit, "UF"),
            "municipio": self._text(end_emit, "xMun"),
            "endereco": endereco,
            "data_emissao": _parse_date_like(self._text_any(ide, ("dhEmi", "dEmi"))),
            "valor_total": _to_float_br(self._text(tot, "vNF") or self._text(tot, "vCF")),
            "total_produtos": _to_float_br(self._text(tot, "vProd")),
            "total_servicos": _to_float_br(self._text(tot, "vServ")),
            "total_icms": _to_float_br(self._text(tot, "vICMS")),
            "total_ipi": _to_float_br(self._text(tot, "vIPI")),
            "total_pis": _to_float_br(self._text(tot, "vPIS")),
            "total_cofins": _to_float_br(self._text(tot, "vCOFINS")),
            # extras
            "natOp": natOp,
            "tpNF": tpNF,
            "cUF": cUF,
            "cMunFG": cMunFG,
            "chave_acesso": (_only_digits(self._get_attr(root, ".//{*}infNFe", "Id")) or None),
        }

    def _extrair_campos_cte(self, root: ET.Element) -> Dict[str, Any]:
        emit = self._find(root, ".//{*}emit")
        rem = self._find(root, ".//{*}rem")  # remetente
        dest = self._find(root, ".//{*}dest") or rem
        vprest = self._find(root, ".//{*}vPrest")
        ide = self._find(root, ".//{*}ide")
        return {
            "emitente_nome": self._text(emit, "xNome"),
            "emitente_cnpj": self._text(emit, "CNPJ"),
            "destinatario_nome": self._text(dest, "xNome"),
            "destinatario_cnpj": self._text(dest, "CNPJ"),
            "municipio": self._text(emit, "xMun") or self._text(dest, "xMun"),
            "uf": self._text(emit, "UF") or self._text(dest, "UF"),
            "data_emissao": _parse_date_like(self._text_any(ide, ("dhEmi", "dEmi"))),
            "valor_total": _to_float_br(self._text(vprest, "vTPrest")),
            "total_produtos": None,
            "total_servicos": None,
            "total_icms": None,
            "total_ipi": None,
            "total_pis": None,
            "total_cofins": None,
            "chave_acesso": (_only_digits(self._get_attr(root, ".//{*}infCTe", "Id")) or None),
        }

    def _extrair_campos_mdfe(self, root: ET.Element) -> Dict[str, Any]:
        emit = self._find(root, ".//{*}emit")
        ide = self._find(root, ".//{*}ide")
        tot = self._find(root, ".//{*}tot")
        return {
            "emitente_nome": self._text(emit, "xNome"),
            "emitente_cnpj": self._text(emit, "CNPJ"),
            "municipio": self._text(emit, "xMun"),
            "uf": self._text(emit, "UF"),
            "data_emissao": _parse_date_like(self._text_any(ide, ("dhEmi", "dEmi")) or self._text(ide, "dIniViagem")),
            "valor_total": _to_float_br(self._text(tot, "vCarga")),
            "total_produtos": None,
            "total_servicos": None,
            "total_icms": None,
            "total_ipi": None,
            "total_pis": None,
            "total_cofins": None,
        }

    def _extrair_campos_cfe(self, root: ET.Element) -> Dict[str, Any]:
        emit = self._find(root, ".//emit") or self._find(root, ".//{*}emit")
        dest = self._find(root, ".//dest") or self._find(root, ".//{*}dest")
        total = self._find(root, ".//total") or self._find(root, ".//{*}total")
        ide = self._find(root, ".//ide") or self._find(root, ".//{*}ide")
        return {
            "emitente_nome": self._text(emit, "xNome"),
            "emitente_cnpj": self._text(emit, "CNPJ"),
            "destinatario_nome": self._text(dest, "xNome"),
            "destinatario_cnpj": self._text(dest, "CNPJ"),
            "municipio": self._text(emit, "xMun"),
            "uf": self._text(emit, "UF"),
            "data_emissao": _parse_date_like(self._text_any(ide, ("dhEmi", "dEmi"))),
            "valor_total": _to_float_br(self._text(total, "vCFe") or self._text(total, "vCFeLei12741") or self._text(total, "vNF")),
            "total_produtos": None,
            "total_servicos": None,
            "total_icms": None,
            "total_ipi": None,
            "total_pis": None,
            "total_cofins": None,
        }

    def _extrair_campos_nfse(self, root: ET.Element) -> Dict[str, Any]:
        """NFSe tem muitos layouts (ABRASF e variações municipais). Usamos buscas tolerantes."""
        prest = self._find(root, ".//{*}PrestadorServico") or self._find(root, ".//{*}Prestador")
        toma = self._find(root, ".//{*}TomadorServico") or self._find(root, ".//{*}Tomador")

        # Nomes
        emit_nome = self._get_text(prest, ".//{*}RazaoSocial") or self._get_text(prest, ".//{*}NomeFantasia") or self._get_text(prest, ".//{*}xNome")
        dest_nome = self._get_text(toma, ".//{*}RazaoSocial") or self._get_text(toma, ".//{*}xNome") or self._get_text(toma, ".//{*}Nome")

        # CNPJ/CPF
        emit_cnpj = self._get_text(prest, ".//{*}Cnpj") or self._get_text(prest, ".//{*}CNPJ")
        emit_cpf = self._get_text(prest, ".//{*}Cpf") or self._get_text(prest, ".//{*}CPF")
        dest_cnpj = self._get_text(toma, ".//{*}Cnpj") or self._get_text(toma, ".//{*}CNPJ")
        dest_cpf = self._get_text(toma, ".//{*}Cpf") or self._get_text(toma, ".//{*}CPF")

        # Endereço
        end_nfse = self._find(prest, ".//{*}Endereco")
        endereco = self._build_address_nfse(end_nfse) if end_nfse is not None else None

        # Totais
        valor_total = self._coalesce(
            _to_float_br(self._get_text(root, ".//{*}ValorServicos")),
            _to_float_br(self._get_text(root, ".//{*}vServ")),
            _to_float_br(self._get_text(root, ".//{*}ValorLiquidoNfse")),
        )

        # Data de emissão/competência
        data_emissao = _parse_date_like(
            self._coalesce(
                self._get_text(root, ".//{*}DataEmissao"),
                self._get_text(root, ".//{*}dtEmissao"),
                self._get_text(root, ".//{*}dhEmi"),
                self._get_text(root, ".//{*}Competencia"),
            )
        )

        return {
            "emitente_nome": emit_nome,
            "emitente_cnpj": emit_cnpj,
            "emitente_cpf": emit_cpf,
            "destinatario_nome": dest_nome,
            "destinatario_cnpj": dest_cnpj,
            "destinatario_cpf": dest_cpf,
            "municipio": self._get_text_local(end_nfse, "xMun") or self._get_text_local(end_nfse, "Municipio"),
            "uf": self._get_text_local(end_nfse, "UF") or self._get_text_local(end_nfse, "Estado"),
            "endereco": endereco,
            "data_emissao": data_emissao,
            "valor_total": valor_total,
            "total_produtos": None,
            "total_servicos": None,
            "total_icms": None,
            "total_ipi": None,
            "total_pis": None,
            "total_cofins": None,
        }

    def _extrair_campos_generico(self, root: ET.Element) -> Dict[str, Any]:
        emit = self._find(root, ".//{*}emit") or self._find(root, ".//emit")
        end_emit = self._find(root, ".//{*}enderEmit") or self._find(root, ".//enderEmit")
        ide = self._find(root, ".//{*}ide") or self._find(root, ".//ide")

        endereco = self._build_address(end_emit) if end_emit is not None else None
        total = self._find(root, ".//{*}total") or self._find(root, ".//total")

        return {
            "emitente_nome": self._text(emit, "xNome"),
            "emitente_cnpj": self._text(emit, "CNPJ"),
            "emitente_cpf": self._text(emit, "CPF"),
            "municipio": self._text(end_emit, "xMun"),
            "uf": self._text(end_emit, "UF"),
            "endereco": endereco,
            "data_emissao": _parse_date_like(self._text_any(ide, ("dhEmi", "dEmi"))),
            "valor_total": _to_float_br(self._text(total, "vNF") or self._text(total, "vCFe") or self._text(total, "vTPrest")),
            "total_produtos": None,
            "total_servicos": None,
            "total_icms": None,
            "total_ipi": None,
            "total_pis": None,
            "total_cofins": None,
        }

    # -------------------- Itens & Impostos --------------------
    def _extrair_itens_impostos(self, root: ET.Element, doc_id: int, tipo: str) -> None:
        """
        Extrai itens/impostos com tolerância. Para NFe/NFCe/CF-e usa det/prod.
        Para CTe/MDF-e geralmente não há itens padronizados -> ignora com segurança.
        NFSe é altamente variável; na ausência de padrão, não insere itens.
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
                qnt = _to_float_br(self._text(prod, "qCom"))
                vun = _to_float_br(self._text(prod, "vUnCom"))
                vtot = _to_float_br(self._text(prod, "vProd"))
                unid = self._text(prod, "uCom")
                cprod = self._text(prod, "cProd")
                cest = self._text(prod, "CEST")

                item_id = self.db.inserir_item(
                    documento_id=doc_id,
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
                                    base_calculo=_to_float_br(bc),
                                    aliquota=_to_float_br(aliq),
                                    valor=_to_float_br(val),
                                )
                            # FCP (se existir)
                            vFCP = self._text(icms_detalhe, "vFCP")
                            pFCP = self._text(icms_detalhe, "pFCP")
                            if vFCP:
                                self.db.inserir_imposto(
                                    item_id=item_id,
                                    tipo_imposto="FCP",
                                    cst=None,
                                    origem=None,
                                    base_calculo=None,
                                    aliquota=_to_float_br(pFCP),
                                    valor=_to_float_br(vFCP),
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
                                base_calculo=_to_float_br(bc),
                                aliquota=_to_float_br(aliq),
                                valor=_to_float_br(val),
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
                                base_calculo=_to_float_br(bc),
                                aliquota=_to_float_br(aliq),
                                valor=_to_float_br(val),
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
                                base_calculo=_to_float_br(bc),
                                aliquota=_to_float_br(aliq),
                                valor=_to_float_br(val),
                            )
        except Exception as e:
            log.warning("Falha ao extrair itens/impostos (doc_id=%s, tipo=%s): %s", doc_id, tipo, e)

    # -------------------- Fallbacks e Utilidades --------------------
    def _coalesce_total(self, root: ET.Element, tipo: str) -> Optional[float]:
        candidatos = []
        if tipo in ("NFe", "NFCe"):
            candidatos += [
                self._get_text(root, ".//{*}total/{*}ICMSTot/{*}vNF"),
                self._get_text(root, ".//{*}total/{*}ICMSTot/{*}vCF"),
                self._get_text(root, ".//{*}total/{*}vNF"),
            ]
        if tipo in ("CTe", "CTeOS"):
            candidatos += [self._get_text(root, ".//{*}vPrest/{*}vTPrest")]
        if tipo == "CF-e":
            candidatos += [
                self._get_text(root, ".//{*}total/{*}vCFe"),
                self._get_text(root, ".//{*}total/{*}vCFeLei12741"),
            ]
        if tipo == "NFSe":
            candidatos += [
                self._get_text(root, ".//{*}ValorServicos"),
                self._get_text(root, ".//{*}vServ"),
                self._get_text(root, ".//{*}ValorLiquidoNfse"),
            ]

        # fallback global
        candidatos += [
            self._get_text(root, ".//*[local-name()='vNF']"),
            self._get_text(root, ".//*[local-name()='vCFe']"),
            self._get_text(root, ".//*[local-name()='vTPrest']"),
            self._get_text(root, ".//*[local-name()='ValorServicos']"),
        ]
        for c in candidatos:
            v = _to_float_br(c)
            if v is not None:
                return v
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

    # -------------------- Falhas --------------------
    def _registrar_xml_invalido(self, nome: str, conteudo: bytes, origem: str, motivo: str, t_start: float) -> int:
        tipo = "xml/invalido"
        status = "quarentena"
        doc_id = -1
        try:
            doc_id = self.db.inserir_documento(
                nome_arquivo=nome,
                tipo=tipo,
                origem=origem,
                hash=self.db.hash_bytes(conteudo),
                status=status,
                data_upload=self.db.now(),
                motivo_rejeicao=motivo,
                caminho_arquivo=str(self.db.save_upload(nome, conteudo)),
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
                self.db.log(
                    "ingestao_xml",
                    usuario="sistema",
                    detalhes=f"doc_id={doc_id}|tipo={tipo}|status={status}|crypto={'on' if getattr(self.cofre,'available',False) else 'off'}",
                )
                self.metrics_agent.registrar_metrica(
                    db=self.db,
                    tipo_documento=tipo,
                    status=status,
                    confianca_media=0.0,
                    tempo_medio=processing_time,
                )
        return doc_id


__all__ = ["AgenteXMLParser"]
