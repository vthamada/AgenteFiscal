# agentes.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple, Iterable, TYPE_CHECKING
from pathlib import Path
import io
import re
import time
import traceback
import xml.etree.ElementTree as ET
import logging
import builtins
import pandas as pd
import json

if TYPE_CHECKING:
    from banco_de_dados import BancoDeDados
    from validacao import ValidadorFiscal
    from memoria import MemoriaSessao
    from seguranca import Cofre # Importação direta para TYPE_CHECKING

# OCR / Imaging (ativados quando instalados no ambiente)
OCR_AVAILABLE = False
PDF_AVAILABLE = False
PDF_RENDERER = None  # 'pdfium' ou 'pdf2image'

try:
    import easyocr  # type: ignore
    import numpy as np  # type: ignore
    from PIL import Image  # type: ignore
    OCR_AVAILABLE = True
except Exception as e:
    OCR_AVAILABLE = False
    logging.warning(f"EasyOCR não disponível: {e}. OCR de imagens/pdf ficará desativado.")

# Renderização de PDF -> imagem (preferir pypdfium2; usar pdf2image como fallback)
try:
    import pypdfium2 as pdfium  # type: ignore
    PDF_AVAILABLE = True
    PDF_RENDERER = "pdfium"
except Exception:
    try:
        from pdf2image import convert_from_bytes  # type: ignore
        PDF_AVAILABLE = True
        PDF_RENDERER = "pdf2image"
    except Exception as e:
        PDF_AVAILABLE = False
        logging.warning(f"Nenhum renderizador de PDF disponível (pypdfium2/pdf2image). Detalhe: {e}")

# LLM
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.language_models import BaseChatModel

# Núcleo do projeto
try:
    from banco_de_dados import BancoDeDados
    from validacao import ValidadorFiscal
    from memoria import MemoriaSessao
    # --- Importação de Segurança ---
    from seguranca import Cofre, carregar_chave_do_env, CRYPTO_OK # Importa Cofre e helpers
    CORE_MODULES_AVAILABLE = True
except ImportError as e:
    CORE_MODULES_AVAILABLE = False
    logging.error(f"FALHA CRÍTICA: Não foi possível importar módulos essenciais do projeto (banco_de_dados, validacao, memoria, seguranca): {e}. O Orchestrator não funcionará corretamente.")
    # Placeholders mínimos
    BancoDeDados = type('BancoDeDados', (object,), {
        "hash_bytes": lambda s, b: "dummy_hash", "now": lambda s: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "save_upload": lambda s, n, c: Path(n), "inserir_documento": lambda s, **kwargs: -1,
        "find_documento_by_hash": lambda s, h: None, "atualizar_documento_campo": lambda s, id, k, v: None,
        "log": lambda s, *args, **kwargs: None, "inserir_extracao": lambda s, **kwargs: None,
        "inserir_item": lambda s, **kwargs: -1, "inserir_imposto": lambda s, **kwargs: None,
        "atualizar_documento_campos": lambda s, id, **kwargs: None, "get_documento": lambda s, id: {},
        "query_table": lambda s, t, **kwargs: pd.DataFrame(),
        "conn": type('Connection', (object,), {"execute": lambda s, *args: None, "commit": lambda s: None}) # Mock de conexão
    })
    # O placeholder do ValidadorFiscal agora precisa aceitar 'cofre'
    ValidadorFiscal = type('ValidadorFiscal', (object,), {
        "__init__": lambda s, cofre=None: None, # Aceita cofre no init
        "validar_documento": lambda s, **kwargs: None
        })
    MemoriaSessao = type('MemoriaSessao', (object,), {"resumo": lambda s: "Histórico indisponível.", "salvar": lambda s, **kwargs: None})
    # Placeholders de Segurança
    CRYPTO_OK = False
    Cofre = type('Cofre', (object,), {
        "__init__": lambda s, key=None: None, "available": False,
        "encrypt_text": lambda s, t: t, "decrypt_text": lambda s, t: t # Retorna o texto puro
    })
    carregar_chave_do_env = lambda var_name="APP_SECRET_KEY": None


log = logging.getLogger("projeto_fiscal.agentes")
if not log.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    log.addHandler(handler)
    log.setLevel(logging.INFO)


# ------------------------------ Utilidades comuns ------------------------------
_WHITESPACE_RE = re.compile(r"\s+", re.S)
_MONEY_CHARS_RE = re.compile(r"[^\d,\-]")

def _norm_ws(texto: str) -> str:
    """Normaliza espaços em branco em uma string."""
    return _WHITESPACE_RE.sub(" ", (texto or "").strip())

def _only_digits(s: Optional[str]) -> Optional[str]:
    """Remove todos os caracteres não numéricos de uma string."""
    return re.sub(r"\D+", "", s) if s else None

def _to_float_br(s: Optional[str]) -> Optional[float]:
    """Converte uma string formatada como número brasileiro (com ',' decimal) para float."""
    if not s: return None
    s2 = s.strip()
    s2 = _MONEY_CHARS_RE.sub("", s2)
    if s2.count(",") == 1 and (s2.count(".") == 0 or s2.rfind(",") > s2.rfind(".")):
        s2 = s2.replace(".", "").replace(",", ".")
    s2 = s2.replace(",", "")
    try: return float(s2)
    except ValueError: return None

def _parse_date_like(s: Optional[str]) -> Optional[str]:
    """Tenta converter uma string de data (DD/MM/AAAA ou AAAA-MM-DD) para o formato AAAA-MM-DD."""
    if not s: return None
    s = s.strip()
    m = re.search(r"(\d{4})[-/](\d{2})[-/](\d{2})(?:[ T]\d{2}:\d{2}:\d{2})?", s)
    if m: return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    m = re.search(r"(\d{2})[-/](\d{2})[-/](\d{4})(?:[ T]\d{2}:\d{2}:\d{2})?", s)
    if m: return f"{m.group(3)}-{m.group(2)}-{m.group(1)}"
    return None


# ------------------------------ Agente XML Parser (Integrado com Criptografia) ------------------------------
class AgenteXMLParser:
    """Interpreta XMLs fiscais, criptografa dados sensíveis e popula o banco."""
    def __init__(self, db: "BancoDeDados", validador: "ValidadorFiscal", cofre: "Cofre", metrics_agent: "MetricsAgent"):
        self.db = db
        self.validador = validador
        self.cofre = cofre
        self.metrics_agent = metrics_agent

    def processar(self, nome: str, conteudo: bytes, origem: str = "upload") -> int:
        """Processa um arquivo XML, extrai dados, criptografa e valida."""
        t_start = time.time()
        doc_id = -1
        tipo = "xml/desconhecido"
        status = "erro"
        motivo_rejeicao = "Falha desconhecida no processamento XML"
        confianca_media = 1.0  # XML é estruturado

        # 1) Parse seguro do XML (UTF-8 -> Latin-1 fallback)
        try:
            try:
                root = ET.fromstring(conteudo)
            except ET.ParseError:
                log.debug("XML '%s' falhou em UTF-8, tentando Latin-1.", nome)
                root = ET.fromstring(conteudo.decode("latin-1", errors="ignore").encode("utf-8"))
        except Exception as e:
            log.warning("Falha ao parsear XML '%s': %s", nome, e)
            motivo_rejeicao = f"XML mal formado: {e}"
            status = "quarentena"
            tipo = "xml/invalido"
            try:
                doc_id = self.db.inserir_documento(
                    nome_arquivo=nome,
                    tipo=tipo,
                    origem=origem,
                    hash=self.db.hash_bytes(conteudo),
                    status=status,
                    data_upload=self.db.now(),
                    motivo_rejeicao=motivo_rejeicao,
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
                        detalhes=f"doc_id={doc_id}|tipo={tipo}|status={status}|crypto={'on' if self.cofre.available else 'off'}",
                    )
                    self.metrics_agent.registrar_metrica(
                        db=self.db,
                        tipo_documento=tipo,
                        status=status,
                        confianca_media=0.0,
                        tempo_medio=processing_time,
                    )
            return doc_id

        # 2) Extração principal
        try:
            tipo = self._detectar_tipo(root)

            # Chave de acesso (infNFe/@Id, infCte/@Id etc.)
            chave_node = root.find(".//{*}infNFe") or root.find(".//{*}infCte")
            chave = chave_node.get("Id") if chave_node is not None else None
            if chave:
                chave = _only_digits(chave)

            # Nós-chave
            emit_node = root.find(".//{*}emit")
            dest_node = root.find(".//{*}dest")
            ide_node = root.find(".//{*}ide")
            total_node = root.find(".//{*}total")
            ender_emit_node = emit_node.find(".//{*}enderEmit") if emit_node is not None else None

            # Emitente/Destinatário - CNPJ/CPF/Nome
            emit_cnpj = self._find_text(emit_node, "CNPJ")
            emit_cpf = self._find_text(emit_node, "CPF")
            dest_cnpj = self._find_text(dest_node, "CNPJ")
            dest_cpf = self._find_text(dest_node, "CPF")

            emit_nome = _norm_ws(self._find_text(emit_node, "xNome") or "")
            dest_nome = _norm_ws(self._find_text(dest_node, "xNome") or "")

            # IE, UF, Município
            inscricao_estadual = self._find_text(emit_node, "IE")
            uf = self._find_text(ender_emit_node, "UF") if ender_emit_node is not None else None
            municipio = self._find_text(ender_emit_node, "xMun") if ender_emit_node is not None else None

            # Datas
            d_emis = self._find_text(ide_node, "dhEmi") or self._find_text(ide_node, "dEmi")

            # Totais (ICMSTot)
            valor_total_str = None
            tot_prod = tot_serv = tot_icms = tot_ipi = tot_pis = tot_cofins = None
            if total_node is not None:
                icms_tot_node = total_node.find(".//{*}ICMSTot")
                if icms_tot_node is not None:
                    valor_total_str = (
                        self._find_text(icms_tot_node, "vNF")
                        or self._find_text(icms_tot_node, "vCF")
                    )
                    # novos totais
                    tot_prod = _to_float_br(self._find_text(icms_tot_node, "vProd"))
                    tot_serv = _to_float_br(self._find_text(icms_tot_node, "vServ"))
                    tot_icms = _to_float_br(self._find_text(icms_tot_node, "vICMS"))
                    tot_ipi = _to_float_br(self._find_text(icms_tot_node, "vIPI"))
                    tot_pis = _to_float_br(self._find_text(icms_tot_node, "vPIS"))
                    tot_cofins = _to_float_br(self._find_text(icms_tot_node, "vCOFINS"))
                elif tipo == "CTe":
                    vprest_node = total_node.find(".//{*}vPrest")
                    if vprest_node is not None:
                        valor_total_str = self._find_text(vprest_node, "vTPrest")

            valor_total = _to_float_br(valor_total_str)

            # Criptografia de identificadores (apenas dígitos)
            emit_cnpj_dig = _only_digits(emit_cnpj)
            emit_cpf_dig = _only_digits(emit_cpf)
            dest_cnpj_dig = _only_digits(dest_cnpj)
            dest_cpf_dig = _only_digits(dest_cpf)

            emit_cnpj_enc = self.cofre.encrypt_text(emit_cnpj_dig) if emit_cnpj_dig else None
            emit_cpf_enc = self.cofre.encrypt_text(emit_cpf_dig) if emit_cpf_dig else None
            dest_cnpj_enc = self.cofre.encrypt_text(dest_cnpj_dig) if dest_cnpj_dig else None
            dest_cpf_enc = self.cofre.encrypt_text(dest_cpf_dig) if dest_cpf_dig else None

            # 3) Inserção do documento
            doc_id = self.db.inserir_documento(
                nome_arquivo=nome,
                tipo=tipo,
                origem=origem,
                hash=self.db.hash_bytes(conteudo),
                chave_acesso=chave,
                status="processando",
                data_upload=self.db.now(),
                data_emissao=_parse_date_like(d_emis),
                emitente_cnpj=emit_cnpj_enc,
                emitente_cpf=emit_cpf_enc,
                emitente_nome=emit_nome,
                destinatario_cnpj=dest_cnpj_enc,
                destinatario_cpf=dest_cpf_enc,
                destinatario_nome=dest_nome,
                inscricao_estadual=inscricao_estadual,
                uf=uf,
                municipio=municipio,
                valor_total=valor_total,
                total_produtos=tot_prod,
                total_servicos=tot_serv,
                total_icms=tot_icms,
                total_ipi=tot_ipi,
                total_pis=tot_pis,
                total_cofins=tot_cofins,
                caminho_arquivo=str(self.db.save_upload(nome, conteudo)),
            )

            # 4) Itens e impostos do XML
            self._extrair_itens_impostos(root, doc_id)

            # 5) Validação fiscal
            self.validador.validar_documento(doc_id=doc_id, db=self.db)

            # 6) Status final
            final_doc_info = self.db.get_documento(doc_id)
            status = final_doc_info.get("status", "processado") if final_doc_info else "erro"

        except Exception as e_proc:
            log.exception("Erro durante o processamento do XML (doc_id %s): %s", doc_id, e_proc)
            motivo_rejeicao = f"Erro no processamento: {e_proc}"
            status = "erro"
            if doc_id > 0:
                self.db.atualizar_documento_campo(doc_id, "status", status)
                self.db.atualizar_documento_campo(doc_id, "motivo_rejeicao", motivo_rejeicao)
        finally:
            # 7) Registro de extração + métricas
            processing_time = time.time() - t_start
            if doc_id > 0:
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
                    detalhes=f"doc_id={doc_id}|tipo={tipo}|status={status}|crypto={'on' if self.cofre.available else 'off'}",
                )
                self.metrics_agent.registrar_metrica(
                    db=self.db,
                    tipo_documento=tipo,
                    status=status,
                    confianca_media=confianca_media,
                    tempo_medio=processing_time,
                )
        return doc_id

    def _detectar_tipo(self, root: ET.Element) -> str:
        tag = root.tag.lower()
        if "}" in tag:
            tag = tag.split("}", 1)[1]
        if "cte" in tag:
            return "CTe"
        if "mdfe" in tag:
            return "MDF-e"
        if "nfe" in tag:
            return "NFe"
        is_nfce = root.find(".//{*}ide/{*}mod") is not None and root.find(".//{*}ide/{*}mod").text == "65"
        if is_nfce:
            return "NFCe"
        if "cfe" in tag:
            return "CF-e"  # SAT
        return "xml/desconhecido"

    def _find_text(self, node: Optional[ET.Element], tag_name: str) -> Optional[str]:
        if node is None:
            return None
        for child in node:
            tag = child.tag
            if "}" in tag:
                tag = tag.split("}", 1)[1]
            if tag.lower() == tag_name.lower():
                return child.text
        return None

    def _find_text_any(self, node: Optional[ET.Element], tag_names: Iterable[str]) -> Optional[str]:
        if node is None:
            return None
        for s in tag_names:
            v = self._find_text(node, s)
            if v:
                return v
        return None

    def _extrair_itens_impostos(self, root: ET.Element, doc_id: int) -> None:
        """Extrai itens/impostos do XML (NFe/CTe etc.) e insere nas tabelas relacionadas."""
        for det in root.findall(".//{*}det"):
            prod = det.find(".//{*}prod")
            imposto = det.find(".//{*}imposto")
            if prod is None:
                continue

            desc = self._find_text(prod, "xProd")
            ncm = self._find_text(prod, "NCM")
            cfop = self._find_text(prod, "CFOP")
            qnt = _to_float_br(self._find_text(prod, "qCom"))
            vun = _to_float_br(self._find_text(prod, "vUnCom"))
            vtot = _to_float_br(self._find_text(prod, "vProd"))
            unid = self._find_text(prod, "uCom")
            cprod = self._find_text(prod, "cProd")
            cest = self._find_text(prod, "CEST")

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
                # ICMS
                icms_node = imposto.find(".//{*}ICMS") or imposto.find(".//{*}ICMSUFDest")
                if icms_node:
                    icms_detalhe = next(iter(icms_node), None)
                    if icms_detalhe is not None:
                        cst = self._find_text(icms_detalhe, "CST") or self._find_text(icms_detalhe, "CSOSN")
                        orig = self._find_text(icms_detalhe, "orig")
                        bc = self._find_text_any(icms_detalhe, ["vBC", "vBCST", "vBCSTRet", "vBCUFDest"])
                        aliq = self._find_text_any(icms_detalhe, ["pICMS", "pICMSST", "pICMSSTRet", "pICMSUFDest", "pICMSInter", "pICMSInterPart"])
                        val = self._find_text_any(icms_detalhe, ["vICMS", "vICMSST", "vICMSSTRet", "vICMSUFDest", "vICMSPartDest", "vICMSPartRemet"])
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

                # IPI
                ipi_node = imposto.find(".//{*}IPI")
                ipi_trib_node = ipi_node.find(".//{*}IPITrib") if ipi_node is not None else None
                if ipi_trib_node:
                    cst = self._find_text(ipi_trib_node, "CST")
                    bc = self._find_text(ipi_trib_node, "vBC")
                    aliq = self._find_text(ipi_trib_node, "pIPI")
                    val = self._find_text(ipi_trib_node, "vIPI")
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
                pis_node = imposto.find(".//{*}PIS")
                pis_aliq_node = pis_node.find(".//{*}PISAliq") if pis_node is not None else None
                if pis_aliq_node:
                    cst = self._find_text(pis_aliq_node, "CST")
                    bc = self._find_text(pis_aliq_node, "vBC")
                    aliq = self._find_text(pis_aliq_node, "pPIS")
                    val = self._find_text(pis_aliq_node, "vPIS")
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
                cofins_node = imposto.find(".//{*}COFINS")
                cofins_aliq_node = cofins_node.find(".//{*}COFINSAliq") if cofins_node is not None else None
                if cofins_aliq_node:
                    cst = self._find_text(cofins_aliq_node, "CST")
                    bc = self._find_text(cofins_aliq_node, "vBC")
                    aliq = self._find_text(cofins_aliq_node, "pCOFINS")
                    val = self._find_text(cofins_aliq_node, "vCOFINS")
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

# ------------------------------ Agente OCR (EasyOCR + pypdfium2 com fallback) ------------------------------
class AgenteOCR:
    def __init__(self):
        self.ocr_ok = OCR_AVAILABLE
        self.pdf_ok = PDF_AVAILABLE
        self.reader = None

        if self.ocr_ok:
            try:
                # GPU=False para funcionar no Streamlit Cloud/CPU
                self.reader = easyocr.Reader(["pt", "en"], gpu=False)  # cacheia modelos
                log.info("OCR (EasyOCR) disponível.")
            except Exception as e:
                self.ocr_ok = False
                log.warning(f"Falha ao inicializar EasyOCR: {e}")
        else:
            log.warning("OCR (EasyOCR) NÃO disponível.")

        if self.pdf_ok:
            if PDF_RENDERER == "pdfium":
                log.info("Renderizador PDF: pypdfium2.")
            elif PDF_RENDERER == "pdf2image":
                log.info("Renderizador PDF: pdf2image.")
        else:
            log.warning("Nenhum renderizador de PDF disponível.")

    def reconhecer(self, nome: str, conteudo: bytes) -> Tuple[str, float]:
        t_start = time.time()
        ext = Path(nome).suffix.lower()
        texto = ""
        conf = 0.0

        try:
            if ext == ".pdf":
                if not (self.ocr_ok and self.pdf_ok):
                    raise RuntimeError("OCR PDF indisponível (EasyOCR ou renderizador de PDF ausente).")
                texto, conf = self._ocr_pdf(conteudo)
            elif ext in {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}:
                if not self.ocr_ok:
                    raise RuntimeError("OCR imagem indisponível (EasyOCR ausente).")
                texto, conf = self._ocr_imagem(conteudo)
            else:
                raise ValueError(f"Extensão não suportada: {ext}")
        except Exception as e:
            log.error("Erro OCR '%s': %s", nome, e)
            raise e
        finally:
            log.info("OCR '%s' (conf: %.2f) em %.2fs", nome, conf, time.time() - t_start)
        return texto, conf

    def _ocr_imagem(self, conteudo: bytes) -> Tuple[str, float]:
        """OCR para imagens com EasyOCR."""
        if not (self.ocr_ok and self.reader):
            return "", 0.0
        try:
            img = Image.open(io.BytesIO(conteudo)).convert("RGB")
            np_img = np.array(img)
            results = self.reader.readtext(np_img, detail=1, paragraph=False)
            texto = " ".join([r[1] for r in results]) if results else ""
            confs = [float(r[2]) for r in results] if results else []
            media = float(np.mean(confs)) if confs else 0.0
            return texto, round(media, 2)
        except Exception as e:
            log.error(f"Erro OCR imagem (EasyOCR): {e}")
            return "", 0.0

    def _ocr_pdf(self, conteudo: bytes) -> Tuple[str, float]:
        """OCR para PDFs: renderiza cada página -> PIL e aplica EasyOCR."""
        if not (self.ocr_ok and self.reader and self.pdf_ok):
            return "", 0.0
        try:
            full_text = []
            confs_all: List[float] = []

            if PDF_RENDERER == "pdfium":
                pdf = pdfium.PdfDocument(io.BytesIO(conteudo))
                for page in pdf:
                    # scale=2 dá ~144 dpi * 2 (boa leitura) sem estourar memória
                    pil_img = page.render(scale=2).to_pil().convert("RGB")
                    np_img = np.array(pil_img)
                    results = self.reader.readtext(np_img, detail=1, paragraph=False)
                    full_text.append(" ".join([r[1] for r in results]) if results else "")
                    confs_all.extend([float(r[2]) for r in results] if results else [])
            else:  # PDF_RENDERER == "pdf2image"
                images = convert_from_bytes(conteudo, dpi=220)
                for pil_img in images:
                    pil_img = pil_img.convert("RGB")
                    np_img = np.array(pil_img)
                    results = self.reader.readtext(np_img, detail=1, paragraph=False)
                    full_text.append(" ".join([r[1] for r in results]) if results else "")
                    confs_all.extend([float(r[2]) for r in results] if results else [])

            texto_final = "\n\n--- Page Break ---\n\n".join([t for t in full_text if t])
            media_conf = float(np.mean(confs_all)) if confs_all else 0.0
            return texto_final, round(media_conf, 2)
        except Exception as e:
            log.error(f"Erro OCR PDF (EasyOCR): {e}")
            return "", 0.0

# ------------------------------ Agente NLP (Extração de Itens OCR Mantida) ------------------------------
class AgenteNLP:
    RE_CNPJ = re.compile(r"\b(\d{2}\.?\d{3}\.?\d{3}/?\d{4}-?\d{2}|\d{14})\b")
    RE_CPF = re.compile(r"\b(\d{3}\.?\d{3}\.?\d{3}-?\d{2}|\d{11})\b")
    RE_IE = re.compile(
        r"\b(?:IE|I\.E\.|INSC(?:RI[ÇC][ÃA]O)?\sESTADUAL)[:\s\-]*([A-Z0-9.\-/]{5,20})\b",
        re.I,
    )
    RE_UF = re.compile(
        r"\b(AC|AL|AP|AM|BA|CE|DF|ES|GO|MA|MT|MS|MG|PA|PB|PR|PE|PI|RJ|RN|RS|RO|RR|SC|SP|SE|TO)\b"
    )
    RE_VALOR_TOTAL = re.compile(
        r"\b(?:VALOR\s+TOTAL\s+DA\s+NOTA|VALOR\s+TOTAL|TOTAL\s+DA\s+NOTA)\s*[:\-]?\s*R?\$\s*([\d.,]+)\b",
        re.I,
    )
    RE_DATA_EMISSAO = re.compile(
        r"\b(?:DATA\s+(?:DE\s+)?EMISS[ÃA]O|EMITIDO\s+EM)\s*[:\-]?\s*(\d{2,4}[-/]\d{2}[-/]\d{2,4})\b",
        re.I,
    )
    RE_ITEM_LINHA = re.compile(
        r"^(?:\d+\s+)?(?P<desc>.+?)\s+(?P<unid>[A-Z]{1,3})\s+(?P<qtd>[\d.,]+)\s+(?P<vun>[\d.,]+)\s+(?P<vtot>[\d.,]+)$",
        re.I | re.M,
    )
    RE_NCM_ITEM = re.compile(r"NCM[:\s]*(\d{8})", re.I)
    RE_CFOP_ITEM = re.compile(r"CFOP[:\s]*(\d{4})", re.I)

    def extrair_campos(self, texto: str) -> Dict[str, Any]:
        """Extrai campos principais a partir do texto OCR (retorna CNPJ/CPF sem criptografia)."""
        t_norm = _norm_ws(texto)
        cnpjs = [_only_digits(m) for m in self.RE_CNPJ.findall(t_norm)]
        cpfs = [_only_digits(m) for m in self.RE_CPF.findall(t_norm)]

        emit_cnpj_cpf = (cnpjs[0] if cnpjs else None) or (cpfs[0] if cpfs else None)
        dest_cnpj_cpf = None
        if len(cnpjs) > 1:
            dest_cnpj_cpf = cnpjs[1]
        elif cnpjs and cpfs:
            dest_cnpj_cpf = cpfs[0]
        elif len(cpfs) > 1:
            dest_cnpj_cpf = cpfs[1]

        m_ie = self.RE_IE.search(t_norm)
        ie = m_ie.group(1).strip() if m_ie else None

        endereco_match = self._match_after(t_norm, ["endereço", "endereco", "logradouro", "rua"], max_len=150)
        municipio_match = self._match_after(t_norm, ["município", "municipio", "cidade"], max_len=80)

        uf = None
        uf_match = self.RE_UF.search(endereco_match[-10:]) if endereco_match else None
        if not uf_match and municipio_match:
            uf_match = self.RE_UF.search(municipio_match[-10:])
        if not uf_match:
            uf_match = self.RE_UF.search(t_norm[-100:])
        uf = uf_match.group(1) if uf_match else None

        razao = self._match_after(t_norm, ["razão social", "razao social", "nome", "emitente"], max_len=100)

        m_valor = self.RE_VALOR_TOTAL.search(t_norm)
        valor_total = _to_float_br(m_valor.group(1)) if m_valor else None

        m_data = self.RE_DATA_EMISSAO.search(t_norm)
        data_emissao = _parse_date_like(m_data.group(1)) if m_data else None

        itens_extraidos, impostos_extraidos = self._extrair_itens_impostos_ocr(texto)

        # Retorna CNPJ/CPF *sem* criptografia aqui (apenas dígitos). A criptografia é feita no Orchestrator.
        return {
            "emitente_cnpj": _only_digits(emit_cnpj_cpf) if len(emit_cnpj_cpf or "") >= 14 else None,
            "emitente_cpf": _only_digits(emit_cnpj_cpf) if len(emit_cnpj_cpf or "") == 11 else None,
            "destinatario_cnpj": _only_digits(dest_cnpj_cpf) if len(dest_cnpj_cpf or "") >= 14 else None,
            "destinatario_cpf": _only_digits(dest_cnpj_cpf) if len(dest_cnpj_cpf or "") == 11 else None,
            "inscricao_estadual": ie,
            "emitente_nome": razao,
            "endereco": endereco_match,
            "uf": uf,
            "municipio": municipio_match,
            "valor_total": valor_total,
            "data_emissao": data_emissao,
            "itens_ocr": itens_extraidos,
            "impostos_ocr": impostos_extraidos,
        }

    def _match_after(self, texto: str, labels: List[str], max_len: int = 80, max_dist: int = 50) -> Optional[str]:
        """Procura o valor que vem após uma das labels fornecidas."""
        texto_lower = texto.lower()
        best_match = None
        min_pos = float("inf")

        for lab in labels:
            lab_lower = lab.lower()
            idx = texto_lower.find(lab_lower)
            if idx == -1:
                continue

            start_value_idx = -1
            end_label_idx = idx + len(lab_lower)
            for i in range(end_label_idx, min(end_label_idx + max_dist, len(texto))):
                if texto[i] in ":|-;\n" or (
                    i == end_label_idx and texto[i].isspace() and not texto[i + 1 : i + 2].isalnum()
                ):
                    start_value_idx = i + 1
                    break

            if start_value_idx == -1:
                continue

            end_idx = texto.find("\n", start_value_idx)
            if end_idx == -1:
                end_idx = len(texto)

            for next_lab in labels:
                next_idx = texto.find(next_lab, start_value_idx, end_idx)
                if next_idx != -1:
                    end_idx = next_idx

            value = texto[start_value_idx:end_idx].strip()
            value = re.sub(r"\s*[|;/-].*$", "", value).strip()
            if value and idx < min_pos:
                min_pos = idx
                best_match = value[:max_len]

        return best_match

    def _extrair_itens_impostos_ocr(self, texto_original: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Extrai linhas de itens e impostos a partir do texto OCR (heurística)."""
        itens: List[Dict[str, Any]] = []
        impostos: List[Dict[str, Any]] = []

        linhas = texto_original.splitlines()
        inicio_itens = -1
        fim_itens = len(linhas)

        header_keywords = ["DESCRIÇÃO", "QTD", "UNIT", "TOTAL", "PRODUTO"]
        for i, linha in enumerate(linhas):
            linha_upper = linha.upper()
            if any(kw in linha_upper for kw in header_keywords):
                if "TOTAL" in linha_upper and len(linha.split()) < 4:
                    continue
                inicio_itens = i + 1
                break

        if inicio_itens == -1:
            log.warning("Início da seção de itens OCR não encontrado.")
            return [], []

        total_keywords = ["TOTAL DOS PRODUTOS", "VALOR TOTAL", "SUBTOTAL"]
        for i in range(inicio_itens, len(linhas)):
            if any(kw in linhas[i].upper() for kw in total_keywords):
                fim_itens = i
                break

        item_idx_counter = 0
        for i in range(inicio_itens, fim_itens):
            linha = linhas[i].strip()
            if not linha:
                continue

            match = self.RE_ITEM_LINHA.search(linha)
            if match:
                item_data = match.groupdict()
                item = {
                    "descricao": _norm_ws(item_data.get("desc", "")),
                    "unidade": item_data.get("unid"),
                    "quantidade": _to_float_br(item_data.get("qtd")),
                    "valor_unitario": _to_float_br(item_data.get("vun")),
                    "valor_total": _to_float_br(item_data.get("vtot")),
                    "ncm": None,
                    "cfop": None,
                }

                linha_seguinte = linhas[i + 1].strip() if (i + 1) < fim_itens else ""
                ncm_match = self.RE_NCM_ITEM.search(linha) or self.RE_NCM_ITEM.search(linha_seguinte)
                cfop_match = self.RE_CFOP_ITEM.search(linha) or self.RE_CFOP_ITEM.search(linha_seguinte)
                if ncm_match:
                    item["ncm"] = ncm_match.group(1)
                if cfop_match:
                    item["cfop"] = cfop_match.group(1)

                itens.append(item)

                icms_match = re.search(r"ICMS.*?(\d+,\d{2})\s*%", linha_seguinte, re.I)
                if icms_match:
                    impostos.append(
                        {
                            "item_idx": item_idx_counter,
                            "tipo_imposto": "ICMS",
                            "aliquota": _to_float_br(icms_match.group(1)),
                        }
                    )

                item_idx_counter += 1

        log.info(f"AgenteNLP: Extraídos {len(itens)} itens via OCR.")
        return itens, impostos

# ------------------------------ Agente Analítico (Sandbox Mantido Aqui - Código Completo) ------------------------------
class SecurityException(Exception): pass
ALLOWED_IMPORTS = {"pandas", "numpy", "matplotlib", "plotly"}

def _restricted_import(name: str, *args, **kwargs):
    """Função de import restrita para o sandbox."""
    # Permite submodulos dos imports permitidos (ex: matplotlib.pyplot)
    root_module = name.split(".")[0]
    if root_module not in ALLOWED_IMPORTS:
        raise SecurityException(f"Importação proibida: {name}")
    return builtins.__import__(name, *args, **kwargs)

# SAFE_BUILTINS final e correto, definido uma vez no nível do módulo.
SAFE_BUILTINS = {k: getattr(builtins, k) for k in (
    "abs", "all", "any", "bool", "dict", "enumerate", "float", "int", "isinstance",
    "len", "list", "max", "min", "print", "range", "round", "set", "sorted",
    "str", "sum", "tuple", "type", "zip",
)}
SAFE_BUILTINS["__import__"] = _restricted_import # Sobrescreve o import padrão

class AgenteAnalitico:
    """Gera e executa código Python via LLM com auto-correção."""
    def __init__(self, llm: BaseChatModel, memoria: MemoriaSessao):
        self.llm = llm
        self.memoria = memoria
        self.last_code: str = "" # Armazena o último código tentado

    def _prompt_inicial(self, catalog: Dict[str, pd.DataFrame]) -> SystemMessage:
        """ Constrói o prompt inicial para a geração de código (Código Completo). """
        schema_lines = []
        # Garante que 'documentos' seja a tabela exemplo se existir
        example_table_name = 'documentos' if 'documentos' in catalog else next(iter(catalog.keys()), 'tabela_exemplo')

        for t, df in catalog.items():
             schema_lines.append(f"- Tabela `{t}` ({df.shape[0]} linhas): Colunas: `{', '.join(map(str, df.columns))}`")
        schema = "\n".join(schema_lines) or "- (Nenhum dado carregado)"
        history = self.memoria.resumo()

        prompt = f"""
        Você é um agente de análise de dados de elite expert em Python. Sua tarefa é gerar código Python robusto e bem formatado para uma função 'solve'.

        **REGRAS CRÍTICAS DE EXECUÇÃO:**
        1.  **CRÍTICO:** Todas as declarações de `import` DEVEM estar DENTRO da função `solve`.
        2.  Imports permitidos: {', '.join(ALLOWED_IMPORTS)}. NENHUM OUTRO será permitido pelo sandbox.
        3.  Use APENAS funções built-in seguras. O sandbox bloqueará outras. Funções como `open()`, `eval()`, `exec()` são PROIBIDAS.
        4.  Acesse dados via `catalog['nome_tabela']`. **SEMPRE** use `.copy()` ao pegar um DataFrame do catalog (ex: `df = catalog['documentos'].copy()`).
        5.  Retorne uma tupla: `(texto: str, tabela: pd.DataFrame | None, figura: plt.Figure | go.Figure | None)`.
        6.  Seja DEFENSIVO: Use `pd.to_numeric(df['coluna'], errors='coerce')` para conversões numéricas. Use `.fillna(0)` ou `.dropna()` apropriadamente. Verifique se as colunas existem antes de usá-las (`if 'coluna' in df.columns:`).
        7.  GRÁFICOS: Prefira `plotly.express as px`. Use `fig.update_layout(width=800, height=500)` para ajustar o tamanho. Para `matplotlib.pyplot as plt`, use `fig, ax = plt.subplots(figsize=(10, 6))` e `plt.tight_layout()` antes de retornar `fig`.
        8.  Se retornar uma `tabela` (DataFrame), o `texto` deve ser um resumo ou título, NÃO a tabela convertida para string (`.to_string()`).
        9.  Manipule datas com `pd.to_datetime(df['coluna_data'], errors='coerce')`.
        10. Os dados de CNPJ/CPF estarão CRIPTOGRAFADOS. Você não pode usá-los para filtros de igualdade ou agrupamento direto. Use outras colunas (UF, tipo, data, valores).

        **ESQUEMA DISPONÍVEL:**
        {schema}

        **HISTÓRICO RECENTE (para contexto):**
        {history}

        **ESTRUTURA OBRIGATÓRIA DA FUNÇÃO:**
        ```python
        def solve(catalog, question):
            # Imports AQUI dentro da função
            import pandas as pd
            import numpy as np
            import matplotlib.pyplot as plt
            import plotly.express as px
            import plotly.graph_objects as go

            # Variáveis de resultado padrão
            text_output = "Análise não pôde ser concluída."
            table_output = None
            figure_output = None

            # Exemplo de acesso seguro aos dados
            if '{example_table_name}' in catalog:
                df = catalog['{example_table_name}'].copy()
            else:
                 return ("Tabela '{example_table_name}' não encontrada no catálogo.", None, None)

            # --- SEU CÓDIGO ROBUSTO DE ANÁLISE VEM AQUI ---
            # Lembre-se das verificações, cópias e tratamento de erros
            try:
                # Exemplo: df['valor_total'] = pd.to_numeric(df['valor_total'], errors='coerce').fillna(0)
                # ... sua lógica ...
                text_output = "# Título da Análise\nDescrição dos resultados..."
                # table_output = df_resultado # Se houver tabela
                # figure_output = fig # Se houver gráfico

            except Exception as e:
                # Captura erros DENTRO do try-except para retornar mensagem amigável
                # Importante para não vazar stack trace completo para o usuário final
                import traceback # Import local permitido dentro do except
                error_details = traceback.format_exc(limit=1) # Pega a última linha do traceback
                text_output = f"Erro durante a análise: {{type(e).__name__}}: {{e}}\nDetalhe: ...{{error_details.splitlines()[-1]}}"
                # Log interno pode ter o traceback completo se necessário (não implementado aqui)

            # Retorna a tupla (texto, tabela, figura)
            return (text_output, table_output, figure_output)
        ```
        Gere APENAS o código Python completo da função `solve`, nada antes ou depois.
        """
        # Usar strip() para remover possíveis linhas em branco antes/depois do prompt
        return SystemMessage(content=prompt.strip())

    def _prompt_correcao(self, failed_code: str, error_message: str) -> SystemMessage:
        """ Constrói o prompt para a correção de código (Código Completo). """
        prompt = f"""
        O código Python gerado anteriormente falhou durante a execução no sandbox seguro. Analise o erro e o código, e reescreva APENAS a função `solve` corrigida.

        **ERRO OCORRIDO:**
        {error_message}

        **CÓDIGO QUE FALHOU:**
        ```python
        {failed_code}
        ```

        **INSTRUÇÕES PARA CORREÇÃO:**
        1.  Verifique se todos os `import` estão DENTRO da função `solve`.
        2.  Confirme que apenas os imports permitidos ({', '.join(ALLOWED_IMPORTS)}) foram usados.
        3.  Certifique-se de que `.copy()` foi usado ao acessar DataFrames do `catalog`.
        4.  Revise o acesso a colunas e tratamento de tipos (use `pd.to_numeric`, `pd.to_datetime` com `errors='coerce'`). Verifique a existência de colunas.
        5.  Garanta que funções built-in não seguras não foram usadas.
        6.  Lembre-se que CNPJ/CPF estão criptografados e não podem ser usados para filtros de igualdade.
        7.  Mantenha a estrutura de retorno `(texto, tabela, figura)`.
        8.  Inclua tratamento de erro `try...except Exception as e:` dentro da função `solve` para retornar mensagens de erro amigáveis.

        Reescreva APENAS o código Python completo da função `solve` corrigida.
        """
        # Usar strip() para remover possíveis linhas em branco antes/depois do prompt
        return SystemMessage(content=prompt.strip())

    def _gerar_codigo(self, pergunta: str, catalog: Dict[str, pd.DataFrame]) -> str:
        """ Gera a primeira versão do código (Código Completo). """
        sys = self._prompt_inicial(catalog)
        hum = HumanMessage(content=f"Pergunta do usuário: {pergunta}")
        try:
            resp = self.llm.invoke([sys, hum]).content.strip()
            # Extrai o bloco de código Python, mesmo com texto antes/depois
            code_match = re.search(r"```python\n(.*?)\n```", resp, re.DOTALL)
            if code_match:
                code = code_match.group(1).strip()
            else:
                 # Se não encontrar o bloco, assume que a resposta inteira é o código (menos provável)
                 log.warning("LLM não retornou um bloco de código python formatado. Tentando usar a resposta inteira.")
                 code = resp.strip()
                 # Adiciona uma verificação básica se a resposta parece ser código Python
                 if not code.startswith("def solve(catalog, question):"):
                     raise ValueError("Resposta do LLM não parece conter a função 'solve' esperada.")

            self.last_code = code # Armazena antes de retornar
            return code
        except Exception as e:
            log.error(f"Erro ao invocar LLM para gerar código: {e}")
            raise RuntimeError(f"Falha na comunicação com LLM: {e}") from e


    def _corrigir_codigo(self, failed_code: str, erro: str) -> str:
        """ Gera uma versão corrigida do código (Código Completo). """
        sys = self._prompt_correcao(failed_code, erro)
        hum = HumanMessage(content="Por favor, corrija a função `solve` baseada no erro e no código fornecido.")
        try:
            resp = self.llm.invoke([sys, hum]).content.strip()
            # Extrai o bloco de código Python da correção
            code_match = re.search(r"```python\n(.*?)\n```", resp, re.DOTALL)
            if code_match:
                code = code_match.group(1).strip()
            else:
                 log.warning("LLM não retornou um bloco de código python formatado na CORREÇÃO. Tentando usar a resposta inteira.")
                 code = resp.strip()
                 if not code.startswith("def solve(catalog, question):"):
                     raise ValueError("Resposta de correção do LLM não parece conter a função 'solve' esperada.")

            self.last_code = code # Armazena antes de retornar
            return code
        except Exception as e:
            log.error(f"Erro ao invocar LLM para corrigir código: {e}")
            # Retorna o código original que falhou se a correção falhar
            raise RuntimeError(f"Falha na comunicação com LLM durante correção: {e}") from e

    def _executar_sandbox(self, code: str, pergunta: str, catalog: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """ Executa o código em sandbox seguro com escopo unificado e builtins restritos (Código Completo). """
        scope = {"__builtins__": SAFE_BUILTINS}
        try:
            scope = {"__builtins__": SAFE_BUILTINS.copy()}
            scope["__builtins__"]["__import__"] = _restricted_import
            exec(code, scope)
        except SecurityException as se:
            log.error(f"Tentativa de execução insegura bloqueada: {se}")
            raise se # Relança a exceção de segurança
        except Exception as e_comp:
            log.error(f"Erro de compilação/execução inicial do código gerado:\n{code}\nErro: {e_comp}")
            raise SyntaxError(f"Erro ao executar código gerado: {e_comp}") from e_comp

        if "solve" not in scope or not callable(scope["solve"]):
            raise RuntimeError("A função `solve` não foi definida corretamente no código gerado.")

        solve_fn = scope["solve"]
        t0 = time.time()
        try:
            # Chama a função 'solve' dentro de um try-except para pegar erros de runtime dela
            texto, tabela, fig = solve_fn({k: v for k, v in catalog.items()}, pergunta)
        except Exception as e_runtime:
             log.error(f"Erro durante a execução da função 'solve':\n{code}\nErro: {e_runtime}")
             # Tenta obter um traceback mais útil
             tb_str = traceback.format_exc(limit=3)
             raise RuntimeError(f"Erro na execução da lógica de 'solve': {e_runtime}\n{tb_str}") from e_runtime

        dt = time.time() - t0

        # Validações básicas do retorno
        if not isinstance(texto, str):
            log.warning(f"Retorno 'texto' não é string, é {type(texto)}. Convertendo.")
            texto = str(texto)
        if tabela is not None and not isinstance(tabela, pd.DataFrame):
             log.warning(f"Retorno 'tabela' não é DataFrame ou None, é {type(tabela)}. Ignorando.")
             tabela = None
        if fig is not None:
             try:
                 # Import dinamicamente para evitar erro de análise estática quando plotly não estiver instalado
                 import importlib
                 # Tenta importar a classe Figure do matplotlib se disponível
                 matplotlib_figure = None
                 try:
                     matplotlib_figure = importlib.import_module("matplotlib.figure")
                 except Exception:
                     matplotlib_figure = None
                 # Tenta importar plotly.graph_objects se disponível
                 go = None
                 try:
                     go = importlib.import_module("plotly.graph_objects")
                 except Exception:
                     go = None

                 is_matplotlib_fig = False
                 if matplotlib_figure is not None:
                     try:
                         is_matplotlib_fig = isinstance(fig, matplotlib_figure.Figure)
                     except Exception:
                         is_matplotlib_fig = False

                 is_plotly_fig = False
                 if go is not None:
                     try:
                         is_plotly_fig = isinstance(fig, go.Figure)
                     except Exception:
                         is_plotly_fig = False

                 if not (is_matplotlib_fig or is_plotly_fig):
                     log.warning(f"'figura' não é suportada: {type(fig)}")
                     fig = None
             except Exception:
                 log.warning("Libs gráficas ausentes para validar figura.")
                 fig = None

        return {"texto": texto, "tabela": tabela, "figuras": [fig] if fig is not None else [], "duracao_s": round(dt, 3), "code": code}

    def responder(self, pergunta: str, catalog: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """ Orquestra a geração, execução e auto-correção do código (Código Completo). """
        max_retries = 2; code_to_run = ""
        try:
            code_to_run = self._gerar_codigo(pergunta, catalog)
            if not code_to_run.strip(): raise ValueError("LLM não gerou nenhum código.")
            for attempt in range(max_retries + 1):
                try:
                    log.info(f"Tentativa {attempt + 1} de executar código para: '{pergunta[:50]}...'")
                    out = self._executar_sandbox(code_to_run, pergunta, catalog)
                    self.memoria.salvar(pergunta, out.get("texto", ""), duracao_s=out.get("duracao_s", 0.0))
                    out["agent_name"] = f"AgenteAnalitico (Tentativa {attempt + 1})"
                    out["summary"] = f"Executou código com sucesso para: '{pergunta[:50]}...'"
                    log.info(f"Execução bem-sucedida na tentativa {attempt + 1}.")
                    return out
                except (SyntaxError, SecurityException, RuntimeError, TypeError, ValueError, KeyError, IndexError, AttributeError) as e1: # Captura erros esperados da execução
                    error_message = f"{type(e1).__name__}: {e1}"; traceback_str = traceback.format_exc(limit=3)
                    log.warning(f"Falha na tentativa {attempt + 1} para '{pergunta[:50]}...': {error_message}\n{traceback_str}")
                    if attempt < max_retries:
                        log.info(f"Solicitando correção ao LLM (tentativa {attempt + 2}/{max_retries + 1}).")
                        try:
                            code_to_run = self._corrigir_codigo(code_to_run, error_message)
                            if not code_to_run.strip(): raise ValueError("LLM não gerou nenhum código de correção.")
                        except Exception as e_corr: # Erro ao tentar corrigir
                             log.error(f"Erro ao tentar obter correção do LLM: {e_corr}")
                             # Decide se tenta executar o código antigo mais uma vez ou falha direto
                             # Neste caso, vamos falhar direto se a correção falhar.
                             raise RuntimeError("Falha ao obter correção do LLM.") from e_corr
                    else:
                        log.error(f"Número máximo de tentativas ({max_retries + 1}) excedido para '{pergunta[:50]}...'. Falha final.")
                        raise e1 # Relança a última exceção da execução
                except Exception as e_inesperado: # Captura outros erros inesperados
                    log.exception(f"Erro inesperado durante a execução/correção (tentativa {attempt + 1}): {e_inesperado}")
                    raise e_inesperado # Relança para o handler final

        except Exception as e_final: # Captura exceções da geração, correção ou a última da execução
            traceback_str = traceback.format_exc(limit=3)
            log.error(f"Falha irrecuperável no AgenteAnalitico para '{pergunta[:50]}...': {type(e_final).__name__}: {e_final}\n{traceback_str}")
            summary = f"Falha final na geração ou auto-correção para: '{pergunta[:50]}...'"
            self.memoria.salvar(pergunta, f"Erro: {type(e_final).__name__}: {e_final}", duracao_s=0.0)
            return {
                "texto": f"Ocorreu um erro irrecuperável ao tentar analisar sua pergunta após {max_retries + 1} tentativas. Detalhe: {type(e_final).__name__}: {e_final}",
                "tabela": None, "figuras": [], "duracao_s": 0.0,
                "code": self.last_code or code_to_run or "", # Retorna o último código tentado
                "agent_name": "AgenteAnalitico (Falha Irrecuperável)", "summary": summary
            }


# ------------------------------ Orchestrator (Com Filtros, Modo Seguro, Reprocessamento e Métricas) ------------------------------
@dataclass
class Orchestrator:
    """ Coordena o pipeline de processamento, criptografa dados e gerencia análise. """
    db: "BancoDeDados"
    validador: "ValidadorFiscal"
    memoria: "MemoriaSessao"
    llm: Optional[BaseChatModel] = None
    cofre: "Cofre" = None
    metrics_agent: "MetricsAgent" = None # Agente de Métricas

    def __post_init__(self):
        """Inicializa os agentes, Cofre e Metrics."""
        if not CORE_MODULES_AVAILABLE:
             log.error("Orchestrator: Módulos CORE ausentes.")
             self.cofre = Cofre(key=None)
             self.validador = ValidadorFiscal(cofre=self.cofre)
        else:
             chave_criptografia = carregar_chave_do_env("APP_SECRET_KEY")
             self.cofre = Cofre(key=chave_criptografia)
             if self.cofre.available: log.info("Criptografia ATIVA.")
             else:
                 log.warning("Criptografia INATIVA.")
                 if not CRYPTO_OK: log.warning("-> Lib 'cryptography' ausente.")
                 if not chave_criptografia: log.warning("-> Var APP_SECRET_KEY ausente/inválida.")
             
             # Instancia o Validador PRIMEIRO, passando o cofre
             self.validador = ValidadorFiscal(cofre=self.cofre)

        # Instancia o Agente de Métricas
        self.metrics_agent = MetricsAgent()

        # Passa o validador, cofre e metrics_agent para o XMLAgent
        self.xml_agent = AgenteXMLParser(self.db, self.validador, self.cofre, self.metrics_agent)
        self.ocr_agent = AgenteOCR()
        self.nlp_agent = AgenteNLP()
        self.analitico = AgenteAnalitico(self.llm, self.memoria) if self.llm else None

        if self.llm: log.info("Agente Analítico INICIALIZADO.")
        else: log.warning("Agente Analítico NÃO inicializado (LLM ausente).")


    def ingestir_arquivo(self, nome: str, conteudo: bytes, origem: str = "web") -> int:
        """ Processa um arquivo, retornando o ID do documento. """
        t_start=time.time(); doc_id=-1; status="erro"; motivo="?"; doc_hash=self.db.hash_bytes(conteudo); ext=Path(nome).suffix.lower(); tipo_doc = ext.strip('.') or 'binario'
        try:
            existing_id=self.db.find_documento_by_hash(doc_hash)
            if existing_id: log.info("Doc '%s' (hash %s...) já existe ID %d. Ignorando.", nome, doc_hash[:8], existing_id); return existing_id
            
            if ext==".xml":
                tipo_doc = "xml" # Será refinado pelo parser
                doc_id=self.xml_agent.processar(nome, conteudo, origem)
            elif ext in {".pdf",".jpg",".jpeg",".png",".tif",".tiff",".bmp"}: 
                doc_id=self._processar_midias(nome, conteudo, origem)
                tipo_doc = Path(nome).suffix.lower().strip('.') # Tipo é definido dentro de _processar_midias
            else:
                motivo=f"Extensão '{ext}' não suportada."; status="quarentena"; log.warning("Arquivo '%s' rejeitado: %s", nome, motivo)
                tipo_doc = "desconhecido"
                doc_id=self.db.inserir_documento(nome_arquivo=nome, tipo=tipo_doc, origem=origem, hash=doc_hash, status=status, data_upload=self.db.now(), motivo_rejeicao=motivo)
                # Registra métrica para arquivo não suportado
                self.metrics_agent.registrar_metrica(db=self.db, tipo_documento=tipo_doc, status=status, confianca_media=0.0, tempo_medio=(time.time()-t_start))
            
            if doc_id > 0:
                doc_info=self.db.get_documento(doc_id);
                if doc_info: status = doc_info.get("status", status) # Pega o status final pós-processamento
        
        except Exception as e:
            log.exception("Falha ingestão '%s': %s", nome, e); motivo=f"Erro: {str(e)}"; status="erro"
            try: # Garante registro do erro
                existing_id_on_error=self.db.find_documento_by_hash(doc_hash)
                if existing_id_on_error: doc_id=existing_id_on_error; self.db.atualizar_documento_campos(doc_id, status="erro", motivo_rejeicao=motivo)
                elif doc_id > 0: self.db.atualizar_documento_campos(doc_id, status="erro", motivo_rejeicao=motivo)
                else: doc_id=self.db.inserir_documento(nome_arquivo=nome, tipo=tipo_doc, origem=origem, hash=doc_hash, status="erro", data_upload=self.db.now(), motivo_rejeicao=motivo)
                # Registra métrica de falha geral
                self.metrics_agent.registrar_metrica(db=self.db, tipo_documento=tipo_doc, status="erro", confianca_media=0.0, tempo_medio=(time.time()-t_start))
            except Exception as db_err: log.error("Erro CRÍTICO ao registrar falha '%s': %s", nome, db_err); return -1
        
        finally: 
            # O registro de métricas agora é feito dentro dos métodos processar/midias
            log.info("Ingestão '%s' (ID: %d, Status: %s, Crypto: %s) em %.2fs", nome, doc_id, status, 'on' if self.cofre.available else 'off', time.time()-t_start)
        
        return doc_id

    def _processar_midias(self, nome: str, conteudo: bytes, origem: str) -> int:
        """ Processa PDF/Imagem via OCR/NLP com criptografia e registro de métricas. """
        doc_id = -1; t_start_proc = time.time(); status_final = "erro"; conf = 0.0; tipo_doc = Path(nome).suffix.lower().strip('.')
        try:
            doc_id = self.db.inserir_documento(
                nome_arquivo=nome, tipo=tipo_doc, origem=origem,
                hash=self.db.hash_bytes(conteudo), status="processando",
                data_upload=self.db.now(), caminho_arquivo=str(self.db.save_upload(nome, conteudo))
            )
            log.info("Processando mídia '%s' (doc_id %d)", nome, doc_id)
            texto=""; t_start_ocr=time.time()
            try: # Etapa OCR
                texto, conf = self.ocr_agent.reconhecer(nome, conteudo); ocr_time = time.time()-t_start_ocr
                log.info(f"OCR doc_id {doc_id}: conf={conf:.2f}, time={ocr_time:.2f}s.")
                self.db.inserir_extracao(documento_id=doc_id, agente="OCRAgent", confianca_media=float(conf), texto_extraido=texto[:50000]+("..."if len(texto)>50000 else ""), linguagem="pt", tempo_processamento=round(ocr_time,3))
            except Exception as e_ocr:
                log.error(f"Falha OCR doc_id {doc_id}: {e_ocr}"); status_final="erro"; self.db.atualizar_documento_campos(doc_id, status=status_final, motivo_rejeicao=f"Falha OCR: {e_ocr}"); self.db.log("ocr_erro","sistema",f"doc_id={doc_id}|erro={e_ocr}"); 
                raise # Relança para o finally registrar a métrica de erro

            if texto:
                try: # Etapa NLP + Save + Validate
                    t_start_nlp = time.time(); log.info("NLP doc_id %d...", doc_id)
                    campos_nlp = self.nlp_agent.extrair_campos(texto); nlp_time = time.time()-t_start_nlp; log.info(f"NLP doc_id {doc_id} em {nlp_time:.2f}s.")
                    itens_ocr=campos_nlp.pop("itens_ocr",[]); impostos_ocr=campos_nlp.pop("impostos_ocr",[])
                    # --- CRIPTOGRAFIA ANTES DE ATUALIZAR ---
                    campos_para_criptografar=["emitente_cnpj","destinatario_cnpj","emitente_cpf","destinatario_cpf"]
                    for campo in campos_para_criptografar:
                        if campo in campos_nlp and campos_nlp[campo]:
                            if getattr(self.cofre, "available", False):
                                campos_nlp[campo] = self.cofre.encrypt_text(campos_nlp[campo])
                            else:
                                log.warning(f"Criptografia desativada - campo '{campo}' salvo em texto puro.")
                    # ----------------------------------------
                    self.db.atualizar_documento_campos(doc_id, **campos_nlp) # Salva dados (cripto ou não)
                    if itens_ocr:
                        log.info(f"Salvando {len(itens_ocr)} itens OCR doc_id {doc_id}."); item_id_map={}
                        for idx, item_data in enumerate(itens_ocr): item_id=self.db.inserir_item(documento_id=doc_id,**item_data); item_id_map[idx]=item_id
                        if impostos_ocr:
                            log.info(f"Salvando {len(impostos_ocr)} impostos OCR doc_id {doc_id}.")
                            for imposto_data in impostos_ocr:
                                item_ocr_idx=imposto_data.pop("item_idx",-1)
                                if item_ocr_idx in item_id_map: self.db.inserir_imposto(item_id=item_id_map[item_ocr_idx],**imposto_data)
                                else: log.warning(f"Imposto OCR s/ item idx={item_ocr_idx}, doc_id {doc_id}.")
                    log.info("Validação doc_id %d pós OCR/NLP...", doc_id)
                    self.validador.validar_documento(doc_id=doc_id, db=self.db)
                    doc_info_after=self.db.get_documento(doc_id); status_depois=doc_info_after.get("status") if doc_info_after else "erro"
                    if status_depois=="revisao_pendente": status_final="revisao_pendente"; log.info(f"Doc {doc_id} para revisão (validação).")
                    else: limiar_conf=0.60; status_final="processado" if conf>=limiar_conf else "revisao_pendente"; log.info(f"Doc {doc_id} {status_final} (Conf OCR: {conf:.2f}).")
                except Exception as e_nlp:
                    log.exception(f"Falha NLP/Save doc_id {doc_id}: {e_nlp}"); status_final="erro"
                    self.db.atualizar_documento_campos(doc_id, status=status_final, motivo_rejeicao=f"Falha NLP/Save: {e_nlp}"); self.db.log("nlp_erro","sistema",f"doc_id={doc_id}|erro={e_nlp}")
                    raise # Relança para o finally registrar a métrica de erro
            else: 
                status_final="revisao_pendente"; log.warning(f"OCR s/ texto doc_id {doc_id} (conf:{conf:.2f}). Revisão pendente."); self.db.atualizar_documento_campos(doc_id, status=status_final, motivo_rejeicao="OCR não extraiu texto.")
            
            if status_final != "erro": self.db.atualizar_documento_campo(doc_id,"status",status_final)
            self.db.log("ingestao_midias","sistema",f"doc_id={doc_id}|conf={conf:.2f}|status={status_final}|crypto={'on' if self.cofre.available else 'off'}")
        
        except Exception as e_outer:
             log.exception(f"Falha geral mídia '{nome}': {e_outer}")
             status_final = "erro" # Garante que o status final seja erro
             if doc_id>0:
                 try: self.db.atualizar_documento_campos(doc_id, status="erro", motivo_rejeicao=f"Falha geral: {e_outer}")
                 except Exception as db_err_f: log.error(f"Erro CRÍTICO ao marcar erro final doc_id {doc_id}: {db_err_f}")
             # Não retorna, deixa o finally registrar a métrica
        finally:
            # --- Registra Métrica no Finally ---
            processing_time = time.time() - t_start_proc
            if doc_id > 0: # Garante que o doc_id foi criado
                self.metrics_agent.registrar_metrica(
                    db=self.db, tipo_documento=tipo_doc, status=status_final,
                    confianca_media=conf, tempo_medio=processing_time
                )
        return doc_id

    def _executar_fast_query(self, pergunta: str, catalog: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Simula o FastQueryAgent . Executa consultas determinísticas simples
        usando Pandas, sem LLM.
        """
        log.info(f"Modo Seguro: Tentando responder com FastQuery: '{pergunta}'")
        pergunta_lower = pergunta.lower()
        df_docs = catalog.get("documentos")
        df_itens = catalog.get("itens")
        
        texto_resposta = "Não foi possível responder a esta pergunta com o 'Modo Seguro' (FastQueryAgent)."
        tabela_resposta = None
        
        try:
            if "contar" in pergunta_lower and "documentos" in pergunta_lower:
                total_docs = len(df_docs) if df_docs is not None else 0
                texto_resposta = f"Total de documentos (processados/revisados) no escopo atual: **{total_docs}**."
            
            elif ("valor total" in pergunta_lower or "soma" in pergunta_lower) and "documentos" in pergunta_lower:
                if df_docs is not None and 'valor_total' in df_docs.columns:
                    # Garante que a coluna é numérica antes de somar
                    df_docs['valor_total_num'] = pd.to_numeric(df_docs['valor_total'], errors='coerce').fillna(0)
                    soma_total = df_docs['valor_total_num'].sum()
                    texto_resposta = f"O valor total somado dos documentos no escopo é: **R$ {soma_total:,.2f}**."
                else:
                    texto_resposta = "A coluna 'valor_total' não está disponível nos documentos para soma."

            elif ("top 5" in pergunta_lower or "top 10" in pergunta_lower) and "valor" in pergunta_lower:
                 n_top = 10 if "top 10" in pergunta_lower else 5
                 if df_docs is not None and 'valor_total' in df_docs.columns and 'emitente_nome' in df_docs.columns:
                     df_docs['valor_total_num'] = pd.to_numeric(df_docs['valor_total'], errors='coerce').fillna(0)
                     top_fornecedores = df_docs.groupby('emitente_nome')['valor_total_num'].sum().nlargest(n_top).reset_index()
                     top_fornecedores = top_fornecedores.rename(columns={"valor_total_num": "Valor Total"})
                     texto_resposta = f"Top {n_top} Emitentes por Valor Total:"
                     tabela_resposta = top_fornecedores
                 else:
                     texto_resposta = f"Não foi possível calcular o Top {n_top} (colunas 'emitente_nome' ou 'valor_total' ausentes)."
            
            # Adicionar outras regras determinísticas aqui (ex: contagem por UF)
            
        except Exception as e:
            log.error(f"Erro no FastQuery: {e}")
            texto_resposta = f"Ocorreu um erro ao tentar executar a consulta rápida: {e}"

        return {
            "texto": texto_resposta,
            "tabela": tabela_resposta,
            "figuras": [],
            "duracao_s": 0.01, # Simulado, pois é rápido
            "code": f"# FastQuery (Determinístico)\n# Pergunta: {pergunta}",
            "agent_name": "FastQueryAgent (Modo Seguro)"
        }

    def responder_pergunta(self, pergunta: str, scope_filters: Optional[Dict[str, Any]] = None, safe_mode: bool = False) -> Dict[str, Any]:
        """ Delega a pergunta analítica para o AgenteAnalitico ou FastQueryAgent. (Código Completo com Filtros) """
        if not self.analitico and not safe_mode: # Precisa do analítico se não for modo seguro
            log.error("Agente Analítico não inicializado.")
            return {"texto": "Erro: Agente analítico não configurado.", "tabela": None, "figuras": []}
        
        catalog: Dict[str, pd.DataFrame] = {}
        try:
            # Constrói cláusula WHERE base
            where_conditions = ["(status = 'processado' OR status = 'revisado')"]
            
            # --- Aplica Filtros de Escopo ---
            if scope_filters:
                uf_escopo = scope_filters.get('uf')
                if uf_escopo and isinstance(uf_escopo, str):
                    where_conditions.append(f"uf = '{uf_escopo.upper()}'")
                
                tipo_escopo = scope_filters.get('tipo')
                if tipo_escopo and isinstance(tipo_escopo, list) and len(tipo_escopo) > 0:
                    tipos_sql = ", ".join([f"'{t}'" for t in tipo_escopo])
                    where_conditions.append(f"tipo IN ({tipos_sql})")
            
            where_clause = " AND ".join(where_conditions)
            log.info(f"Carregando catálogo para LLM com filtro: {where_clause}")
            # ---------------------------------
            
            catalog["documentos"] = self.db.query_table("documentos", where=where_clause)
            
            if not catalog["documentos"].empty:
                 doc_ids = tuple(catalog["documentos"]['id'].unique().tolist()); doc_ids_sql = ', '.join(map(str, doc_ids))
                 catalog["itens"] = self.db.query_table("itens", where=f"documento_id IN ({doc_ids_sql})")
                 if not catalog["itens"].empty:
                    item_ids = tuple(catalog["itens"]['id'].unique().tolist()); item_ids_sql = ', '.join(map(str, item_ids))
                    catalog["impostos"] = self.db.query_table("impostos", where=f"item_id IN ({item_ids_sql})")
                 else: catalog["impostos"] = pd.DataFrame(columns=['id','item_id','tipo_imposto','cst','origem','base_calculo','aliquota','valor'])
            else:
                 catalog["itens"] = pd.DataFrame(columns=['id','documento_id','descricao','ncm','cest','cfop','quantidade','unidade','valor_unitario','valor_total','codigo_produto'])
                 catalog["impostos"] = pd.DataFrame(columns=['id','item_id','tipo_imposto','cst','origem','base_calculo','aliquota','valor'])
        except Exception as e:
            log.exception(f"Falha ao montar catálogo com filtros: {e}"); return {"texto": f"Erro ao carregar dados com filtros: {e}", "tabela": None, "figuras": []}
        
        if catalog["documentos"].empty:
            log.info("Nenhum documento válido para análise (considerando filtros)."); return {"texto": "Não há documentos válidos (status 'processado' ou 'revisado') que correspondam aos filtros selecionados para análise.", "tabela": None, "figuras": []}
        
        # --- Lógica de Modo Seguro (FastQueryAgent) ---
        if safe_mode:
            # Tenta responder com lógica determinística
            return self._executar_fast_query(pergunta, catalog)
        # ---------------------------------------------

        # Se safe_mode=False, continua para o AgenteAnalitico (LLM)
        if not self.analitico:
             log.error("Modo Seguro desativado, mas Agente Analítico (LLM) não está configurado.")
             return {"texto": "Erro: Modo Seguro desativado, mas o Agente Analítico (LLM) não está configurado.", "tabela": None, "figuras": []}

        log.info("Iniciando AgenteAnalitico (Filtros: %s): '%s...'", scope_filters, pergunta[:100])
        # Passa o catálogo filtrado para o agente
        return self.analitico.responder(pergunta, catalog)


    def revalidar_documento(self, documento_id: int) -> Dict[str, Any]:
        """ Aciona a revalidação de um documento específico. (Código Completo) """
        try:
            doc = self.db.get_documento(documento_id)
            if not doc: log.warning("Revalidar: Doc ID %d não encontrado.", documento_id); return {"ok": False, "mensagem": f"Documento ID {documento_id} não encontrado."}
            status_anterior = doc.get('status')
            log.info("Iniciando revalidação doc_id %d (status: %s)", documento_id, status_anterior)
            
            # O ValidadorFiscal já foi inicializado com o Cofre
            self.validador.validar_documento(doc_id=documento_id, db=self.db, force_revalidation=True)
            
            doc_depois = self.db.get_documento(documento_id); novo_status = doc_depois.get('status') if doc_depois else 'desconhecido'
            self.db.log("revalidacao", "usuario_sistema", f"doc_id={documento_id}|status_anterior={status_anterior}|status_novo={novo_status}|timestamp={self.db.now()}")
            log.info("Revalidação doc_id %d concluída. Novo status: %s", documento_id, novo_status)
            return {"ok": True, "mensagem": f"Documento revalidado. Novo status: {novo_status}."}
        except Exception as e:
            log.exception("Falha ao revalidar doc_id %d: %s", documento_id, e); return {"ok": False, "mensagem": f"Falha ao revalidar: {e}"}

    def reprocessar_documento(self, documento_id: int) -> Dict[str, Any]:
        """
        Deleta um documento e seus dados associados, e tenta re-ingerir o arquivo original.
        (Implementação da Tela 5)
        """
        log.info(f"Iniciando reprocessamento para doc_id {documento_id}...")
        try:
            doc_original = self.db.get_documento(documento_id)
            if not doc_original:
                return {"ok": False, "mensagem": f"Documento ID {documento_id} não encontrado."}
            
            caminho_arquivo_str = doc_original.get('caminho_arquivo')
            if not caminho_arquivo_str:
                return {"ok": False, "mensagem": f"Documento ID {documento_id} não possui caminho de arquivo original salvo."}

            caminho_arquivo = Path(caminho_arquivo_str)
            if not caminho_arquivo.exists():
                return {"ok": False, "mensagem": f"Arquivo original '{caminho_arquivo_str}' não encontrado no disco."}

            # Lê o conteúdo do arquivo original
            nome_arquivo_original = doc_original.get('nome_arquivo', caminho_arquivo.name)
            origem_original = doc_original.get('origem', 'reprocessamento')
            conteudo_original = caminho_arquivo.read_bytes()

            # Deleta o documento antigo do banco de dados
            # Isso deve deletar em cascata itens, impostos, extrações (ON DELETE CASCADE)
            self.db.conn.execute("DELETE FROM documentos WHERE id = ?", (documento_id,))
            self.db.conn.commit()
            log.info(f"Documento ID {documento_id} e dados associados deletados do banco.")
            
            # (Opcional: deletar o arquivo físico antigo de /data/uploads se o hash for mudar ou para limpar)
            # Por enquanto, vamos manter o arquivo original no lugar.

            try:
                if caminho_arquivo.exists():
                    caminho_arquivo.unlink()
                    log.info(f"Arquivo físico '{caminho_arquivo}' removido durante reprocessamento.")
            except Exception as e_clean:
                log.warning(f"Não foi possível excluir arquivo físico '{caminho_arquivo}': {e_clean}")

            # Re-ingere o arquivo
            log.info(f"Re-ingerindo arquivo '{nome_arquivo_original}'...")
            novo_doc_id = self.ingestir_arquivo(
                nome=nome_arquivo_original,
                conteudo=conteudo_original,
                origem=origem_original
            )
            
            # Verifica se a re-ingestão falhou (ex: hash já existe - o que não deveria acontecer)
            if novo_doc_id == documento_id:
                 msg = f"Reprocessamento falhou. O documento ID {documento_id} não pôde ser deletado e re-inserido."
                 log.error(msg)
                 return {"ok": False, "mensagem": msg}

            novo_doc_info = self.db.get_documento(novo_doc_id)
            novo_status = novo_doc_info.get('status') if novo_doc_info else 'desconhecido'

            msg = f"Reprocessamento concluído. ID antigo: {documento_id}. Novo ID: {novo_doc_id} (Status: {novo_status})."
            log.info(msg)
            self.db.log("reprocessamento", "usuario_sistema", f"doc_id_antigo={documento_id}|doc_id_novo={novo_doc_id}|status={novo_status}")
            return {"ok": True, "mensagem": msg, "novo_id": novo_doc_id}

        except Exception as e:
            log.exception(f"Falha ao reprocessar doc_id {documento_id}: {e}")
            return {"ok": False, "mensagem": f"Falha ao reprocessar: {e}"}

# ------------------------------ Agente de Métricas (Implementação) ------------------------------

class MetricsAgent:
    """
    Agente responsável por calcular e persistir métricas de performance e qualidade.
    Agora também agrega dados fiscais e de volume de processamento.
    """
    def __init__(self):
        log.info("MetricsAgent inicializado.")

    def registrar_metrica(self, db: BancoDeDados, tipo_documento: str, status: str, 
                          confianca_media: float, tempo_medio: float):
        """
        Registra ou atualiza métricas agregadas na tabela 'metricas'.
        Além das métricas anteriores, coleta dados de ICMS/IPI/PIS/COFINS médios.
        """
        try:
            # 1. Taxas básicas
            taxa_revisao = 1.0 if status == 'revisao_pendente' else 0.0
            taxa_erro = 1.0 if status in ('erro', 'quarentena') else 0.0

            # 2. Busca dados recentes do tipo de documento (para enriquecer métricas)
            df_docs = db.query_table("documentos", where=f"tipo = '{tipo_documento}'")
            if df_docs.empty:
                # Nenhum documento do tipo ainda — salva métrica simples
                db.inserir_metrica(
                    tipo_documento=tipo_documento,
                    acuracia_media=confianca_media,
                    taxa_revisao=taxa_revisao,
                    taxa_erro=taxa_erro,
                    tempo_medio=tempo_medio
                )
                log.debug(f"Métrica simples registrada: tipo={tipo_documento}, status={status}")
                return

            # 3. Cálculo de agregados
            total_docs = len(df_docs)
            media_conf = confianca_media
            media_tempo = tempo_medio
            media_valor_total = df_docs["valor_total"].mean() if "valor_total" in df_docs else 0.0

            # 4. Agregados fiscais
            media_icms = df_docs["total_icms"].mean() if "total_icms" in df_docs else 0.0
            media_ipi = df_docs["total_ipi"].mean() if "total_ipi" in df_docs else 0.0
            media_pis = df_docs["total_pis"].mean() if "total_pis" in df_docs else 0.0
            media_cofins = df_docs["total_cofins"].mean() if "total_cofins" in df_docs else 0.0

            # 5. Relações percentuais fiscais
            taxa_icms_media = (media_icms / media_valor_total * 100) if media_valor_total > 0 else 0.0
            taxa_ipi_media = (media_ipi / media_valor_total * 100) if media_valor_total > 0 else 0.0
            taxa_pis_media = (media_pis / media_valor_total * 100) if media_valor_total > 0 else 0.0
            taxa_cofins_media = (media_cofins / media_valor_total * 100) if media_valor_total > 0 else 0.0

            # 6. Prepara campo meta_json (JSON agregando KPIs)
            meta = {
                "total_documentos": total_docs,
                "media_valor_total": media_valor_total,
                "media_icms": media_icms,
                "media_ipi": media_ipi,
                "media_pis": media_pis,
                "media_cofins": media_cofins,
                "taxa_icms_media": taxa_icms_media,
                "taxa_ipi_media": taxa_ipi_media,
                "taxa_pis_media": taxa_pis_media,
                "taxa_cofins_media": taxa_cofins_media,
            }

            # 7. Insere métrica consolidada
            db.inserir_metrica(
                tipo_documento=tipo_documento,
                acuracia_media=confianca_media,
                taxa_revisao=taxa_revisao,
                taxa_erro=taxa_erro,
                tempo_medio=media_tempo,
                meta_json=json.dumps(meta, ensure_ascii=False)
            )

            log.debug(
                f"Métrica registrada: tipo={tipo_documento}, conf={confianca_media:.2f}, "
                f"tempo={tempo_medio:.2f}s, impostos médios ICMS={media_icms:.2f}, PIS={media_pis:.2f}"
            )

        except Exception as e:
            log.error(f"Falha ao registrar métrica: {e}")
            # Evita quebrar pipeline principal