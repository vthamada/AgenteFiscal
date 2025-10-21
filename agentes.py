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

if TYPE_CHECKING:
    from banco_de_dados import BancoDeDados
    from validacao import ValidadorFiscal
    from memoria import MemoriaSessao

# OCR / Imaging (ativados quando instalados no ambiente)
try:
    import pytesseract  # type: ignore
    from PIL import Image, ImageOps, ImageFilter # type: ignore
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    logging.warning("Bibliotecas Pillow ou pytesseract não encontradas. Funcionalidade de OCR de imagens desativada.")

try:
    from pdf2image import convert_from_bytes # type: ignore
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    logging.warning("Biblioteca pdf2image não encontrada. Funcionalidade de OCR de PDF desativada.")

# LLM
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.language_models import BaseChatModel

# Núcleo do projeto
# Importações com tratamento de erro e placeholders mínimos
try:
    from banco_de_dados import BancoDeDados
    from validacao import ValidadorFiscal
    from memoria import MemoriaSessao
    CORE_MODULES_AVAILABLE = True
except ImportError as e:
    CORE_MODULES_AVAILABLE = False
    logging.error(f"FALHA CRÍTICA: Não foi possível importar módulos essenciais do projeto (banco_de_dados, validacao, memoria): {e}. O Orchestrator não funcionará corretamente.")
    # Define placeholders mínimos para o código não quebrar na inicialização, mas funcionalidade será limitada.
    BancoDeDados = type('BancoDeDados', (object,), {
        "hash_bytes": lambda s, b: "dummy_hash",
        "now": lambda s: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "save_upload": lambda s, n, c: Path(n),
        "inserir_documento": lambda s, **kwargs: -1,
        "find_documento_by_hash": lambda s, h: None,
        "atualizar_documento_campo": lambda s, id, k, v: None,
        "log": lambda s, *args, **kwargs: None,
        "inserir_extracao": lambda s, **kwargs: None,
        "inserir_item": lambda s, **kwargs: -1,
        "inserir_imposto": lambda s, **kwargs: None,
        "atualizar_documento_campos": lambda s, id, **kwargs: None,
        "get_documento": lambda s, id: {},
        "query_table": lambda s, t, **kwargs: pd.DataFrame() # Retorna DF vazio
    })
    ValidadorFiscal = type('ValidadorFiscal', (object,), {"validar_documento": lambda s, **kwargs: None})
    MemoriaSessao = type('MemoriaSessao', (object,), {"resumo": lambda s: "Histórico indisponível (módulo de memória não carregado).", "salvar": lambda s, **kwargs: None})


log = logging.getLogger("projeto_fiscal.agentes")
if not log.handlers:
    # Configuração de logging mais robusta
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    log.addHandler(handler)
    log.setLevel(logging.INFO)


# ------------------------------ Utilidades comuns ------------------------------
_WHITESPACE_RE = re.compile(r"\s+", re.S)
_MONEY_CHARS_RE = re.compile(r"[^\d,\-]") # Para limpar valores monetários antes de _to_float_br

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
    # Limpa caracteres não numéricos, exceto vírgula, ponto e hífen (para negativos)
    s2 = _MONEY_CHARS_RE.sub("", s2)
    # Heurística para tratar milhares com ponto e decimal com vírgula
    if s2.count(",") == 1 and (s2.count(".") == 0 or s2.rfind(",") > s2.rfind(".")):
        s2 = s2.replace(".", "").replace(",", ".")
    # Remove vírgulas restantes (caso fossem separadores de milhar sem ponto)
    s2 = s2.replace(",", "")
    try: return float(s2)
    except ValueError: return None

def _parse_date_like(s: Optional[str]) -> Optional[str]:
    """Tenta converter uma string de data (DD/MM/AAAA ou AAAA-MM-DD) para o formato AAAA-MM-DD."""
    if not s: return None
    s = s.strip()
    # Tenta AAAA-MM-DD (ou AAAA/MM/DD) com ou sem hora
    m = re.search(r"(\d{4})[-/](\d{2})[-/](\d{2})(?:[ T]\d{2}:\d{2}:\d{2})?", s)
    if m: return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    # Tenta DD/MM/AAAA (ou DD-MM-AAAA) com ou sem hora
    m = re.search(r"(\d{2})[-/](\d{2})[-/](\d{4})(?:[ T]\d{2}:\d{2}:\d{2})?", s)
    if m: return f"{m.group(3)}-{m.group(2)}-{m.group(1)}"
    return None


# ------------------------------ Agente XML Parser (Aprimorado) ------------------------------
class AgenteXMLParser:
    """Interpreta XMLs fiscais e popula o banco, com suporte a namespaces."""
    def __init__(self, db: "BancoDeDados", validador: "ValidadorFiscal"):
        """Inicializa o AgenteXMLParser."""
        self.db = db
        self.validador = validador

    def processar(self, nome: str, conteudo: bytes, origem: str = "upload") -> int:
        """Processa um arquivo XML, extrai dados e os valida."""
        t_start = time.time()
        doc_id = -1
        tipo = "xml/desconhecido" # Valor padrão
        status = "erro"
        motivo_rejeicao = "Falha desconhecida no processamento XML"
        try:
            try:
                root = ET.fromstring(conteudo)
            except ET.ParseError:
                # Tenta decodificar como latin-1 se utf-8 falhar
                log.debug("XML '%s' falhou em UTF-8, tentando Latin-1.", nome)
                root = ET.fromstring(conteudo.decode('latin-1', errors='ignore').encode('utf-8'))
        except Exception as e:
            log.warning("Falha ao parsear XML '%s': %s", nome, e)
            motivo_rejeicao = f"XML mal formado: {e}"
            status = "quarentena"
            doc_id = self.db.inserir_documento(
                nome_arquivo=nome, tipo="xml/invalido", origem=origem,
                hash=self.db.hash_bytes(conteudo), status=status,
                data_upload=self.db.now(), motivo_rejeicao=motivo_rejeicao
            )
            return doc_id # Retorna o ID do documento em quarentena

        try:
            tipo = self._detectar_tipo(root)

            chave_node = root.find('.//{*}infNFe') or root.find('.//{*}infCte') # Namespace wildcard
            chave = chave_node.get('Id') if chave_node is not None else None
            if chave:
                chave = _only_digits(chave) # Garante apenas dígitos para chave de acesso

            emit_node = root.find('.//{*}emit')
            dest_node = root.find('.//{*}dest')
            ide_node = root.find('.//{*}ide')
            total_node = root.find('.//{*}total') # Nó total pode variar

            emit_cnpj = self._find_text(emit_node, 'CNPJ')
            dest_cnpj_cpf = self._find_text(dest_node, 'CNPJ') or self._find_text(dest_node, 'CPF')
            emit_nome = _norm_ws(self._find_text(emit_node, 'xNome') or "")
            dest_nome = _norm_ws(self._find_text(dest_node, 'xNome') or "")
            d_emis = self._find_text(ide_node, 'dhEmi') or self._find_text(ide_node, 'dEmi')

            # Lógica mais robusta para valor total
            valor_total_str = None
            if tipo in ["NFe", "NFCe", "CF-e"] and total_node is not None:
                icms_tot_node = total_node.find('.//{*}ICMSTot')
                if icms_tot_node is not None:
                    valor_total_str = self._find_text(icms_tot_node, 'vNF') or self._find_text(icms_tot_node, 'vCF') # Adiciona vCF
            elif tipo == "CTe" and total_node is not None:
                 vprest_node = total_node.find('.//{*}vPrest')
                 if vprest_node is not None:
                    valor_total_str = self._find_text(vprest_node, 'vTPrest')

            valor_total = _to_float_br(valor_total_str)

            doc_id = self.db.inserir_documento(
                nome_arquivo=nome, tipo=tipo, origem=origem, hash=self.db.hash_bytes(conteudo),
                chave_acesso=chave, status="processando", data_upload=self.db.now(), # Inicia como processando
                data_emissao=_parse_date_like(d_emis),
                emitente_cnpj=_only_digits(emit_cnpj), emitente_nome=emit_nome,
                destinatario_cnpj=_only_digits(dest_cnpj_cpf), destinatario_nome=dest_nome,
                valor_total=valor_total,
                caminho_arquivo=str(self.db.save_upload(nome, conteudo))
            )

            self._extrair_itens_impostos(root, doc_id)
            self.validador.validar_documento(doc_id=doc_id, db=self.db) # Valida após extrair tudo

            # Atualiza status final após validação (validador pode alterar para revisao_pendente)
            final_doc_info = self.db.get_documento(doc_id)
            status = final_doc_info.get("status", "processado") if final_doc_info else "erro"

        except Exception as e_proc:
            log.exception("Erro durante o processamento do XML (doc_id %d): %s", doc_id, e_proc)
            motivo_rejeicao = f"Erro no processamento: {e_proc}"
            status = "erro"
            if doc_id > 0: # Atualiza o documento se ele chegou a ser criado
                self.db.atualizar_documento_campo(doc_id, "status", status)
                self.db.atualizar_documento_campo(doc_id, "motivo_rejeicao", motivo_rejeicao)
            # Se doc_id ainda for -1, um novo registro de erro será criado no finally

        finally:
            processing_time = time.time() - t_start
            if doc_id > 0: # Registra extração apenas se o documento foi criado
                 self.db.inserir_extracao(
                    documento_id=doc_id, agente="XMLParser", confianca_media=1.0, # XML é estruturado
                    texto_extraido=None, linguagem="pt", tempo_processamento=round(processing_time, 3)
                )
                 self.db.log("ingestao_xml", usuario="sistema", detalhes=f"doc_id={doc_id}|tipo={tipo}|status={status}")
            else: # Se nem o doc_id foi gerado, registra uma falha genérica
                 doc_id = self.db.inserir_documento(
                        nome_arquivo=nome, tipo="xml/erro_desconhecido", origem=origem,
                        hash=self.db.hash_bytes(conteudo), status="erro",
                        data_upload=self.db.now(), motivo_rejeicao=motivo_rejeicao
                    )
                 self.db.log("ingestao_xml_falha", usuario="sistema", detalhes=f"arquivo={nome}|erro={motivo_rejeicao}")

        return doc_id

    def _detectar_tipo(self, root: ET.Element) -> str:
        """Detecta o tipo de documento fiscal (ignorando namespace)."""
        tag = root.tag.lower()
        if '}' in tag: tag = tag.split('}', 1)[1]
        if "cte" in tag: return "CTe"
        if "mdfe" in tag: return "MDF-e"
        if "nfe" in tag: return "NFe"
        # NFCe pode estar dentro de NFe, verificar nós internos se necessário
        is_nfce = root.find('.//{*}ide/{*}mod') is not None and root.find('.//{*}ide/{*}mod').text == '65'
        if is_nfce: return "NFCe"
        if "cfe" in tag: return "CF-e" # SAT
        return "xml/desconhecido" # Padronizado

    def _find_text(self, node: Optional[ET.Element], tag_name: str) -> Optional[str]:
        """Encontra o texto do filho direto com o nome da tag (ignorando namespace)."""
        if node is None: return None
        for child in node:
            tag = child.tag
            if '}' in tag: tag = tag.split('}', 1)[1]
            if tag.lower() == tag_name.lower():
                return child.text
        return None

    def _find_text_any(self, node: Optional[ET.Element], tag_names: Iterable[str]) -> Optional[str]:
        """Encontra o texto do primeiro filho direto que corresponda a qualquer uma das tags."""
        if node is None: return None
        for s in tag_names:
            v = self._find_text(node, s)
            if v: return v
        return None

    def _extrair_itens_impostos(self, root: ET.Element, doc_id: int) -> None:
        """Itera sobre os elementos 'det' do XML para extrair itens e impostos."""
        for det in root.findall('.//{*}det'): # Namespace wildcard
            prod = det.find('.//{*}prod')
            imposto = det.find('.//{*}imposto')
            if prod is None: continue

            # Extração dos campos do item
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
                documento_id=doc_id, descricao=desc, ncm=ncm, cest=cest, cfop=cfop,
                quantidade=qnt, unidade=unid, valor_unitario=vun, valor_total=vtot, codigo_produto=cprod
            )

            if imposto is not None:
                # Extração de ICMS (mais robusta)
                # Tenta ICMS primeiro, depois ICMSUFDest se o primeiro não tiver valor
                icms_node = imposto.find('.//{*}ICMS') or imposto.find('.//{*}ICMSUFDest')
                if icms_node:
                    icms_detalhe = next(iter(icms_node), None)
                    if icms_detalhe is not None:
                        cst = self._find_text(icms_detalhe, "CST") or self._find_text(icms_detalhe, "CSOSN")
                        orig = self._find_text(icms_detalhe, "orig")
                        # Busca por diferentes nomes de campos de base de cálculo, alíquota e valor
                        bc = self._find_text_any(icms_detalhe, ["vBC", "vBCST", "vBCSTRet", "vBCUFDest"])
                        aliq = self._find_text_any(icms_detalhe, ["pICMS", "pICMSST", "pICMSSTRet", "pICMSUFDest", "pICMSInter", "pICMSInterPart"])
                        val = self._find_text_any(icms_detalhe, ["vICMS", "vICMSST", "vICMSSTRet", "vICMSUFDest", "vICMSPartDest", "vICMSPartRemet"])
                        if val is not None: # Insere mesmo se for zero
                             self.db.inserir_imposto(
                                item_id=item_id, tipo_imposto="ICMS", cst=cst, origem=orig,
                                base_calculo=_to_float_br(bc), aliquota=_to_float_br(aliq), valor=_to_float_br(val)
                            )

                # Extração de IPI
                ipi_node = imposto.find('.//{*}IPI')
                ipi_trib_node = ipi_node.find('.//{*}IPITrib') if ipi_node is not None else None
                if ipi_trib_node:
                    cst = self._find_text(ipi_trib_node, "CST")
                    bc = self._find_text(ipi_trib_node, "vBC")
                    aliq = self._find_text(ipi_trib_node, "pIPI")
                    val = self._find_text(ipi_trib_node, "vIPI")
                    if val is not None:
                        self.db.inserir_imposto(
                            item_id=item_id, tipo_imposto="IPI", cst=cst, origem=None,
                            base_calculo=_to_float_br(bc), aliquota=_to_float_br(aliq), valor=_to_float_br(val)
                        )
                # Extração de PIS
                pis_node = imposto.find('.//{*}PIS')
                pis_aliq_node = pis_node.find('.//{*}PISAliq') if pis_node is not None else None # Exemplo PISAliq
                if pis_aliq_node:
                    cst = self._find_text(pis_aliq_node, "CST")
                    bc = self._find_text(pis_aliq_node, "vBC")
                    aliq = self._find_text(pis_aliq_node, "pPIS")
                    val = self._find_text(pis_aliq_node, "vPIS")
                    if val is not None:
                        self.db.inserir_imposto(
                            item_id=item_id, tipo_imposto="PIS", cst=cst, origem=None,
                            base_calculo=_to_float_br(bc), aliquota=_to_float_br(aliq), valor=_to_float_br(val)
                        )

                # Extração de COFINS (similar ao PIS)
                cofins_node = imposto.find('.//{*}COFINS')
                cofins_aliq_node = cofins_node.find('.//{*}COFINSAliq') if cofins_node is not None else None
                if cofins_aliq_node:
                    cst = self._find_text(cofins_aliq_node, "CST")
                    bc = self._find_text(cofins_aliq_node, "vBC")
                    aliq = self._find_text(cofins_aliq_node, "pCOFINS")
                    val = self._find_text(cofins_aliq_node, "vCOFINS")
                    if val is not None:
                        self.db.inserir_imposto(
                            item_id=item_id, tipo_imposto="COFINS", cst=cst, origem=None,
                            base_calculo=_to_float_br(bc), aliquota=_to_float_br(aliq), valor=_to_float_br(val)
                        )

# ------------------------------ Agente OCR (Aprimorado) ------------------------------
class AgenteOCR:
    """Executa OCR em PDFs/Imagens com cálculo de confiança aprimorado."""
    def __init__(self):
        self.ocr_ok = OCR_AVAILABLE
        self.pdf_ok = PDF_AVAILABLE
        if self.ocr_ok: log.info("OCR (Tesseract/Pillow) disponível.")
        else: log.warning("OCR (Tesseract/Pillow) NÃO disponível. Funcionalidade de imagem desativada.")
        if self.pdf_ok: log.info("Conversor PDF (pdf2image) disponível.")
        else: log.warning("Conversor PDF (pdf2image) NÃO disponível. Funcionalidade de PDF desativada.")

    def reconhecer(self, nome: str, conteudo: bytes) -> Tuple[str, float]:
        """Reconhece texto de um arquivo (PDF ou Imagem)."""
        t_start = time.time()
        ext = Path(nome).suffix.lower()
        texto = ""
        conf = 0.0
        try:
            if ext == ".pdf":
                if not self.pdf_ok: raise RuntimeError("Conversor PDF->Imagem indisponível (pdf2image ausente).")
                texto, conf = self._ocr_pdf(conteudo)
            elif ext in {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}:
                if not self.ocr_ok: raise RuntimeError("OCR indisponível (pytesseract/Pillow ausentes).")
                texto, conf = self._ocr_imagem(conteudo)
            else:
                raise ValueError(f"Extensão de arquivo não suportada para OCR: {ext}")
        except Exception as e:
            log.error("Erro no reconhecimento OCR para '%s': %s", nome, e)
            raise e # Relança a exceção para o Orchestrator tratar
        finally:
            log.info("OCR concluído para '%s' (conf: %.2f) em %.2fs", nome, conf, time.time() - t_start)

        return texto, conf

    def _preprocess_image(self, img: Image.Image) -> Image.Image:
        """Aplica pré-processamento para melhorar OCR."""
        try:
            gray = img.convert('L')
            # Ajuste de limiar pode ser necessário dependendo da qualidade da imagem
            bw = gray.point(lambda x: 0 if x < 180 else 255, '1')
            # Considerar outras técnicas: deskew, noise removal, etc.
            return bw
        except Exception as e:
            log.warning("Falha no pré-processamento da imagem: %s", e)
            return img

    def _ocr_imagem(self, conteudo: bytes) -> Tuple[str, float]:
        """Executa OCR em imagem com cálculo de confiança."""
        if not self.ocr_ok: return "", 0.0
        try:
            img = Image.open(io.BytesIO(conteudo))
            img_proc = self._preprocess_image(img)
            # Tesseract config: '--psm 6' assume um bloco uniforme de texto. Pode precisar ajustar.
            ocr_data = pytesseract.image_to_data(img_proc, lang='por', config='--psm 6', output_type=pytesseract.Output.DICT)
            confidences = [int(c) for i, c in enumerate(ocr_data['conf']) if int(c) > -1 and ocr_data['text'][i].strip()]
            avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
            texto = pytesseract.image_to_string(img_proc, lang='por', config='--psm 6')
            return texto, round(avg_conf / 100.0, 2)
        except Exception as e:
            log.error("Erro durante OCR da imagem: %s", e)
            return "", 0.0

    def _ocr_pdf(self, conteudo: bytes) -> Tuple[str, float]:
        """Executa OCR em PDF com cálculo de confiança."""
        if not self.pdf_ok: return "", 0.0
        try:
            images = convert_from_bytes(conteudo, dpi=200) # dpi=200 é um bom equilíbrio, pode ajustar
            full_text = []
            total_conf = 0.0
            num_valid_pages = 0

            for i, img in enumerate(images):
                log.debug("Processando OCR da página %d do PDF", i + 1)
                img_proc = self._preprocess_image(img)
                ocr_data = pytesseract.image_to_data(img_proc, lang='por', config='--psm 6', output_type=pytesseract.Output.DICT)
                confidences = [int(c) for idx, c in enumerate(ocr_data['conf']) if int(c) > -1 and ocr_data['text'][idx].strip()]
                if confidences:
                    avg_conf_page = sum(confidences) / len(confidences)
                    total_conf += avg_conf_page
                    num_valid_pages += 1
                page_text = pytesseract.image_to_string(img_proc, lang='por', config='--psm 6')
                full_text.append(page_text)

            final_text = "\n\n--- Page Break ---\n\n".join(full_text)
            final_avg_conf = (total_conf / num_valid_pages) if num_valid_pages > 0 else 0.0
            return final_text, round(final_avg_conf / 100.0, 2)
        except Exception as e:
            log.error("Erro durante OCR do PDF: %s", e)
            return "", 0.0


# ------------------------------ Agente NLP (Aprimorado com Extração de Itens OCR) ------------------------------
class AgenteNLP:
    """Extrai campos fiscais de texto bruto usando regex aprimoradas."""
    RE_CNPJ = re.compile(r"\b(\d{2}\.?\d{3}\.?\d{3}/?\d{4}-?\d{2}|\d{14})\b")
    RE_CPF = re.compile(r"\b(\d{3}\.?\d{3}\.?\d{3}-?\d{2}|\d{11})\b")
    RE_IE = re.compile(r"\b(?:IE|I\.E\.|INSC(?:RI[ÇC][ÃA]O)?\sESTADUAL)[:\s\-]*([A-Z0-9.\-/]{5,20})\b", re.I)
    RE_UF = re.compile(r"\b(AC|AL|AP|AM|BA|CE|DF|ES|GO|MA|MT|MS|MG|PA|PB|PR|PE|PI|RJ|RN|RS|RO|RR|SC|SP|SE|TO)\b")
    RE_VALOR_TOTAL = re.compile(r"\b(?:VALOR\s+TOTAL\s+DA\s+NOTA|VALOR\s+TOTAL|TOTAL\s+DA\s+NOTA)\s*[:\-]?\s*R?\$\s*([\d.,]+)\b", re.I)
    RE_DATA_EMISSAO = re.compile(r"\b(?:DATA\s+(?:DE\s+)?EMISS[ÃA]O|EMITIDO\s+EM)\s*[:\-]?\s*(\d{2,4}[-/]\d{2}[-/]\d{2,4})\b", re.I)

    # --- NOVAS REGEX PARA ITENS (Exemplo Simplificado - pode precisar de muitos ajustes) ---
    # Captura linhas que começam com um código (opcional), descrição, unidade, qtd, v.unit, v.total
    # Exemplo: 001 PRODUTO A UN 1,000 10,00 10,00
    # Exemplo: DESC PROD B KG 2.5 5,50 13,75
    RE_ITEM_LINHA = re.compile(
        r"^(?:\d+\s+)?(?P<desc>.+?)\s+" # Descrição (não gulosa)
        r"(?P<unid>[A-Z]{1,3})\s+" # Unidade (1-3 letras maiúsculas)
        r"(?P<qtd>[\d.,]+)\s+" # Quantidade
        r"(?P<vun>[\d.,]+)\s+" # Valor unitário
        r"(?P<vtot>[\d.,]+)$", # Valor total
        re.IGNORECASE | re.MULTILINE
    )
    # Regex para NCM e CFOP podem ser mais difíceis de capturar de forma genérica em OCR
    RE_NCM_ITEM = re.compile(r"NCM[:\s]*(\d{8})", re.I)
    RE_CFOP_ITEM = re.compile(r"CFOP[:\s]*(\d{4})", re.I)

    def extrair_campos(self, texto: str) -> Dict[str, Any]:
        """Extrai os principais campos de cabeçalho e itens/impostos do texto."""
        t_norm = _norm_ws(texto) # Normaliza espaços gerais primeiro

        # --- Extração de Cabeçalho (como antes) ---
        cnpjs = [_only_digits(m) for m in self.RE_CNPJ.findall(t_norm)]
        cpfs = [_only_digits(m) for m in self.RE_CPF.findall(t_norm)]

        emit_cnpj_cpf = (cnpjs[0] if cnpjs else None) or (cpfs[0] if cpfs else None)
        dest_cnpj_cpf = None
        if len(cnpjs) > 1: dest_cnpj_cpf = cnpjs[1]
        elif cnpjs and cpfs: dest_cnpj_cpf = cpfs[0]
        elif len(cpfs) > 1: dest_cnpj_cpf = cpfs[1]

        m_ie = self.RE_IE.search(t_norm)
        ie = m_ie.group(1).strip() if m_ie else None

        endereco_match = self._match_after(t_norm, ["endereço", "endereco", "logradouro", "rua"], max_len=150)
        municipio_match = self._match_after(t_norm, ["município", "municipio", "cidade"], max_len=80)

        uf = None
        uf_match = self.RE_UF.search(endereco_match[-10:] if endereco_match else "")
        if not uf_match and municipio_match: uf_match = self.RE_UF.search(municipio_match[-10:])
        if not uf_match: uf_match = self.RE_UF.search(t_norm[-100:]) # Tenta no final do doc
        uf = uf_match.group(1) if uf_match else None

        razao = self._match_after(t_norm, ["razão social", "razao social", "nome", "emitente"], max_len=100)

        m_valor = self.RE_VALOR_TOTAL.search(t_norm)
        valor_total = _to_float_br(m_valor.group(1)) if m_valor else None

        m_data = self.RE_DATA_EMISSAO.search(t_norm)
        data_emissao = _parse_date_like(m_data.group(1)) if m_data else None

        # --- NOVA Extração de Itens e Impostos (OCR) ---
        itens_extraidos, impostos_extraidos = self._extrair_itens_impostos_ocr(texto) # Passa o texto original com quebras de linha

        return {
            "emitente_cnpj": emit_cnpj_cpf if len(emit_cnpj_cpf or "") >= 14 else None,
            "emitente_cpf": emit_cnpj_cpf if len(emit_cnpj_cpf or "") == 11 else None,
            "destinatario_cnpj": dest_cnpj_cpf if len(dest_cnpj_cpf or "") >= 14 else None,
            "destinatario_cpf": dest_cnpj_cpf if len(dest_cnpj_cpf or "") == 11 else None,
            "inscricao_estadual": ie,
            "emitente_nome": razao,
            "endereco": endereco_match,
            "uf": uf,
            "municipio": municipio_match,
            "valor_total": valor_total,
            "data_emissao": data_emissao,
            # --- NOVOS CAMPOS ---
            "itens_ocr": itens_extraidos,
            "impostos_ocr": impostos_extraidos,
        }

    def _match_after(self, texto: str, labels: List[str], max_len: int = 80, max_dist: int = 50) -> Optional[str]:
        """Encontra o texto após um dos labels, considerando separadores e limites."""
        texto_lower = texto.lower()
        best_match = None
        min_pos = float('inf')

        for lab in labels:
            lab_lower = lab.lower()
            idx = texto_lower.find(lab_lower)
            if idx != -1:
                start_value_idx = -1
                # Separadores aprimorados: :, -, |, ;, \n e espaço (se for fim da label)
                end_label_idx = idx + len(lab_lower)
                for i in range(end_label_idx, min(end_label_idx + max_dist, len(texto))):
                    # Se achar um separador OU se o caractere após label for espaço e não letra/num
                    if texto[i] in ':|-;\n' or (i == end_label_idx and texto[i].isspace() and not texto[i+1:i+2].isalnum()):
                        start_value_idx = i + 1
                        break

                if start_value_idx != -1:
                    # Tenta encontrar o fim da linha ou o próximo label conhecido
                    end_idx = texto.find('\n', start_value_idx)
                    if end_idx == -1: end_idx = len(texto)

                    # Verifica se outro label aparece antes do fim da linha
                    for next_lab in labels:
                         # Busca o próximo label a partir do início do valor atual
                         next_idx = texto.find(next_lab, start_value_idx, end_idx)
                         if next_idx != -1:
                             # Se encontrou, limita o fim do valor atual ao início do próximo label
                             end_idx = next_idx

                    value = texto[start_value_idx : end_idx].strip()
                    # Remove lixo comum após valores (ex: códigos, barras)
                    value = re.sub(r'\s*[|;/-].*$', '', value).strip()
                    # Limita o tamanho e verifica se é o primeiro match encontrado
                    if value and idx < min_pos:
                        min_pos = idx
                        best_match = value[:max_len]

        return best_match

    def _extrair_itens_impostos_ocr(self, texto_original: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Tenta extrair itens e impostos de um bloco de texto OCR.
        Retorna (lista_de_itens, lista_de_impostos_associados_por_item_idx).
        Esta é uma implementação EXEMPLO e SIMPLIFICADA.
        """
        itens: List[Dict[str, Any]] = []
        impostos: List[Dict[str, Any]] = []
        linhas = texto_original.splitlines()

        # Encontra a seção de itens (exemplo: busca por cabeçalho típico)
        inicio_itens = -1
        fim_itens = len(linhas)
        header_keywords = ["DESCRIÇÃO", "QTD", "UNIT", "TOTAL", "PRODUTO"]
        for i, linha in enumerate(linhas):
            linha_upper = linha.upper()
            if any(kw in linha_upper for kw in header_keywords):
                 if "TOTAL" in linha_upper and len(linha.split()) < 4: # Provável linha de totais, não cabeçalho
                     continue
                 inicio_itens = i + 1
                 break # Assume que o primeiro header encontrado marca o início

        if inicio_itens == -1:
            log.warning("Não foi possível identificar o início da seção de itens no OCR.")
            return [], []

        # Tenta encontrar o fim da seção de itens (ex: linha de totais)
        total_keywords = ["TOTAL DOS PRODUTOS", "VALOR TOTAL", "SUBTOTAL"]
        for i in range(inicio_itens, len(linhas)):
            if any(kw in linhas[i].upper() for kw in total_keywords):
                fim_itens = i
                break

        # Processa as linhas dentro da seção de itens
        item_idx_counter = 0 # Para associar impostos
        for i in range(inicio_itens, fim_itens):
            linha = linhas[i].strip()
            if not linha: continue # Pula linhas vazias

            match = self.RE_ITEM_LINHA.search(linha)
            if match:
                item_data = match.groupdict()
                item = {
                    "descricao": _norm_ws(item_data.get("desc", "")),
                    "unidade": item_data.get("unid"),
                    "quantidade": _to_float_br(item_data.get("qtd")),
                    "valor_unitario": _to_float_br(item_data.get("vun")),
                    "valor_total": _to_float_br(item_data.get("vtot")),
                    # Tenta extrair NCM/CFOP da linha atual ou da próxima
                    "ncm": None,
                    "cfop": None,
                }
                # Tenta buscar NCM/CFOP na própria linha ou na seguinte (layout comum)
                linha_seguinte = linhas[i+1].strip() if (i+1) < fim_itens else ""
                ncm_match = self.RE_NCM_ITEM.search(linha) or self.RE_NCM_ITEM.search(linha_seguinte)
                cfop_match = self.RE_CFOP_ITEM.search(linha) or self.RE_CFOP_ITEM.search(linha_seguinte)
                if ncm_match: item["ncm"] = ncm_match.group(1)
                if cfop_match: item["cfop"] = cfop_match.group(1)

                itens.append(item)

                # --- Extração Simplificada de Impostos (Exemplo: ICMS na linha seguinte) ---
                # Procura por padrões como "ICMS: 18,00%" ou "BC: 10,00 Vlr: 1,80"
                # Esta parte é ALTAMENTE dependente do layout e precisaria ser muito mais robusta
                icms_match = re.search(r"ICMS.*?(\d+,\d{2})\s*%", linha_seguinte, re.I)
                if icms_match:
                     impostos.append({
                         "item_idx": item_idx_counter, # Associa ao último item adicionado
                         "tipo_imposto": "ICMS",
                         "aliquota": _to_float_br(icms_match.group(1)),
                         # Outros campos (BC, Valor) precisariam de regex mais complexas
                     })
                # Adicionar lógica similar para IPI, PIS, COFINS se os padrões forem identificáveis

                item_idx_counter += 1 # Incrementa para o próximo item
            else:
                # Se a linha não bateu com o regex de item, pode ser continuação da descrição
                # ou informações adicionais (NCM, impostos soltos)
                # Poderia tentar anexar à descrição do item anterior ou processar separadamente
                pass

        log.info(f"AgenteNLP: Extraídos {len(itens)} itens via OCR.")
        return itens, impostos


# ------------------------------ Agente Analítico (LLM → Sandbox - Final e Seguro) ------------------------------
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
    # Funções adicionais úteis e seguras podem ser adicionadas aqui se necessário
    # Ex: 'divmod', 'pow', 'repr'
)}
SAFE_BUILTINS["__import__"] = _restricted_import # Sobrescreve o import padrão

class AgenteAnalitico:
    """Gera e executa código Python via LLM com auto-correção."""
    def __init__(self, llm: BaseChatModel, memoria: MemoriaSessao):
        self.llm = llm
        self.memoria = memoria
        self.last_code: str = "" # Armazena o último código tentado

    def _prompt_inicial(self, catalog: Dict[str, pd.DataFrame]) -> SystemMessage:
        """ Constrói o prompt inicial para a geração de código. """
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
        6.  Seja DEFENSIVO: Use `pd.to_numeric(df['coluna'], errors='coerce')` para conversões numéricas. Use `.fillna(0)` ou `.dropna()` apropriadamente. Verifique se as colunas existem antes de usá-las.
        7.  GRÁFICOS: Prefira `plotly.express as px`. Use `fig.update_layout(width=800, height=500)` para ajustar o tamanho. Para `matplotlib.pyplot as plt`, use `fig, ax = plt.subplots(figsize=(10, 6))` e `plt.tight_layout()` antes de retornar `fig`.
        8.  Se retornar uma `tabela` (DataFrame), o `texto` deve ser um resumo ou título, NÃO a tabela convertida para string (`.to_string()`).
        9.  Manipule datas com `pd.to_datetime(df['coluna_data'], errors='coerce')`.

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
                text_output = f"Erro durante a análise: {type(e).__name__}: {e}"
                # Pode logar o erro completo se necessário, mas não o retorne diretamente ao usuário

            # Retorna a tupla (texto, tabela, figura)
            return (text_output, table_output, figure_output)
        ```
        Gere APENAS o código Python completo da função `solve`, nada antes ou depois.
        """
        return SystemMessage(content=prompt)

    def _prompt_correcao(self, failed_code: str, error_message: str) -> SystemMessage:
        """ Constrói o prompt para a correção de código. """
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
        4.  Revise o acesso a colunas e tratamento de tipos (use `pd.to_numeric`, `pd.to_datetime` com `errors='coerce'`).
        5.  Garanta que funções built-in não seguras não foram usadas.
        6.  Verifique a lógica da análise para possíveis erros (divisão por zero, índices inválidos, etc.).
        7.  Mantenha a estrutura de retorno `(texto, tabela, figura)`.

        Reescreva APENAS o código Python completo da função `solve` corrigida.
        """
        return SystemMessage(content=prompt)

    def _gerar_codigo(self, pergunta: str, catalog: Dict[str, pd.DataFrame]) -> str:
        """ Gera a primeira versão do código. """
        sys = self._prompt_inicial(catalog)
        hum = HumanMessage(content=f"Pergunta do usuário: {pergunta}")
        resp = self.llm.invoke([sys, hum]).content.strip()
        # Extrai o bloco de código Python, mesmo com texto antes/depois
        code_match = re.search(r"```python\n(.*?)\n```", resp, re.DOTALL)
        if code_match:
            code = code_match.group(1).strip()
        else:
             # Se não encontrar o bloco, assume que a resposta inteira é o código (menos provável)
             log.warning("LLM não retornou um bloco de código python formatado. Tentando usar a resposta inteira.")
             code = resp.strip()

        self.last_code = code # Armazena antes de retornar
        return code

    def _corrigir_codigo(self, failed_code: str, erro: str) -> str:
        """ Gera uma versão corrigida do código. """
        # Nota: O prompt de correção já contém o código que falhou.
        # Não precisamos passar o catálogo novamente, mas o LLM deve lembrar das regras.
        sys = self._prompt_correcao(failed_code, erro)
        hum = HumanMessage(content="Por favor, corrija a função `solve` baseada no erro e no código fornecido.")
        resp = self.llm.invoke([sys, hum]).content.strip()
        # Extrai o bloco de código Python da correção
        code_match = re.search(r"```python\n(.*?)\n```", resp, re.DOTALL)
        if code_match:
            code = code_match.group(1).strip()
        else:
             log.warning("LLM não retornou um bloco de código python formatado na CORREÇÃO. Tentando usar a resposta inteira.")
             code = resp.strip()

        self.last_code = code # Armazena antes de retornar
        return code

    def _executar_sandbox(self, code: str, pergunta: str, catalog: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """ Executa o código em sandbox seguro com escopo unificado e builtins restritos. """
        # Cria um escopo novo para cada execução, incluindo os builtins seguros
        # É crucial que SAFE_BUILTINS seja passado aqui para restringir o ambiente
        scope = {"__builtins__": SAFE_BUILTINS}

        # Executa o código da função 'solve' dentro do escopo seguro
        # O próprio código da função fará os imports restritos
        exec(code, scope)

        # Verifica se a função 'solve' foi definida no escopo
        if "solve" not in scope or not callable(scope["solve"]):
            raise RuntimeError("A função `solve` não foi definida corretamente no código gerado.")

        solve_fn = scope["solve"]
        t0 = time.time()
        # Chama a função 'solve' passando o catálogo (cópia profunda é feita dentro do solve se o LLM seguir a regra)
        # e a pergunta original.
        texto, tabela, fig = solve_fn({k: v for k, v in catalog.items()}, pergunta) # Passa o catálogo original
        dt = time.time() - t0

        # Validações básicas do retorno
        if not isinstance(texto, str):
            log.warning(f"Retorno 'texto' não é string, é {type(texto)}. Convertendo.")
            texto = str(texto)
        if tabela is not None and not isinstance(tabela, pd.DataFrame):
             log.warning(f"Retorno 'tabela' não é DataFrame ou None, é {type(tabela)}. Ignorando.")
             tabela = None
        # Validação da figura é mais complexa, aceita matplotlib ou plotly por enquanto
        if fig is not None:
             try:
                 import matplotlib.figure
                 import plotly.graph_objects as go
                 if not isinstance(fig, (matplotlib.figure.Figure, go.Figure)):
                    log.warning(f"Retorno 'figura' não é Matplotlib ou Plotly Figure, é {type(fig)}. Ignorando.")
                    fig = None
             except ImportError: # Se libs gráficas não estiverem instaladas
                 log.warning("Bibliotecas gráficas não disponíveis para validar tipo da figura.")
                 fig = None


        # Padronizado para chaves em português
        return {
            "texto": texto,
            "tabela": tabela,
            "figuras": [fig] if fig is not None else [], # Sempre retorna lista
            "duracao_s": round(dt, 3),
            "code": code, # Retorna o código que foi executado com sucesso
        }

    def responder(self, pergunta: str, catalog: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """ Orquestra a geração, execução e auto-correção do código. """
        max_retries = 2 # Número de tentativas de correção
        code_to_run = ""

        try:
            code_to_run = self._gerar_codigo(pergunta, catalog)
            if not code_to_run.strip():
                 raise ValueError("LLM não gerou nenhum código.")

            for attempt in range(max_retries + 1):
                try:
                    log.info(f"Tentativa {attempt + 1} de executar código para: '{pergunta}'")
                    out = self._executar_sandbox(code_to_run, pergunta, catalog)
                    # Sucesso! Salva na memória e retorna
                    self.memoria.salvar(pergunta, out.get("texto", ""), duracao_s=out.get("duracao_s", 0.0))
                    out["agent_name"] = f"AgenteAnalitico (Tentativa {attempt + 1})"
                    out["summary"] = f"Executou código com sucesso para: '{pergunta}'"
                    log.info(f"Execução bem-sucedida na tentativa {attempt + 1}.")
                    return out

                except Exception as e1:
                    error_message = f"{type(e1).__name__}: {e1}"
                    traceback_str = traceback.format_exc(limit=3)
                    log.warning(f"Falha na tentativa {attempt + 1} para '{pergunta}': {error_message}\n{traceback_str}")

                    if attempt < max_retries:
                        log.info(f"Solicitando correção ao LLM (tentativa {attempt + 2}/{max_retries + 1}).")
                        code_to_run = self._corrigir_codigo(code_to_run, error_message) # Pede correção
                        if not code_to_run.strip():
                             raise ValueError("LLM não gerou nenhum código de correção.")
                    else:
                        log.error(f"Número máximo de tentativas ({max_retries + 1}) excedido. Falha final.")
                        raise e1 # Relança a última exceção após esgotar tentativas

        except Exception as e_final:
            # Captura exceções da geração de código inicial, da correção ou a última exceção da execução
            traceback_str = traceback.format_exc(limit=3)
            log.error(f"Falha irrecuperável no AgenteAnalitico para '{pergunta}': {type(e_final).__name__}: {e_final}\n{traceback_str}")
            summary = f"Falha final na geração ou auto-correção para: '{pergunta}'"
            self.memoria.salvar(pergunta, f"Erro: {type(e_final).__name__}: {e_final}", duracao_s=0.0)
            return {
                "texto": f"Ocorreu um erro irrecuperável ao tentar analisar sua pergunta após {max_retries + 1} tentativas. Detalhe: {type(e_final).__name__}: {e_final}",
                "tabela": None,
                "figuras": [],
                "duracao_s": 0.0,
                "code": self.last_code or code_to_run or "", # Retorna o último código tentado
                "agent_name": "AgenteAnalitico (Falha Irrecuperável)",
                "summary": summary
            }


# ------------------------------ Orchestrator (Aprimorado e Completo) ------------------------------
@dataclass
class Orchestrator:
    """ Coordena o pipeline de processamento de documentos e análise. """
    db: "BancoDeDados"
    validador: "ValidadorFiscal"
    memoria: "MemoriaSessao"
    llm: Optional[BaseChatModel] = None

    def __post_init__(self):
        """Inicializa os agentes."""
        if not CORE_MODULES_AVAILABLE:
             log.error("Orchestrator não pode ser inicializado corretamente devido a dependências ausentes.")
             # Considerar levantar uma exceção aqui para impedir a execução se módulos core falharem
             # raise RuntimeError("Dependências críticas (banco_de_dados, validacao, memoria) ausentes.")
        self.xml_agent = AgenteXMLParser(self.db, self.validador)
        self.ocr_agent = AgenteOCR()
        # Passa o banco de dados para o AgenteNLP para que ele possa salvar itens/impostos
        self.nlp_agent = AgenteNLP() # Não precisa mais do DB aqui, a lógica de salvar foi para _processar_midias
        self.analitico = AgenteAnalitico(self.llm, self.memoria) if self.llm else None
        if self.llm: log.info("Agente Analítico (LLM) INICIALIZADO.")
        else: log.warning("Agente Analítico (LLM) NÃO inicializado (LLM não fornecido).")


    def ingestir_arquivo(self, nome: str, conteudo: bytes, origem: str = "web") -> int:
        """ Processa um arquivo de entrada, retornando o ID do documento. """
        t_start = time.time()
        doc_id = -1
        status = "erro"
        motivo = "Falha desconhecida na ingestão"
        doc_hash = self.db.hash_bytes(conteudo)
        ext = Path(nome).suffix.lower()

        try:
            existing_id = self.db.find_documento_by_hash(doc_hash)
            if existing_id:
                log.info("Documento '%s' (hash %s...) já existe com ID %d. Ignorando.", nome, doc_hash[:8], existing_id)
                return existing_id

            if ext == ".xml":
                doc_id = self.xml_agent.processar(nome, conteudo, origem)
            elif ext in {".pdf", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}:
                doc_id = self._processar_midias(nome, conteudo, origem)
            else:
                motivo = f"Extensão '{ext}' não suportada para ingestão."
                status = "quarentena"
                log.warning("Arquivo '%s' rejeitado: %s", nome, motivo)
                doc_id = self.db.inserir_documento(
                    nome_arquivo=nome, tipo="desconhecido", origem=origem, hash=doc_hash,
                    chave_acesso=None, status=status, data_upload=self.db.now(), motivo_rejeicao=motivo
                )

            # Busca o status final do banco após o processamento ter ocorrido (ou falhado)
            if doc_id > 0:
                 doc_info = self.db.get_documento(doc_id)
                 if doc_info: status = doc_info.get("status", status)

        except Exception as e:
            log.exception("Falha crítica na ingestão de %s: %s", nome, e)
            motivo = f"Erro inesperado: {str(e)}"
            status = "erro" # Marcar como erro geral
            try:
                # Tenta garantir que um registro de erro exista, atualizando se já foi criado
                existing_id_on_error = self.db.find_documento_by_hash(doc_hash)
                if existing_id_on_error:
                    doc_id = existing_id_on_error
                    self.db.atualizar_documento_campos(doc_id, status="erro", motivo_rejeicao=motivo)
                elif doc_id > 0 : # Se foi criado mas falhou depois
                     self.db.atualizar_documento_campos(doc_id, status="erro", motivo_rejeicao=motivo)
                else: # Cria um novo registro de erro se nada foi criado ainda
                    doc_id = self.db.inserir_documento(
                        nome_arquivo=nome, tipo=ext.strip('.') or 'binario', origem=origem, hash=doc_hash,
                        chave_acesso=None, status="erro", data_upload=self.db.now(), motivo_rejeicao=motivo
                    )
            except Exception as db_err:
                 log.error("Erro CRÍTICO ao registrar falha de ingestão para '%s': %s", nome, db_err)
                 return -1 # Indica falha grave

        finally:
            log.info("Ingestão de '%s' concluída (ID: %d, Status Final: %s) em %.2fs",
                     nome, doc_id, status, time.time() - t_start)

        return doc_id

    def _processar_midias(self, nome: str, conteudo: bytes, origem: str) -> int:
        """ Processa arquivos PDF ou Imagem via OCR e NLP, incluindo itens/impostos. """
        doc_id = -1 # Inicializa doc_id
        try:
            # Cria o registro inicial do documento ANTES do OCR/NLP
            doc_id = self.db.inserir_documento(
                nome_arquivo=nome, tipo=Path(nome).suffix.lower().strip('.'), origem=origem,
                hash=self.db.hash_bytes(conteudo), status="processando", # Começa como processando
                data_upload=self.db.now(), caminho_arquivo=str(self.db.save_upload(nome, conteudo))
            )
            log.info("Iniciando processamento de mídia para '%s' (doc_id %d)", nome, doc_id)

            texto = ""
            conf = 0.0
            t_start_ocr = time.time()

            # --- Etapa OCR ---
            try:
                texto, conf = self.ocr_agent.reconhecer(nome, conteudo)
                ocr_time = time.time() - t_start_ocr
                log.info(f"OCR para doc_id {doc_id} concluído com confiança {conf:.2f} em {ocr_time:.2f}s.")
                self.db.inserir_extracao(
                    documento_id=doc_id, agente="OCRAgent", confianca_media=float(conf),
                    # Limita o tamanho do texto salvo para evitar sobrecarga no DB
                    texto_extraido=texto[:50000] + ("..." if len(texto) > 50000 else ""),
                    linguagem="pt", tempo_processamento=round(ocr_time, 3)
                )
            except Exception as e_ocr:
                log.error(f"Falha na etapa de OCR para doc_id {doc_id}: {e_ocr}")
                self.db.atualizar_documento_campos(doc_id, status="erro", motivo_rejeicao=f"Falha no OCR: {e_ocr}")
                self.db.log("ocr_erro", "sistema", f"doc_id={doc_id}|erro={e_ocr}")
                return doc_id # Retorna o ID com status de erro

            # --- Etapa NLP (Cabeçalho e Itens) ---
            status_final = "erro" # Status padrão se algo falhar daqui pra frente
            if texto:
                try:
                    t_start_nlp = time.time()
                    log.info("Iniciando NLP para doc_id %d.", doc_id)
                    # Extrai cabeçalho E itens/impostos do texto OCR
                    campos_nlp = self.nlp_agent.extrair_campos(texto)
                    nlp_time = time.time() - t_start_nlp
                    log.info(f"NLP para doc_id {doc_id} concluído em {nlp_time:.2f}s.")

                    # Separa itens/impostos dos campos de cabeçalho
                    itens_ocr = campos_nlp.pop("itens_ocr", [])
                    impostos_ocr = campos_nlp.pop("impostos_ocr", [])

                    # Atualiza campos do cabeçalho no banco
                    self.db.atualizar_documento_campos(doc_id, **campos_nlp)

                    # --- Persistência de Itens e Impostos OCR ---
                    if itens_ocr:
                        log.info(f"Salvando {len(itens_ocr)} itens extraídos via OCR para doc_id {doc_id}.")
                        item_id_map = {} # Mapeia índice da lista OCR para ID do banco
                        for idx, item_data in enumerate(itens_ocr):
                            item_id = self.db.inserir_item(documento_id=doc_id, **item_data)
                            item_id_map[idx] = item_id

                        if impostos_ocr:
                            log.info(f"Salvando {len(impostos_ocr)} impostos associados aos itens OCR para doc_id {doc_id}.")
                            for imposto_data in impostos_ocr:
                                item_ocr_idx = imposto_data.pop("item_idx", -1)
                                if item_ocr_idx in item_id_map:
                                    self.db.inserir_imposto(item_id=item_id_map[item_ocr_idx], **imposto_data)
                                else:
                                     log.warning(f"Imposto OCR não pôde ser associado a um item válido (índice {item_ocr_idx}), doc_id {doc_id}.")

                    # --- Validação Final ---
                    log.info("Iniciando validação fiscal para doc_id %d após OCR/NLP.", doc_id)
                    self.validador.validar_documento(doc_id=doc_id, db=self.db)

                    # Re-busca o status após validação
                    doc_info_after_validation = self.db.get_documento(doc_id)
                    status_depois_validacao = doc_info_after_validation.get("status") if doc_info_after_validation else "erro"

                    # Decide o status final: mantém 'revisao_pendente' da validação ou usa confiança do OCR
                    if status_depois_validacao == "revisao_pendente":
                        status_final = "revisao_pendente"
                        log.info(f"Documento {doc_id} marcado para revisão devido a inconsistências de validação.")
                    else:
                        # Limiar de confiança para marcar para revisão se a validação passou
                        limiar_confianca = 0.60
                        status_final = "processado" if conf >= limiar_confianca else "revisao_pendente"
                        if status_final == "revisao_pendente":
                             log.info(f"Documento {doc_id} marcado para revisão devido à baixa confiança do OCR ({conf:.2f} < {limiar_confianca:.2f}).")
                        else:
                             log.info(f"Documento {doc_id} processado com sucesso (Conf OCR: {conf:.2f}).")

                except Exception as e_nlp:
                    log.exception(f"Falha na etapa de NLP ou persistência de itens para doc_id {doc_id}: {e_nlp}")
                    status_final = "erro"
                    self.db.atualizar_documento_campos(doc_id, status=status_final, motivo_rejeicao=f"Falha no NLP/Save: {e_nlp}")
                    self.db.log("nlp_erro", "sistema", f"doc_id={doc_id}|erro={e_nlp}")

            else: # Caso OCR não retorne texto
                 status_final = "revisao_pendente" # Marcar para revisão se OCR falhou
                 log.warning(f"OCR não extraiu texto para doc_id {doc_id} (conf: {conf:.2f}). Marcado para revisão.")
                 self.db.atualizar_documento_campos(doc_id, status=status_final, motivo_rejeicao="OCR não extraiu texto.")

            # Atualiza o status final (se não foi erro)
            if status_final != "erro":
                 self.db.atualizar_documento_campo(doc_id, "status", status_final)
            self.db.log("ingestao_midias", "sistema", f"doc_id={doc_id}|conf={conf:.2f}|status_final={status_final}")

        except Exception as e_outer:
             # Captura erros na criação inicial do documento ou outras falhas inesperadas
             log.exception(f"Falha geral no processamento de mídia para '{nome}': {e_outer}")
             # Se doc_id foi criado, marca como erro. Senão, não há o que fazer aqui.
             if doc_id > 0:
                 try:
                     self.db.atualizar_documento_campos(doc_id, status="erro", motivo_rejeicao=f"Falha geral: {e_outer}")
                 except Exception as db_err_final:
                     log.error(f"Erro CRÍTICO ao tentar marcar doc_id {doc_id} como erro final: {db_err_final}")
             return doc_id if doc_id > 0 else -1 # Retorna ID se foi criado, senão -1

        return doc_id


    def responder_pergunta(self, pergunta: str) -> Dict[str, Any]:
        """ Delega a pergunta analítica para o AgenteAnalitico. """
        if not self.analitico:
            log.error("Agente Analítico não inicializado. Verifique a configuração do LLM.")
            return {"texto": "Erro interno: Agente analítico não configurado.", "tabela": None, "figuras": []}

        catalog: Dict[str, pd.DataFrame] = {}
        try:
            # Carrega apenas documentos com status 'processado' ou 'revisado' (assumindo que 'revisado' significa OK após correção manual)
            # Você pode ajustar os status válidos aqui conforme necessário
            where_clause = "status = 'processado' OR status = 'revisado'"
            catalog["documentos"] = self.db.query_table("documentos", where=where_clause)

            if not catalog["documentos"].empty:
                 doc_ids = tuple(catalog["documentos"]['id'].unique().tolist())
                 # Formatação segura para cláusula IN do SQL
                 doc_ids_sql = ', '.join(map(str, doc_ids))

                 catalog["itens"] = self.db.query_table("itens", where=f"documento_id IN ({doc_ids_sql})")
                 if not catalog["itens"].empty:
                    item_ids = tuple(catalog["itens"]['id'].unique().tolist())
                    item_ids_sql = ', '.join(map(str, item_ids))
                    catalog["impostos"] = self.db.query_table("impostos", where=f"item_id IN ({item_ids_sql})")
                 else:
                    catalog["impostos"] = pd.DataFrame(columns=['id', 'item_id', 'tipo_imposto', 'cst', 'origem', 'base_calculo', 'aliquota', 'valor']) # DF vazio com schema
            else:
                 # Define DFs vazios com schema esperado se não houver documentos válidos
                 catalog["itens"] = pd.DataFrame(columns=['id', 'documento_id', 'descricao', 'ncm', 'cest', 'cfop', 'quantidade', 'unidade', 'valor_unitario', 'valor_total', 'codigo_produto'])
                 catalog["impostos"] = pd.DataFrame(columns=['id', 'item_id', 'tipo_imposto', 'cst', 'origem', 'base_calculo', 'aliquota', 'valor'])

        except Exception as e:
            log.exception("Falha ao montar catálogo do banco de dados para análise: %s", e)
            return {"texto": f"Erro ao carregar dados para análise: {e}", "tabela": None, "figuras": []}

        if catalog["documentos"].empty:
            log.info("Nenhum documento com status 'processado' ou 'revisado' encontrado no banco para análise.")
            return {"texto": "Não há documentos válidos (status 'processado' ou 'revisado') no banco para realizar a análise.", "tabela": None, "figuras": []}

        log.info("Iniciando AgenteAnalitico para a pergunta: '%s'", pergunta[:100] + "...") # Log truncado
        return self.analitico.responder(pergunta, catalog)

    def revalidar_documento(self, documento_id: int) -> Dict[str, Any]:
        """ Aciona a revalidação de um documento específico. """
        try:
            doc = self.db.get_documento(documento_id)
            if not doc:
                 log.warning("Tentativa de revalidar documento inexistente: ID %d", documento_id)
                 return {"ok": False, "mensagem": f"Documento com ID {documento_id} não encontrado."}

            log.info("Iniciando revalidação para doc_id %d (status atual: %s)", documento_id, doc.get('status'))
            # Força a validação mesmo que já esteja 'processado'
            self.validador.validar_documento(doc_id=documento_id, db=self.db, force_revalidation=True) # Adicionar 'force' se necessário na lógica do validador

            # Busca o novo status após a revalidação
            doc_depois = self.db.get_documento(documento_id)
            novo_status = doc_depois.get('status') if doc_depois else 'desconhecido'

            # Idealmente, o usuário logado seria registrado aqui
            self.db.log("revalidacao", "usuario_sistema", f"doc_id={documento_id}|status_anterior={doc.get('status')}|status_novo={novo_status}|timestamp={self.db.now()}")
            log.info("Revalidação concluída para doc_id %d. Novo status: %s", documento_id, novo_status)
            return {"ok": True, "mensagem": f"Documento revalidado. Novo status: {novo_status}."}
        except Exception as e:
            log.exception("Falha ao revalidar doc_id %d: %s", documento_id, e)
            return {"ok": False, "mensagem": f"Falha ao revalidar: {e}"}