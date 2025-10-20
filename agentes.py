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

# OCR / Imaging (ativados quando instalados no ambiente)
try:
    import pytesseract  # type: ignore
    from PIL import Image, ImageOps, ImageFilter  # type: ignore
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    logging.warning("Bibliotecas Pillow ou pytesseract não encontradas. Funcionalidade de OCR de imagens desativada.")

try:
    from pdf2image import convert_from_bytes  # type: ignore
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
    s2 = re.sub(r"[^\d,.\-]", "", s2)
    if s2.count(",") == 1 and (s2.count(".") == 0 or s2.rfind(",") > s2.rfind(".")):
        s2 = s2.replace(".", "").replace(",", ".")
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
    def __init__(self, db: BancoDeDados, validador: ValidadorFiscal):
        """Inicializa o AgenteXMLParser."""
        self.db = db
        self.validador = validador

    def processar(self, nome: str, conteudo: bytes, origem: str = "upload") -> int:
        """Processa um arquivo XML, extrai dados e os valida."""
        t_start = time.time()
        doc_id = -1
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
            bw = gray.point(lambda x: 0 if x < 180 else 255, '1')
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
            ocr_data = pytesseract.image_to_data(img_proc, lang='por', output_type=pytesseract.Output.DICT)
            confidences = [int(c) for i, c in enumerate(ocr_data['conf']) if int(c) > -1 and ocr_data['text'][i].strip()]
            avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
            texto = pytesseract.image_to_string(img_proc, lang='por')
            return texto, round(avg_conf / 100.0, 2)
        except Exception as e:
            log.error("Erro durante OCR da imagem: %s", e)
            return "", 0.0

    def _ocr_pdf(self, conteudo: bytes) -> Tuple[str, float]:
        """Executa OCR em PDF com cálculo de confiança."""
        if not self.pdf_ok: return "", 0.0
        try:
            images = convert_from_bytes(conteudo, dpi=200) 
            full_text = []
            total_conf = 0.0
            num_valid_pages = 0
            
            for i, img in enumerate(images):
                log.debug("Processando OCR da página %d do PDF", i + 1)
                img_proc = self._preprocess_image(img)
                ocr_data = pytesseract.image_to_data(img_proc, lang='por', output_type=pytesseract.Output.DICT)
                confidences = [int(c) for idx, c in enumerate(ocr_data['conf']) if int(c) > -1 and ocr_data['text'][idx].strip()]
                if confidences:
                    avg_conf_page = sum(confidences) / len(confidences)
                    total_conf += avg_conf_page
                    num_valid_pages += 1
                page_text = pytesseract.image_to_string(img_proc, lang='por')
                full_text.append(page_text)
            
            final_text = "\n\n--- Page Break ---\n\n".join(full_text)
            final_avg_conf = (total_conf / num_valid_pages) if num_valid_pages > 0 else 0.0
            return final_text, round(final_avg_conf / 100.0, 2)
        except Exception as e:
            log.error("Erro durante OCR do PDF: %s", e)
            return "", 0.0


# ------------------------------ Agente NLP (Aprimorado) ------------------------------
class AgenteNLP:
    """Extrai campos fiscais de texto bruto usando regex aprimoradas."""
    RE_CNPJ = re.compile(r"\b(\d{2}\.?\d{3}\.?\d{3}/?\d{4}-?\d{2}|\d{14})\b")
    RE_CPF = re.compile(r"\b(\d{3}\.?\d{3}\.?\d{3}-?\d{2}|\d{11})\b")
    RE_IE = re.compile(r"\b(?:IE|I\.E\.|INSC(?:RI[ÇC][ÃA]O)?\sESTADUAL)[:\s\-]*([A-Z0-9.\-/]{5,20})\b", re.I)
    RE_UF = re.compile(r"\b(AC|AL|AP|AM|BA|CE|DF|ES|GO|MA|MT|MS|MG|PA|PB|PR|PE|PI|RJ|RN|RS|RO|RR|SC|SP|SE|TO)\b")
    RE_VALOR_TOTAL = re.compile(r"\b(?:VALOR\s+TOTAL\s+DA\s+NOTA|VALOR\s+TOTAL|TOTAL\s+DA\s+NOTA)\s*[:\-]?\s*R?\$\s*([\d.,]+)\b", re.I)
    RE_DATA_EMISSAO = re.compile(r"\b(?:DATA\s+(?:DE\s+)?EMISS[ÃA]O|EMITIDO\s+EM)\s*[:\-]?\s*(\d{2,4}[-/]\d{2}[-/]\d{2,4})\b", re.I)
    # TODO: Adicionar regex para itens e impostos se necessário

    def extrair_campos(self, texto: str) -> Dict[str, Any]:
        """Extrai os principais campos de cabeçalho do texto."""
        t_norm = _norm_ws(texto)
        
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
        if endereco_match: m_uf = self.RE_UF.search(endereco_match[-10:])
        if not uf and municipio_match: m_uf = self.RE_UF.search(municipio_match[-10:])
        if not uf: m_uf = self.RE_UF.search(t_norm)
        uf = m_uf.group(1) if m_uf else None
        
        razao = self._match_after(t_norm, ["razão social", "razao social", "nome", "emitente"], max_len=100)
        
        m_valor = self.RE_VALOR_TOTAL.search(t_norm)
        valor_total = _to_float_br(m_valor.group(1)) if m_valor else None

        m_data = self.RE_DATA_EMISSAO.search(t_norm)
        data_emissao = _parse_date_like(m_data.group(1)) if m_data else None

        # A extração de itens e impostos de texto OCR é complexa e omitida nesta versão.
        
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
                # Separadores aprimorados: :, -, |, ;, \n
                for i in range(idx + len(lab_lower), min(idx + len(lab_lower) + max_dist, len(texto))):
                    if texto[i] in ':|-;\n': 
                        start_value_idx = i + 1
                        break
                
                if start_value_idx != -1:
                    end_idx = texto.find('\n', start_value_idx)
                    if end_idx == -1: end_idx = len(texto)
                    
                    for next_lab in labels: 
                        next_idx = texto.find(next_lab, start_value_idx, end_idx)
                        if next_idx != -1: end_idx = next_idx

                    value = texto[start_value_idx : end_idx].strip()
                    value = re.sub(r'\s*[|;].*$', '', value).strip() # Remove lixo após | ou ;
                    if value and idx < min_pos:
                         min_pos = idx
                         best_match = value[:max_len] 

        return best_match


# ------------------------------ Agente Analítico (LLM → Sandbox - Final e Seguro) ------------------------------
class SecurityException(Exception): pass
ALLOWED_IMPORTS = {"pandas", "numpy", "matplotlib", "plotly"}

def _restricted_import(name: str, *args, **kwargs):
    """Função de import restrita para o sandbox."""
    if name.split(".")[0] not in ALLOWED_IMPORTS:
        raise SecurityException(f"Importação proibida: {name}")
    return builtins.__import__(name, *args, **kwargs)

# SAFE_BUILTINS final e correto, definido uma vez no nível do módulo.
SAFE_BUILTINS = {k: getattr(builtins, k) for k in (
    "abs", "all", "any", "bool", "dict", "enumerate", "float", "int", "isinstance", 
    "len", "list", "max", "min", "print", "range", "round", "set", "sorted", 
    "str", "sum", "tuple", "type", "zip"
)}
SAFE_BUILTINS["__import__"] = _restricted_import

class AgenteAnalitico:
    """Gera e executa código Python via LLM com auto-correção."""
    def __init__(self, llm: BaseChatModel, memoria: MemoriaSessao):
        self.llm = llm
        self.memoria = memoria
        self.last_code: str = ""

    def _prompt_inicial(self, catalog: Dict[str, pd.DataFrame]) -> SystemMessage:
        """ Constrói o prompt inicial para a geração de código. """
        schema_lines = []
        example_table_name = next(iter(catalog.keys()), 'tabela_exemplo')
        for t, df in catalog.items():
            schema_lines.append(f"- Tabela `{t}` ({df.shape[0]} linhas): Colunas: `{', '.join(map(str, df.columns))}`")
        schema = "\n".join(schema_lines) or "- (Nenhum dado carregado)"
        history = self.memoria.resumo()

        prompt = f"""
        Você é um agente de análise de dados de elite expert em Python. Sua tarefa é gerar código Python robusto e bem formatado para uma função 'solve'.

        **REGRAS CRÍTICAS DE EXECUÇÃO:**
        1.  **CRÍTICO:** Todas as declarações de `import` DEVEM estar DENTRO da função `solve`.
        2.  Imports permitidos: {', '.join(ALLOWED_IMPORTS)}. NENHUM OUTRO será permitido pelo sandbox.
        3.  Use APENAS funções built-in seguras. O sandbox bloqueará outras.
        4.  Acesse dados via `catalog['nome_tabela']`. Use `.copy()`.
        5.  Retorne: `(texto: str, tabela: pd.DataFrame|None, figura: plt.Figure|go.Figure|None)`.
        6.  Seja DEFENSIVO: Use `pd.to_numeric(..., errors='coerce').dropna()`.
        7.  GRÁFICOS: Prefira `plotly.express as px`. Tamanho: `fig.update_layout(width=800, height=500)`. Use `plt.tight_layout()` para matplotlib (`figsize=(10,6)`).
        8.  NÃO use `.to_string()` no `texto` se retornar `tabela`.

        **ESQUEMA:**
        {schema}

        **HISTÓRICO:**
        {history}
        
        **ESTRUTURA OBRIGATÓRIA:**
        ```python
        def solve(catalog, question):
            # Imports AQUI
            import pandas as pd
            import numpy as np
            import matplotlib.pyplot as plt
            import plotly.express as px
            import plotly.graph_objects as go
            
            df = catalog['{example_table_name}'].copy() 
            
            # --- Seu código robusto ---
            
            text_output = "# Análise"
            table_output = None
            figure_output = None 

            return (text_output, table_output, figure_output)
        ```
        Gere APENAS o código Python completo da função `solve`.
        """
        return SystemMessage(content=prompt)

    def _prompt_correcao(self, failed_code: str, error_message: str) -> SystemMessage:
        """ Constrói o prompt para a correção de código. """
        prompt = f"""
        O código abaixo falhou. Reescreva a função solve corrigida, mantendo as regras originais.

        ERRO: {error_message}

        CÓDIGO:
        ```python
        {failed_code}
        ```
        Reescreva APENAS a função `solve` corrigida. Imports DENTRO dela. Lembre-se das regras.
        """
        return SystemMessage(content=prompt)

    def _gerar_codigo(self, pergunta: str, catalog: Dict[str, pd.DataFrame]) -> str:
        """ Gera a primeira versão do código. """
        sys = self._prompt_inicial(catalog)
        hum = HumanMessage(content=f"Pergunta do usuário: {pergunta}")
        resp = self.llm.invoke([sys, hum]).content.strip()
        code = resp.removeprefix("```python").removesuffix("```").strip()
        self.last_code = code
        return code

    def _corrigir_codigo(self, failed_code: str, erro: str) -> str:
        """ Gera uma versão corrigida do código. """
        sys = self._prompt_correcao(failed_code, erro)
        hum = HumanMessage(content="Corrija a função `solve`.")
        resp = self.llm.invoke([sys, hum]).content.strip()
        code = resp.removeprefix("```python").removesuffix("```").strip()
        self.last_code = code
        return code

    def _executar_sandbox(self, code: str, pergunta: str, catalog: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """ Executa o código em sandbox seguro com escopo unificado e builtins restritos. """
        # Escopo unificado com SAFE_BUILTINS garante a segurança correta
        scope = {"__builtins__": SAFE_BUILTINS}
        exec(code, scope) 

        if "solve" not in scope or not callable(scope["solve"]):
            raise RuntimeError("A função `solve` não foi definida corretamente.")

        solve_fn = scope["solve"]
        t0 = time.time()
        texto, tabela, fig = solve_fn({k: v.copy() for k, v in catalog.items()}, pergunta)
        dt = time.time() - t0

        # Padronizado para chaves em português
        return {
            "texto": str(texto) if texto is not None else "",
            "tabela": tabela if isinstance(tabela, pd.DataFrame) else None,
            "figuras": [fig] if fig is not None else [],
            "duracao_s": round(dt, 3),
            "code": code,
        }

    def responder(self, pergunta: str, catalog: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """ Orquestra a geração, execução e auto-correção do código. """
        max_retries = 2
        
        try:
            code = self._gerar_codigo(pergunta, catalog)
            
            for attempt in range(max_retries + 1):
                try:
                    out = self._executar_sandbox(code, pergunta, catalog)
                    self.memoria.salvar(pergunta, out.get("texto", ""), duracao_s=out.get("duracao_s", 0.0))
                    out["agent_name"] = f"AgenteAnalitico (Tentativa {attempt + 1})"
                    out["summary"] = f"Executou código para: '{pergunta}'"
                    return out 
                
                except Exception as e1:
                    error_message = f"{type(e1).__name__}: {e1}"
                    log.warning("Falha na tentativa %d para '%s': %s", attempt + 1, pergunta, error_message)
                    if attempt < max_retries:
                        code = self._corrigir_codigo(code, error_message)
                    else:
                        raise e1 
        
        except Exception as e_final:
            traceback_str = traceback.format_exc(limit=3) 
            log.error("Falha final no AgenteAnalitico para '%s': %s\n%s", pergunta, e_final, traceback_str)
            summary = f"Falha final na auto-correção para: '{pergunta}'"
            self.memoria.salvar(pergunta, f"Erro: {e_final}", duracao_s=0.0)
            return {
                "texto": f"O agente não conseguiu corrigir o próprio código após {max_retries + 1} tentativas. Erro final: {e_final}",
                "tabela": None,
                "figuras": [],
                "duracao_s": 0.0,
                "code": self.last_code or "",
                "agent_name": "AgenteAnalitico (Falha Final)",
                "summary": summary
            }

# ------------------------------ Orchestrator (Aprimorado e Completo) ------------------------------
@dataclass
class Orchestrator:
    """ Coordena o pipeline de processamento de documentos e análise. """
    db: BancoDeDados
    validador: ValidadorFiscal
    memoria: MemoriaSessao
    llm: Optional[BaseChatModel] = None

    def __post_init__(self):
        """Inicializa os agentes."""
        if not CORE_MODULES_AVAILABLE:
             log.error("Orchestrator não pode ser inicializado corretamente devido a dependências ausentes.")
             # Poderia levantar uma exceção aqui para impedir a execução
             # raise RuntimeError("Dependências críticas (banco_de_dados, validacao, memoria) ausentes.")
        self.xml_agent = AgenteXMLParser(self.db, self.validador)
        self.ocr_agent = AgenteOCR()
        self.nlp_agent = AgenteNLP()
        self.analitico = AgenteAnalitico(self.llm, self.memoria) if self.llm else None

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
                log.info("Documento '%s' (hash %s) já existe com ID %d. Ignorando.", nome, doc_hash, existing_id)
                return existing_id

            if ext == ".xml":
                doc_id = self.xml_agent.processar(nome, conteudo, origem)
            elif ext in {".pdf", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}:
                doc_id = self._processar_midias(nome, conteudo, origem)
            else:
                motivo = "Extensão não suportada."
                status = "quarentena"
                doc_id = self.db.inserir_documento(
                    nome_arquivo=nome, tipo="desconhecido", origem=origem, hash=doc_hash,
                    chave_acesso=None, status=status, data_upload=self.db.now(), motivo_rejeicao=motivo
                )
            
            # Busca o status final do banco após o processamento ter ocorrido (ou falhado)
            if doc_id > 0:
                 doc_info = self.db.get_documento(doc_id)
                 if doc_info: status = doc_info.get("status", status)

        except Exception as e:
            log.exception("Falha na ingestão de %s: %s", nome, e)
            motivo = str(e)
            try:
                # Tenta garantir que um registro de erro exista
                existing_id_on_error = self.db.find_documento_by_hash(doc_hash)
                if existing_id_on_error:
                    doc_id = existing_id_on_error
                    self.db.atualizar_documento_campo(doc_id, "status", "erro")
                    self.db.atualizar_documento_campo(doc_id, "motivo_rejeicao", motivo)
                elif doc_id <= 0: # Apenas cria se não existir e não foi criado na tentativa
                     doc_id = self.db.inserir_documento(
                        nome_arquivo=nome, tipo=ext.strip('.'), origem=origem, hash=doc_hash,
                        chave_acesso=None, status="erro", data_upload=self.db.now(), motivo_rejeicao=motivo
                    )
            except Exception as db_err:
                 log.error("Erro CRÍTICO ao registrar falha de ingestão para '%s': %s", nome, db_err)
                 return -1 
        finally:
            log.info("Ingestão de '%s' concluída (ID: %d, Status: %s) em %.2fs", 
                     nome, doc_id, status, time.time() - t_start)
            
        return doc_id

    def _processar_midias(self, nome: str, conteudo: bytes, origem: str) -> int:
        """ Processa arquivos PDF ou Imagem via OCR e NLP. """
        doc_id = self.db.inserir_documento(
            nome_arquivo=nome, tipo=Path(nome).suffix.lower().strip('.'), origem=origem,
            hash=self.db.hash_bytes(conteudo), status="processando",
            data_upload=self.db.now(), caminho_arquivo=str(self.db.save_upload(nome, conteudo))
        )

        texto = ""
        conf = 0.0
        t_start = time.time()
        try:
            texto, conf = self.ocr_agent.reconhecer(nome, conteudo)
            ocr_time = time.time() - t_start
            
            self.db.inserir_extracao(
                documento_id=doc_id, agente="OCRAgent", confianca_media=float(conf),
                texto_extraido=texto[:100000], linguagem="pt", tempo_processamento=round(ocr_time, 3)
            )

            status_final = "erro" # Status padrão se algo falhar
            if texto: 
                log.info("Texto extraído para doc_id %d. Iniciando NLP.", doc_id)
                campos = self.nlp_agent.extrair_campos(texto)
                log.info("Campos NLP extraídos para doc_id %d: %s", doc_id, list(campos.keys()))
                self.db.atualizar_documento_campos(doc_id, **campos) 
                
                log.info("Iniciando validação para doc_id %d.", doc_id)
                self.validador.validar_documento(doc_id=doc_id, db=self.db) 
                
                # Re-busca o status após validação, pois ela pode alterá-lo
                doc_info_after_validation = self.db.get_documento(doc_id)
                status_depois_validacao = doc_info_after_validation.get("status") if doc_info_after_validation else "erro"
                
                # Se a validação não marcou como pendente, usa a confiança do OCR
                if status_depois_validacao != "revisao_pendente":
                    status_final = "processado" if conf >= 0.6 else "revisao_pendente"
                else:
                    status_final = "revisao_pendente" # Mantém se a validação falhou
            else:
                 status_final = "revisao_pendente" # OCR falhou em extrair texto
                 log.warning("OCR não extraiu texto para doc_id %d (conf: %.2f)", doc_id, conf)

            self.db.atualizar_documento_campo(doc_id, "status", status_final)
            self.db.log("ingestao_midias", "sistema", f"doc_id={doc_id}|conf={conf:.2f}|status={status_final}")

        except Exception as e:
            log.exception("Falha no processamento de mídia para doc_id %d: %s", doc_id, e)
            self.db.atualizar_documento_campo(doc_id, "status", "erro")
            self.db.log("ocr_erro", "sistema", f"doc_id={doc_id}|erro={e}")

        return doc_id

    def responder_pergunta(self, pergunta: str) -> Dict[str, Any]:
        """ Delega a pergunta analítica para o AgenteAnalitico. """
        if not self.analitico:
            log.error("Agente Analítico não inicializado. Verifique a configuração do LLM.")
            return {"texto": "Erro interno: Agente analítico não configurado.", "tabela": None, "figuras": []}

        catalog: Dict[str, pd.DataFrame] = {}
        try:
            # Carrega apenas documentos com status 'processado' ou 'revisado' (assumindo revisão como OK)
            where_clause = "status = 'processado' OR status = 'revisado'" # Exemplo, ajuste conforme seu status
            catalog["documentos"] = self.db.query_table("documentos", where=where_clause)
            
            if not catalog["documentos"].empty:
                 doc_ids = tuple(catalog["documentos"]['id'].unique().tolist())
                 doc_ids_sql = f"({doc_ids[0]})" if len(doc_ids) == 1 else str(doc_ids)
                 
                 catalog["itens"] = self.db.query_table("itens", where=f"documento_id IN {doc_ids_sql}")
                 if not catalog["itens"].empty:
                    item_ids = tuple(catalog["itens"]['id'].unique().tolist())
                    item_ids_sql = f"({item_ids[0]})" if len(item_ids) == 1 else str(item_ids)
                    catalog["impostos"] = self.db.query_table("impostos", where=f"item_id IN {item_ids_sql}")
                 else:
                     catalog["impostos"] = pd.DataFrame() 
            else:
                 catalog["itens"] = pd.DataFrame()
                 catalog["impostos"] = pd.DataFrame()
                 
        except Exception as e:
            log.exception("Falha ao montar catálogo do banco de dados para análise: %s", e)
            return {"texto": f"Erro ao carregar dados para análise: {e}", "tabela": None, "figuras": []}

        if catalog["documentos"].empty:
            log.info("Nenhum documento válido encontrado no banco para análise.")
            return {"texto": "Não há documentos válidos no banco para realizar a análise. Verifique o status dos documentos processados.", "tabela": None, "figuras": []}
        
        log.info("Iniciando AgenteAnalitico para a pergunta: '%s'", pergunta)
        return self.analitico.responder(pergunta, catalog)

    def revalidar_documento(self, documento_id: int) -> Dict[str, Any]:
        """ Aciona a revalidação de um documento específico. """
        try:
            doc = self.db.get_documento(documento_id)
            if not doc:
                 log.warning("Tentativa de revalidar documento inexistente: ID %d", documento_id)
                 return {"ok": False, "mensagem": f"Documento com ID {documento_id} não encontrado."}
                 
            log.info("Iniciando revalidação para doc_id %d", documento_id)
            self.validador.validar_documento(doc_id=documento_id, db=self.db)
            # Idealmente, o usuário logado seria registrado aqui
            self.db.log("revalidacao", "usuario_sistema", f"doc_id={documento_id}|timestamp={self.db.now()}") 
            log.info("Revalidação concluída para doc_id %d", documento_id)
            return {"ok": True, "mensagem": "Documento revalidado com sucesso."}
        except Exception as e:
            log.exception("Falha ao revalidar doc_id %d: %s", documento_id, e)
            return {"ok": False, "mensagem": f"Falha ao revalidar: {e}"}