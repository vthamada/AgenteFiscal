# agentes/agente_ocr.py

from __future__ import annotations

import io
import logging
import time
import re
import hashlib
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
from PIL import Image, ImageFilter, ImageOps

log = logging.getLogger("agente_fiscal.agentes")

try:
    from langchain_core.language_models.chat_models import BaseChatModel  # type: ignore
except Exception:  # pragma: no cover
    BaseChatModel = object  # type: ignore


# ------------------------- Disponibilidade OCR/PDF -------------------------
try:
    import easyocr  # type: ignore
    OCR_AVAILABLE = True
except Exception as e:  # pragma: no cover
    OCR_AVAILABLE = False
    easyocr = None  # type: ignore
    log.debug(f"easyocr indisponível: {e}")

PDF_RENDERER: str | None = None
PDF_AVAILABLE = False
pdfium = None
convert_from_bytes = None

# Preferimos pypdfium2 (extrai texto e renderiza). Fallback: pdf2image.
try:
    import pypdfium2 as pdfium  # type: ignore
    PDF_RENDERER = "pdfium"
    PDF_AVAILABLE = True
except Exception:
    try:
        from pdf2image import convert_from_bytes  # type: ignore
        PDF_RENDERER = "pdf2image"
        PDF_AVAILABLE = True
    except Exception as e:  # pragma: no cover
        log.debug(f"Nenhum renderizador/extração PDF disponível: {e}")

# Fallback extra de extração de texto nativo (sem renderizar) via pdfminer.six
_PDFMINER_AVAILABLE = False
try:
    from pdfminer.high_level import extract_text as pdfminer_extract_text  # type: ignore
    _PDFMINER_AVAILABLE = True
except Exception as e:
    log.debug(f"pdfminer.six indisponível: {e}")


# ------------------------------- Agente OCR --------------------------------
class AgenteOCR:
    """
    OCR com EasyOCR. Para PDFs:
      1) tenta extrair TEXTO NATIVO (pypdfium2); se ruim, tenta pdfminer.six;
      2) se insuficiente/ruim, renderiza (~300DPI) e roda OCR (1..2 passes adaptativos).

    Para imagens: pré-processamento adaptativo + 1..2 passes OCR.

    Se uma LLM for fornecida, executa correção cognitiva leve (pós-processamento),
    chamada somente quando necessário (gating por confiança/sinais de baixa qualidade).
    """

    # Heurísticas de “texto nativo suficiente”
    _MIN_NATIVE_LEN = 20
    _MIN_NATIVE_TOKENS = 6   # tokens (palavras) mínimas
    _MIN_NATIVE_LINES = 2    # linhas mínimas

    # Tamanhos/limites
    _MAX_TEXT_CHARS = 400_000  # corta para evitar estourar memória/logs

    # Cache interno (na sessão do processo)
    _CACHE_MAX_ENTRIES = 128
    _cache: Dict[str, Tuple[str, float, float]] = {}  # key -> (texto, conf, timestamp)

    def __init__(
        self,
        llm: Optional["BaseChatModel"] = None,
        *,
        aplicar_llm_quando_conf_abaixo: float = 0.98,
    ):
        # Core flags
        self.ocr_ok = OCR_AVAILABLE
        self.pdf_ok = PDF_AVAILABLE
        self.reader = None

        # Camada cognitiva (opcional)
        self.llm: Optional["BaseChatModel"] = llm
        self.modo_cognitivo: bool = llm is not None

        # Normaliza limiar para [0, 1]
        try:
            self._limiar_llm = float(aplicar_llm_quando_conf_abaixo)
        except Exception:
            self._limiar_llm = 0.98
        self._limiar_llm = max(0.0, min(1.0, self._limiar_llm))

        # Estatísticas da última execução (debug/telemetria)
        self.last_stats: Dict[str, Any] = {
            "pages": 0,
            "ocr_blocks": 0,
            "used_native_pdf_text": False,
            "avg_confidence": 0.0,
            "llm_correction_applied": False,
            "native_lines": 0,
            "native_tokens": 0,
            "passes_ocr": 0,
            "cache_hit": False,
            "native_source": None,  # "pdfium" | "pdfminer" | None
        }

        # Inicialização do OCR
        if self.ocr_ok:
            try:
                # GPU=False para funcionar em CPU
                self.reader = easyocr.Reader(["pt", "en"], gpu=False)  # type: ignore
                log.info("OCR (EasyOCR) disponível.")
            except Exception as e:
                self.ocr_ok = False
                self.reader = None
                log.warning(f"Falha ao inicializar EasyOCR: {e}")
        else:
            log.warning("OCR (EasyOCR) NÃO disponível.")

        if self.pdf_ok:
            log.info(f"Renderizador PDF: {PDF_RENDERER}.")
        else:
            log.warning("Nenhum renderizador de PDF disponível.")

    # ------------------------------- Público --------------------------------
    def reconhecer(self, nome: str, conteudo: bytes) -> Tuple[str, float]:
        t_start = time.time()
        self._reset_stats()

        ext = Path(nome).suffix.lower()
        cache_key = self._make_cache_key(nome, conteudo)
        if cache_key in self._cache:
            texto_cached, conf_cached, _ts = self._cache[cache_key]
            self.last_stats["cache_hit"] = True
            self.last_stats["avg_confidence"] = float(conf_cached or 0.0)
            log.info("OCR cache HIT para '%s'", nome)
            return texto_cached, float(conf_cached or 0.0)

        texto = ""
        conf = 0.0

        try:
            if ext == ".pdf":
                if not self.pdf_ok and not _PDFMINER_AVAILABLE:
                    raise RuntimeError("Nenhum extrator/renderizador PDF disponível.")
                texto, conf = self._processar_pdf(conteudo)
            elif ext in {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}:
                if not self.ocr_ok:
                    raise RuntimeError("OCR para imagem indisponível.")
                texto, conf = self._ocr_imagem(conteudo)
            else:
                raise ValueError(f"Extensão não suportada para OCR: {ext}")

            # Pós-processamento + correções semânticas leves
            texto = self._postprocess_text(texto)
            texto = self._fix_semantic_artifacts(texto)

            # Correção cognitiva opcional (pós-processamento) — gating
            if self._should_use_llm(texto, conf):
                texto, conf = self._corrigir_texto_ocr(texto, conf)

            # Segurança de retorno
            texto = (texto or "").strip()
            if len(texto) > self._MAX_TEXT_CHARS:
                texto = texto[: self._MAX_TEXT_CHARS] + "\n...[TRUNCADO]"

            # Guarda em cache
            self._cache_put(cache_key, texto, float(conf or 0.0))

            return texto, float(conf or 0.0)

        except Exception as e:
            log.error("Erro OCR '%s': %s", nome, e)
            # Retorna vazio, mas consistente
            return "", 0.0
        finally:
            log.info(
                "OCR '%s' → conf=%.2f | pages=%s | blocks=%s | native=%s | native_src=%s | llm=%s | cache=%s | %.2fs",
                nome,
                float(self.last_stats.get("avg_confidence", conf)),
                int(self.last_stats.get("pages", 0)),
                int(self.last_stats.get("ocr_blocks", 0)),
                bool(self.last_stats.get("used_native_pdf_text", False)),
                str(self.last_stats.get("native_source")),
                bool(self.last_stats.get("llm_correction_applied", False)),
                bool(self.last_stats.get("cache_hit", False)),
                time.time() - t_start,
            )

    # ---------------------------- PDF Helpers -------------------------------
    def _pdf_text_nativo_pdfium(self, content: bytes) -> str:
        """Extração de texto nativo usando pypdfium2."""
        if PDF_RENDERER != "pdfium" or pdfium is None:
            return ""
        txt_total: List[str] = []
        try:
            pdf = pdfium.PdfDocument(io.BytesIO(content))
            self.last_stats["pages"] = len(pdf)
            log.debug("PDF aberto (pdfium): %s páginas.", len(pdf))
            for idx, page in enumerate(pdf):
                try:
                    textpage = page.get_textpage()
                    # get_text_range() => texto ordenado visualmente
                    txt = (textpage.get_text_range() or "").strip()
                    if txt:
                        txt_total.append(txt)
                    textpage.close()
                except Exception as e:
                    log.debug(f"Falha texto nativo na página {idx + 1} (pdfium): {e}")
            pdf.close()
        except Exception as e:
            log.debug(f"Falha ao extrair texto nativo via pdfium: {e}")
        return "\n".join(txt_total).strip()

    def _pdf_text_nativo_pdfminer(self, content: bytes) -> str:
        """Extração de texto nativo usando pdfminer.six (fallback)."""
        if not _PDFMINER_AVAILABLE:
            return ""
        try:
            txt = pdfminer_extract_text(io.BytesIO(content)) or ""
            if txt:
                # pdfminer costuma manter quebras; ainda assim aplicamos leve normalização
                txt = txt.replace("\r", "\n")
            return txt.strip()
        except Exception as e:
            log.debug(f"Falha pdfminer_extract_text: {e}")
            return ""

    def _pdf_text_nativo(self, content: bytes) -> str:
        """
        Extração de texto nativo do PDF (sem OCR). Primeiro pdfium; se vazio/ruim,
        tenta pdfminer.six. Retorna string (pode ser vazia).
        """
        texto = ""
        src = None

        # 1) pdfium
        if PDF_RENDERER == "pdfium" and pdfium is not None:
            texto = self._pdf_text_nativo_pdfium(content)
            src = "pdfium"

        # 2) pdfminer fallback se texto estiver muito pobre
        if (not texto or self._looks_monoblock(texto)) and _PDFMINER_AVAILABLE:
            txt2 = self._pdf_text_nativo_pdfminer(content)
            if len(txt2) > len(texto):
                texto = txt2
                src = "pdfminer"

        # Heurística de "monobloco" → insere quebras em rótulos comuns
        if texto and (self._looks_monoblock(texto)):
            texto = self._inject_linebreaks_labels(texto)

        # Estatística de qualidade do texto nativo
        self.last_stats["native_lines"] = self._count_lines(texto)
        self.last_stats["native_tokens"] = self._count_tokens(texto)
        self.last_stats["native_source"] = src
        return texto

    def _render_paginas_pdf(self, content: bytes, scale: float = 2.4, dpi_fallback: int = 300) -> List[Image.Image]:
        """
        Renderiza cada página em ~300DPI. Usa pdfium quando possível;
        fallback para pdf2image.
        """
        imagens: List[Image.Image] = []
        try:
            if PDF_RENDERER == "pdfium" and pdfium is not None:
                pdf = pdfium.PdfDocument(io.BytesIO(content))
                self.last_stats["pages"] = len(pdf)
                for page in pdf:
                    pil_img = page.render(scale=scale).to_pil().convert("RGB")
                    imagens.append(pil_img)
                pdf.close()
            elif PDF_RENDERER == "pdf2image" and convert_from_bytes is not None:
                imagens = [im.convert("RGB") for im in convert_from_bytes(content, dpi=dpi_fallback)]
                self.last_stats["pages"] = len(imagens)
            else:
                log.error("Nenhum renderizador disponível para PDF.")
        except Exception as e:
            log.error(f"Falha ao renderizar PDF: {e}")
        return imagens

    def _processar_pdf(self, content: bytes) -> Tuple[str, float]:
        """
        Estratégia:
          1) Extrai texto nativo (se houver) e avalia suficiência/qualidade (pdfium→pdfminer);
             - Se suficiente: aceita (conf≈0.99).
             - Se monobloco pobre/curto: tenta aplicar quebras; se ainda ruim → fallback OCR.
          2) OCR em páginas renderizadas (1..2 passes adaptativos).
        """
        # 1) Texto nativo primeiro (melhor qualidade e estrutura)
        texto_nativo = self._pdf_text_nativo(content)
        if self._is_native_usable(texto_nativo):
            self.last_stats["used_native_pdf_text"] = True
            self.last_stats["avg_confidence"] = 0.99
            log.debug("Texto nativo suficiente; pulando OCR.")
            return texto_nativo.strip(), 0.99

        # 2) OCR em páginas renderizadas
        if not (self.ocr_ok and self.reader):
            return "", 0.0

        # Passo A: render padrão
        paginas = self._render_paginas_pdf(content, scale=2.4, dpi_fallback=300)
        texto_ocr, conf_ocr, blocks = self._ocr_paginas(paginas, pass_name="pass1")
        self.last_stats["passes_ocr"] = 1

        # Passo B (fallback): se confiança fraca/sem texto, reforça binarização + upscale
        if (not texto_ocr) or conf_ocr < 0.55:
            paginas2 = [self._preprocess_pil(img, aggressive=True) for img in paginas]
            texto_ocr2, conf_ocr2, blocks2 = self._ocr_paginas(paginas2, pass_name="pass2_aggressive")
            if conf_ocr2 > conf_ocr or len(texto_ocr2) > len(texto_ocr):
                texto_ocr, conf_ocr, blocks = texto_ocr2, conf_ocr2, blocks2
            self.last_stats["passes_ocr"] = 2

        self.last_stats["ocr_blocks"] = int(blocks)
        self.last_stats["avg_confidence"] = float(round(conf_ocr, 2))
        return texto_ocr.strip(), float(round(conf_ocr, 2))

    # ------------------------ OCR Helpers (PDF/Imagem) ----------------------
    def _ocr_paginas(self, paginas: List[Image.Image], pass_name: str) -> Tuple[str, float, int]:
        """
        Executa OCR página a página e retorna (texto, confiança_media_ponderada, total_blocos).
        """
        if not paginas:
            return "", 0.0, 0
        if not (self.ocr_ok and self.reader):
            return "", 0.0, 0

        textos: List[str] = []
        confs: List[float] = []
        pesos: List[int] = []
        total_blocos = 0

        for idx, pil_img in enumerate(paginas, start=1):
            # Pré-processamento leve para cada página desta passada
            pil_proc = self._preprocess_pil(pil_img, aggressive=False)
            np_img = np.array(pil_proc)

            try:
                results = self.reader.readtext(np_img, detail=1, paragraph=True)  # type: ignore
                total_blocos += len(results or [])
                page_parts: List[str] = []
                if results:
                    for r in results:
                        # r: [bbox, text, conf]
                        t = (r[1] or "").strip()
                        c = float(r[2])
                        if t:
                            page_parts.append(t)
                            confs.append(c)
                            pesos.append(max(1, len(t)))
                page_text = " ".join(page_parts)
                textos.append(page_text)
                log.debug("[%s] Página %d: %d blocos OCR.", pass_name, idx, len(results or []))
            except Exception as e:
                log.debug(f"[{pass_name}] OCR falhou em uma página ({idx}): {e}")

        # Junta e limpa duplicidades simples entre páginas
        texto_final = "\n\n--- Page Break ---\n\n".join([t for t in textos if t]).strip()
        texto_final = self._dedup_repeated_blocks(texto_final)

        # Média ponderada por tamanho do texto reconhecido (mais robusta)
        media_conf = float(np.average(confs, weights=pesos)) if (confs and pesos and sum(pesos) > 0) else 0.0
        media_conf = float(round(media_conf, 2))
        return texto_final, media_conf, total_blocos

    # ------------------------ Imagem & Pré-processo -------------------------
    def _preprocess_pil(self, pil_img: Image.Image, aggressive: bool = False) -> Image.Image:
        """
        Pré-processamento adaptativo e seguro:
          - grayscale
          - sharpen
          - optional auto-contrast
          - upscale suave para melhorar OCR em fontes pequenas
          - binarização adaptativa (Otsu) quando contraste baixo
          - modo aggressive aplica binarização forte e leve dilatação
        """
        try:
            img = pil_img.convert("L")  # grayscale
        except Exception:
            img = ImageOps.grayscale(pil_img)

        # Nitidez leve
        try:
            img = img.filter(ImageFilter.SHARPEN)
        except Exception:
            pass

        # Auto contraste (suave)
        try:
            img = ImageOps.autocontrast(img, cutoff=1)
        except Exception:
            pass

        # Upscale suave para textos pequenos
        w, h = img.size
        if w < 1200:
            scale = 1200 / float(w)
            img = img.resize((int(w * scale), int(h * scale)))

        # Binarização adaptativa (Otsu) se contraste baixo
        try:
            lo, hi = img.getextrema()
            if (hi - lo) < 60 or aggressive:
                np_img = np.array(img)
                thr = self._otsu_threshold(np_img)
                np_bin = (np_img > max(120, thr)).astype(np.uint8) * 255
                img = Image.fromarray(np_bin, mode="L")
                # Em modo agressivo, leve "espessamento" com max filter
                if aggressive:
                    img = img.filter(ImageFilter.MaxFilter(3))
        except Exception:
            pass

        return img

    @staticmethod
    def _otsu_threshold(gray: np.ndarray) -> int:
        """Limiar de Otsu para array 8-bit."""
        if gray.ndim != 2:
            gray = gray.mean(axis=2).astype(np.uint8)
        hist, _ = np.histogram(gray, bins=256, range=(0, 256))
        total = gray.size
        sum_total = np.dot(np.arange(256), hist)
        sum_b, w_b, w_f, var_max, threshold = 0.0, 0.0, 0.0, 0.0, 0
        for t in range(256):
            w_b += hist[t]
            if w_b == 0:
                continue
            w_f = total - w_b
            if w_f == 0:
                break
            sum_b += t * hist[t]
            m_b = sum_b / w_b
            m_f = (sum_total - sum_b) / w_f
            var_between = w_b * w_f * (m_b - m_f) ** 2
            if var_between > var_max:
                var_max = var_between
                threshold = t
        return threshold

    def _ocr_imagem(self, content: bytes) -> Tuple[str, float]:
        """
        OCR para imagens soltas: 1..2 passes (normal + agressivo) com
        pós-processamento idêntico ao PDF.
        """
        if not (self.ocr_ok and self.reader):
            return "", 0.0
        try:
            pil = Image.open(io.BytesIO(content)).convert("RGB")
        except Exception as e:
            log.error(f"Falha ao abrir imagem: {e}")
            return "", 0.0

        # Passo 1
        pil1 = self._preprocess_pil(pil, aggressive=False)
        texto1, conf1, blocks1 = self._ocr_paginas([pil1], pass_name="img_pass1")

        # Passo 2 (se necessário)
        if (not texto1) or conf1 < 0.55:
            pil2 = self._preprocess_pil(pil, aggressive=True)
            texto2, conf2, blocks2 = self._ocr_paginas([pil2], pass_name="img_pass2_aggressive")
            if conf2 > conf1 or len(texto2) > len(texto1):
                texto1, conf1, blocks1 = texto2, conf2, blocks2

        self.last_stats["pages"] = 1
        self.last_stats["ocr_blocks"] = int(blocks1)
        self.last_stats["avg_confidence"] = float(round(conf1, 2))
        return texto1.strip(), float(round(conf1, 2))

    # ---------------------- Pós-processamento de Texto ----------------------
    def _postprocess_text(self, texto: str) -> str:
        """
        - Normaliza espaços/quebras
        - Des‐hifeniza quebras artificiais de OCR
        - Insere quebras entre rótulos fiscais quando vier monobloco
        - Remove artefatos comuns
        - Deduplica linhas repetidas consecutivas
        """
        if not texto:
            return ""

        t = texto.replace("\r", "\n")
        # Normaliza múltiplas quebras
        t = re.sub(r"\n{3,}", "\n\n", t)
        # Des-hifenização: 'palavra-\ncontinua' -> 'palavracontinua'
        t = re.sub(r"(\w)[\--–]\n(\w)", r"\1\2", t)
        # Espaços múltiplos -> simples (mas preserva quebras)
        t = re.sub(r"[ \t]{2,}", " ", t)

        # Se ainda parecer monobloco, injeta quebras por rótulos
        if self._count_lines(t) < self._MIN_NATIVE_LINES:
            t = self._inject_linebreaks_labels(t)

        # Remoção de lixo comum
        t = t.replace("\x00", " ").strip()

        # Dedup de linhas repetidas consecutivas (artefatos)
        lines = [ln.strip() for ln in t.split("\n")]
        deduped: List[str] = []
        prev = None
        for ln in lines:
            if ln and ln != prev:
                deduped.append(ln)
            prev = ln
        t = "\n".join(deduped)

        # Limita tamanho para logs/LLM downstream
        if len(t) > self._MAX_TEXT_CHARS:
            t = t[: self._MAX_TEXT_CHARS] + "\n...[TRUNCADO]"
        return t

    def _fix_semantic_artifacts(self, texto: str) -> str:
        """
        Correções semânticas leves e conservadoras para erros comuns de OCR
        que afetam campos e valores (sem "inventar" dados).
        """
        if not texto:
            return ""

        t = texto

        # 1) Normalização de valores monetários (troca vírgula/ponto mal lidos)
        #    Exemplos: "R$ 1.234,56", "R$1,234.56" (americano) → preferimos BR
        def _fix_currency(m: re.Match) -> str:
            val = m.group(1)
            # Remove espaços extras
            val = re.sub(r"\s+", "", val)
            # Se parece no formato americano (1,234.56), converte para BR (1.234,56)
            if re.match(r"^\d{1,3}(,\d{3})+(\.\d{2})$", val):
                val = val.replace(",", "_").replace(".", ",").replace("_", ".")
            # Se tem múltiplos pontos e vírgula, prioriza padrão BR
            return f"R$ {val}"

        t = re.sub(r"R\$\s*([0-9][0-9\., ]+)", _fix_currency, t)

        # 2) Correções contextuais para CNPJ/CPF (O↔0, I↔1) somente em dígitos
        def _fix_digits_context(s: str) -> str:
            s = re.sub(r"(?<=\d)[Oo](?=\d)", "0", s)  # 1O23 -> 1023
            s = re.sub(r"(?<=\d)[Ii](?=\d)", "1", s)  # 1i23 -> 1123
            return s

        t = re.sub(
            r"(CNPJ\s*[:\-]?\s*)([0-9OIo\.\-\/ ]{11,20})",
            lambda m: m.group(1) + _fix_digits_context(m.group(2)),
            t,
            flags=re.I,
        )
        t = re.sub(
            r"(CPF\s*[:\-]?\s*)([0-9OIo\.\- ]{9,15})",
            lambda m: m.group(1) + _fix_digits_context(m.group(2)),
            t,
            flags=re.I,
        )

        # 3) Datas com separadores trocados (ex.: 29-07-2022 ou 29/07/22) → mantém, mas remove espaços errados
        t = re.sub(r"(\d{2})\s*([\/\-])\s*(\d{2})\s*([\/\-])\s*(\d{2,4})", r"\1\2\3\4\5", t)

        return t

    def _dedup_repeated_blocks(self, texto: str) -> str:
        """
        Deduplica trechos idênticos grandes que por vezes aparecem repetidos
        após concatenação de blocos OCR.
        """
        if not texto:
            return ""
        # Estratégia simples: remove duplicidade de parágrafos adjacentes idênticos
        parts = [p.strip() for p in texto.split("\n\n") if p.strip()]
        out: List[str] = []
        prev = None
        for p in parts:
            if p != prev:
                out.append(p)
            prev = p
        return "\n\n".join(out)

    # ---------------------- Heurísticas de qualidade ------------------------
    @staticmethod
    def _count_lines(texto: str) -> int:
        return texto.count("\n") + 1 if texto else 0

    @staticmethod
    def _count_tokens(texto: str) -> int:
        return len(re.findall(r"\w+", texto or ""))

    def _looks_monoblock(self, texto: str) -> bool:
        if not texto:
            return True
        if "\n" not in texto:
            return True
        return self._count_lines(texto) < self._MIN_NATIVE_LINES

    def _inject_linebreaks_labels(self, texto: str) -> str:
        """
        Insere quebras antes de rótulos fiscais comuns quando o texto veio “colar de frases”.
        """
        if not texto:
            return ""

        # Garante separador antes de labels típicos
        labels = [
            r"CNPJ", r"CPF", r"IE", r"I\.E\.", r"Inscri[çc][aã]o\s+Estadual",
            r"Endere[çc]o", r"Logradouro", r"Rua", r"Avenida",
            r"Munic[ií]pio", r"Cidade", r"UF", r"CEP",
            r"Data\s+de\s+Emiss[aã]o", r"Emiss[aã]o", r"Compet[êe]ncia",
            r"Chave\s+de\s+Acesso", r"chNFe", r"chCTe",
            r"Valor\s+Total", r"Total\s+da\s+Nota", r"Valor\s+L[ií]quido", r"Valor\s+a\s+Pagar",
        ]
        pattern = r"\s*(" + r"|".join(labels) + r")\s*[:\-]"
        # quebra antes do label se não houver quebra
        texto = re.sub(pattern, r"\n\1: ", texto, flags=re.I)
        # Normaliza “ : ” redundantes
        texto = re.sub(r"\s+:\s+", ": ", texto)
        # Remove espaços antes de quebras
        texto = re.sub(r"[ \t]+\n", "\n", texto)
        return texto

    def _is_native_usable(self, texto: str) -> bool:
        """
        Decide se o texto nativo é suficiente.
        """
        if not texto:
            return False
        if len(texto) < self._MIN_NATIVE_LEN:
            return False
        if self.last_stats.get("native_tokens", 0) < self._MIN_NATIVE_TOKENS:
            return False
        # Se “monobloco” após tentativa de injeção de quebras, ainda rejeita
        if self.last_stats.get("native_lines", 0) < self._MIN_NATIVE_LINES:
            return False
        # Sinal mínimo de metadado fiscal
        if not re.search(r"\b(CNPJ|CPF|Chave\s+de\s+Acesso|chNFe|NF-?e|NFC-?e|NFSe)\b", texto, re.I):
            # não reprova, mas reduz confiança; ainda consideramos "ruim"
            return False
        return True

    # --------------------------- Cache helpers ------------------------------
    @staticmethod
    def _make_cache_key(nome: str, content: bytes) -> str:
        h = hashlib.sha256()
        h.update(content)
        # extensão influencia estratégias
        h.update(("|" + Path(nome).suffix.lower()).encode("utf-8"))
        return h.hexdigest()

    def _cache_put(self, key: str, texto: str, conf: float) -> None:
        if len(self._cache) >= self._CACHE_MAX_ENTRIES:
            # removemos arbitrariamente a 1ª chave (cache simples FIFO)
            try:
                first_key = next(iter(self._cache.keys()))
                self._cache.pop(first_key, None)
            except Exception:
                self._cache.clear()
        self._cache[key] = (texto, conf, time.time())

    def _reset_stats(self) -> None:
        self.last_stats = {
            "pages": 0,
            "ocr_blocks": 0,
            "used_native_pdf_text": False,
            "avg_confidence": 0.0,
            "llm_correction_applied": False,
            "native_lines": 0,
            "native_tokens": 0,
            "passes_ocr": 0,
            "cache_hit": False,
            "native_source": None,
        }

    # ---------------------- Correção Cognitiva (LLM) ------------------------
    def _should_use_llm(self, texto: str, conf: float) -> bool:
        """
        Decide de forma conservadora quando acionar a LLM.
        - confiança baixa
        - texto muito curto
        - sinais de monobloco / artefatos
        """
        if not self.modo_cognitivo or not self.llm:
            return False
        if not (texto or "").strip():
            return False

        tokens = self._count_tokens(texto)
        monobloco = self._looks_monoblock(texto)

        if conf < max(0.85, self._limiar_llm):  # exige confiança de fato alta para pular LLM
            return True
        if tokens < 30:
            return True
        if monobloco:
            return True
        # Sinais de artefato: muitos '____' ou blocos repetidos
        if re.search(r"[_]{3,}", texto):
            return True
        return False

    def _corrigir_texto_ocr(self, texto: str, conf: float) -> Tuple[str, float]:
        """
        Usa LLM (se configurada) para revisar e corrigir erros comuns de OCR
        (caracteres trocados, espaçamento, quebras de linha), quando a
        confiança do OCR estiver abaixo de um limiar.

        - Não altera o conteúdo semântico fiscal (não inventar dados).
        - Mantém números, datas e rótulos fiscais (CNPJ, CPF, NF-e, R$, CFOP, NCM, etc.).
        """
        if not self.modo_cognitivo or not self.llm:
            return texto, conf

        prompt = (
            "Você é um assistente especializado em correção de texto obtido via OCR de documentos fiscais brasileiros.\n"
            "Corrija apenas erros visuais/tipográficos (caracteres trocados, espaços, hifenização) e quebras de linha.\n"
            "Não invente informações. Preserve números, datas e rótulos fiscais (CNPJ, CPF, NF-e, CFOP, NCM, R$ etc.).\n"
            "Retorne SOMENTE o texto corrigido, sem comentários.\n\n"
            f"Texto OCR (até 5k chars):\n{(texto or '')[:5000]}\n"
        )

        try:
            # LangChain: .invoke(prompt) retorna um objeto com .content
            resposta = self.llm.invoke(prompt)  # type: ignore[attr-defined]
            texto_corrigido = getattr(resposta, "content", None) or str(resposta)
            texto_corrigido = (texto_corrigido or "").strip()
            if texto_corrigido:
                self.last_stats["llm_correction_applied"] = True
                texto_corrigido = self._postprocess_text(texto_corrigido)
                texto_corrigido = self._fix_semantic_artifacts(texto_corrigido)
                # Ajuste pequeno e conservador de confiança
                conf_corrigida = min(1.0, max(conf, conf + 0.05))
                self.last_stats["avg_confidence"] = conf_corrigida
                return texto_corrigido, conf_corrigida
        except Exception as e:
            log.warning(f"LLM falhou na correção OCR: {e}")
        return texto, conf


__all__ = ["AgenteOCR"]
