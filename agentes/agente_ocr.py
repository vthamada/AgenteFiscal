# agentes/agente_ocr.py
from __future__ import annotations

import io
import logging
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageFilter

log = logging.getLogger("agente_fiscal.agentes")

# ------------------------- Disponibilidade OCR/PDF -------------------------
try:
    import easyocr  # type: ignore
    OCR_AVAILABLE = True
except Exception as e:
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
    except Exception as e:
        log.debug(f"Nenhum renderizador/extração PDF disponível: {e}")


# ------------------------------- Agente OCR --------------------------------
class AgenteOCR:
    """
    OCR com EasyOCR. Para PDFs:
      1) tenta extrair TEXTO NATIVO (quando disponível) usando pypdfium2;
      2) se quase não houver texto, renderiza em ~300DPI e roda OCR na página inteira.
    Para imagens: pré-processamento leve + OCR.
    """

    def __init__(self):
        self.ocr_ok = OCR_AVAILABLE
        self.pdf_ok = PDF_AVAILABLE
        self.reader = None

        if self.ocr_ok:
            try:
                # GPU=False para funcionar em CPU (Streamlit Cloud etc.)
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
        ext = Path(nome).suffix.lower()
        texto = ""
        conf = 0.0

        try:
            if ext == ".pdf":
                if not self.pdf_ok:
                    raise RuntimeError("Renderizador PDF ausente.")
                texto, conf = self._processar_pdf(conteudo)
            elif ext in {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}:
                if not self.ocr_ok:
                    raise RuntimeError("OCR para imagem indisponível.")
                texto, conf = self._ocr_imagem(conteudo)
            else:
                raise ValueError(f"Extensão não suportada para OCR: {ext}")
            return texto, conf
        except Exception as e:
            log.error("Erro OCR '%s': %s", nome, e)
            raise
        finally:
            log.info("OCR '%s' (conf: %.2f) em %.2fs", nome, conf, time.time() - t_start)

    # ---------------------------- PDF Helpers -------------------------------
    def _pdf_text_nativo(self, content: bytes) -> str:
        """
        Extração de texto nativo do PDF (sem OCR).
        Requer pypdfium2. Retorna string (pode ser vazia).
        """
        if PDF_RENDERER != "pdfium" or pdfium is None:
            return ""

        txt_total: List[str] = []
        try:
            pdf = pdfium.PdfDocument(io.BytesIO(content))
            for page in pdf:
                textpage = page.get_textpage()
                try:
                    txt = textpage.get_text_range()
                    if txt and txt.strip():
                        txt_total.append(txt)
                finally:
                    textpage.close()
            pdf.close()
        except Exception as e:
            log.debug(f"Falha ao extrair texto nativo via pdfium: {e}")
        return "\n".join(txt_total).strip()

    def _render_paginas_pdf(self, content: bytes) -> List[Image.Image]:
        """
        Renderiza cada página em ~300DPI. Usa pdfium quando possível;
        fallback para pdf2image.
        """
        imagens: List[Image.Image] = []
        try:
            if PDF_RENDERER == "pdfium" and pdfium is not None:
                # scale≈2.4 -> ~300DPI (boa legibilidade sem pesar demais)
                pdf = pdfium.PdfDocument(io.BytesIO(content))
                for page in pdf:
                    pil_img = page.render(scale=2.4).to_pil().convert("RGB")
                    imagens.append(pil_img)
                pdf.close()
            elif PDF_RENDERER == "pdf2image" and convert_from_bytes is not None:
                imagens = [im.convert("RGB") for im in convert_from_bytes(content, dpi=300)]
            else:
                log.error("Nenhum renderizador disponível para PDF.")
        except Exception as e:
            log.error(f"Falha ao renderizar PDF: {e}")
        return imagens

    def _processar_pdf(self, content: bytes) -> Tuple[str, float]:
        """
        Estratégia:
          1) Extrai texto nativo (se houver). Se vier suficiente, retorna com alta confiança.
          2) Se não houver, realiza OCR por página inteira com pré-processamento leve.
        """
        # 1) Texto nativo primeiro (melhor qualidade e estrutura)
        texto_nativo = self._pdf_text_nativo(content)
        if texto_nativo and len(texto_nativo) >= 20:
            return texto_nativo, 0.99  # confiança praticamente máxima para texto digital

        # 2) OCR em páginas renderizadas
        if not (self.ocr_ok and self.reader):
            return "", 0.0

        paginas = self._render_paginas_pdf(content)
        if not paginas:
            return "", 0.0

        textos: List[str] = []
        confs: List[float] = []

        for pil_img in paginas:
            pil_proc = self._preprocess_pil(pil_img)
            np_img = np.array(pil_proc)  # grayscale
            try:
                # Página inteira, preservando ordem com paragraph=True
                results = self.reader.readtext(np_img, detail=1, paragraph=True)  # type: ignore
                page_text = " ".join([r[1] for r in results]) if results else ""
                textos.append(page_text)
                confs.extend([float(r[2]) for r in results] if results else [])
            except Exception as e:
                log.debug(f"OCR falhou em uma página: {e}")

        texto_final = "\n\n--- Page Break ---\n\n".join([t for t in textos if t]).strip()
        media_conf = float(np.mean(confs)) if confs else 0.0
        return texto_final, round(media_conf, 2)

    # ------------------------ Imagem & Pré-processo -------------------------
    def _preprocess_pil(self, pil_img: Image.Image) -> Image.Image:
        """
        Pré-processamento leve e seguro:
          - grayscale
          - sharpen
          - upscale suave para textos pequenos
          - binarização opcional somente se contraste muito baixo
        """
        img = pil_img.convert("L")  # grayscale
        img = img.filter(ImageFilter.SHARPEN)

        w, h = img.size
        # Upscale suave para melhorar OCR em fontes pequenas
        if w < 1200:
            scale = 1200 / float(w)
            img = img.resize((int(w * scale), int(h * scale)))

        # Binarização somente se contraste for baixo
        try:
            lo, hi = img.getextrema()
            if (hi - lo) < 60:
                img = img.point(lambda x: 255 if x > 180 else 0)
        except Exception:
            pass

        return img

    def _ocr_imagem(self, content: bytes) -> Tuple[str, float]:
        if not (self.ocr_ok and self.reader):
            return "", 0.0
        try:
            pil = Image.open(io.BytesIO(content)).convert("RGB")
            pil = self._preprocess_pil(pil)
            np_img = np.array(pil)  # grayscale após preprocess
            results = self.reader.readtext(np_img, detail=1, paragraph=True)  # type: ignore
            texto = " ".join([r[1] for r in results]) if results else ""
            confs = [float(r[2]) for r in results] if results else []
            media = float(np.mean(confs)) if confs else 0.0
            return texto.strip(), round(media, 2)
        except Exception as e:
            log.error(f"Erro OCR imagem (EasyOCR): {e}")
            return "", 0.0


__all__ = ["AgenteOCR"]
