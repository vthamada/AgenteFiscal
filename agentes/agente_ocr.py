# agentes/ocr.py

from __future__ import annotations
import io
import logging
import time
from pathlib import Path
from typing import List, Tuple
import numpy as np
from PIL import Image, ImageFilter

# ---------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------
log = logging.getLogger("projeto_fiscal.agentes")

# ---------------------------------------------------------------------
# Autodetecção de OCR/PDF (standalone)
# ---------------------------------------------------------------------
try:
    import easyocr  # type: ignore
    OCR_AVAILABLE = True
except Exception as e:
    OCR_AVAILABLE = False
    log.debug(f"easyocr indisponível: {e}")

PDF_RENDERER = None
PDF_AVAILABLE = False
pdfium = None
convert_from_bytes = None

# pdfium primeiro (melhor performance), depois pdf2image
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
        log.debug(f"Renderizador PDF indisponível: {e}")

# ---------------------------------------------------------------------
# Agente OCR
# ---------------------------------------------------------------------
class AgenteOCR:
    """OCR usando EasyOCR + pypdfium2 (preferencial) ou pdf2image como fallback."""

    def __init__(self):
        self.ocr_ok = OCR_AVAILABLE
        self.pdf_ok = PDF_AVAILABLE
        self.reader = None

        if self.ocr_ok:
            try:
                # GPU=False para funcionar em CPU
                self.reader = easyocr.Reader(["pt", "en"], gpu=False)
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
            raise
        finally:
            log.info("OCR '%s' (conf: %.2f) em %.2fs", nome, conf, time.time() - t_start)
        return texto, conf

    # ---------- Pré-processamento leve (numpy/PIL) ----------
    def _preprocess_pil(self, pil_img: "Image.Image") -> "Image.Image":
        try:
            img = pil_img.convert("L")  # grayscale
            # sharpen leve
            img = img.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=3))
            # binarização simples (Otsu manual com numpy)
            np_img = np.array(img)
            hist, _ = np.histogram(np_img.flatten(), bins=256, range=(0, 256))
            total = np_img.size
            sum_total = (np.arange(256) * hist).sum()
            sumB = 0.0
            wB = 0.0
            varMax = 0.0
            threshold = 127
            for t in range(256):
                wB += hist[t]
                if wB == 0:
                    continue
                wF = total - wB
                if wF == 0:
                    break
                sumB += t * hist[t]
                mB = sumB / wB
                mF = (sum_total - sumB) / wF
                varBetween = wB * wF * (mB - mF) ** 2
                if varBetween > varMax:
                    varMax = varBetween
                    threshold = t
            np_bin = (np_img > threshold).astype("uint8") * 255
            return Image.fromarray(np_bin)
        except Exception:
            # fallback simples: retorno original em L
            return pil_img.convert("L")

    def _ocr_imagem(self, conteudo: bytes) -> Tuple[str, float]:
        if not (self.ocr_ok and self.reader):
            return "", 0.0
        try:
            pil = Image.open(io.BytesIO(conteudo)).convert("RGB")
            # upscale leve ajuda o OCR
            w, h = pil.size
            if max(w, h) < 1500:
                pil = pil.resize((w * 2, h * 2))
            pil = self._preprocess_pil(pil)
            np_img = np.array(pil)
            results = self.reader.readtext(np_img, detail=1, paragraph=False)
            texto = " ".join([r[1] for r in results]) if results else ""
            confs = [float(r[2]) for r in results] if results else []
            media = float(np.mean(confs)) if confs else 0.0
            return texto, round(media, 2)
        except Exception as e:
            log.error(f"Erro OCR imagem (EasyOCR): {e}")
            return "", 0.0

    def _ocr_pdf(self, conteudo: bytes) -> Tuple[str, float]:
        if not (self.ocr_ok and self.reader and self.pdf_ok):
            return "", 0.0
        try:
            full_text: List[str] = []
            confs_all: List[float] = []

            if PDF_RENDERER == "pdfium" and pdfium is not None:
                pdf = pdfium.PdfDocument(io.BytesIO(conteudo))
                for page in pdf:
                    pil_img = page.render(scale=2).to_pil().convert("RGB")
                    pil_img = self._preprocess_pil(pil_img)
                    np_img = np.array(pil_img)
                    H, W = np_img.shape
                    slices = 3
                    step = H // slices
                    results_all = []
                    for s in range(slices):
                        a = s * step
                        b = (s + 1) * step if s < slices - 1 else H
                        crop = np_img[a:b, :]
                        results = self.reader.readtext(crop, detail=1, paragraph=False)
                        results_all.extend(results)
                    texto_p = " ".join([r[1] for r in results_all]) if results_all else ""
                    full_text.append(texto_p)
                    confs_all.extend([float(r[2]) for r in results_all] if results_all else [])
            elif PDF_RENDERER == "pdf2image" and convert_from_bytes is not None:
                images = convert_from_bytes(conteudo, dpi=220)
                for pil_img in images:
                    pil_img = pil_img.convert("RGB")
                    pil_img = self._preprocess_pil(pil_img)
                    np_img = np.array(pil_img)
                    H, W = np_img.shape
                    slices = 3
                    step = H // slices
                    results_all = []
                    for s in range(slices):
                        a = s * step
                        b = (s + 1) * step if s < slices - 1 else H
                        crop = np_img[a:b, :]
                        results = self.reader.readtext(crop, detail=1, paragraph=False)
                        results_all.extend(results)
                    texto_p = " ".join([r[1] for r in results_all]) if results_all else ""
                    full_text.append(texto_p)
                    confs_all.extend([float(r[2]) for r in results_all] if results_all else [])
            else:
                log.error("Nenhum renderizador de PDF ativo.")
                return "", 0.0

            texto_final = "\n\n--- Page Break ---\n\n".join([t for t in full_text if t])
            media_conf = float(np.mean(confs_all)) if confs_all else 0.0
            return texto_final, round(media_conf, 2)
        except Exception as e:
            log.error(f"Erro OCR PDF (EasyOCR): {e}")
            return "", 0.0


__all__ = ["AgenteOCR"]
