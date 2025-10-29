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

# ===== Integração com modelos_llm (wrappers centralizados) =====
_HAS_LLM = False
_llm_correct_compat = None   # type: ignore
_llm_identity = None         # type: ignore

try:
    # shim compatível que retorna {"text": "...", "confidence": ...}
    from modelos_llm import ocr_correct_text_compat as _llm_correct_compat  # type: ignore
    from modelos_llm import get_llm_identity as _llm_identity               # type: ignore
    _HAS_LLM = True
except Exception:
    _HAS_LLM = False
    _llm_correct_compat = None
    _llm_identity = None

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
    from pdfminer_high_level import extract_text as pdfminer_extract_text  # type: ignore
    _PDFMINER_AVAILABLE = True
except Exception as e:
    try:
        from pdfminer.high_level import extract_text as pdfminer_extract_text  # type: ignore
        _PDFMINER_AVAILABLE = True
    except Exception as e2:
        log.debug(f"pdfminer.six indisponível: {e or e2}")

# ------------------------------- Agente OCR --------------------------------
class AgenteOCR:
    """
    OCR robusto para documentos fiscais:
      1) tenta TEXTO NATIVO (pdfium → pdfminer) e aceita com heurísticas estáveis;
      2) se insuficiente, renderiza (~300DPI) e roda EasyOCR (1..2 passes adaptativos);
      3) aplica limpeza leve, correções semânticas conservadoras e, opcionalmente, LLM.

    Cognitivo (conforme mapa):
      - Gating LLM por confiança/monobloco/tamanho;
      - Métricas de qualidade; cache;
      - Saída OCR→NLP com meta padronizada.
    """

    # Heurísticas de “texto nativo suficiente”
    _MIN_NATIVE_LEN = 20
    _MIN_NATIVE_TOKENS = 6
    _MIN_NATIVE_LINES = 2

    # Tamanhos/limites
    _MAX_TEXT_CHARS = 400_000  # corta para evitar estourar memória/logs

    # Cache interno (sessão do processo)
    _CACHE_MAX_ENTRIES = 128
    _cache: Dict[str, Tuple[str, float, float]] = {}  # key -> (texto, conf, timestamp)

    def __init__(
        self,
        llm: Optional["BaseChatModel"] = None,
        *,
        aplicar_llm_quando_conf_abaixo: float = 0.90,
        langs: Optional[List[str]] = None,
        use_gpu: bool = False,
    ):
        self.ocr_ok = OCR_AVAILABLE
        self.pdf_ok = PDF_AVAILABLE
        self.reader = None

        # Camada cognitiva (opcional)
        self.llm: Optional["BaseChatModel"] = llm
        self.modo_cognitivo: bool = llm is not None

        try:
            self._limiar_llm = float(aplicar_llm_quando_conf_abaixo)
        except Exception:
            self._limiar_llm = 0.90
        self._limiar_llm = max(0.0, min(1.0, self._limiar_llm))

        # Estatísticas da última execução (telemetria)
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
            "native_source": None,   # "pdfium" | "pdfminer" | None
            "native_rejected_reason": None,
            "ocr_engine": None,      # "easyocr"|"pdfium"|"pdfminer"
            "llm_provider": None,
            "llm_model": None,
        }

        if self.llm and _llm_identity:
            try:
                ident = _llm_identity(self.llm)  # type: ignore
                self.last_stats["llm_provider"] = ident.get("provider")
                self.last_stats["llm_model"] = ident.get("model")
            except Exception:
                pass

        # Inicialização do OCR
        if self.ocr_ok:
            try:
                langs = langs or ["pt", "en"]
                self.reader = easyocr.Reader(langs, gpu=bool(use_gpu))  # type: ignore
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
        """
        Retrocompatível: retorna apenas (texto, confiança).
        Para meta OCR→NLP, use reconhecer_com_meta().
        """
        texto, ocr_meta = self.reconhecer_com_meta(nome, conteudo)
        return texto, float(ocr_meta.get("avg_confidence", 0.0))

    def reconhecer_com_meta(self, nome: str, conteudo: bytes) -> Tuple[str, Dict[str, Any]]:
        """
        Retorna (texto, ocr_meta) com o CONTRATO OCR→NLP:
        {
          "engine": "pdfium|pdfminer|easyocr|cache|error",
          "avg_confidence": float,
          "pages": int,
          "blocks": int,
          "llm_correction_applied": bool
        }
        """
        t_start = time.time()
        self._reset_stats()

        ext = Path(nome).suffix.lower()
        cache_key = self._make_cache_key(nome, conteudo)
        if cache_key in self._cache:
            texto_cached, conf_cached, _ts = self._cache[cache_key]
            self.last_stats["cache_hit"] = True
            self.last_stats["avg_confidence"] = float(conf_cached or 0.0)
            ocr_meta = self._build_ocr_meta(
                engine="cache",
                confidence=float(conf_cached or 0.0),
                texto=texto_cached,
            )
            log.info("OCR cache HIT para '%s'", nome)
            return texto_cached, ocr_meta

        texto = ""
        conf = 0.0
        engine = None

        try:
            if ext == ".pdf":
                if not self.pdf_ok and not _PDFMINER_AVAILABLE:
                    raise RuntimeError("Nenhum extrator/renderizador PDF disponível.")
                texto, conf = self._processar_pdf(conteudo)
                engine = self.last_stats.get("ocr_engine")
            elif ext in {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}:
                if not self.ocr_ok:
                    raise RuntimeError("OCR para imagem indisponível.")
                texto, conf = self._ocr_imagem(conteudo)
                engine = "easyocr"
                self.last_stats["ocr_engine"] = engine
            else:
                raise ValueError(f"Extensão não suportada para OCR: {ext}")

            # Pós-processamento e correções leves
            texto = self._postprocess_text(texto)
            texto = self._fix_semantic_artifacts(texto)
            texto = self._normalize_layout_fiscal(texto)

            # Correção cognitiva (gating)
            if self._should_use_llm(texto, conf):
                texto, conf = self._corrigir_texto_ocr(texto, conf)

            # Segurança e truncagem
            texto = (texto or "").strip()
            if len(texto) > self._MAX_TEXT_CHARS:
                texto = texto[: self._MAX_TEXT_CHARS] + "\n...[TRUNCADO]"

            # Cache
            self._cache_put(cache_key, texto, float(conf or 0.0))

            # Meta no formato do contrato
            ocr_meta = self._build_ocr_meta(
                engine=self.last_stats.get("ocr_engine") or engine or "unknown",
                confidence=float(conf or 0.0),
                texto=texto,
            )
            return texto, ocr_meta

        except Exception as e:
            log.error("Erro OCR '%s': %s", nome, e)
            ocr_meta = self._build_ocr_meta(
                engine=self.last_stats.get("ocr_engine") or engine or "error",
                confidence=0.0,
                texto="",
            )
            return "", ocr_meta
        finally:
            log.info(
                "OCR '%s' → conf=%.2f | pages=%s | blocks=%s | native=%s | native_src=%s | native_reject=%s | llm=%s | cache=%s | %.2fs",
                nome,
                float(self.last_stats.get("avg_confidence", conf)),
                int(self.last_stats.get("pages", 0)),
                int(self.last_stats.get("ocr_blocks", 0)),
                bool(self.last_stats.get("used_native_pdf_text", False)),
                str(self.last_stats.get("native_source")),
                str(self.last_stats.get("native_rejected_reason")),
                bool(self.last_stats.get("llm_correction_applied", False)),
                bool(self.last_stats.get("cache_hit", False)),
                time.time() - t_start,
            )

    # ---------------------------- PDF Helpers -------------------------------
    def _pdf_text_nativo_pdfium(self, content: bytes) -> str:
        if PDF_RENDERER != "pdfium" or pdfium is None:
            return ""
        txt_total: List[str] = []
        try:
            pdf = pdfium.PdfDocument(io.BytesIO(content))
            self.last_stats["pages"] = len(pdf)
            for idx, page in enumerate(pdf):
                try:
                    textpage = page.get_textpage()
                    txt = (textpage.get_text_range() or "").strip()
                    if txt:
                        txt_total.append(txt)
                    textpage.close()
                except Exception as e:
                    log.debug(f"Falha texto nativo na página {idx + 1} (pdfium): {e}")
            pdf.close()
            if txt_total:
                self.last_stats["ocr_engine"] = "pdfium"
                # pdfium não quebra em "blocos" OCR — assegura blocks>=1
                try:
                    if int(self.last_stats.get("ocr_blocks") or 0) <= 0:
                        self.last_stats["ocr_blocks"] = 1
                except Exception:
                    self.last_stats["ocr_blocks"] = 1
        except Exception as e:
            log.debug(f"Falha ao extrair texto nativo via pdfium: {e}")
        return "\n".join(txt_total).strip()

    def _pdf_text_nativo_pdfminer(self, content: bytes) -> str:
        if not _PDFMINER_AVAILABLE:
            return ""
        try:
            txt = pdfminer_extract_text(io.BytesIO(content)) or ""
            if txt:
                txt = txt.replace("\r", "\n")
                self.last_stats["ocr_engine"] = "pdfminer"
                # blocks mínimo para manter heurísticas estáveis
                try:
                    if int(self.last_stats.get("ocr_blocks") or 0) <= 0:
                        self.last_stats["ocr_blocks"] = 1
                except Exception:
                    self.last_stats["ocr_blocks"] = 1
            return txt.strip()
        except Exception as e:
            log.debug(f"pdfminer_extract_text falhou: {e}")
            return ""

    def _pdf_text_nativo(self, content: bytes) -> str:
        texto = ""
        src = None

        if PDF_RENDERER == "pdfium" and pdfium is not None:
            texto = self._pdf_text_nativo_pdfium(content)
            src = "pdfium"

        if (not texto or self._looks_monoblock(texto)) and _PDFMINER_AVAILABLE:
            txt2 = self._pdf_text_nativo_pdfminer(content)
            if len(txt2) > len(texto):
                texto = txt2
                src = "pdfminer"

        if texto and (self._looks_monoblock(texto)):
            texto = self._inject_linebreaks_labels(texto)

        self.last_stats["native_lines"] = self._count_lines(texto)
        self.last_stats["native_tokens"] = self._count_tokens(texto)
        self.last_stats["native_source"] = src
        return texto

    def _render_paginas_pdf(self, content: bytes, scale: float = 2.4, dpi_fallback: int = 300) -> List[Image.Image]:
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
        texto_nativo = self._pdf_text_nativo(content)

        if texto_nativo:
            usable, reason = self._is_native_usable_with_reason(texto_nativo)
            if usable:
                self.last_stats["used_native_pdf_text"] = True
                self.last_stats["avg_confidence"] = 0.99
                # garantir meta consistente para fluxo downstream
                try:
                    blocks = int(self.last_stats.get("ocr_blocks") or 0)
                except Exception:
                    blocks = 0
                if blocks <= 0:
                    self.last_stats["ocr_blocks"] = 1
                if not self.last_stats.get("ocr_engine"):
                    self.last_stats["ocr_engine"] = self.last_stats.get("native_source") or "pdf"
                return texto_nativo.strip(), 0.99
            else:
                self.last_stats["native_rejected_reason"] = reason

        if not (self.ocr_ok and self.reader):
            if texto_nativo:
                self.last_stats["ocr_engine"] = self.last_stats.get("native_source")
                # fallback nativo: blocks >= 1
                try:
                    if int(self.last_stats.get("ocr_blocks") or 0) <= 0:
                        self.last_stats["ocr_blocks"] = 1
                except Exception:
                    self.last_stats["ocr_blocks"] = 1
            return (texto_nativo or "").strip(), 0.0

        paginas = self._render_paginas_pdf(content, scale=2.4, dpi_fallback=300)
        texto_ocr, conf_ocr, blocks = self._ocr_paginas(paginas, pass_name="pass1")
        self.last_stats["passes_ocr"] = 1
        self.last_stats["ocr_engine"] = "easyocr"

        if (not texto_ocr) or conf_ocr < 0.55:
            paginas2 = [self._preprocess_pil(img, aggressive=True) for img in paginas]
            texto_ocr2, conf_ocr2, blocks2 = self._ocr_paginas(paginas2, pass_name="pass2_aggressive")
            if conf_ocr2 > conf_ocr or len(texto_ocr2) > len(texto_ocr):
                texto_ocr, conf_ocr, blocks = texto_ocr2, conf_ocr2, blocks2
            self.last_stats["passes_ocr"] = 2

        if (not (texto_ocr or "").strip()) and (texto_nativo or "").strip():
            self.last_stats["native_rejected_reason"] = "used_as_fallback"
            self.last_stats["avg_confidence"] = 0.60
            self.last_stats["ocr_engine"] = self.last_stats.get("native_source")
            # fallback nativo: blocks >= 1
            try:
                blocks_fb = int(self.last_stats.get("ocr_blocks") or 0)
            except Exception:
                blocks_fb = 0
            if blocks_fb <= 0:
                self.last_stats["ocr_blocks"] = 1
            return texto_nativo.strip(), 0.60

        self.last_stats["ocr_blocks"] = int(blocks)
        self.last_stats["avg_confidence"] = float(round(conf_ocr, 2))
        return (texto_ocr or "").strip(), float(round(conf_ocr, 2))

    # ------------------------ OCR Helpers (PDF/Imagem) ----------------------
    def _ocr_paginas(self, paginas: List[Image.Image], pass_name: str) -> Tuple[str, float, int]:
        if not paginas:
            return "", 0.0, 0
        if not (self.ocr_ok and self.reader):
            return "", 0.0, 0

        textos: List[str] = []
        confs: List[float] = []
        pesos: List[int] = []
        total_blocos = 0

        for idx, pil_img in enumerate(paginas, start=1):
            pil_proc = self._preprocess_pil(pil_img, aggressive=False)
            np_img = np.array(pil_proc)

            try:
                results = self.reader.readtext(np_img, detail=1, paragraph=False)  # type: ignore
                total_blocos += len(results or [])
                page_parts: List[str] = []
                if results:
                    for r in results:
                        t = (r[1] or "").strip()
                        t = re.sub(r"[\u200B-\u200D\u2060\x0c]+", "", t)
                        c = float(r[2])
                        if t:
                            page_parts.append(t)
                            confs.append(c)
                            pesos.append(max(1, len(t)))
                page_text = "\n".join(page_parts)
                textos.append(page_text)
                log.debug("[%s] Página %d: %d blocos OCR.", pass_name, idx, len(results or []))
            except Exception as e:
                log.debug(f"[{pass_name}] OCR falhou em uma página ({idx}): {e}")

        texto_final = "\n\n--- Page Break ---\n\n".join([t for t in textos if t]).strip()
        texto_final = self._dedup_repeated_blocks(texto_final)

        media_conf = float(np.average(confs, weights=pesos)) if (confs and pesos and sum(pesos) > 0) else 0.0
        media_conf = float(round(media_conf, 2))
        return texto_final, media_conf, total_blocos

    # ------------------------ Imagem & Pré-processo -------------------------
    def _preprocess_pil(self, pil_img: Image.Image, aggressive: bool = False) -> Image.Image:
        try:
            img = pil_img.convert("L")
        except Exception:
            img = ImageOps.grayscale(pil_img)

        try:
            img = img.filter(ImageFilter.SHARPEN)
        except Exception:
            pass

        try:
            img = ImageOps.autocontrast(img, cutoff=1)
        except Exception:
            pass

        w, h = img.size
        if w < 1200:
            scale = 1200 / float(w)
            img = img.resize((int(w * scale), int(h * scale)))

        try:
            lo, hi = img.getextrema()
            if (hi - lo) < 60 or aggressive:
                np_img = np.array(img)
                thr = self._otsu_threshold(np_img)
                np_bin = (np_img > max(120, thr)).astype(np.uint8) * 255
                img = Image.fromarray(np_bin, mode="L")
                if aggressive:
                    img = img.filter(ImageFilter.MaxFilter(3))
        except Exception:
            pass

        return img

    @staticmethod
    def _otsu_threshold(gray: np.ndarray) -> int:
        if gray.ndim != 2:
            gray = gray.mean(axis=2).astype(np.uint8)
        hist, _ = np.histogram(gray, bins=256, range=(0, 256))
        total = gray.size
        sum_total = np.dot(np.arange(256), hist)
        sum_b, w_b, var_max, threshold = 0.0, 0.0, 0.0, 0.0, 0
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
        if not (self.ocr_ok and self.reader):
            return "", 0.0
        try:
            pil = Image.open(io.BytesIO(content)).convert("RGB")
        except Exception as e:
            log.error(f"Falha ao abrir imagem: {e}")
            return "", 0.0

        pil1 = self._preprocess_pil(pil, aggressive=False)
        texto1, conf1, blocks1 = self._ocr_paginas([pil1], pass_name="img_pass1")

        if (not texto1) or conf1 < 0.55:
            pil2 = self._preprocess_pil(pil, aggressive=True)
            texto2, conf2, blocks2 = self._ocr_paginas([pil2], pass_name="img_pass2_aggressive")
            if conf2 > conf1 or len(texto2) > len(texto1):
                texto1, conf1, blocks1 = texto2, conf2, blocks2

        self.last_stats["pages"] = 1
        self.last_stats["ocr_blocks"] = int(blocks1)
        self.last_stats["avg_confidence"] = float(round(conf1, 2))
        self.last_stats["ocr_engine"] = "easyocr"
        return texto1.strip(), float(round(conf1, 2))

    # ---------------------- Pós-processamento de Texto ----------------------
    def _postprocess_text(self, texto: str) -> str:
        if not texto:
            return ""

        t = texto.replace("\r", "\n")
        t = re.sub(r"\n{3,}", "\n\n", t)
        t = re.sub(r"(\w)[\-\–]\n(\w)", r"\1\2", t)
        t = re.sub(r"[ \t]{2,}", " ", t)
        t = re.sub(r"[\u200B-\u200D\u2060\x0c]+", "", t)

        if self._count_lines(t) < self._MIN_NATIVE_LINES:
            t = self._inject_linebreaks_labels(t)

        t = t.replace("\x00", " ").strip()

        lines = [ln.strip() for ln in t.split("\n")]
        deduped: List[str] = []
        prev = None
        for ln in lines:
            if ln and ln != prev:
                deduped.append(ln)
            prev = ln
        t = "\n".join(deduped)

        if len(t) > self._MAX_TEXT_CHARS:
            t = t[: self._MAX_TEXT_CHARS] + "\n...[TRUNCADO]"
        return t

    def _normalize_layout_fiscal(self, texto: str) -> str:
        if not texto:
            return ""
        t = texto

        t = re.sub(r"(?i)\b(Produto|Descrição|Descricao|Qtde|Quantidade|Unid\.?|Unitário|Unitario|Vlr\s*Unit\.?|Valor\s*Unit[aá]rio|Total|Vlr\s*Total)\b",
                   r"\n\1", t)

        def _cols_to_pipes(line: str) -> str:
            if len(re.findall(r"\s{2,}", line)) >= 2:
                line = re.sub(r"\s{2,}", " | ", line.strip())
            return line

        t = "\n".join(_cols_to_pipes(ln) for ln in t.split("\n"))
        t = re.sub(r"\s*\|\s*\|\s*", " | ", t)
        return t

    def _fix_semantic_artifacts(self, texto: str) -> str:
        if not texto:
            return ""
        t = texto

        def _fix_currency(m: re.Match) -> str:
            val = m.group(1)
            val = re.sub(r"\s+", "", val)
            if re.match(r"^\d{1,3}(,\d{3})+(\.\d{2})$", val):
                val = val.replace(",", "_").replace(".", ",").replace("_", ".")
            return f"R$ {val}"

        t = re.sub(r"R\$\s*([0-9][0-9\., ]+)", _fix_currency, t)

        def _fix_digits_context(s: str) -> str:
            s = re.sub(r"(?<=\d)[Oo](?=\d)", "0", s)
            s = re.sub(r"(?<=\d)[Ii](?=\d)", "1", s)
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

        t = re.sub(r"(\d{2})\s*([\/\-])\s*(\d{2})\s*([\/\-])\s*(\d{2,4})", r"\1\2\3\4\5", t)
        return t

    def _dedup_repeated_blocks(self, texto: str) -> str:
        if not texto:
            return ""
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
        if not texto:
            return ""
        labels = [
            r"CNPJ", r"CPF", r"IE", r"I\.E\.", r"Inscri[çc][aã]o\s+Estadual",
            r"Endere[çc]o", r"Logradouro", r"Rua", r"Avenida",
            r"Munic[ií]pio", r"Cidade", r"UF", r"CEP",
            r"Data\s+de\s+Emiss[aã]o", r"Emiss[aã]o", r"Compet[êe]ncia",
            r"Chave\s+de\s+Acesso", r"chNFe", r"chCTe",
            r"Valor\s+Total", r"Total\s+da\s+Nota", r"Valor\s+L[ií]quido", r"Valor\s+a\s+Pagar",
        ]
        pattern = r"\s*(" + r"|".join(labels) + r")\s*[:\-]"
        texto = re.sub(pattern, r"\n\1: ", texto, flags=re.I)
        texto = re.sub(r"\s+:\s+", ": ", texto)
        texto = re.sub(r"[ \t]+\n", "\n", texto)
        return texto

    def _is_native_usable_with_reason(self, texto: str) -> Tuple[bool, Optional[str]]:
        if not texto:
            return False, "empty"
        if len(texto) < self._MIN_NATIVE_LEN:
            return False, "too_short"
        if self.last_stats.get("native_tokens", 0) < self._MIN_NATIVE_TOKENS:
            return False, "too_few_tokens"
        if self.last_stats.get("native_lines", 0) < self._MIN_NATIVE_LINES:
            return False, "too_few_lines"

        signals = 0
        if re.search(r"\b(CNPJ|CPF)\b", texto, re.I):
            signals += 1
        if re.search(r"\b(chave\s+de\s+acesso|chNFe|chCTe|NF-?e|NFC-?e|NFSe)\b", texto, re.I):
            signals += 1
        if re.search(r"\b\d{2}[\/\-]\d{2}[\/\-]\d{2,4}\b", texto):
            signals += 1
        if re.search(r"R\$\s*[0-9][0-9\., ]+", texto):
            signals += 1
        if signals < 1:
            return False, "too_few_fiscal_signals"

        return True, None

    def _is_native_usable(self, texto: str) -> bool:
        ok, _ = self._is_native_usable_with_reason(texto)
        return ok

    # --------------------------- Cache helpers ------------------------------
    @staticmethod
    def _make_cache_key(nome: str, content: bytes) -> str:
        h = hashlib.sha256()
        h.update(content)
        h.update(("|" + Path(nome).suffix.lower()).encode("utf-8"))
        return h.hexdigest()

    def _cache_put(self, key: str, texto: str, conf: float) -> None:
        if len(self._cache) >= self._CACHE_MAX_ENTRIES:
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
            "native_rejected_reason": None,
            "ocr_engine": None,
            "llm_provider": self.last_stats.get("llm_provider") if isinstance(self.last_stats, dict) else None,
            "llm_model": self.last_stats.get("llm_model") if isinstance(self.last_stats, dict) else None,
        }

    # ---------------------- Correção Cognitiva (LLM) ------------------------
    def _should_use_llm(self, texto: str, conf: float) -> bool:
        if not self.modo_cognitivo or not self.llm:
            return False
        if not (texto or "").strip():
            return False

        tokens = self._count_tokens(texto)
        monobloco = self._looks_monoblock(texto)

        if conf < max(0.85, self._limiar_llm):
            return True
        if tokens < 30:
            return True
        if monobloco:
            return True
        if re.search(r"[_]{3,}", texto):
            return True
        return False

    def _corrigir_texto_ocr(self, texto: str, conf: float) -> Tuple[str, float]:
        if not self.modo_cognitivo or not self.llm:
            return texto, conf

        # Usa o shim compatível do modelos_llm
        if _HAS_LLM and _llm_correct_compat:
            try:
                result = _llm_correct_compat(self.llm, noisy_text=texto, temperature=0.0, max_chars_user=5000)  # type: ignore
                if isinstance(result, dict):
                    txt = (result.get("text") or "").strip()
                    if txt:
                        self.last_stats["llm_correction_applied"] = True
                        txt = self._normalize_layout_fiscal(self._fix_semantic_artifacts(self._postprocess_text(txt)))
                        new_conf = float(result.get("confidence") or conf)
                        new_conf = min(1.0, max(conf, new_conf, conf + 0.05))
                        self.last_stats["avg_confidence"] = new_conf
                        return txt, new_conf
            except Exception as e:
                log.warning(f"Wrapper ocr_correct_text_compat falhou; fallback .invoke(): {e}")

        # Fallback simples com .invoke()
        prompt = (
            "Você é um assistente especializado em correção de texto obtido via OCR de documentos fiscais brasileiros.\n"
            "Corrija apenas erros visuais/tipográficos e quebras; não invente dados. Retorne SOMENTE o texto corrigido.\n\n"
            f"Texto OCR (até 5k chars):\n{(texto or '')[:5000]}\n"
        )
        try:
            resposta = self.llm.invoke(prompt)  # type: ignore[attr-defined]
            texto_corrigido = getattr(resposta, "content", None) or str(resposta)
            texto_corrigido = (texto_corrigido or "").strip()
            if texto_corrigido:
                self.last_stats["llm_correction_applied"] = True
                texto_corrigido = self._normalize_layout_fiscal(self._fix_semantic_artifacts(self._postprocess_text(texto_corrigido)))
                conf_corrigida = min(1.0, max(conf, conf + 0.05))
                self.last_stats["avg_confidence"] = conf_corrigida
                return texto_corrigido, conf_corrigida
        except Exception as e:
            log.warning(f"LLM falhou na correção OCR: {e}")
        return texto, conf

    # ---------------------- Métricas & Saída (contrato) ---------------------
    def _compute_quality_metrics(self, texto: str) -> Dict[str, float]:
        lines = max(1, self._count_lines(texto))
        tokens = self._count_tokens(texto)
        text_density = round(tokens / float(lines), 4)

        if not texto:
            entropy = 0.0
        else:
            from math import log2
            s = texto
            freq: Dict[str, int] = {}
            for ch in s:
                freq[ch] = freq.get(ch, 0) + 1
            total = float(len(s))
            probs = [c / total for c in freq.values()]
            entropy = -sum(p * log2(p) for p in probs if p > 0)
            entropy = round(entropy, 4)

        avg_char_conf = float(self.last_stats.get("avg_confidence") or 0.0)
        return {"text_density": text_density, "entropy_score": entropy, "avg_char_conf": round(avg_char_conf, 4)}

    def _guess_language(self, texto: str) -> str:
        if re.search(r"[áéíóúãõâêôçÁÉÍÓÚÃÕÂÊÔÇ]", texto or ""):
            return "pt-BR"
        if "CNPJ" in (texto or "") or "CPF" in (texto or ""):
            return "pt-BR"
        return "pt-BR"

    def _needs_reprocess(self, texto: str, confidence: float) -> bool:
        few_lines = self._count_lines(texto) < 3
        few_tokens = self._count_tokens(texto) < 30
        monoblock = self._looks_monoblock(texto)
        low_conf = confidence < 0.60
        return bool(low_conf or ((few_lines or few_tokens) and monoblock))

    def _build_ocr_meta(self, *, engine: str, confidence: float, texto: Optional[str] = None) -> Dict[str, Any]:
        """
        Constrói o META no formato do contrato OCR→NLP (mínimo),
        mantendo a telemetria detalhada em self.last_stats.
        Se 'texto' for fornecido, atualiza métricas derivadas com base nele.
        """
        try:
            if texto is not None:
                _ = self._compute_quality_metrics(texto)
        except Exception:
            pass

        return {
            "engine": engine,
            "avg_confidence": float(round(self.last_stats.get("avg_confidence", confidence), 4)),
            "pages": int(self.last_stats.get("pages", 0)),
            "blocks": int(self.last_stats.get("ocr_blocks", 0)),
            "llm_correction_applied": bool(self.last_stats.get("llm_correction_applied", False)),
        }

__all__ = ["AgenteOCR"]
