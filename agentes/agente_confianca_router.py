# agentes/router.py

from __future__ import annotations
from typing import Any, Dict

class AgenteConfiancaRouter:
    """
    Decide status final e fonte com base na confiança do OCR e presença de campos críticos.
    """
    def decidir(self, conf_ocr: float, campos: Dict[str, Any], xml_encontrado: bool = False) -> Dict[str, Any]:
        tem_basico = bool(campos.get("valor_total") and campos.get("data_emissao"))
        if xml_encontrado:
            return {"status": "processado", "fonte": "xml"}
        conf_ocr = conf_ocr or 0.0

        if conf_ocr >= 0.70 and tem_basico:
            return {"status": "processado", "fonte": "ocr_nlp"}
        if conf_ocr >= 0.55 and tem_basico:
            return {"status": "revisao_pendente", "fonte": "ocr_nlp"}
        return {"status": "revisao_pendente", "fonte": "ocr_llm" if not tem_basico else "ocr_nlp"}


__all__ = ["AgenteConfiancaRouter"]
