# agentes/risk_agent.py
from __future__ import annotations

from typing import Dict, Any, List
import math

import pandas as pd


def _safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def detect_duplicates(db, doc_id: int) -> Dict[str, Any]:
    doc = db.get_documento(int(doc_id)) or {}
    if not doc:
        return {"duplicates": []}
    emit = doc.get("emitente_cnpj") or doc.get("emitente_cpf") or ""
    numero = str(doc.get("numero_nota") or "").strip()
    serie = str(doc.get("serie") or "").strip()
    valor = _safe_float(doc.get("valor_total"))
    if not emit or not numero:
        return {"duplicates": []}
    sql = (
        "SELECT id FROM documentos WHERE id <> ? AND COALESCE(emitente_cnpj, emitente_cpf) = ? "
        "AND numero_nota = ? AND COALESCE(serie,'') = ? AND ABS(COALESCE(valor_total,0) - ?) < 0.01"
    )
    rows = db.conn.execute(sql, (int(doc_id), emit, numero, serie, valor)).fetchall()
    ids = [int(r[0]) for r in rows]
    return {"duplicates": ids}


def compute_risk_score(db, doc_id: int) -> Dict[str, Any]:
    doc = db.get_documento(int(doc_id)) or {}
    if not doc:
        return {"risk_score": 0.0, "signals": []}
    signals: List[str] = []
    score = 0.0

    # 1) Duplicidade
    dups = detect_duplicates(db, int(doc_id)).get("duplicates", [])
    if dups:
        signals.append(f"Possível duplicidade com IDs {dups[:3]}{'...' if len(dups) > 3 else ''}")
        score += 0.5

    # 2) Campos críticos
    misses = 0
    for k in ("emitente_cnpj", "valor_total", "data_emissao"):
        v = doc.get(k)
        if v in (None, "", [], {}):
            misses += 1
    if misses:
        signals.append(f"Campos críticos ausentes: {misses}")
        score += 0.2

    # 3) Valor e datas anômalas
    vtot = _safe_float(doc.get("valor_total"))
    if vtot <= 0:
        signals.append("Valor total não positivo")
        score += 0.2
    # data futura? (texto YYYY-MM-DD)
    try:
        from datetime import date
        from datetime import datetime as dt
        if doc.get("data_emissao"):
            d = dt.fromisoformat(str(doc.get("data_emissao"))).date()
            if d > date.today():
                signals.append("Data de emissão no futuro")
                score += 0.2
    except Exception:
        pass

    # 4) Impostos ausentes em base relevante (heurística)
    try:
        imp = db.query_table("impostos", where=(
            "item_id IN (SELECT id FROM itens WHERE documento_id = %d)" % int(doc_id)
        ))
        if vtot > 1000 and (imp is None or imp.empty):
            signals.append("Sem impostos por item em documento de alto valor")
            score += 0.2
    except Exception:
        pass

    score = max(0.0, min(1.0, score))
    return {"risk_score": round(score, 3), "signals": signals}
