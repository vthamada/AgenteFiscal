# memoria.py

from __future__ import annotations
from typing import List
from textwrap import shorten

from banco_de_dados import BancoDeDados


class MemoriaSessao:
    """
    Memória simples persistida em SQLite:
    - salva(pergunta, resposta_resumo, duracao_s)
    - resumo(): string compacta das últimas interações
    """

    def __init__(self, db: BancoDeDados):
        self.db = db

    def salvar(self, pergunta: str, resposta_resumo: str, duracao_s: float = 0.0) -> None:
        self.db.conn.execute(
            "INSERT INTO memoria (pergunta, resposta_resumo, duracao_s) VALUES (?, ?, ?)",
            (pergunta, resposta_resumo, float(duracao_s)),
        )
        self.db.conn.commit()

    def resumo(self, limite: int = 6) -> str:
        cur = self.db.conn.cursor()
        cur.execute("SELECT pergunta, resposta_resumo FROM memoria ORDER BY id DESC LIMIT ?", (int(limite),))
        rows = cur.fetchall()
        partes: List[str] = []
        for r in rows:
            p = shorten(r["pergunta"] or "", width=80, placeholder="…")
            a = shorten(r["resposta_resumo"] or "", width=100, placeholder="…")
            partes.append(f"- Q: {p}\n  A: {a}")
        return "\n".join(reversed(partes)) or "(sem histórico)"
