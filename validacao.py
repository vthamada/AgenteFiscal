# validacao.py

from __future__ import annotations
from typing import Optional, Dict, Any
import re

from banco_de_dados import BancoDeDados


def _only_digits(s: Optional[str]) -> str:
    return re.sub(r"\D+", "", s or "")


def _valida_cnpj(cnpj: Optional[str]) -> bool:
    """
    Validação de CNPJ (dígito verificador).
    """
    c = _only_digits(cnpj)
    if len(c) != 14 or len(set(c)) == 1:
        return False

    def dv_calc(numeros: str, pesos: str) -> int:
        soma = sum(int(n) * p for n, p in zip(numeros, map(int, pesos)))
        resto = soma % 11
        return 0 if resto < 2 else 11 - resto

    dv1 = dv_calc(c[:12], "543298765432")
    dv2 = dv_calc(c[:12] + str(dv1), "6543298765432")
    return c[-2:] == f"{dv1}{dv2}"


class ValidadorFiscal:
    """
    Regras mínimas de validação e normalização para a PoC:
    - CNPJ emitente/destinatário (quando presentes) com dígito verificador válido.
    - Se houver itens, a soma dos itens deve bater com valor_total do documento (tolerância).
    - Preenche status "revisao_pendente" quando algo estiver inconsistente.
    """

    TOLERANCIA_VALOR = 0.05  # 5 centavos

    def validar_documento(self, *, doc_id: int | None = None, doc: Dict[str, Any] | None = None,
                          db: BancoDeDados, **kwargs) -> None:
        if doc_id is None and doc is None:
            raise ValueError("Informe doc_id ou doc.")

        if doc is None:
            doc = db.get_documento(int(doc_id))
            if not doc:
                raise ValueError(f"Documento {doc_id} não encontrado.")

        # 1) CNPJ(s)
        inconsistencias: list[str] = []
        emit = doc.get("emitente_cnpj") or ""
        dest = doc.get("destinatario_cnpj") or ""

        if emit and not _valida_cnpj(emit):
            inconsistencias.append("CNPJ do emitente inválido.")
        if dest and len(_only_digits(dest)) == 14 and not _valida_cnpj(dest):
            inconsistencias.append("CNPJ do destinatário inválido.")

        # 2) Totais (se houver itens)
        try:
            import pandas as pd  # local
            itens_df = db.query_table("itens", where=f"documento_id = {int(doc['id'])}")
            if not itens_df.empty:
                soma_itens = float(itens_df["valor_total"].fillna(0).sum())
                total_doc = float(doc.get("valor_total") or 0.0)
                if total_doc > 0:
                    if abs(soma_itens - total_doc) > self.TOLERANCIA_VALOR:
                        inconsistencias.append(
                            f"Inconsistência de totais: soma_itens={soma_itens:.2f} "
                            f"vs valor_total_doc={total_doc:.2f}"
                        )
        except Exception:
            # Não quebra validação por erro de leitura
            pass

        # 3) Status
        if inconsistencias:
            db.atualizar_documento_campo(int(doc["id"]), "status", "revisao_pendente")
            db.log("validacao_inconsistente", "sistema", f"doc_id={doc['id']}|motivos={'; '.join(inconsistencias)}")
        else:
            # Se ainda estiver como "processando" ou vazio, marca como "processado"
            status_atual = doc.get("status") or ""
            if status_atual in ("", "processando", "quarentena"):
                db.atualizar_documento_campo(int(doc["id"]), "status", "processado")
            db.log("validacao_ok", "sistema", f"doc_id={doc['id']}")
