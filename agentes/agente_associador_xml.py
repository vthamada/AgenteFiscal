# agentes/agente_associador_xml.py

from __future__ import annotations
import logging
import math
import re
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING, List, Iterable

from .utils import _only_digits, _norm_ws

# Tolerância quando pandas não está disponível
try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    class _DFMock:
        @property
        def empty(self): return True
        def __getitem__(self, _): return self
        def astype(self, *_a, **_k): return self
        def str(self): return self
        def contains(self, *_a, **_k): return self
        def map(self, *_a, **_k): return self
        def to_dict(self): return {}
        def sort_values(self, *_, **__): return self
        def iloc(self): return self
        def __getattr__(self, _): return self
        def unique(self): return []
        def fillna(self, *_a, **_k): return self
        def sum(self): return 0.0
        def get(self, *_a, **_k): return self
        def __iter__(self): return iter([])
        def tolist(self): return []
        @property
        def columns(self): return []
    pd = type("pandas", (), {"DataFrame": _DFMock, "Series": _DFMock, "to_numeric": lambda *a, **k: 0})  # type: ignore

if TYPE_CHECKING:
    from banco_de_dados import BancoDeDados  # apenas para type hints
    # opcional: modelos de LLM/embeddings
    from langchain_core.language_models.chat_models import BaseChatModel  # type: ignore

log = logging.getLogger("agente_fiscal.agentes")


class AgenteAssociadorXML:
    """
    Agente de Alinhamento Semântico: associa um PDF/imagem (OCR) a um XML existente no banco.

    Estratégias (em ordem):
      1) chave de acesso exata (do OCR ou detectada no texto),
      2) URL/QR com chNFe/chCTe=...,
      3) (valor_total, data_emissao) ± tolerâncias com desempate por nome/UF do emitente,
      4) heurística (emitente_nome normalizado + valor_total).

    NOVO:
      • Pós-validação semântica OCR↔XML (embeddings/LLM quando disponível; fallback determinístico).
      • Explicações de divergência por campo (edit distance, diferença de valor/data, etc.).
      • 'match_score' agregado (0..1) e flags para o Orchestrator.
    """

    RE_QR_CHAVE = re.compile(r"(?:chNFe|chCTe)=([0-9]{44})", re.I)
    RE_CHAVE_SECA = re.compile(r"\b(\d{44})\b")

    # Campos principais que o associador pode enriquecer diretamente do XML/DB
    # (REMOVIDOS os genéricos 'uf', 'municipio', 'endereco')
    _CAMPOS_CORE: List[str] = [
        # Identificação básica
        "tipo", "chave_acesso", "numero_nota", "serie", "modelo", "natureza_operacao",
        # Partes
        "emitente_nome", "emitente_cnpj", "emitente_cpf",
        "destinatario_nome", "destinatario_cnpj", "destinatario_cpf",
        # Datas e totais
        "data_emissao", "valor_total",
        "total_produtos", "total_servicos",
        "total_icms", "total_ipi", "total_pis", "total_cofins",
        # Complementos úteis
        "valor_descontos", "valor_frete", "valor_seguro", "valor_outros", "valor_liquido",
        # Transporte
        "modalidade_frete", "placa_veiculo", "uf_veiculo", "peso_bruto", "peso_liquido", "qtd_volumes",
        # Pagamento – expandido
        "forma_pagamento", "valor_pagamento", "valor_troco", "condicao_pagamento", "meio_pagamento", "bandeira_cartao",
        # XML / autorização
        "caminho_xml", "versao_schema", "ambiente", "protocolo_autorizacao", "data_autorizacao", "cstat", "xmotivo",
        "responsavel_tecnico",
        # Telemetria OCR
        "ocr_tipo",
    ]

    _CAMPOS_SUGERIDOS_DETALHE: List[str] = [
        "emitente_ie", "emitente_im", "emitente_uf", "emitente_municipio", "emitente_endereco",
        "destinatario_ie", "destinatario_im", "destinatario_uf", "destinatario_municipio", "destinatario_endereco",
        "condicao_pagamento", "meio_pagamento", "bandeira_cartao", "valor_troco",
    ]

    # campos usados na comparação semântica (com pesos)
    _CAMPOS_SIMILARIDADE: List[Tuple[str, float]] = [
        ("emitente_nome", 0.28),
        ("emitente_cnpj", 0.18),
        ("destinatario_nome", 0.12),
        ("destinatario_cnpj", 0.12),
        ("valor_total", 0.18),
        ("data_emissao", 0.12),
    ]

    def __init__(self,
                 db: "BancoDeDados",
                 *,
                 llm: Optional["BaseChatModel"] = None,
                 embedder: Any = None):
        """
        :param llm: (opcional) gerador de explicações curtas; não é necessário para a pontuação.
        :param embedder: (opcional) objeto com método .embed_query(text: str) -> List[float]
        """
        self.db = db
        self.llm = llm
        self.embedder = embedder  # sentence-transformers / langchain Embeddings / etc.

    # ---------------------------- Público ----------------------------

    def tentar_associar_pdf(
        self,
        doc_id: int,
        campos_parciais: Dict[str, Any],
        texto_ocr: str = ""
    ) -> Dict[str, Any]:
        """
        Retorna campos enriquecidos do XML encontrado e atualiza o próprio documento com a chave (se aplicável).
        Inclui metadados de associação em `__assoc_meta__` com divergências explicadas e match_score.
        """
        try:
            parcial = dict(campos_parciais or {})
            chave_ocr = parcial.get("chave_acesso") or self._extrair_chave_do_texto(texto_ocr)
            valor = parcial.get("valor_total")
            data = parcial.get("data_emissao")
            nome_emit = self._safe_name(parcial.get("emitente_nome"))
            uf_emit = (parcial.get("emitente_uf") or parcial.get("destinatario_uf") or "").strip().upper() or None

            candidato: Optional[Dict[str, Any]] = None
            estrategia = "nao_associado"
            score_heur = 0.0

            # 1) Por chave
            if chave_ocr:
                cand = self._procurar_por_chave(chave_ocr)
                if cand:
                    candidato = cand
                    estrategia = "chave"
                    score_heur = 1.0

            # 2) Fallback por (valor, data)
            if candidato is None and (valor is not None or data):
                cand = self._procurar_por_valor_data(valor, data, preferir_xml=True, valor_tol=0.02, dias_tol=1)
                if cand:
                    candidato = cand
                    estrategia = "valor_data"
                    score_heur = 0.85 + self._bonus_nome_uf(cand, nome_emit, uf_emit)
                    score_heur = min(score_heur, 0.97)

            # 3) Heurística (nome + valor)
            if candidato is None and nome_emit and valor is not None:
                cand = self._procurar_por_nome_valor(nome_emit, float(valor), uf_emit)
                if cand:
                    candidato = cand
                    estrategia = "nome_valor"
                    score_heur = 0.75 + self._bonus_nome_uf(cand, nome_emit, uf_emit)
                    score_heur = min(score_heur, 0.90)

            # 4) Nada encontrado
            if not candidato:
                parcial.setdefault("__assoc_meta__", {
                    "status": "nao_associado",
                    "por": "nenhum",
                    "score": 0.0,
                    "match_score": 0.0,
                    "divergencias": [],
                    "reprocessar_sugerido": True,
                    "motivo": "Sem candidatos por chave/valor/data/nome.",
                })
                return parcial

            # ========= NOVO: avaliação semântica + explicações =========
            match_score, divergencias = self._avaliar_semelhanca(parcial, candidato)

            # Monta enriquecimento e meta
            enriquecidos, meta = self._consolidar(parcial, candidato, estrategia=estrategia, score=score_heur)
            meta.update({
                "match_score": float(round(match_score, 3)),
                "divergencias": divergencias,
            })

            # Sinalização para Orchestrator (baixo match)
            meta["reprocessar_sugerido"] = bool(
                (estrategia != "chave" and match_score < 0.72) or
                (estrategia == "chave" and match_score < 0.55)  # chave encontrada mas metadados destoam
            )

            return self._finalizar(doc_id, enriquecidos, meta)

        except Exception as e:
            log.warning(f"Associador: falha geral na associação doc_id={doc_id}: {e}")
            out = dict(campos_parciais or {})
            out.setdefault("__assoc_meta__", {"status": "erro", "score": 0.0, "match_score": 0.0, "motivo": str(e)})
            return out

    # ---------------------------- Internos ----------------------------

    def _bonus_nome_uf(self, row: Dict[str, Any], nome_emit_norm: Optional[str], uf_emit: Optional[str]) -> float:
        bonus = 0.0
        if nome_emit_norm:
            cand_nome = self._safe_name(row.get("emitente_nome"))
            if cand_nome and cand_nome == nome_emit_norm:
                bonus += 0.05
        if uf_emit:
            cand_uf = (row.get("emitente_uf") or "").strip().upper() or None
            if cand_uf and cand_uf == uf_emit:
                bonus += 0.03
        return bonus

    def _finalizar(self, doc_id: int, enriquecidos: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
        # Atualiza apenas a chave no documento "alvo" (o restante é persistido pelo Orchestrator)
        try:
            chave = enriquecidos.get("chave_acesso")
            if chave:
                self.db.atualizar_documento_campos(doc_id, chave_acesso=chave)
        except Exception as e:
            log.debug(f"Associador: não foi possível atualizar chave do doc_id={doc_id}: {e}")
        enriquecidos["__assoc_meta__"] = meta
        return enriquecidos

    def _consolidar(self, parcial: Dict[str, Any], encontrado: Dict[str, Any], *, estrategia: str, score: float) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Enriquecimento:
        - Prioriza valores já presentes em `parcial` quando não-nulos;
        - Puxa campos do registro 'documentos' encontrado;
        - Se existir 'documentos_detalhes', usa-o para cobrir lacunas (IE/IM/endereços específicos etc.).
        - Não reintroduz campos genéricos (uf/municipio/endereco).
        - Converte 'troco' legado em 'valor_troco' quando necessário.
        """
        base = dict(parcial or {})

        # 1) Tenta carregar detalhes do documento encontrado (quando a tabela existe)
        detalhes_map = self._coletar_detalhes(encontrado.get("id"))

        # 2) Monta pacote de enriquecimento a partir do row de documentos
        enr_core: Dict[str, Any] = {}
        for k in self._CAMPOS_CORE:
            val = encontrado.get(k)
            if val not in (None, "", []):
                enr_core[k] = val

        # Converter 'troco' → 'valor_troco' se vier legado no core
        if "troco" in encontrado and not encontrado.get("valor_troco"):
            tv = encontrado.get("troco")
            if tv not in (None, "", []):
                enr_core["valor_troco"] = tv

        # 3) Acrescenta campos de detalhe sugeridos (quando houver)
        for k in self._CAMPOS_SUGERIDOS_DETALHE:
            v = detalhes_map.get(k)
            if v not in (None, "", []):
                enr_core[k] = v

        # 4) Evitar reintroduzir genéricos
        for k in ("uf", "municipio", "endereco"):
            enr_core.pop(k, None)

        # 5) Mescla (parcial vence onde já tiver informação não-nula)
        for k, v in enr_core.items():
            if base.get(k) in (None, "", []):
                base[k] = v

        meta = {
            "status": "associado",
            "por": estrategia,
            "score": float(round(score, 3)),
            "documento_alvo_id": int(encontrado.get("id", 0) or 0),
            "usou_detalhes": bool(detalhes_map),
        }
        return base, meta

    # ------------------------ Consulta ao banco ------------------------

    def _tabela_existe(self, nome: str) -> bool:
        try:
            cur = self.db.conn.cursor()
            cur.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1", (nome,))
            return cur.fetchone() is not None
        except Exception:
            return False

    def _coletar_detalhes(self, documento_id: Optional[int]) -> Dict[str, Any]:
        if not documento_id or not self._tabela_existe("documentos_detalhes"):
            return {}
        try:
            df = self.db.query_table("documentos_detalhes", where=f"documento_id = {int(documento_id)}")
            if getattr(df, "empty", True):
                return {}
            out: Dict[str, Any] = {}
            if {"chave", "valor"}.issubset(set(df.columns)):  # layout chave/valor
                for _, row in df.iterrows():
                    k = str(row.get("chave") or "").strip()
                    v = row.get("valor")
                    if k:
                        out[k] = v
            else:
                out = df.iloc[0].to_dict()
            return out
        except Exception as e:
            log.debug(f"Associador: falha ao coletar detalhes do doc_id={documento_id}: {e}")
            return {}

    def _procurar_por_chave(self, chave: Optional[str]) -> Optional[Dict[str, Any]]:
        if not chave:
            return None
        chave = _only_digits(chave) or ""
        if len(chave) != 44:
            return None
        try:
            df = self.db.query_table("documentos", where=f"chave_acesso = '{chave}'")
            if getattr(df, "empty", True):
                return None
            tipo_series = df["tipo"].astype(str)
            mask_xml = tipo_series.str.contains("xml|nfe|nfce|cte|cfe|nfse", case=False, na=False)
            df_xml = df[mask_xml] if hasattr(df, "__getitem__") else df
            row = (df_xml if not getattr(df_xml, "empty", True) else df).iloc[0].to_dict()
            return row
        except Exception as e:
            log.warning(f"Associador: falha query por chave {chave}: {e}")
            return None

    def _procurar_por_valor_data(
        self,
        valor: Optional[float],
        data: Optional[str],
        *,
        preferir_xml: bool = True,
        valor_tol: float = 0.02,
        dias_tol: int = 1
    ) -> Optional[Dict[str, Any]]:
        if valor is None and not data:
            return None

        clauses = []
        if data:
            base = str(data).strip().split("T")[0]
            clauses.append(f"(data_emissao BETWEEN date('{base}', '-{int(dias_tol)} day') AND date('{base}', '+{int(dias_tol)} day'))")
        if valor is not None:
            try:
                v = float(valor)
                clauses.append(f"(ABS(CAST(valor_total AS REAL) - {v:.2f}) <= {float(valor_tol):.2f} OR valor_total = {v:.2f})")
            except Exception:
                pass
        sql = " AND ".join(clauses) if clauses else "1=1"

        try:
            df = self.db.query_table("documentos", where=sql)
            if getattr(df, "empty", True):
                return None

            if preferir_xml and "tipo" in getattr(df, "columns", []):
                df = df.sort_values(
                    by=["tipo"],
                    key=lambda s: s.astype(str).str.contains("xml|nfe|nfce|cte|cfe|nfse", case=False, na=False),
                    ascending=False,
                )

            if "valor_total" in getattr(df, "columns", []) and valor is not None:
                df["_diff_valor"] = (pd.to_numeric(df["valor_total"], errors="coerce") - float(valor)).abs()
                df = df.sort_values(by=["_diff_valor"])
            row = df.iloc[0].to_dict()
            if isinstance(row, dict):
                row.pop("_diff_valor", None)
            return row
        except Exception as e:
            log.warning(f"Associador: falha query por valor/data: {e}")
            return None

    def _procurar_por_nome_valor(self, nome_norm: str, valor: float, uf_emit: Optional[str]) -> Optional[Dict[str, Any]]:
        try:
            df = self.db.query_table(
                "documentos",
                where=f"(ABS(CAST(valor_total AS REAL) - {float(valor):.2f}) <= 0.02 OR valor_total = {float(valor):.2f})"
            )
            if getattr(df, "empty", True):
                return None

            if "emitente_nome" in getattr(df, "columns", []):
                df["__emit_norm"] = df["emitente_nome"].astype(str).map(self._safe_name)
            else:
                df["__emit_norm"] = None

            cand = df[df["__emit_norm"].astype(str).str.contains(re.escape(nome_norm), na=False)]

            if uf_emit:
                if "emitente_uf" in getattr(cand, "columns", []):
                    uf_series = cand["emitente_uf"].astype(str).str.upper()
                    mask_uf = (uf_series == uf_emit)
                    cand = cand[mask_uf] if hasattr(cand, "__getitem__") else cand

            if getattr(cand, "empty", True):
                cand = df  # fallback

            if "tipo" in getattr(cand, "columns", []):
                mask_xml = cand["tipo"].astype(str).str.contains("xml|nfe|nfce|cte|cfe|nfse", case=False, na=False)
                cand_xml = cand[mask_xml]
            else:
                cand_xml = cand

            row_df = (cand_xml if not getattr(cand_xml, "empty", True) else cand)
            row = row_df.iloc[0].to_dict()
            if isinstance(row, dict):
                row.pop("__emit_norm", None)
            return row
        except Exception as e:
            log.debug(f"Associador: nome/valor falhou: {e}")
            return None

    # ---------------------------- Similaridade & Diferenças ----------------------------

    def _avaliar_semelhanca(self, ocr: Dict[str, Any], xml: Dict[str, Any]) -> Tuple[float, List[Dict[str, Any]]]:
        """
        Calcula score agregado (0..1) e explica divergências campo a campo.
        Preferimos embeddings/LLM quando disponíveis; fallback determinístico quando não.
        """
        pesos = dict(self._CAMPOS_SIMILARIDADE)
        partes: List[Tuple[float, float, Dict[str, Any]]] = []  # (peso, sim, explicacao)

        for campo, peso in self._CAMPOS_SIMILARIDADE:
            v_ocr = ocr.get(campo)
            v_xml = xml.get(campo)

            if campo in ("emitente_nome", "destinatario_nome"):
                sim, exp = self._sim_textual(v_ocr, v_xml, label=campo)
            elif campo in ("emitente_cnpj", "destinatario_cnpj"):
                sim, exp = self._sim_cnpj(v_ocr, v_xml, label=campo)
            elif campo == "valor_total":
                sim, exp = self._sim_valor(v_ocr, v_xml, label=campo)
            elif campo == "data_emissao":
                sim, exp = self._sim_data(v_ocr, v_xml, label=campo)
            else:
                sim, exp = self._sim_textual(v_ocr, v_xml, label=campo)

            partes.append((peso, sim, exp))

        # score ponderado
        num = sum(peso * sim for peso, sim, _ in partes)
        den = sum(pesos.values()) or 1.0
        score = max(0.0, min(1.0, num / den))

        divergencias: List[Dict[str, Any]] = []
        for _, sim, exp in partes:
            if exp and sim < 0.9:  # só explica quando não está “perfeito”
                divergencias.append(exp)

        return score, divergencias

    # ---- Semântica Textual

    def _sim_textual(self, a: Any, b: Any, *, label: str) -> Tuple[float, Dict[str, Any]]:
        sa = self._normalize_for_text(a)
        sb = self._normalize_for_text(b)
        if not sa and not sb:
            return 1.0, {"campo": label, "status": "ausente_ambos", "detalhe": "Sem valor nos dois lados."}
        if not sa or not sb:
            return 0.0, {"campo": label, "erro": "ausente", "detalhe": "Valor presente apenas em um lado."}

        sim = self._semantic_similarity(sa, sb)
        if sim >= 0.95:
            return sim, {"campo": label, "ok": True}

        # explicação com Levenshtein normalizado
        ed, max_len = self._levenshtein(sa, sb)
        detalhe = f"{ed} edições em {max_len} chars"
        hint = None
        if 0 < ed <= 3 and sim >= 0.8:
            hint = "Provável ruído de OCR."
        return sim, {"campo": label, "erro": "divergencia_texto", "detalhe": detalhe, "sugestao": hint}

    def _semantic_similarity(self, sa: str, sb: str) -> float:
        # 1) embeddings quando disponíveis
        try:
            if self.embedder is not None and hasattr(self.embedder, "embed_query"):
                va = self.embedder.embed_query(sa)  # type: ignore
                vb = self.embedder.embed_query(sb)  # type: ignore
                return self._cosine(va, vb)
        except Exception:
            pass

        # 2) fallback determinístico: Jaccard tokens + Levenshtein normalizado
        toks_a = set(self._tokenize(sa))
        toks_b = set(self._tokenize(sb))
        jacc = len(toks_a & toks_b) / max(1, len(toks_a | toks_b))
        ed, max_len = self._levenshtein(sa, sb)
        lev_norm = 1.0 - (ed / max(1, max_len))
        # combinação conservadora
        return max(0.0, min(1.0, 0.6 * lev_norm + 0.4 * jacc))

    # ---- CNPJ/numéricos

    def _sim_cnpj(self, a: Any, b: Any, *, label: str) -> Tuple[float, Dict[str, Any]]:
        da = _only_digits(str(a) if a is not None else "")
        db = _only_digits(str(b) if b is not None else "")
        if not da and not db:
            return 1.0, {"campo": label, "status": "ausente_ambos"}
        if not da or not db:
            return 0.0, {"campo": label, "erro": "ausente"}
        if da == db:
            return 1.0, {"campo": label, "ok": True}
        # tolera até 2 dígitos diferentes como possível OCR
        diffs = sum(1 for x, y in zip(da, db) if x != y) + abs(len(da) - len(db))
        sim = max(0.0, 1.0 - (diffs / max(len(da), len(db), 14)))
        sug = "Possível erro de OCR em dígitos." if diffs <= 2 else None
        return sim, {"campo": label, "erro": "cnpj_diferente", "detalhe": f"{diffs} dígitos divergentes", "sugestao": sug}

    def _sim_valor(self, a: Any, b: Any, *, label: str) -> Tuple[float, Dict[str, Any]]:
        try:
            fa = float(str(a).replace(".", "").replace(",", "."))
        except Exception:
            fa = None
        try:
            fb = float(str(b).replace(".", "").replace(",", "."))
        except Exception:
            fb = None

        if fa is None and fb is None:
            return 1.0, {"campo": label, "status": "ausente_ambos"}
        if fa is None or fb is None:
            return 0.0, {"campo": label, "erro": "ausente"}

        if fa == 0 and fb == 0:
            return 1.0, {"campo": label, "ok": True}

        diff = abs(fa - fb)
        base = max(1.0, max(abs(fa), abs(fb)))
        rel = diff / base
        sim = max(0.0, 1.0 - min(1.0, rel * 2.0))  # queda rápida com a diferença relativa

        if rel <= 0.01:
            return sim, {"campo": label, "ok": True}
        return sim, {"campo": label, "erro": "valor_diverge", "detalhe": f"Δ={diff:.2f} (rel={rel:.2%})"}

    def _sim_data(self, a: Any, b: Any, *, label: str) -> Tuple[float, Dict[str, Any]]:
        sa = str(a or "").strip().split("T")[0] or ""
        sb = str(b or "").strip().split("T")[0] or ""
        if not sa and not sb:
            return 1.0, {"campo": label, "status": "ausente_ambos"}
        if not sa or not sb:
            return 0.0, {"campo": label, "erro": "ausente"}
        if sa == sb:
            return 1.0, {"campo": label, "ok": True}
        # tolera ±1 dia como ruído (fuso/horário de autorização)
        try:
            from datetime import date, timedelta
            ya, ma, da = [int(x) for x in sa.split("-")]
            yb, mb, dbb = [int(x) for x in sb.split("-")]
            da_ = date(ya, ma, da)
            db_ = date(yb, mb, dbb)
            delta = abs((da_ - db_).days)
            if delta <= 1:
                return 0.85, {"campo": label, "erro": "data_proxima", "detalhe": f"diferença de {delta} dia(s)", "sugestao": "Possível timezone/autorização."}
        except Exception:
            pass
        # fallback textual
        ed, mx = self._levenshtein(sa, sb)
        sim = 1.0 - (ed / max(1, mx))
        return sim, {"campo": label, "erro": "data_diverge", "detalhe": f"Levenshtein={ed}/{mx}"}

    # ---------------------------- Text/Math utils ----------------------------

    @staticmethod
    def _tokenize(s: str) -> Iterable[str]:
        s = re.sub(r"[^a-z0-9\s]", " ", s.lower())
        return [t for t in re.split(r"\s+", s) if t]

    @staticmethod
    def _normalize_for_text(v: Any) -> str:
        if v is None:
            return ""
        s = _norm_ws(str(v)).strip()
        s = re.sub(r"\s+", " ", s)
        return s

    @staticmethod
    def _cosine(a: Iterable[float], b: Iterable[float]) -> float:
        av = list(a); bv = list(b)
        if not av or not bv or len(av) != len(bv):
            return 0.0
        num = sum(x*y for x, y in zip(av, bv))
        na = math.sqrt(sum(x*x for x in av)); nb = math.sqrt(sum(y*y for y in bv))
        if na == 0 or nb == 0:
            return 0.0
        return max(0.0, min(1.0, num / (na * nb)))

    @staticmethod
    def _levenshtein(a: str, b: str) -> Tuple[int, int]:
        # Levenshtein simples O(len(a)*len(b))
        if a == b:
            return 0, max(len(a), len(b))
        if not a:
            return len(b), len(b)
        if not b:
            return len(a), len(a)
        m, n = len(a), len(b)
        dp = list(range(n + 1))
        for i in range(1, m + 1):
            prev = dp[0]
            dp[0] = i
            for j in range(1, n + 1):
                cur = dp[j]
                cost = 0 if a[i - 1] == b[j - 1] else 1
                dp[j] = min(dp[j] + 1,      # deletion
                            dp[j - 1] + 1,  # insertion
                            prev + cost)    # substitution
                prev = cur
        return dp[n], max(m, n)

    # ---------------------------- Text/Regex helpers ----------------------------

    def _extrair_chave_do_texto(self, texto: str) -> Optional[str]:
        if not texto:
            return None
        m = self.RE_QR_CHAVE.search(texto)
        if m:
            ch = _only_digits(m.group(1)) or ""
            return ch if len(ch) == 44 else None

        # procura qualquer sequência de 44 dígitos
        for match in re.finditer(self.RE_CHAVE_SECA, texto):
            ch = _only_digits(match.group(1)) or ""
            if len(ch) == 44:
                return ch
        return None

    @staticmethod
    def _safe_name(s: Optional[str]) -> Optional[str]:
        if not s:
            return None
        s = _norm_ws(str(s)).strip().lower()
        s = re.sub(r"\b(ltda|me|eireli|s\/a|s\.a\.|sa|epp|mei|comercial|comércio|comercio|holding|grupo)\b", "", s)
        s = re.sub(r"[^a-z0-9\s]", "", s)
        s = re.sub(r"\s+", " ", s)
        return s or None

__all__ = ["AgenteAssociadorXML"]
