# app.py
from __future__ import annotations

import os
import io
import json
import zipfile
import hashlib
import traceback
from pathlib import Path
from typing import Iterable, Optional, Dict, Any, List, Tuple

import streamlit as st
import pandas as pd
from dotenv import load_dotenv

# =================== Config da Página & Tema ===================
st.set_page_config(
    page_title="Agente Fiscal - I2A2",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(
    """
    <style>
      /* evita “corte” no topo e melhora o respiro geral */
      .block-container { padding-top: .5rem !important; }
      .stAlert div[data-baseweb="notification"] { padding: 0.6rem 0.8rem; }

      /* header “grudado” no topo, útil em telas menores */
      header[data-testid="stHeader"]{
        position: sticky; top: 0; z-index: 100;
        background: #fff; border-bottom: 1px solid #eee;
      }

      /* tabs quebram linha para não cortar rótulos */
      .stTabs [data-baseweb="tab-list"]{
        flex-wrap: wrap; gap: .25rem;
      }

      .chip {
        display:inline-block; padding:.25rem .6rem; border-radius:999px;
        color:#fff; font-size:.80rem; margin-right:.35rem; margin-bottom:.3rem;
      }
      .chip-green { background:#16a34a; }
      .chip-yellow{ background:#eab308; }
      .chip-red   { background:#ef4444; }
      .chip-blue  { background:#2563eb; }
      .pill {
        display:inline-block; padding:.2rem .5rem; border-radius:8px; font-size:.80rem;
        border:1px solid #E5E7EB; background:#F9FAFB; margin-right:.35rem; margin-bottom:.3rem;
      }
      .hgroup { display:flex; align-items:center; gap:.75rem; flex-wrap:wrap; }
      .soft { color:#6b7280; }
      .badge { font-size:.75rem; padding:.15rem .4rem; border-radius:6px; border:1px solid #E5E7EB; background:#F3F4F6; }
      .timeline li { margin-bottom:.25rem; }
      .chat-bubble-user { background:#e9f3ff; border:1px solid #cfe4ff; padding:10px 12px; border-radius:12px; }
      .chat-bubble-assistant { background:#f6f6f6; border:1px solid #e6e6e6; padding:10px 12px; border-radius:12px; }
    </style>
    """,
    unsafe_allow_html=True,
)

load_dotenv()
REGRAS_FISCAIS_PATH = Path("regras_fiscais.yaml")

# =================== Projeto (núcleo) ===================
from banco_de_dados import BancoDeDados
from validacao import ValidadorFiscal
from memoria import MemoriaSessao
from orchestrator import Orchestrator

# =================== LLM (opcional) ===================
try:
    from modelos_llm import make_llm, GEMINI_MODELS, OPENAI_MODELS, OPENROUTER_MODELS
except Exception:
    def make_llm(*args, **kwargs):
        raise RuntimeError("Módulo de LLM indisponível.")
    GEMINI_MODELS, OPENAI_MODELS, OPENROUTER_MODELS = [], [], []

# =================== Helpers ===================
def hash_password(plain: str) -> str:
    return hashlib.sha256((plain or "").encode("utf-8")).hexdigest()

def mask_doc(doc: str | None) -> str | None:
    if not doc:
        return doc
    s = "".join([c for c in str(doc) if c.isdigit()])
    if len(s) == 14:
        return f"{s[:2]}.{s[2:5]}.{s[5:8]}/{s[8:12]}-{s[12:]}"
    if len(s) == 11:
        return f"{s[:3]}.{s[3:6]}.{s[6:9]}-{s[9:]}"
    return doc

def _strip_str(x: Any) -> Any:
    return x.strip() if isinstance(x, str) else x

def tidy_dataframe(
    df: Optional[pd.DataFrame],
    expected_cols: Optional[Iterable[str]] = None,
    mask_id_cols: bool = False
) -> pd.DataFrame:
    if df is None:
        df = pd.DataFrame()
    if df.empty and expected_cols:
        df = pd.DataFrame(columns=list(expected_cols))
    df2 = df.copy()

    if expected_cols:
        for c in expected_cols:
            if c not in df2.columns:
                df2[c] = ""

    str_cols = [c for c in df2.columns if df2[c].dtype == "object"]
    if str_cols:
        df2[str_cols] = df2[str_cols].fillna("")
        for c in str_cols:
            df2[c] = df2[c].map(_strip_str)

    if mask_id_cols:
        for col in ("emitente_cnpj", "destinatario_cnpj", "emitente_cpf", "destinatario_cpf"):
            if col in df2.columns:
                df2[col] = df2[col].apply(mask_doc)
    return df2

def mask_df_id_cols(df: pd.DataFrame) -> pd.DataFrame:
    return tidy_dataframe(df, mask_id_cols=True)

def status_chip(text: str, color: str) -> str:
    cls = {"green":"chip-green","yellow":"chip-yellow","red":"chip-red","blue":"chip-blue"}.get(color, "chip-blue")
    return f"<span class='chip {cls}'>{text}</span>"

def score_pill(label: str, val: Optional[float]) -> str:
    if val is None:
        return f"<span class='pill'><b>{label}</b>: —</span>"
    c = "green" if val >= 0.80 else ("yellow" if val >= 0.60 else "red")
    return f"<span class='pill'><b>{label}</b>: <span class='soft'>{val:.2f}</span></span>"

def configure_llm(provider: str | None, model: str | None, api_key: str | None):
    try:
        if not provider or not model:
            st.session_state.llm_status_message = "Selecione Provedor e Modelo LLM."
            return None
        key_to_use = api_key or os.getenv(f"{provider.upper()}_API_KEY")
        if not key_to_use:
            st.session_state.llm_status_message = f"Chave API para {provider} não encontrada."
            return None
        llm = make_llm(provider=provider, model=model, api_key=key_to_use)
        st.session_state.llm_status_message = f"LLM {provider}/{model} ATIVO."
        return llm
    except Exception as e:
        st.session_state.llm_status_message = f"Erro ao configurar LLM: {e}"
        return None

# =================== Estado de Sessão ===================
def _ensure_session_defaults():
    defaults = {
        "logged_in": False,
        "user_profile": None,
        "user_name": None,
        "admin_just_created": False,
        "llm_instance": None,
        "llm_status_message": "LLM não configurado.",

        # Upload
        "upload_nonce": 0,
        "last_ingested_hashes": set(),

        # Filtros persistentes
        "filter_status": "Todos",
        "filter_tipo": "",
        "filter_uf": "",
        "filter_date_from": None,
        "filter_date_to": None,

        # Saved views
        "saved_views": {},  # name -> dict(filters)
        "saved_view_name": "",

        # Revisão/edição
        "doc_id_revisao": 0,
        "edited_doc_data": {},
        "edited_items_data": pd.DataFrame(),

        # Chat
        "chat_llm_history": [],
        "chat_scope": {"uf": None, "tipo": None},
        "chat_safe_mode": False,
        "chat_show_code": True,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
_ensure_session_defaults()

# =================== Serviços (cacheados) ===================
@st.cache_resource
def get_core_services():
    db = BancoDeDados()
    memoria = MemoriaSessao(db)
    validador = ValidadorFiscal(regras_path=REGRAS_FISCAIS_PATH)

    try:
        df_users = db.query_table("usuarios")
        if df_users.empty:
            admin_email = "admin@i2a2.academy"
            admin_pass = "admin123"
            db.inserir_usuario(
                nome="Admin Padrão",
                email=admin_email,
                perfil="admin",
                senha_hash=hash_password(admin_pass),
            )
            st.session_state.admin_just_created = True
    except Exception as e:
        st.warning(f"Não foi possível checar/criar admin padrão: {e}")

    return db, memoria, validador

# =================== Autenticação ===================
def attempt_login(db: BancoDeDados, email: str, senha: str) -> bool:
    if not email or not senha:
        return False
    try:
        users = tidy_dataframe(db.query_table("usuarios"))
        match = users[users["email"].str.lower() == email.strip().lower()]
        if match.empty:
            return False
        user = match.iloc[0].to_dict()
        if hash_password(senha) == user.get("senha_hash"):
            st.session_state.logged_in = True
            st.session_state.user_profile = user.get("perfil", "operador")
            st.session_state.user_name = user.get("nome", "Usuário")
            return True
        return False
    except Exception as e:
        st.error(f"Erro durante o login: {e}")
        return False

# =================== UI Comum ===================
def ui_header(db: BancoDeDados):
    left, right = st.columns([5, 3])
    with left:
        st.markdown("### 📄 Agente Fiscal – Orquestração Cognitiva, OCR, XML, Validação & Insights")
        st.caption("Pipeline fiscal multimodal com agentes inteligentes, blackboard e decisões adaptativas.")
    with right:
        st.markdown("#### ⚙️ Ambiente")
        st.write(f"- **LLM**: {st.session_state.llm_status_message}")
        # KPIs rápidos
        try:
            df_docs = db.query_table("documentos")
            total = len(df_docs)
            proc = int((df_docs["status"] == "processado").sum()) if "status" in df_docs.columns else 0
            rev  = int((df_docs["status"] == "revisao_pendente").sum()) if "status" in df_docs.columns else 0
            err  = int((df_docs["status"] == "erro").sum()) if "status" in df_docs.columns else 0
            st.markdown(
                f"<div class='hgroup'>"
                f"{status_chip(f'Processados: {proc}', 'green')}"
                f"{status_chip(f'Revisão: {rev}', 'yellow')}"
                f"{status_chip(f'Erros: {err}', 'red')}"
                f"{status_chip(f'Total: {total}', 'blue')}"
                f"</div>",
                unsafe_allow_html=True
            )
        except Exception:
            pass

def ui_sidebar():
    with st.sidebar:
        st.markdown(f"**👤 {st.session_state.user_name or 'Usuário'}**")
        st.caption(f"Perfil: {st.session_state.user_profile or '—'}")

        if st.button("Sair", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.user_profile = None
            st.session_state.user_name = None
            try:
                st.cache_resource.clear()
            except Exception:
                pass
            st.rerun()

        st.divider()
        st.header("🧠 Configuração LLM")
        st.caption(f"Status atual: {st.session_state.llm_status_message}")
        providers = ["", "gemini", "openai", "openrouter"]
        prov = st.selectbox("Provedor", providers, index=0, key="llm_provider")
        models: list[str] = []
        if prov == "gemini":
            models = GEMINI_MODELS
        elif prov == "openai":
            models = OPENAI_MODELS
        elif prov == "openrouter":
            models = OPENROUTER_MODELS
        model = st.selectbox("Modelo", models, index=0 if models else -1, key="llm_model")
        key = st.text_input("Chave API (opcional; usa env se vazio)", type="password", key="llm_api_key")
        if st.button("Aplicar", use_container_width=True):
            st.session_state.llm_instance = configure_llm(prov, model, key)
            try:
                st.cache_resource.clear()
            except Exception:
                pass
            st.rerun()

        st.divider()
        st.header("🔎 Views Salvas")
        sv_name = st.text_input("Nome da View", key="saved_view_name")
        col_sv1, col_sv2 = st.columns(2)
        with col_sv1:
            if st.button("Salvar View", use_container_width=True, disabled=(not sv_name.strip())):
                st.session_state.saved_views[sv_name.strip()] = {
                    "status": st.session_state.filter_status,
                    "tipo": st.session_state.filter_tipo,
                    "uf": st.session_state.filter_uf,
                    "date_from": st.session_state.filter_date_from,
                    "date_to": st.session_state.filter_date_to,
                }
                st.success(f"View '{sv_name.strip()}' salva.")
        with col_sv2:
            if st.button("Limpar Views", use_container_width=True):
                st.session_state.saved_views = {}
                st.info("Views salvas limpas.")

        if st.session_state.saved_views:
            pick = st.selectbox("Carregar View", ["—"] + list(st.session_state.saved_views.keys()))
            if pick and pick != "—":
                cfg = st.session_state.saved_views[pick]
                st.session_state.filter_status = cfg.get("status", "Todos")
                st.session_state.filter_tipo   = cfg.get("tipo", "")
                st.session_state.filter_uf     = cfg.get("uf", "")
                st.session_state.filter_date_from = cfg.get("date_from")
                st.session_state.filter_date_to   = cfg.get("date_to")
                st.experimental_rerun()

# =================== Seções ===================
def tab_processar(orch: Orchestrator, db: BancoDeDados):
    """Mantida para possível reuso/rota separada, mas não utilizada no tabstrip."""
    st.subheader("📤 Upload & Processamento")
    up_col1, up_col2 = st.columns([3, 2])
    with up_col1:
        uploaded_files = st.file_uploader(
            "Selecione XML / PDF / Imagem",
            type=["xml", "pdf", "jpg", "jpeg", "png", "tif", "tiff", "bmp"],
            accept_multiple_files=True,
            key=f"uploader_{st.session_state.upload_nonce}",
            help="Você pode selecionar múltiplos arquivos.",
        )
    with up_col2:
        origem = st.text_input("Origem (rótulo livre)", value="upload_ui")
        ingest = st.button(
            f"Ingerir {len(uploaded_files) if uploaded_files else 0} Arquivo(s)",
            use_container_width=True,
            type="primary",
            disabled=not uploaded_files,
        )
        limpar = st.button("Limpar fila", use_container_width=True)
        if limpar:
            st.session_state.upload_nonce += 1
            st.rerun()

    if ingest and uploaded_files:
        all_ok = True
        prog = st.progress(0, text="Iniciando ingestão...")
        # hashes existentes
        try:
            df_exist = db.query_table("documentos")
            existing_hashes = set((df_exist["hash"].dropna().astype(str).tolist() if "hash" in df_exist.columns else []))
        except Exception:
            existing_hashes = set()

        for i, up in enumerate(uploaded_files):
            prog.progress((i + 1) / len(uploaded_files), text=f"Processando: {up.name} ({i+1}/{len(uploaded_files)})...")
            try:
                file_hash = hashlib.sha256(up.getvalue()).hexdigest()
                if file_hash in existing_hashes or file_hash in st.session_state.last_ingested_hashes:
                    st.info(f"'{up.name}' ignorado (duplicado).")
                    continue
                doc_id = orch.processar_automatico(up.name, up.getvalue(), origem=origem)
                doc_info = orch.db.get_documento(doc_id)
                status = (doc_info or {}).get("status", "desconhecido")
                existing_hashes.add(file_hash)
                st.session_state.last_ingested_hashes.add(file_hash)
                if status in ("revisao_pendente", "erro", "quarentena"):
                    all_ok = False
                    st.warning(f"ID {doc_id} ('{up.name}'): **{status}**")
                else:
                    st.success(f"ID {doc_id} ('{up.name}'): **{status}**")
            except Exception as e:
                all_ok = False
                st.error(f"Falha em '{up.name}': {e}")
        prog.empty()
        if all_ok:
            st.success("✅ Todos os arquivos foram ingeridos com sucesso.")
        else:
            st.warning("⚠️ Alguns arquivos falharam ou precisam de revisão.")
        st.session_state.upload_nonce += 1
        st.rerun()

def _montar_where_docs() -> Optional[str]:
    parts = []
    if st.session_state.filter_status != "Todos":
        parts.append(f"status = '{st.session_state.filter_status}'")
    if st.session_state.filter_tipo.strip():
        parts.append(f"tipo = '{st.session_state.filter_tipo.strip()}'")
    if st.session_state.filter_uf.strip():
        uf = st.session_state.filter_uf.strip().upper()
        parts.append(f"(emitente_uf = '{uf}' OR destinatario_uf = '{uf}')")
    # datas (se armazenadas como text YYYY-MM-DD)
    d1, d2 = st.session_state.filter_date_from, st.session_state.filter_date_to
    if d1:
        parts.append(f"(data_emissao >= '{pd.to_datetime(d1).date()}')")
    if d2:
        parts.append(f"(data_emissao <= '{pd.to_datetime(d2).date()}')")
    return " AND ".join(parts) if parts else None

def _read_meta(doc: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return json.loads(doc.get("meta_json") or "{}")
    except Exception:
        return {}

def _meta_scores(meta: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    try:
        coverage = float(((meta.get("nlp") or {}).get("coverage", None)))
    except Exception:
        coverage = None
    try:
        match_score = float(((meta.get("associacao") or {}).get("match_score", None)))
    except Exception:
        match_score = None
    try:
        sanity = float(((meta.get("normalizer") or {}).get("sanity_score", None)) or ((meta.get("context") or {}).get("sanity_score", None)))
    except Exception:
        sanity = None
    return coverage, sanity, match_score

def _download_pacote_documento(db: BancoDeDados, doc_id: int) -> bytes:
    doc = db.get_documento(doc_id) or {}
    itens = db.query_table("itens", where=f"documento_id = {doc_id}") or pd.DataFrame()
    impostos = pd.DataFrame()
    if not itens.empty:
        ids = tuple(itens["id"].unique().tolist())
        if ids:
            ids_sql = ", ".join(map(str, ids))
            impostos = db.query_table("impostos", where=f"item_id IN ({ids_sql})")
    meta = _read_meta(doc)
    # zip em memória
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("documento.json", json.dumps(doc, ensure_ascii=False, indent=2))
        z.writestr("meta.json", json.dumps(meta, ensure_ascii=False, indent=2))
        z.writestr("itens.csv", itens.to_csv(index=False).encode("utf-8"))
        z.writestr("impostos.csv", (impostos.to_csv(index=False).encode("utf-8") if not impostos.empty else b""))
        # inclui caminho do arquivo original se existir (apenas referência)
        if doc.get("caminho_arquivo"):
            z.writestr("origem.txt", str(doc.get("caminho_arquivo")))
    buf.seek(0)
    return buf.read()

def _extracao_ocr_text(db: BancoDeDados, doc_id: int) -> str:
    try:
        df = db.query_table("extracoes", where=f"documento_id = {doc_id} AND agente = 'OCRAgent'")
        if not df.empty and "texto_extraido" in df.columns:
            return str(df.iloc[-1]["texto_extraido"] or "")[:200000]
    except Exception:
        pass
    return ""

def _render_cognitive_timeline(meta: Dict[str, Any]):
    decisions = ((meta.get("blackboard") or {}).get("decisions")) or []
    if not decisions:
        st.info("Sem decisões registradas no blackboard para este documento.")
        return
    st.markdown("**🧠 Timeline Cognitiva (decisions log)**")
    st.markdown("<ul class='timeline'>", unsafe_allow_html=True)
    for d in decisions:
        msg = d.get("msg") or ""
        st.markdown(f"<li><span class='badge'>{int(d.get('ts',0))}</span> {msg}</li>", unsafe_allow_html=True)
    st.markdown("</ul>", unsafe_allow_html=True)

def tab_auditoria(orch: Orchestrator, db: BancoDeDados):
    st.subheader("📚 Documentos & Auditoria Assistida")

    # Upload integrado (substitui aba "Processar")
    with st.expander("📤 Upload & Processamento", expanded=False):
        up_col1, up_col2 = st.columns([3, 2])
        with up_col1:
            uploaded_files = st.file_uploader(
                "Selecione XML / PDF / Imagem",
                type=["xml", "pdf", "jpg", "jpeg", "png", "tif", "tiff", "bmp"],
                accept_multiple_files=True,
                key=f"uploader_{st.session_state.upload_nonce}",
                help="Você pode selecionar múltiplos arquivos.",
            )
        with up_col2:
            origem = st.text_input("Origem (rótulo livre)", value="upload_ui")
            ingest = st.button(
                f"Ingerir {len(uploaded_files) if uploaded_files else 0} Arquivo(s)",
                use_container_width=True,
                type="primary",
                disabled=not uploaded_files,
            )
            limpar = st.button("Limpar fila", use_container_width=True)
            if limpar:
                st.session_state.upload_nonce += 1
                st.rerun()

        if ingest and uploaded_files:
            all_ok = True
            prog = st.progress(0, text="Iniciando ingestão...")
            # hashes existentes
            try:
                df_exist = db.query_table("documentos")
                existing_hashes = set((df_exist["hash"].dropna().astype(str).tolist() if "hash" in df_exist.columns else []))
            except Exception:
                existing_hashes = set()

            for i, up in enumerate(uploaded_files):
                prog.progress((i + 1) / len(uploaded_files), text=f"Processando: {up.name} ({i+1}/{len(uploaded_files)})...")
                try:
                    file_hash = hashlib.sha256(up.getvalue()).hexdigest()
                    if file_hash in existing_hashes or file_hash in st.session_state.last_ingested_hashes:
                        st.info(f"'{up.name}' ignorado (duplicado).")
                        continue
                    doc_id = orch.processar_automatico(up.name, up.getvalue(), origem=origem)
                    doc_info = orch.db.get_documento(doc_id)
                    status = (doc_info or {}).get("status", "desconhecido")
                    existing_hashes.add(file_hash)
                    st.session_state.last_ingested_hashes.add(file_hash)
                    if status in ("revisao_pendente", "erro", "quarentena"):
                        all_ok = False
                        st.warning(f"ID {doc_id} ('{up.name}'): **{status}**")
                    else:
                        st.success(f"ID {doc_id} ('{up.name}'): **{status}**")
                except Exception as e:
                    all_ok = False
                    st.error(f"Falha em '{up.name}': {e}")
            prog.empty()
            if all_ok:
                st.success("✅ Todos os arquivos foram ingeridos com sucesso.")
            else:
                st.warning("⚠️ Alguns arquivos falharam ou precisam de revisão.")
            st.session_state.upload_nonce += 1
            st.rerun()

    # Filtros persistentes
    f1, f2, f3, f4, f5 = st.columns([1.1, 1.1, 0.8, 1.1, 1.1])
    with f1:
        st.session_state.filter_status = st.selectbox(
            "Status",
            ["Todos", "processado", "revisado", "revisao_pendente", "quarentena", "erro"],
            index=["Todos","processado","revisado","revisao_pendente","quarentena","erro"].index(st.session_state.filter_status)
        )
    with f2:
        st.session_state.filter_tipo = st.text_input("Tipo (NFe, CTe, pdf...)", value=st.session_state.filter_tipo)
    with f3:
        st.session_state.filter_uf = st.text_input("UF", value=st.session_state.filter_uf)
    with f4:
        st.session_state.filter_date_from = st.date_input("De", value=st.session_state.filter_date_from)
    with f5:
        st.session_state.filter_date_to = st.date_input("Até", value=st.session_state.filter_date_to)

    where_clause = _montar_where_docs()

    # Grid de documentos
    try:
        expected = [
            "id","status","motivo_rejeicao","chave_acesso","data_emissao","valor_total",
            # Emitente
            "emitente_nome","emitente_cnpj","emitente_cpf","emitente_ie",
            "emitente_uf","emitente_municipio","emitente_bairro","emitente_logradouro",
            "emitente_numero","emitente_cep",
            # Destinatário
            "destinatario_nome","destinatario_cnpj","destinatario_cpf","destinatario_ie",
            "destinatario_uf","destinatario_municipio","destinatario_bairro","destinatario_logradouro",
            "destinatario_numero","destinatario_cep",
            # Meta
            "tipo","serie","numero_nota","nome_arquivo","origem"
        ]
        df_docs_raw = db.query_table("documentos", where=where_clause)
        df_docs = tidy_dataframe(df_docs_raw, expected_cols=expected, mask_id_cols=True)
        if "data_emissao" in df_docs.columns:
            df_docs["data_emissao"] = pd.to_datetime(df_docs["data_emissao"], errors="coerce").dt.date

        cols_show = [
            "id","nome_arquivo","tipo","status","data_emissao","valor_total",
            # Emitente
            "emitente_nome","emitente_cnpj","emitente_cpf","emitente_uf","emitente_municipio",
            "emitente_logradouro","emitente_numero","emitente_bairro","emitente_cep",
            # Destinatário
            "destinatario_nome","destinatario_cnpj","destinatario_cpf","destinatario_uf","destinatario_municipio",
            "destinatario_logradouro","destinatario_numero","destinatario_bairro","destinatario_cep",
            # Demais
            "serie","numero_nota","chave_acesso","origem","motivo_rejeicao"
        ]
        cols_exist = [c for c in cols_show if c in df_docs.columns]
        st.data_editor(
            df_docs[cols_exist],
            use_container_width=True, height=430, disabled=True,
            column_config={
                "data_emissao": st.column_config.DateColumn("Data Emissão", format="YYYY-MM-DD"),
                "valor_total":  st.column_config.NumberColumn("Valor Total (R$)", format="R$ %.2f", step=0.01),
            },
            key="grid_docs_ro"
        )
    except Exception as e:
        st.error(f"Erro ao listar documentos: {e}")
        st.code(traceback.format_exc(), language="python")
        return

    st.markdown("---")
    st.subheader("🔎 Detalhe do Documento e Auditoria")

    d1, d2, d3 = st.columns([1, 1, 2])
    with d1:
        doc_id = st.number_input("Documento ID", min_value=0, step=1, key="doc_id_detail")
    with d2:
        abrir = st.button("Abrir Detalhe", type="primary", use_container_width=True, disabled=(doc_id == 0))
    with d3:
        if doc_id > 0:
            try:
                data_zip = _download_pacote_documento(db, int(doc_id))
                st.download_button(
                    "⬇️ Baixar Pacote do Documento",
                    data=data_zip,
                    file_name=f"documento_{int(doc_id)}.zip",
                    mime="application/zip",
                    use_container_width=True
                )
            except Exception:
                pass

    if abrir and doc_id > 0:
        doc = db.get_documento(int(doc_id))
        if not doc:
            st.warning("Documento não encontrado.")
            return

        meta = _read_meta(doc)
        coverage, sanity, match_score = _meta_scores(meta)

        doc_status = doc.get("status")
        status_color = "green" if doc_status in ("processado", "revisado") else ("yellow" if doc_status == "revisao_pendente" else "red")
        st.markdown(
            f"<div class='hgroup'>"
            f"{status_chip(f'Status: {doc_status}', status_color)}"
            f"{score_pill('Cobertura NLP', coverage)}"
            f"{score_pill('Sanidade', sanity)}"
            f"{score_pill('Match XML', match_score)}"
            f"</div>", unsafe_allow_html=True
        )

        # Split view: OCR ↔ XML/Associação
        col_left, col_right = st.columns([3, 2], gap="large")
        with col_left:
            st.markdown("#### 📄 Documento (OCR & Texto)")
            header_df = tidy_dataframe(pd.DataFrame([doc]), mask_id_cols=True)
            if "data_emissao" in header_df.columns:
                header_df["data_emissao"] = pd.to_datetime(header_df["data_emissao"], errors="coerce").dt.date
            st.data_editor(header_df[[
                c for c in [
                    "id","nome_arquivo","tipo","data_emissao","valor_total",
                    # Emitente
                    "emitente_nome","emitente_cnpj","emitente_cpf","emitente_uf","emitente_municipio",
                    "emitente_logradouro","emitente_numero","emitente_bairro","emitente_cep",
                    # Destinatário
                    "destinatario_nome","destinatario_cnpj","destinatario_cpf","destinatario_uf","destinatario_municipio",
                    "destinatario_logradouro","destinatario_numero","destinatario_bairro","destinatario_cep",
                    # Estado
                    "status","serie","numero_nota","chave_acesso","motivo_rejeicao"
                ] if c in header_df.columns
            ]], use_container_width=True, height=220, disabled=True, key=f"doc_header_{doc_id}")

            st.markdown("**Texto OCR (amostra)**")
            ocr_text = _extracao_ocr_text(db, int(doc_id))
            st.text_area("OCR", ocr_text[:20000], height=240)

        with col_right:
            st.markdown("#### 🧷 XML & Associação")
            assoc = (meta.get("associacao") or {})
            if assoc:
                st.json(assoc)
                diffs = assoc.get("diffs") or assoc.get("divergencias")
                if diffs:
                    st.markdown("**Divergências OCR ↔ XML**")
                    st.json(diffs)
            else:
                st.info("Sem metadados de associação registrados.")

            st.markdown("#### ✅ Validações")
            try:
                st.write(f"- **CFOP**: {doc.get('cfop','—')}")
                st.write(f"- **Emitente UF / Dest UF**: {doc.get('emitente_uf','—')} → {doc.get('destinatario_uf','—')}")
                st.write(f"- **Valor Total**: {doc.get('valor_total','—')}")
            except Exception:
                pass

        st.markdown("---")
        col_actions = st.columns(3)
        if col_actions[0].button("✅ Aprovar (Processado)", use_container_width=True):
            try:
                db.atualizar_documento_campo(int(doc_id), "status", "processado")
                db.atualizar_documento_campo(int(doc_id), "motivo_rejeicao", "Aprovado manualmente")
                db.log("revisao_aprovada", st.session_state.user_name or "revisor_ui", f"doc_id={int(doc_id)} processado.")
                st.success("Documento aprovado.")
            except Exception as e:
                st.error(f"Falha ao aprovar: {e}")

        if col_actions[1].button("🔁 Reprocessar", use_container_width=True):
            with st.spinner("Reprocessando..."):
                out = orch.reprocessar_documento(int(doc_id))
                st.info(out.get("mensagem"))

        if col_actions[2].button("♻️ Revalidar", use_container_width=True):
            with st.spinner("Revalidando..."):
                out = orch.revalidar_documento(int(doc_id))
                if out.get("ok"):
                    st.success(out.get("mensagem"))
                else:
                    st.error(out.get("mensagem"))

        st.markdown("---")
        with st.expander("🧠 Blackboard & Meta (decisions, pipeline, nlp, context)", expanded=False):
            if meta:
                st.json(meta)
            else:
                st.info("Sem meta_json no documento.")
        with st.expander("🧭 Timeline Cognitiva (Blackboard Decisions)", expanded=True):
            _render_cognitive_timeline(meta)

def tab_analises(orch: Orchestrator, db: BancoDeDados):
    st.subheader("🤖 Análises & LLM (Chat)")
    st.markdown("**Filtros de Escopo (opcionais)**")
    f1, f2 = st.columns(2)
    with f1:
        uf_scope = st.text_input("UF (ex: SP, RJ):", key="llm_scope_uf")
    with f2:
        tipo_scope = st.multiselect("Tipo:", ["NFe", "NFCe", "CTe", "pdf", "png", "jpg"], key="llm_scope_tipo")

    c1, c2 = st.columns(2)
    with c1:
        safe_mode = st.toggle("Modo Seguro (sem IA)", value=st.session_state.chat_safe_mode, key="llm_safe_mode")
    with c2:
        show_code = st.toggle("Mostrar Código/Query", value=st.session_state.chat_show_code, key="llm_show_code")

    st.session_state.chat_scope = {"uf": uf_scope or None, "tipo": tipo_scope or None}
    st.session_state.chat_safe_mode = safe_mode
    st.session_state.chat_show_code = show_code

    st.markdown("---")
    if (orch.analitico and st.session_state.llm_instance) and not safe_mode:
        st.success(f"LLM Ativo: {st.session_state.llm_status_message}")
    elif safe_mode:
        st.info("Modo Seguro habilitado — respostas sem IA.")

    history: List[Dict[str, Any]] = st.session_state.chat_llm_history or []
    def _render_chat_message(role: str, content: str):
        bubble_class = "chat-bubble-user" if role == "user" else "chat-bubble-assistant"
        with st.chat_message(role):
            st.markdown(f"<div class='{bubble_class}'>{content}</div>", unsafe_allow_html=True)

    for msg in history:
        _render_chat_message(msg["role"], msg["content"])
        if msg.get("table") is not None and isinstance(msg["table"], pd.DataFrame) and not msg["table"].empty:
            st.data_editor(tidy_dataframe(msg["table"]), use_container_width=True, height=320, disabled=True, key=f"chat_tbl_{id(msg)}")
        if st.session_state.chat_show_code and msg.get("code"):
            with st.expander("Código/Query utilizado nesta resposta"):
                st.code(msg["code"], language="python")

    user_input = st.chat_input("Pergunte algo… ex.: 'Top 5 emitentes por valor total'")
    col_l, col_r = st.columns([7, 3])
    with col_r:
        if st.button("🧹 Limpar conversa", use_container_width=True):
            st.session_state.chat_llm_history = []
            st.rerun()

    if user_input:
        history.append({"role": "user", "content": user_input})
        st.session_state.chat_llm_history = history
        _render_chat_message("user", user_input)
        with st.spinner("Analisando..."):
            try:
                scope = st.session_state.chat_scope
                out = orch.responder_pergunta(user_input, scope_filters=scope, safe_mode=safe_mode)
                answer_text = out.get("texto", "*Nenhuma resposta gerada.*")
                answer_table = out.get("tabela")
                answer_code = out.get("code", "")
                agent_name = out.get("agent_name", "N/A")
                dur = out.get("duracao_s", 0.0)
                assistant_msg = f"**Agente:** {agent_name}  \n**Tempo:** {float(dur):.2f}s\n\n{answer_text}"
                _render_chat_message("assistant", assistant_msg)
                if isinstance(answer_table, pd.DataFrame) and not answer_table.empty:
                    st.data_editor(tidy_dataframe(answer_table), use_container_width=True, height=360, disabled=True, key=f"ans_tbl_{len(history)}")
                if show_code and answer_code:
                    with st.expander("Código/Query desta resposta"):
                        st.code(answer_code, language="python")
                history.append({"role": "assistant","content": assistant_msg,"table": answer_table if isinstance(answer_table, pd.DataFrame) else None,"code": answer_code})
                st.session_state.chat_llm_history = history
            except Exception as e:
                st.error(f"Falha ao responder: {e}")
                st.code(traceback.format_exc(), language="python")

def tab_metricas(orch: Orchestrator, db: BancoDeDados):
    st.subheader("📊 Métricas & Insights")
    title_col, act_col = st.columns([7, 3])
    with title_col:
        st.caption("Acompanhe a qualidade do processamento, tempos e taxas de revisão/erro.")
    with act_col:
        gerar_insights = st.button("Gerar Insights (LLM)", type="primary", use_container_width=True)

    try:
        df_metricas_raw = db.query_table("metricas")
        df_metricas = tidy_dataframe(
            df_metricas_raw,
            expected_cols=["tipo_documento","acuracia_media","taxa_revisao","tempo_medio","taxa_erro"]
        )
        tipos = df_metricas["tipo_documento"].unique().tolist() if not df_metricas.empty else []
        c1, c2 = st.columns(2)
        with c1:
            _ = st.date_input("Período", (pd.Timestamp.now() - pd.DateOffset(days=30), pd.Timestamp.now()), key="metric_date")
        with c2:
            _ = st.selectbox("Tipo de Documento", ["Todos"] + tipos, key="metric_doctype")

        if df_metricas.empty:
            st.info("Nenhuma métrica registrada.")
        else:
            for col in ["acuracia_media","taxa_revisao","tempo_medio","taxa_erro"]:
                if col in df_metricas.columns:
                    df_metricas[col] = pd.to_numeric(df_metricas[col], errors="coerce").fillna(0)

            acur = df_metricas["acuracia_media"].mean() * 100 if "acuracia_media" in df_metricas.columns else 0.0
            tx_rev = df_metricas["taxa_revisao"].mean() * 100 if "taxa_revisao" in df_metricas.columns else 0.0
            t_med = df_metricas["tempo_medio"].mean() if "tempo_medio" in df_metricas.columns else 0.0
            total = len(df_metricas)
            tx_err = df_metricas["taxa_erro"].mean() * 100 if "taxa_erro" in df_metricas.columns else 0.0

            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("Acurácia Média (Conf.)", f"{acur:.1f}%")
            k2.metric("Taxa de Revisão", f"{tx_rev:.1f}%")
            k3.metric("Tempo Médio Proc.", f"{t_med:.2f}s")
            k4.metric("Eventos", f"{total}")
            k5.metric("Taxa de Erro", f"{tx_err:.1f}%")

            st.markdown("---")
            g1, g2 = st.columns(2)
            with g1:
                st.markdown("**Acurácia Média por Tipo**")
                if "acuracia_media" in df_metricas.columns and "tipo_documento" in df_metricas.columns:
                    st.bar_chart(df_metricas.groupby("tipo_documento")["acuracia_media"].mean())
                else:
                    st.info("Dados insuficientes.")
            with g2:
                st.markdown("**Taxa de Erro por Tipo**")
                if "taxa_erro" in df_metricas.columns and "tipo_documento" in df_metricas.columns:
                    st.bar_chart(df_metricas.groupby("tipo_documento")["taxa_erro"].mean() * 100)
                else:
                    st.info("Dados insuficientes.")

        st.markdown("---")
        st.subheader("⚠️ Documentos com Baixa Confiança (< 70%)")
        df_baixa_raw = db.query_table("extracoes", where="confianca_media < 0.7")
        df_baixa = tidy_dataframe(
            df_baixa_raw,
            expected_cols=["documento_id","agente","confianca_media","tempo_processamento"]
        )
        if not df_baixa.empty:
            st.data_editor(df_baixa, use_container_width=True, height=220, disabled=True, key="grid_baixa_conf")
        else:
            st.info("Nenhum documento com confiança inferior a 70% encontrado.")

        if gerar_insights:
            if orch.analitico and st.session_state.llm_instance:
                if df_metricas.empty:
                    st.warning("Não há métricas para analisar.")
                else:
                    with st.spinner("Gerando insights..."):
                        kpis = {
                            "acuracia_media": float(acur),
                            "taxa_revisao": float(tx_rev),
                            "tempo_medio": float(t_med),
                            "taxa_erro": float(tx_err),
                            "total_eventos": int(total),
                        }
                        d1 = df_metricas.groupby("tipo_documento")["acuracia_media"].mean().to_json() if "acuracia_media" in df_metricas.columns else "{}"
                        d2 = df_metricas.groupby("tipo_documento")["taxa_erro"].mean().to_json() if "taxa_erro" in df_metricas.columns else "{}"
                        prompt = f"""
                        Analise os seguintes KPIs:
                        KPIs Principais: {kpis}
                        Acurácia por Tipo: {d1}
                        Taxa de Erro por Tipo: {d2}
                        Forneça 2 a 3 insights acionáveis, em português, sucintos.
                        """
                        out = orch.responder_pergunta(prompt, scope_filters={}, safe_mode=False)
                        st.markdown(out.get("texto", "Não foi possível gerar insights."))
            else:
                st.warning("O LLM não está configurado na barra lateral.")
    except Exception as e:
        st.error(f"Erro ao carregar métricas: {e}")
        st.code(traceback.format_exc(), language="python")

def tab_admin(db: BancoDeDados):
    st.subheader("⚙️ Administração")
    a1, a2 = st.tabs(["Gerenciar Usuários", "Criar Novo Usuário"])

    with a1:
        try:
            df_users_raw = db.query_table("usuarios")
            df_users = tidy_dataframe(df_users_raw, expected_cols=["id","nome","email","perfil"])
            cols_disp = ["id", "nome", "email", "perfil"]
            cols_exist = [c for c in cols_disp if c in df_users.columns]
            if not df_users.empty:
                edited = st.data_editor(
                    df_users[cols_exist],
                    use_container_width=True,
                    disabled=["id", "email"],
                    column_config={
                        "perfil": st.column_config.SelectboxColumn(
                            "Perfil", options=["operador", "conferente", "admin"], required=True
                        )
                    },
                    key="editor_usuarios",
                )
                _, ac_right = st.columns([8, 2])
                with ac_right:
                    salvar = st.button("Salvar", type="primary", use_container_width=True)
                if salvar:
                    orig = df_users[cols_exist].set_index("id")
                    edit = edited.set_index("id")
                    changes = 0
                    for uid, row_new in edit.iterrows():
                        if uid in orig.index and not orig.loc[uid].equals(row_new):
                            if uid == 1 and row_new.get("perfil") != "admin":
                                st.error("Não é permitido alterar o perfil do usuário ID 1 (admin padrão).")
                                continue
                            db.conn.execute(
                                "UPDATE usuarios SET nome = ?, perfil = ? WHERE id = ?",
                                (row_new.get("nome"), row_new.get("perfil"), int(uid)),
                            )
                            db.conn.commit()
                            db.log("update_usuario", "admin_ui", f"Usuário ID {uid} atualizado.")
                            changes += 1
                    if changes:
                        st.success(f"{changes} usuário(s) atualizado(s).")
                        st.rerun()
                    else:
                        st.info("Nenhuma alteração.")
            else:
                st.info("Nenhum usuário cadastrado.")

            st.markdown("**Deletar Usuário**")
            c1, c2, c3 = st.columns([1, 2, 2])
            with c1:
                uid_del = st.number_input("ID", min_value=0, step=1, key="user_delete_id")
            with c2:
                confirmar = st.checkbox("Confirmo a exclusão", key="chk_confirma_delete")
            with c3:
                if st.button("Deletar", disabled=(uid_del == 0 or not confirmar), use_container_width=True):
                    if uid_del == 1:
                        st.error("Não é permitido deletar o usuário ID 1 (admin padrão).")
                    else:
                        try:
                            db.conn.execute("DELETE FROM usuarios WHERE id = ?", (int(uid_del),))
                            db.conn.commit()
                            st.success(f"Usuário ID {int(uid_del)} deletado.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Erro ao deletar usuário: {e}")
        except Exception as e:
            st.error(f"Erro ao carregar usuários: {e}")

    with a2:
        with st.form("form_novo_usuario"):
            nome = st.text_input("Nome")
            email = st.text_input("Email")
            perfil = st.selectbox("Perfil", ["operador", "conferente", "admin"])
            senha = st.text_input("Senha", type="password")
            senha_conf = st.text_input("Confirmar Senha", type="password")
            sub = st.form_submit_button("Criar")
            if sub:
                if not nome or not email or not perfil or not senha:
                    st.warning("Todos os campos são obrigatórios.")
                elif senha != senha_conf:
                    st.error("As senhas não conferem.")
                else:
                    try:
                        db.inserir_usuario(
                            nome=nome.strip(),
                            email=email.strip(),
                            perfil=perfil,
                            senha_hash=hash_password(senha),
                        )
                        st.success(f"Usuário '{nome}' ({email}) criado com perfil '{perfil}'.")
                        db.log("criacao_usuario", "admin_ui", f"Usuário {email} criado.")
                    except Exception as e:
                        st.error(f"Erro ao criar usuário (email pode já existir): {e}")

    st.markdown("---")
    st.subheader("📝 Logs do Sistema")
    try:
        limit = st.number_input("Quantidade", min_value=10, max_value=1000, value=100, step=10, key="log_limit")
        logs_raw = db.query_table("logs")
        logs = tidy_dataframe(logs_raw, expected_cols=["id","timestamp","categoria","autor","mensagem"])
        if not logs.empty:
            st.data_editor(logs.sort_values("id", ascending=False).head(limit), use_container_width=True, height=420, disabled=True, key="grid_logs")
        else:
            st.info("Ainda não há logs registrados.")
    except Exception as e:
        st.error(f"Erro ao carregar logs: {e}")

# =================== MAIN ===================
def main():
    db, memoria, validador = get_core_services()

    if st.session_state.admin_just_created:
        st.toast("Admin padrão (admin@i2a2.academy / admin123) foi criado!", icon="🎉")
        st.session_state.admin_just_created = False

    if not st.session_state.logged_in:
        st.markdown("<h1 style='text-align:center; margin-top: 12px;'>Agente Fiscal - Login</h1>", unsafe_allow_html=True)
        c = st.columns([1, 1.5, 1])[1]
        with c:
            with st.container():
                login_tab, register_tab = st.tabs(["Login", "Registrar"])
                with login_tab:
                    with st.form("login_form"):
                        email = st.text_input("Email", placeholder="admin@i2a2.academy")
                        senha = st.text_input("Senha", type="password", placeholder="••••••••")
                        submitted = st.form_submit_button("Entrar")
                        if submitted:
                            if attempt_login(db, email, senha):
                                st.success("Login bem-sucedido!")
                                st.rerun()
                            else:
                                st.error("Email ou senha inválidos.")
                with register_tab:
                    with st.form("register_form"):
                        st.markdown("Criar uma nova conta (perfil: **operador**).")
                        nome = st.text_input("Nome Completo")
                        email = st.text_input("Email")
                        senha = st.text_input("Senha", type="password")
                        senha_conf = st.text_input("Confirmar Senha", type="password")
                        sub = st.form_submit_button("Registrar")
                        if sub:
                            if not nome or not email or not senha:
                                st.warning("Todos os campos são obrigatórios.")
                            elif senha != senha_conf:
                                st.error("As senhas não conferem.")
                            else:
                                try:
                                    db.inserir_usuario(
                                        nome=nome.strip(),
                                        email=email.strip(),
                                        perfil="operador",
                                        senha_hash=hash_password(senha),
                                    )
                                    st.success("Usuário criado! Vá para a aba Login para entrar.")
                                    db.log("registro_usuario", "sistema", f"Usuário {email} registrado.")
                                except Exception as e:
                                    st.error(f"Erro ao criar usuário (email pode já existir): {e}")
        return

    orch = Orchestrator(
        db=db,
        validador=validador,
        memoria=memoria,
        llm=st.session_state.llm_instance,
    )

    ui_header(db)
    ui_sidebar()

    # Tabs sem a aba "📤 Processar" — upload fica dentro da Auditoria
    tabs = st.tabs([
        "📚 Auditoria",
        "🤖 Análises",
        "📊 Métricas",
        "⚙️ Administração",
    ])

    with tabs[0]:
        tab_auditoria(orch, db)
    with tabs[1]:
        tab_analises(orch, db)
    with tabs[2]:
        tab_metricas(orch, db)
    with tabs[3]:
        tab_admin(db)

if __name__ == "__main__":
    main()
