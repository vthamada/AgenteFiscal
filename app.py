# app.py
from __future__ import annotations

import os
import hashlib
import traceback
from pathlib import Path

import streamlit as st
import pandas as pd
from dotenv import load_dotenv

# ---------------------- Config da Página ----------------------
st.set_page_config(
    page_title="Agente Fiscal - I2A2",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(
    """
    <style>
    .stAlert div[data-baseweb="notification"] { padding: 0.6rem 0.8rem; }
    .block-container { padding-top: 1.6rem; }
    .uploadedFile { display: none; } /* esconde chips de upload duplicados */
    </style>
    """,
    unsafe_allow_html=True,
)

load_dotenv()
REGRAS_FISCAIS_PATH = Path("regras_fiscais.yaml")

# ---------------------- Projeto (núcleo) ----------------------
from banco_de_dados import BancoDeDados
from validacao import ValidadorFiscal
from memoria import MemoriaSessao
from orchestrator import Orchestrator

# ---------------------- LLM (opcional) ----------------------
try:
    from modelos_llm import make_llm, GEMINI_MODELS, OPENAI_MODELS, OPENROUTER_MODELS
except Exception:
    # Caso o módulo de LLM não esteja presente, definimos placeholders
    def make_llm(*args, **kwargs):
        raise RuntimeError("Módulo de LLM indisponível. Instale/Configure modelos_llm.py")

    GEMINI_MODELS, OPENAI_MODELS, OPENROUTER_MODELS = [], [], []

# ---------------------- Helpers ----------------------
def hash_password(plain: str) -> str:
    """Hash simples (sha256) para senha de usuário (demonstração)."""
    return hashlib.sha256((plain or "").encode("utf-8")).hexdigest()

def mask_doc(doc: str | None) -> str | None:
    """Máscara simples para CNPJ/CPF exibidos em tabelas (apenas visual)."""
    if not doc:
        return doc
    s = "".join([c for c in str(doc) if c.isdigit()])
    if len(s) == 14:  # CNPJ
        return f"{s[:2]}.{s[2:5]}.{s[5:8]}/{s[8:12]}-{s[12:]}"
    if len(s) == 11:  # CPF
        return f"{s[:3]}.{s[3:6]}.{s[6:9]}-{s[9:]}"
    return doc

def mask_df_id_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica máscara de CNPJ/CPF em colunas padrão (apenas visual)."""
    if df is None or getattr(df, "empty", True):
        return df if df is not None else pd.DataFrame()
    df2 = df.copy()
    for col in ("emitente_cnpj", "destinatario_cnpj", "emitente_cpf", "destinatario_cpf"):
        if col in df2.columns:
            df2[col] = df2[col].apply(mask_doc)
    return df2

# ---------------------- Estado de Sessão ----------------------
def _ensure_session_defaults():
    defaults = {
        "edited_doc_data": {},
        "edited_items_data": pd.DataFrame(),
        "doc_id_revisao": 0,
        "llm_instance": None,
        "llm_status_message": "LLM não configurado.",
        "logged_in": False,
        "user_profile": None,
        "user_name": None,
        "admin_just_created": False,
        "edited_users_data": pd.DataFrame(),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_ensure_session_defaults()

# ---------------------- Serviços (cacheados) ----------------------
@st.cache_resource
def get_core_services():
    """
    Inicializa DB, Memória e Validador.
    Cria admin padrão se tabela estiver vazia.
    """
    db = BancoDeDados()
    memoria = MemoriaSessao(db)
    validador = ValidadorFiscal(regras_path=REGRAS_FISCAIS_PATH)

    # Seed admin padrão (apenas se tabela estiver vazia)
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

def configure_llm(provider: str | None, model: str | None, api_key: str | None):
    """Configura o LLM e retorna a instância (ou None)."""
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

# ---------------------- Autenticação ----------------------
def attempt_login(db: BancoDeDados, email: str, senha: str) -> bool:
    if not email or not senha:
        return False
    try:
        users = db.query_table("usuarios")
        if users.empty:
            return False
        match = users[users["email"].str.lower() == email.lower()]
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

# ---------------------- UI: Cabeçalho e Sidebar ----------------------
def ui_header():
    top_left, top_right = st.columns([5, 3])
    with top_left:
        st.markdown("### 📄 Agente Fiscal – Ingestão, OCR, XML, Validação & Análises")
        st.caption("Plataforma unificada para processar documentos fiscais, validar e gerar insights.")
    with top_right:
        st.markdown("#### ⚙️ Ambiente")
        st.write(f"- **LLM**: {st.session_state.llm_status_message}")

def ui_sidebar_llm():
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

# ---------------------- Seções (Abas) ----------------------
def tab_processamento(orch: Orchestrator, db: BancoDeDados):
    st.subheader("📤 Upload & Processamento")

    up_col1, up_col2 = st.columns([3, 2])
    with up_col1:
        uploaded_files = st.file_uploader(
            "Selecione XML / PDF / Imagem",
            type=["xml", "pdf", "jpg", "jpeg", "png", "tif", "tiff", "bmp"],
            accept_multiple_files=True,
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

    if ingest and uploaded_files:
        all_ok = True
        prog = st.progress(0, text="Iniciando ingestão...")
        for i, up in enumerate(uploaded_files):
            prog.progress(
                (i + 1) / len(uploaded_files),
                text=f"Processando: {up.name} ({i+1}/{len(uploaded_files)})...",
            )
            try:
                # Roteamento automático: XML -> parser, senão OCR/NLP
                doc_id = orch.processar_automatico(up.name, up.getvalue(), origem=origem)
                doc_info = orch.db.get_documento(doc_id)
                status = (doc_info or {}).get("status", "desconhecido")
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
        st.rerun()

    st.markdown("---")
    st.subheader("📚 Documentos")
    fcol1, fcol2, fcol3 = st.columns([1.2, 1, 1.2])
    with fcol1:
        status_opts = ["Todos", "processado", "revisado", "revisao_pendente", "quarentena", "erro"]
        status_sel = st.selectbox("Status:", status_opts, key="doc_status_filter")
    with fcol2:
        tipo_sel = st.text_input("Filtrar Tipo (ex: NFe, CTe, pdf):", key="doc_tipo_filter")
    with fcol3:
        uf_sel = st.text_input("Filtrar UF (ex: SP):", key="doc_uf_filter")

    where = []
    if status_sel != "Todos":
        where.append(f"status = '{status_sel}'")
    if tipo_sel.strip():
        where.append(f"tipo = '{tipo_sel.strip()}'")
    if uf_sel.strip():
        where.append(f"uf = '{uf_sel.strip().upper()}'")
    where_clause = " AND ".join(where) if where else None

    try:
        df_docs = db.query_table("documentos", where=where_clause)
        st.dataframe(mask_df_id_cols(df_docs), use_container_width=True, height=430)
    except Exception as e:
        st.error(f"Erro na consulta de documentos: {e}")
        st.code(traceback.format_exc(), language="python")

    st.markdown("---")
    st.subheader("🔎 Itens & Impostos por Documento (consulta rápida)")
    cc1, cc2, _ = st.columns([1, 1, 6])
    with cc1:
        doc_id_q = st.number_input("Documento ID:", min_value=0, step=1, key="doc_id_q_itens")
    with cc2:
        consultar = st.button("Consultar", type="primary", use_container_width=True, disabled=(doc_id_q == 0))
    if consultar and doc_id_q > 0:
        try:
            header = db.get_documento(int(doc_id_q))
            if header:
                st.write(f"**Documento ID: {doc_id_q}** — Status: **{header.get('status')}**")
                cols = st.columns(2)
                with cols[0]:
                    itens = db.query_table("itens", where=f"documento_id = {int(doc_id_q)}")
                    st.markdown("**Itens**")
                    if not itens.empty:
                        st.dataframe(itens, use_container_width=True, height=360)
                    else:
                        st.info("Nenhum item encontrado.")
                with cols[1]:
                    impostos = pd.DataFrame()
                    if not itens.empty:
                        ids = tuple(itens["id"].unique().tolist())
                        if ids:
                            ids_sql = ", ".join(map(str, ids))
                            impostos = db.query_table("impostos", where=f"item_id IN ({ids_sql})")
                    st.markdown("**Impostos**")
                    if not impostos.empty:
                        st.dataframe(impostos, use_container_width=True, height=360)
                    elif not itens.empty:
                        st.info("Nenhum imposto associado aos itens.")
                    else:
                        st.info("Consulte os itens primeiro.")
            else:
                st.warning(f"Documento com ID {doc_id_q} não encontrado.")
        except Exception as e:
            st.error(f"Erro ao consultar: {e}")
            st.code(traceback.format_exc(), language="python")

def tab_revisao(orch: Orchestrator, db: BancoDeDados):
    st.subheader("🧐 Revisão Fiscal")
    c1, c2, _ = st.columns([1, 1, 3])
    with c1:
        tipo = st.selectbox("Tipo:", ["Todos", "NFe", "NFCe", "CTe", "pdf", "jpg", "png"], key="rev_tipo")
    with c2:
        uf = st.text_input("UF (ex: SP):", key="rev_uf")

    where_rev = "status = 'revisao_pendente'"
    if tipo != "Todos":
        where_rev += f" AND tipo = '{tipo}'"
    if uf:
        where_rev += f" AND uf = '{uf.upper()}'"

    try:
        pend = db.query_table("documentos", where=where_rev)
        if pend.empty:
            st.info("🎉 Nenhum documento pendente de revisão com os filtros atuais.")
            return

        st.dataframe(mask_df_id_cols(pend), use_container_width=True, height=250)
        ids = pend["id"].tolist()
        st.session_state.doc_id_revisao = st.selectbox(
            "Selecione o Documento para revisar:",
            options=[0] + ids,
            index=0,
            key="select_doc_revisao",
        )
        if st.session_state.doc_id_revisao > 0:
            doc_id = int(st.session_state.doc_id_revisao)
            st.markdown(f"--- \n#### 📝 Editando Documento ID: {doc_id}")
            doc = db.get_documento(doc_id)
            itens = db.query_table("itens", where=f"documento_id = {doc_id}")
            if not doc:
                st.error("Documento não encontrado.")
                return

            # Remove campos não editáveis
            doc_edit = {
                k: v
                for k, v in dict(doc).items()
                if k not in ("id", "hash", "caminho_arquivo", "data_upload")
            }

            t1, t2, t3 = st.tabs(["Cabeçalho", "Itens do Documento", "Recomendações (LLM)"])
            with t1:
                st.session_state.edited_doc_data = st.data_editor(
                    pd.DataFrame([doc_edit]),
                    use_container_width=True,
                    num_rows="fixed",
                    key=f"editor_doc_{doc_id}",
                )
            with t2:
                cols = [
                    "descricao",
                    "ncm",
                    "cfop",
                    "quantidade",
                    "unidade",
                    "valor_unitario",
                    "valor_total",
                    "codigo_produto",
                ]
                itens_disp = (
                    pd.DataFrame(columns=cols)
                    if itens.empty
                    else itens[[c for c in cols if c in itens.columns]]
                )
                st.session_state.edited_items_data = st.data_editor(
                    itens_disp, use_container_width=True, num_rows="dynamic", key=f"editor_items_{doc_id}"
                )
            with t3:
                st.info("💡 Sugestões automáticas do Agente (LLM).")
                if st.button("Gerar Sugestões (LLM)", use_container_width=False):
                    if orch.analitico and st.session_state.llm_instance:
                        with st.spinner("Analisando inconsistências..."):
                            prompt = (
                                f"O documento ID {doc_id} está em revisão (motivo: {doc.get('motivo_rejeicao')}). "
                                f"Analise os dados do cabeçalho {doc_edit} e itens {itens.to_dict('records')} "
                                f"e sugira correções fiscais (ex: CFOP, NCM, UF) ou de OCR."
                            )
                            out = orch.responder_pergunta(prompt, scope_filters={}, safe_mode=False)
                            st.markdown(out.get("texto", "Não foi possível gerar sugestões."))
                    else:
                        st.warning("O LLM não está configurado na barra lateral.")

            st.markdown("---")
            cA, cB, cC = st.columns(3)

            if cA.button("💾 Salvar & Marcar Revisado", type="primary", use_container_width=True):
                try:
                    # Cabeçalho
                    if not st.session_state.edited_doc_data.empty:
                        new_fields = st.session_state.edited_doc_data.iloc[0].to_dict()
                        for k, v in new_fields.items():
                            old = doc.get(k)
                            if str(old) != str(v):
                                db.inserir_revisao(
                                    documento_id=doc_id,
                                    campo=f"documento.{k}",
                                    valor_anterior=str(old),
                                    valor_corrigido=str(v),
                                    usuario=st.session_state.user_name or "revisor_ui",
                                )
                        db.atualizar_documento_campos(doc_id, **new_fields)

                    # Itens (recria com base na edição)
                    old_item_ids = tuple(itens["id"].unique().tolist()) if not itens.empty else ()
                    db.conn.execute("DELETE FROM itens WHERE documento_id = ?", (doc_id,))
                    if old_item_ids:
                        placeholders = ", ".join("?" for _ in old_item_ids)
                        db.conn.execute(f"DELETE FROM impostos WHERE item_id IN ({placeholders})", old_item_ids)
                    db.conn.commit()

                    for idx, row in st.session_state.edited_items_data.iterrows():
                        rowd = {k: v for k, v in row.to_dict().items() if pd.notna(v)}
                        if rowd:
                            db.inserir_item(documento_id=doc_id, **rowd)
                            db.inserir_revisao(
                                documento_id=doc_id,
                                campo=f"item[{idx}]",
                                valor_anterior="(item recriado)",
                                valor_corrigido=str(rowd),
                                usuario=st.session_state.user_name or "revisor_ui",
                            )

                    db.atualizar_documento_campo(doc_id, "status", "revisado")
                    db.atualizar_documento_campo(doc_id, "motivo_rejeicao", "Corrigido manualmente")
                    db.log("revisao_concluida", st.session_state.user_name or "revisor_ui", f"doc_id={doc_id} revisado.")
                    st.success(f"Documento ID {doc_id} atualizado e marcado como 'revisado'.")
                    st.session_state.doc_id_revisao = 0
                    st.rerun()
                except Exception as e:
                    st.error(f"Erro ao salvar revisões: {e}")
                    st.code(traceback.format_exc(), language="python")

            if cB.button("✅ Aprovar (Processado)", use_container_width=True):
                try:
                    db.atualizar_documento_campo(doc_id, "status", "processado")
                    db.atualizar_documento_campo(doc_id, "motivo_rejeicao", "Aprovado manualmente")
                    db.log("revisao_aprovada", st.session_state.user_name or "revisor_ui", f"doc_id={doc_id} processado.")
                    st.success(f"Documento ID {doc_id} marcado como 'processado'.")
                    st.session_state.doc_id_revisao = 0
                    st.rerun()
                except Exception as e:
                    st.error(f"Falha ao aprovar: {e}")

            if cC.button("🔁 Reprocessar", help="Re-extrai os dados do arquivo.", use_container_width=True):
                with st.spinner(f"Reprocessando documento ID {doc_id}..."):
                    try:
                        out = orch.reprocessar_documento(doc_id)
                        if out.get("ok"):
                            st.success(out.get("mensagem"))
                        else:
                            st.error(out.get("mensagem"))
                        st.session_state.doc_id_revisao = 0
                        st.rerun()
                    except Exception as e:
                        st.error(f"Falha ao acionar reprocessamento: {e}")
                        st.code(traceback.format_exc(), language="python")
    except Exception as e:
        st.error(f"Erro ao carregar documentos pendentes: {e}")
        st.code(traceback.format_exc(), language="python")

def tab_analises(orch: Orchestrator, db: BancoDeDados):
    st.subheader("🤖 Análises & LLM")
    st.markdown("**Filtros de Escopo (Opcional):**")
    f1, f2 = st.columns(2)
    with f1:
        uf_scope = st.text_input("UF (ex: SP, RJ):", key="llm_scope_uf")
    with f2:
        tipo_scope = st.multiselect("Tipo:", ["NFe", "NFCe", "CTe", "pdf", "png", "jpg"], key="llm_scope_tipo")

    c1, c2 = st.columns(2)
    with c1:
        safe_mode = st.toggle(
            "Modo Seguro (sem IA)",
            value=False,
            key="llm_safe_mode",
            help="Tenta responder usando lógica interna rápida (sem LLM).",
        )
    with c2:
        show_code = st.toggle("Mostrar Código/Query", value=True, key="llm_show_code")

    st.markdown("---")
    if (orch.analitico and st.session_state.llm_instance) or safe_mode:
        if orch.analitico and st.session_state.llm_instance:
            st.success(f"LLM Ativo: {st.session_state.llm_status_message}")
        else:
            st.info("LLM não configurado. Apenas o 'Modo Seguro' funcionará.")

        pergunta = st.text_area(
            "Sua Pergunta:",
            height=120,
            placeholder="Ex: Qual o valor total por UF dos documentos processados nos últimos 30 dias?",
        )
        bar_left, bar_clear, bar_exec = st.columns([6, 2, 2])
        with bar_clear:
            limpar = st.button("Limpar", use_container_width=True)
        with bar_exec:
            executar = st.button("Executar", type="primary", use_container_width=True, disabled=not pergunta.strip())

        if limpar:
            st.session_state["llm_scope_uf"] = ""
            st.session_state["llm_scope_tipo"] = []
            st.rerun()

        if executar:
            with st.spinner("Executando análise..."):
                try:
                    scope = {"uf": uf_scope or None, "tipo": tipo_scope or None}
                    out = orch.responder_pergunta(pergunta, scope_filters=scope, safe_mode=safe_mode)
                    st.info(f"Concluído em {out.get('duracao_s', 0):.2f}s (Agente: {out.get('agent_name', 'N/A')})")
                    st.markdown("**Resposta:**")
                    st.markdown(out.get("texto", "*Nenhum texto retornado.*"))

                    tabela = out.get("tabela")
                    if isinstance(tabela, pd.DataFrame) and not tabela.empty:
                        st.markdown("**Tabela de Dados:**")
                        st.dataframe(tabela, use_container_width=True, height=360)

                    figs = out.get("figuras") or []
                    if figs:
                        st.markdown("**Gráfico(s):**")
                        for f in figs:
                            try:
                                import plotly.graph_objects as go
                                import matplotlib.figure
                                if isinstance(f, go.Figure):
                                    st.plotly_chart(f, use_container_width=True)
                                elif isinstance(f, matplotlib.figure.Figure):
                                    st.pyplot(f)
                                else:
                                    st.warning(f"Tipo de figura não suportado: {type(f)}")
                            except ImportError:
                                st.warning("Bibliotecas gráficas não instaladas.")
                            except Exception as e:
                                st.error(f"Erro ao exibir figura: {e}")
                    if show_code:
                        with st.expander("Ver Código Executado / Query"):
                            st.code(out.get("code", "# Nenhum código disponível"), language="python")
                except Exception as e:
                    st.error(f"Falha ao responder: {e}")
                    st.code(traceback.format_exc(), language="python")
    else:
        st.warning(f"LLM não está ativo. Status: {st.session_state.llm_status_message}")

    st.markdown("---")
    with st.expander("🧠 Memória LLM (Histórico de Perguntas)", expanded=False):
        try:
            mem = db.query_table("memoria")
            if not mem.empty:
                st.dataframe(mem.sort_values("id", ascending=False), use_container_width=True, height=400)
            else:
                st.info("Sem registros de memória ainda.")
        except Exception as e:
            st.error(f"Erro ao carregar memória: {e}")

def tab_metricas(orch: Orchestrator, db: BancoDeDados):
    st.subheader("📊 Métricas & Insights")
    title_col, act_col = st.columns([7, 3])
    with title_col:
        st.caption("Acompanhe a qualidade do processamento, tempos e taxas de revisão/erro.")
    with act_col:
        gerar_insights = st.button("Gerar Insights (LLM)", type="primary", use_container_width=True)

    try:
        df_metricas_raw = db.query_table("metricas")
        tipos = df_metricas_raw["tipo_documento"].unique().tolist() if not df_metricas_raw.empty else []
        c1, c2 = st.columns(2)
        with c1:
            _ = st.date_input(
                "Período",
                (pd.Timestamp.now() - pd.DateOffset(days=30), pd.Timestamp.now()),
                key="metric_date",
            )
        with c2:
            _ = st.selectbox("Tipo de Documento", ["Todos"] + tipos, key="metric_doctype")

        df_metricas = df_metricas_raw.copy()
        if df_metricas.empty:
            st.info("Nenhuma métrica registrada no banco de dados (ou nos filtros selecionados).")
        else:
            acur = df_metricas["acuracia_media"].mean() * 100
            tx_rev = df_metricas["taxa_revisao"].mean() * 100
            t_med = df_metricas["tempo_medio"].mean()
            total = len(df_metricas)
            tx_err = df_metricas["taxa_erro"].mean() * 100

            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("Acurácia Média (Conf.)", f"{acur:.1f}%")
            k2.metric("Taxa de Revisão", f"{tx_rev:.1f}%")
            k3.metric("Tempo Médio Proc.", f"{t_med:.2f}s")
            k4.metric("Total de Eventos", f"{total}")
            k5.metric("Taxa de Erro", f"{tx_err:.1f}%")

            st.markdown("---")
            g1, g2 = st.columns(2)
            with g1:
                st.markdown("**Acurácia Média por Tipo**")
                st.bar_chart(df_metricas.groupby("tipo_documento")["acuracia_media"].mean())
            with g2:
                st.markdown("**Taxa de Erro por Tipo**")
                st.bar_chart(df_metricas.groupby("tipo_documento")["taxa_erro"].mean() * 100)

        st.markdown("---")
        st.subheader("⚠️ Documentos com Baixa Confiança (< 70%)")
        df_baixa = db.query_table("extracoes", where="confianca_media < 0.7")
        if not df_baixa.empty:
            st.dataframe(df_baixa, use_container_width=True, height=220)
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
                        d1 = df_metricas.groupby("tipo_documento")["acuracia_media"].mean().to_json()
                        d2 = df_metricas.groupby("tipo_documento")["taxa_erro"].mean().to_json()
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
            df_users = db.query_table("usuarios")
            cols_disp = ["id", "nome", "email", "perfil"]
            cols_exist = [c for c in cols_disp if c in df_users.columns]
            if not df_users.empty:
                st.session_state.edited_users_data = st.data_editor(
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
                ac_left, ac_right = st.columns([8, 2])
                with ac_right:
                    salvar = st.button("Salvar alterações", type="primary", use_container_width=True)
                if salvar:
                    orig = df_users[cols_exist].set_index("id")
                    edit = st.session_state.edited_users_data.set_index("id")
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
                    st.success(f"{changes} usuário(s) atualizado(s).") if changes else st.info("Nenhuma alteração.")
                    if changes:
                        st.rerun()
            else:
                st.info("Nenhum usuário cadastrado.")

            st.markdown("**Deletar Usuário**")
            c1, c2, c3 = st.columns([1, 2, 2])
            with c1:
                uid_del = st.number_input("ID do Usuário:", min_value=0, step=1, key="user_delete_id")
            with c2:
                confirmar = st.checkbox("Confirmo a exclusão", key="chk_confirma_delete")
            with c3:
                if st.button(
                    "Deletar Usuário",
                    disabled=(uid_del == 0 or not confirmar),
                    use_container_width=True,
                ):
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
            sub = st.form_submit_button("Criar Usuário")
            if sub:
                if not nome or not email or not perfil or not senha:
                    st.warning("Todos os campos são obrigatórios.")
                elif senha != senha_conf:
                    st.error("As senhas não conferem.")
                else:
                    try:
                        db.inserir_usuario(
                            nome=nome,
                            email=email,
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
        limit = st.number_input(
            "Número de logs recentes:", min_value=10, max_value=1000, value=100, step=10, key="log_limit"
        )
        logs = db.query_table("logs")
        if not logs.empty:
            st.dataframe(logs.sort_values("id", ascending=False).head(limit), use_container_width=True, height=420)
        else:
            st.info("Ainda não há logs registrados.")
    except Exception as e:
        st.error(f"Erro ao carregar logs: {e}")

# ---------------------- MAIN ----------------------
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
                                        nome=nome,
                                        email=email,
                                        perfil="operador",
                                        senha_hash=hash_password(senha),
                                    )
                                    st.success("Usuário criado! Vá para a aba Login para entrar.")
                                    db.log("registro_usuario", "sistema", f"Usuário {email} registrado.")
                                except Exception as e:
                                    st.error(f"Erro ao criar usuário (email pode já existir): {e}")
        return

    # Logado — instancia Orchestrator com LLM opcional
    orch = Orchestrator(
        db=db,
        validador=validador,
        memoria=memoria,
        llm=st.session_state.llm_instance,
    )

    ui_header()
    ui_sidebar_llm()

    # Navegação principal
    tabs = st.tabs([
        "📄 Processamento",
        "🧐 Revisão Fiscal",
        "🤖 Análises & LLM",
        "📊 Métricas & Insights",
        "⚙️ Administração",
    ])

    with tabs[0]:
        tab_processamento(orch, db)
    with tabs[1]:
        tab_revisao(orch, db)
    with tabs[2]:
        tab_analises(orch, db)
    with tabs[3]:
        tab_metricas(orch, db)
    with tabs[4]:
        tab_admin(db)

if __name__ == "__main__":
    main()
