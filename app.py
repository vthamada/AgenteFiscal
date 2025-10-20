# app.py

from __future__ import annotations
from pathlib import Path
import io
import traceback
import streamlit as st
import pandas as pd

from banco_de_dados import BancoDeDados
from validacao import ValidadorFiscal
from memoria import MemoriaSessao
from agentes import Orchestrator


st.set_page_config(page_title="Projeto Fiscal - PoC", layout="wide")


@st.cache_resource
def get_services():
    db = BancoDeDados()
    validador = ValidadorFiscal()
    memoria = MemoriaSessao(db)
    # LLM opcional (None). Se quiser ligar, passe um BaseChatModel em Orchestrator(...)
    orch = Orchestrator(db=db, validador=validador, memoria=memoria, llm=None)
    return db, validador, memoria, orch


def ui_header():
    st.title("📄 Projeto Fiscal – Ingestão, OCR, XML, Validação & Análises")
    st.caption("PoC: processa XML/Imagens/PDFs, extrai dados e permite perguntas analíticas (LLM opcional).")


def ui_sidebar_upload(orch: Orchestrator):
    st.sidebar.header("Upload de arquivos")
    up = st.sidebar.file_uploader(
        "Selecione um arquivo XML / PDF / imagem", type=["xml", "pdf", "jpg", "jpeg", "png", "tif", "tiff", "bmp"]
    )
    origem = st.sidebar.text_input("Origem (rótulo livre)", value="web")
    if up is not None:
        if st.sidebar.button("Ingerir arquivo"):
            try:
                doc_id = orch.ingestir_arquivo(up.name, up.getvalue(), origem=origem)
                if doc_id > 0:
                    st.sidebar.success(f"Documento ingerido com sucesso. ID = {doc_id}")
                else:
                    st.sidebar.warning(f"Ingestão concluída com ID={doc_id}. Verifique os logs/status.")
            except Exception as e:
                st.sidebar.error(f"Falha na ingestão: {e}")
                st.sidebar.exception(e)


def ui_tabs(orch: Orchestrator, db: BancoDeDados):
    tabs = st.tabs(["📚 Documentos", "🧾 Itens & Impostos", "🤖 Perguntas (LLM)", "📝 Logs", "🧠 Memória"])

    # ---------------------- Documentos ----------------------
    with tabs[0]:
        st.subheader("Documentos")
        where = st.text_input("Filtro SQL (WHERE)", value="")
        try:
            df = db.query_table("documentos", where=where or None)
        except Exception as e:
            st.error(f"Erro na consulta: {e}")
            df = pd.DataFrame()
        st.dataframe(df, use_container_width=True, height=420)

        col1, col2, col3 = st.columns(3)
        with col1:
            doc_id_sel = st.number_input("ID do documento para ações", min_value=0, step=1)
        with col2:
            if st.button("🔁 Revalidar documento"):
                if doc_id_sel:
                    out = orch.revalidar_documento(int(doc_id_sel))
                    if out.get("ok"):
                        st.success(out.get("mensagem"))
                    else:
                        st.warning(out.get("mensagem"))
        with col3:
            st.caption("Use o ID exibido na tabela para revalidar.")

    # ---------------------- Itens & Impostos ----------------------
    with tabs[1]:
        st.subheader("Itens & Impostos por Documento")
        doc_id_q = st.number_input("Documento ID", min_value=0, step=1)
        colA, colB = st.columns(2)
        if st.button("Consultar Itens/Impostos"):
            try:
                itens = db.query_table("itens", where=f"documento_id = {int(doc_id_q)}")
                impostos = pd.DataFrame()
                if not itens.empty:
                    item_ids = tuple(itens["id"].tolist())
                    item_ids_sql = f"({item_ids[0]})" if len(item_ids) == 1 else str(item_ids)
                    impostos = db.query_table("impostos", where=f"item_id IN {item_ids_sql}")
                with colA:
                    st.markdown("**Itens**")
                    st.dataframe(itens, use_container_width=True, height=360)
                with colB:
                    st.markdown("**Impostos**")
                    st.dataframe(impostos, use_container_width=True, height=360)
            except Exception as e:
                st.error(f"Erro ao consultar: {e}")
                st.exception(e)

    # ---------------------- Perguntas (LLM) ----------------------
    with tabs[2]:
        st.subheader("Perguntas analíticas (LLM → sandbox)")
        if orch.analitico is None:
            st.info("LLM não está configurada no Orchestrator. Configure um BaseChatModel para ativar esta aba.")
        else:
            pergunta = st.text_area("Pergunta", placeholder="Ex.: Qual foi o top 10 NCM por valor_total nos últimos 30 dias?")
            if st.button("Executar"):
                try:
                    out = orch.responder_pergunta(pergunta)
                    st.success("Execução concluída.")
                    st.write("**Texto:**")
                    st.write(out.get("texto", ""))
                    tabela = out.get("tabela")
                    if isinstance(tabela, pd.DataFrame) and not tabela.empty:
                        st.write("**Tabela:**")
                        st.dataframe(tabela, use_container_width=True, height=360)
                    figs = out.get("figuras") or []
                    if figs:
                        st.write("**Figura:**")
                        for f in figs:
                            try:
                                import plotly.graph_objects as go  # lazy
                                if isinstance(f, go.Figure):
                                    st.plotly_chart(f, use_container_width=True)
                                else:
                                    st.pyplot(f)
                            except Exception:
                                st.pyplot(f)
                except Exception as e:
                    st.error(f"Falha ao responder: {e}")
                    st.code(traceback.format_exc(), language="python")

    # ---------------------- Logs ----------------------
    with tabs[3]:
        st.subheader("Logs")
        try:
            logs = db.query_table("logs")
            st.dataframe(logs.sort_values("id", ascending=False), use_container_width=True, height=420)
        except Exception as e:
            st.error(f"Erro ao carregar logs: {e}")

    # ---------------------- Memória ----------------------
    with tabs[4]:
        st.subheader("Memória do agente analítico")
        try:
            mem = db.query_table("memoria")
            st.dataframe(mem.sort_values("id", ascending=False), use_container_width=True, height=420)
        except Exception as e:
            st.error(f"Erro ao carregar memória: {e}")


def main():
    db, validador, memoria, orch = get_services()
    ui_header()
    ui_sidebar_upload(orch)
    ui_tabs(orch, db)


if __name__ == "__main__":
    main()
