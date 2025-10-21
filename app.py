# app.py

from __future__ import annotations
from pathlib import Path
import io
import traceback
import streamlit as st
import pandas as pd
import os # Importado para ler variáveis de ambiente

# Importações do projeto
from banco_de_dados import BancoDeDados
from validacao import ValidadorFiscal
from memoria import MemoriaSessao
from agentes import Orchestrator
# Importações para configuração do LLM
from modelos_llm import make_llm, GEMINI_MODELS, OPENAI_MODELS, OPENROUTER_MODELS
from langchain_core.language_models import BaseChatModel

st.set_page_config(page_title="Projeto Fiscal - I2A2", layout="wide")

# --- Estado da Sessão ---
# Inicializa o estado da sessão para armazenar dados editáveis e ID selecionado
if 'edited_doc_data' not in st.session_state:
    st.session_state.edited_doc_data = {}
if 'edited_items_data' not in st.session_state:
    st.session_state.edited_items_data = pd.DataFrame()
if 'doc_id_revisao' not in st.session_state:
    st.session_state.doc_id_revisao = 0
if 'llm_instance' not in st.session_state:
    st.session_state.llm_instance = None
if 'llm_status_message' not in st.session_state:
    st.session_state.llm_status_message = "LLM não configurado."

# --- Configuração dos Serviços ---
@st.cache_resource # Mantém o cache dos serviços básicos
def get_base_services():
    """Retorna instâncias básicas (DB, Validador, Memoria) que não dependem do LLM."""
    db = BancoDeDados()
    validador = ValidadorFiscal()
    memoria = MemoriaSessao(db)
    return db, validador, memoria

def configure_llm(provider, model, api_key):
    """Tenta configurar o LLM e retorna a instância ou None."""
    try:
        if not provider or not model:
            st.session_state.llm_status_message = "Selecione Provedor e Modelo LLM."
            return None
        # Prioriza a chave da UI, senão tenta variável de ambiente
        key_to_use = api_key or os.getenv(f"{provider.upper()}_API_KEY")
        if not key_to_use:
            st.session_state.llm_status_message = f"Chave API para {provider} não encontrada na UI ou variáveis de ambiente."
            return None

        llm = make_llm(provider=provider, model=model, api_key=key_to_use)
        st.session_state.llm_status_message = f"LLM {provider}/{model} ATIVO."
        print(f"LLM Configurado: {provider}/{model}") # Log no console
        return llm
    except Exception as e:
        st.session_state.llm_status_message = f"Erro ao configurar LLM: {e}"
        print(f"Erro LLM: {e}") # Log no console
        return None

# --- Interface Principal ---

def ui_header():
    st.title("📄 Projeto Fiscal – Ingestão, OCR, XML, Validação & Análises")
    st.caption("I2A2 - PoC: processa XML/Imagens/PDFs, valida dados, permite revisão e análises com LLM.")

def ui_sidebar(orch: Orchestrator):
    st.sidebar.header("📤 Upload de Arquivos")
    up = st.sidebar.file_uploader(
        "Selecione XML / PDF / Imagem", type=["xml", "pdf", "jpg", "jpeg", "png", "tif", "tiff", "bmp"]
    )
    origem = st.sidebar.text_input("Origem (rótulo livre)", value="upload_ui")

    if up is not None:
        if st.sidebar.button(f"Ingerir '{up.name}'"):
            with st.spinner("Processando arquivo..."):
                try:
                    doc_id = orch.ingestir_arquivo(up.name, up.getvalue(), origem=origem)
                    doc_info = orch.db.get_documento(doc_id)
                    status = doc_info.get('status') if doc_info else 'desconhecido'
                    if doc_id > 0:
                        st.sidebar.success(f"Documento ID {doc_id} ingerido. Status: **{status}**")
                        # Força refresh se um doc pendente for adicionado
                        if status == 'revisao_pendente': st.rerun()
                    else:
                        st.sidebar.warning(f"Ingestão retornou ID={doc_id}. Verifique logs.")
                except Exception as e:
                    st.sidebar.error(f"Falha na ingestão: {e}")
                    st.sidebar.exception(e)

    st.sidebar.divider()
    st.sidebar.header("🧠 Configuração LLM")
    st.sidebar.caption(f"Status Atual: {st.session_state.llm_status_message}")

    providers = ["gemini", "openai", "openrouter"]
    selected_provider = st.sidebar.selectbox("Provedor LLM", options=providers, index=0, key="llm_provider")

    models: list[str] = []
    if selected_provider == "gemini": models = GEMINI_MODELS
    elif selected_provider == "openai": models = OPENAI_MODELS
    elif selected_provider == "openrouter": models = OPENROUTER_MODELS

    selected_model = st.sidebar.selectbox("Modelo", options=models, index=0 if models else -1, key="llm_model")
    api_key_input = st.sidebar.text_input("Chave API (opcional, usa var. ambiente se vazio)", type="password", key="llm_api_key")

    if st.sidebar.button("Aplicar Configuração LLM"):
        st.session_state.llm_instance = configure_llm(selected_provider, selected_model, api_key_input)
        # Força a recriação do orchestrator com o novo LLM
        st.cache_resource.clear()
        st.rerun() # Recarrega a página para atualizar o orchestrator e status

def ui_tabs(orch: Orchestrator, db: BancoDeDados):
    tab_list = [
        "📚 Documentos",
        "🧐 Revisão Pendente", # TELA 4 ADICIONADA
        "🧾 Itens & Impostos",
        "🤖 Perguntas (LLM)",
        "📊 Métricas", # TELA 6 ADICIONADA (Placeholder)
        "⚙️ Administração", # TELA 7 ADICIONADA (Placeholder)
        "📝 Logs",
        "🧠 Memória LLM",
    ]
    tabs = st.tabs(tab_list)

    # --- TELA 1: Documentos ---
    with tabs[0]:
        st.subheader("Documentos Processados")
        # Filtro de status
        status_filter_options = ["Todos", "processado", "revisado", "revisao_pendente", "quarentena", "erro"]
        selected_status = st.selectbox("Filtrar por Status:", status_filter_options)

        where_clause = ""
        if selected_status != "Todos":
            where_clause = f"status = '{selected_status}'"

        try:
            df_docs = db.query_table("documentos", where=where_clause or None)
            st.dataframe(df_docs, use_container_width=True, height=420)
        except Exception as e:
            st.error(f"Erro na consulta de documentos: {e}")
            st.exception(e)

        st.markdown("---")
        st.subheader("Ações")
        col1, col2, col3 = st.columns([1,1,2])
        with col1:
            doc_id_acao = st.number_input("ID do Documento para Ações:", min_value=1, step=1, key="doc_id_acao_geral")
        with col2:
            st.write("") # Espaçador
            st.write("") # Espaçador
            if st.button("🔁 Revalidar Documento", key="btn_revalidar_geral", disabled=not doc_id_acao):
                out = orch.revalidar_documento(int(doc_id_acao))
                if out.get("ok"):
                    st.success(out.get("mensagem"))
                    st.rerun() # Recarrega para atualizar a tabela
                else:
                    st.warning(out.get("mensagem"))
        with col3:
            st.caption("Selecione um ID da tabela acima e clique em 'Revalidar' para reprocessar as regras de validação.")

    # --- TELA 4: Revisão Pendente ---
    with tabs[1]:
        st.subheader("Documentos Pendentes de Revisão")
        try:
            df_pendentes = db.query_table("documentos", where="status = 'revisao_pendente'")
            if df_pendentes.empty:
                st.info("🎉 Nenhum documento pendente de revisão no momento.")
            else:
                st.dataframe(df_pendentes, use_container_width=True, height=250)

                doc_ids_pendentes = df_pendentes['id'].tolist()
                st.session_state.doc_id_revisao = st.selectbox(
                    "Selecione o ID do Documento para Revisar:",
                    options=doc_ids_pendentes,
                    index=0,
                    key="select_doc_revisao"
                )

                if st.session_state.doc_id_revisao:
                    doc_id_rev = int(st.session_state.doc_id_revisao)
                    st.markdown(f"#### Revisando Documento ID: {doc_id_rev}")

                    # Carrega dados atuais do DB
                    doc_data = db.get_documento(doc_id_rev)
                    items_data = db.query_table("itens", where=f"documento_id = {doc_id_rev}")
                    # TODO: Carregar impostos se a edição for necessária

                    if not doc_data:
                        st.error("Documento não encontrado.")
                    else:
                        st.markdown("**Dados do Cabeçalho:**")
                        # Usar st.data_editor para edição direta (experimental)
                        # Remove colunas não editáveis diretamente ou problemáticas para o editor
                        editable_doc_fields = {k: v for k, v in doc_data.items() if k not in ['id', 'hash', 'caminho_arquivo', 'data_upload']}
                        st.session_state.edited_doc_data = st.data_editor(
                            pd.DataFrame([editable_doc_fields]), # Precisa ser um DataFrame
                            use_container_width=True,
                            num_rows="dynamic", # Permite editar a linha
                            key=f"editor_doc_{doc_id_rev}" # Chave única para evitar resets
                        )

                        st.markdown("**Itens do Documento:**")
                        # Colunas a serem exibidas/editadas no data_editor para itens
                        item_cols_to_edit = ['descricao', 'ncm', 'cfop', 'quantidade', 'unidade', 'valor_unitario', 'valor_total', 'codigo_produto']
                        st.session_state.edited_items_data = st.data_editor(
                            items_data[item_cols_to_edit], # Passa o DataFrame filtrado
                            use_container_width=True,
                            num_rows="dynamic", # Permite adicionar/remover/editar linhas
                             key=f"editor_items_{doc_id_rev}" # Chave única
                        )

                        if st.button("💾 Salvar Correções e Marcar como Revisado", key=f"save_rev_{doc_id_rev}"):
                            try:
                                # --- Salvar Cabeçalho ---
                                if not st.session_state.edited_doc_data.empty:
                                    updated_doc_fields = st.session_state.edited_doc_data.iloc[0].to_dict()
                                    # Registrar alterações na tabela 'revisoes'
                                    for key, new_value in updated_doc_fields.items():
                                        old_value = doc_data.get(key)
                                        if str(old_value) != str(new_value): # Compara como string para evitar problemas de tipo
                                            db.inserir_revisao(
                                                documento_id=doc_id_rev, campo=f"documento.{key}",
                                                valor_anterior=str(old_value), valor_corrigido=str(new_value),
                                                usuario="revisor_ui" # TODO: Trocar por usuário logado
                                            )
                                    # Atualizar no banco
                                    db.atualizar_documento_campos(doc_id_rev, **updated_doc_fields)

                                # --- Salvar Itens ---
                                # Compara linha por linha com o original (se existir) ou insere novas
                                # Esta lógica pode ser complexa dependendo se IDs são mantidos
                                # Simplificação: Assume que o editor mantém a ordem ou deleta/insere
                                # Primeiro, deleta itens antigos para simplificar (não ideal para auditoria!)
                                # Ideal seria comparar IDs se o editor os mantivesse, ou fazer diff
                                db.conn.execute("DELETE FROM itens WHERE documento_id = ?", (doc_id_rev,))
                                db.conn.commit()

                                # Insere os itens editados
                                for index, row in st.session_state.edited_items_data.iterrows():
                                    item_dict = row.to_dict()
                                    # Remover valores NaN ou None antes de inserir
                                    item_dict_clean = {k: v for k, v in item_dict.items() if pd.notna(v)}
                                    new_item_id = db.inserir_item(documento_id=doc_id_rev, **item_dict_clean)
                                    # Registrar como revisão (ex: item novo ou modificado)
                                    db.inserir_revisao(
                                        documento_id=doc_id_rev, campo=f"item[{index}]",
                                        valor_anterior="(original)", valor_corrigido=str(item_dict_clean),
                                        usuario="revisor_ui"
                                    )
                                    # TODO: Adicionar lógica para salvar impostos associados se editados

                                # --- Atualizar Status ---
                                db.atualizar_documento_campo(doc_id_rev, "status", "revisado")
                                db.log("revisao_concluida", "revisor_ui", f"doc_id={doc_id_rev} marcado como revisado.")
                                st.success(f"Documento ID {doc_id_rev} atualizado e marcado como 'revisado'.")
                                # Limpa estado e recarrega
                                st.session_state.doc_id_revisao = 0
                                st.rerun()

                            except Exception as e_save:
                                st.error(f"Erro ao salvar revisões: {e_save}")
                                st.exception(e_save)

        except Exception as e:
            st.error(f"Erro ao carregar documentos pendentes: {e}")
            st.exception(e)

    # --- TELA 2: Itens & Impostos --- (Agora é a terceira aba)
    with tabs[2]:
        st.subheader("Consultar Itens & Impostos por Documento")
        doc_id_q_itens = st.number_input("Documento ID:", min_value=1, step=1, key="doc_id_q_itens")
        colA, colB = st.columns(2)

        if st.button("Consultar Detalhes", key="btn_consultar_itens", disabled=not doc_id_q_itens):
            try:
                doc_header = db.get_documento(int(doc_id_q_itens))
                if doc_header:
                    st.write(f"**Detalhes para Documento ID: {doc_id_q_itens} (Status: {doc_header.get('status')})**")
                    itens = db.query_table("itens", where=f"documento_id = {int(doc_id_q_itens)}")
                    impostos = pd.DataFrame() # Inicializa vazio

                    with colA:
                        st.markdown("**Itens**")
                        if not itens.empty:
                            st.dataframe(itens, use_container_width=True, height=360)
                            # Prepara IDs para buscar impostos
                            item_ids = tuple(itens["id"].unique().tolist())
                            item_ids_sql = ', '.join(map(str, item_ids))
                            impostos = db.query_table("impostos", where=f"item_id IN ({item_ids_sql})")
                        else:
                            st.info("Nenhum item encontrado para este documento.")

                    with colB:
                        st.markdown("**Impostos**")
                        if not impostos.empty:
                            st.dataframe(impostos, use_container_width=True, height=360)
                        elif not itens.empty:
                             st.info("Nenhum imposto associado aos itens deste documento.")
                        else:
                             st.info("Consulte os itens primeiro.")
                else:
                    st.warning(f"Documento com ID {doc_id_q_itens} não encontrado.")

            except Exception as e:
                st.error(f"Erro ao consultar itens/impostos: {e}")
                st.exception(e)

    # --- TELA 3: Perguntas (LLM) --- (Agora é a quarta aba)
    with tabs[3]:
        st.subheader("Perguntas Analíticas (LLM → Sandbox)")
        # Verifica se a instância LLM foi criada com sucesso na sidebar
        if orch.analitico and st.session_state.llm_instance:
            st.success(f"LLM Ativo: {st.session_state.llm_status_message}")
            pergunta = st.text_area("Sua Pergunta:", height=100, placeholder="Ex: Qual o valor total por UF dos documentos processados?")

            if st.button("Executar Análise", key="btn_executar_llm", disabled=not pergunta.strip()):
                with st.spinner("O Agente Analítico está pensando..."):
                    try:
                        out = orch.responder_pergunta(pergunta)
                        st.info(f"Análise concluída em {out.get('duracao_s', 0):.2f}s (Agente: {out.get('agent_name', 'N/A')})")

                        st.markdown("**Resposta:**")
                        st.markdown(out.get("texto", "*Nenhum texto retornado.*"))

                        tabela = out.get("tabela")
                        if isinstance(tabela, pd.DataFrame) and not tabela.empty:
                            st.markdown("**Tabela de Dados:**")
                            st.dataframe(tabela, use_container_width=True, height=360)

                        figs = out.get("figuras") or []
                        if figs:
                            st.markdown("**Gráfico(s):**")
                            for i, f in enumerate(figs):
                                try:
                                    import plotly.graph_objects as go # Lazy import
                                    import matplotlib.figure
                                    if isinstance(f, go.Figure):
                                        st.plotly_chart(f, use_container_width=True)
                                    elif isinstance(f, matplotlib.figure.Figure):
                                        st.pyplot(f)
                                    else:
                                        st.warning(f"Tipo de figura não suportado: {type(f)}")
                                except ImportError:
                                    st.warning("Bibliotecas Plotly ou Matplotlib não instaladas para exibir figura.")
                                except Exception as e_fig:
                                    st.error(f"Erro ao exibir figura {i+1}: {e_fig}")

                        with st.expander("Ver Código Executado"):
                            st.code(out.get("code", "# Nenhum código disponível"), language="python")

                    except Exception as e:
                        st.error(f"Falha ao executar a análise: {e}")
                        st.code(traceback.format_exc(), language="python")
        else:
            # Mensagem se LLM não estiver ativo
            st.warning(f"LLM não está ativo. Verifique a configuração na sidebar. Status: {st.session_state.llm_status_message}")
            st.info("Configure um provedor LLM e sua chave API na sidebar e clique em 'Aplicar Configuração LLM' para habilitar esta funcionalidade.")


    # --- TELA 6: Métricas --- (Placeholder)
    with tabs[4]:
        st.subheader("Métricas e Monitoramento")
        st.info("🚧 Funcionalidade de Métricas ainda não implementada.")
        st.write("Esta tela exibirá KPIs como acurácia média por campo, taxa de revisão humana, tempo médio de processamento, etc., consultando a tabela `metricas`.")

    # --- TELA 7: Administração --- (Placeholder)
    with tabs[5]:
        st.subheader("Administração e Segurança")
        st.info("🚧 Funcionalidade de Administração ainda não implementada.")
        st.write("Esta tela permitirá o gerenciamento de usuários (tabela `usuarios`), controle de acesso baseado em perfis, e configurações de segurança (como rotação de chaves, políticas de retenção).")


    # --- TELA 5: Logs --- (Agora penúltima aba)
    with tabs[6]:
        st.subheader("Logs do Sistema")
        try:
            # Adiciona opção para limitar número de logs exibidos
            limit = st.number_input("Número de logs recentes para exibir:", min_value=10, max_value=1000, value=100, step=10)
            logs_df = db.query_table("logs")
            st.dataframe(logs_df.sort_values("id", ascending=False).head(limit), use_container_width=True, height=420)
        except Exception as e:
            st.error(f"Erro ao carregar logs: {e}")

    # --- Memória LLM --- (Agora última aba)
    with tabs[7]:
        st.subheader("Histórico de Perguntas (Memória LLM)")
        try:
            mem = db.query_table("memoria")
            st.dataframe(mem.sort_values("id", ascending=False), use_container_width=True, height=420)
        except Exception as e:
            st.error(f"Erro ao carregar memória: {e}")


# --- Função Principal ---
def main():
    # Pega serviços básicos (DB, Validador, Memoria) do cache
    db, validador, memoria = get_base_services()

    # O LLM é configurado dinamicamente via sidebar e armazenado no session_state
    # Cria o orchestrator com o LLM do estado da sessão (pode ser None)
    orch = Orchestrator(db=db, validador=validador, memoria=memoria, llm=st.session_state.llm_instance)

    # Renderiza UI
    ui_header()
    ui_sidebar(orch) # Passa o orchestrator atualizado para a sidebar
    ui_tabs(orch, db)


if __name__ == "__main__":
    main()