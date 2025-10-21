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
# Importações para Criptografia e Mascaramento
from seguranca import Cofre, carregar_chave_do_env, mascarar_documento_fiscal, CRYPTO_OK, sha256_text # Importa sha256_text para senhas

st.set_page_config(page_title="Projeto Fiscal - I2A2", layout="wide")

# Define o caminho para as regras fiscais
REGRAS_FISCAIS_PATH = Path("regras_fiscais.yaml")

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
if 'app_cofre' not in st.session_state:
    st.session_state.app_cofre = None # Armazena o Cofre na sessão
if 'toast_exibido' not in st.session_state: # --- Adicionado para controle do toast ---
    st.session_state.toast_exibido = False

# --- Configuração dos Serviços ---

@st.cache_resource # Cacheia todos os serviços base (DB, Memoria, Cofre, Validador)
def get_core_services():
    """
    Inicializa e cacheia os serviços centrais que não mudam durante a sessão.
    Isso inclui DB, Memoria, Cofre e o Validador (que depende do Cofre e das Regras).
    """
    db = BancoDeDados()
    memoria = MemoriaSessao(db)
    
    # Carrega a chave *uma vez* e instancia o Cofre
    chave_criptografia = carregar_chave_do_env("APP_SECRET_KEY")
    cofre = Cofre(key=chave_criptografia)
    st.session_state.app_cofre = cofre # Armazena no estado da sessão para acesso fácil
    
    # --- CORREÇÃO: Chamadas st.toast REMOVIDAS daqui ---

    # Instancia o Validador, passando o Cofre e o caminho das regras
    validador = ValidadorFiscal(cofre=cofre, regras_path=REGRAS_FISCAIS_PATH)
    
    return db, memoria, cofre, validador

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

# --- Funções Auxiliares da UI ---

def descriptografar_e_mascarar_df(df: pd.DataFrame, cofre: Cofre) -> pd.DataFrame:
    """Descriptografa e mascara colunas sensíveis de um DataFrame para exibição."""
    if df.empty: # Não verifica mais o cofre aqui, deixa a função interna tratar
        return df
    
    df_display = df.copy()
    
    # Função segura para aplicar descriptografia e mascaramento
    def decrypt_and_mask(encrypted_value):
        if not encrypted_value or not isinstance(encrypted_value, str):
            return encrypted_value # Retorna valores não-string (ex: None)
        
        decrypted = encrypted_value
        if cofre.available:
            try:
                decrypted = cofre.decrypt_text(encrypted_value)
            except Exception as e:
                print(f"Erro em decrypt_and_mask (decrypt): {e}")
                return "Erro Cripto"
        
        # Se a descriptografia falhou (retornou o original cripto) ou não estava ativa,
        # ainda tentamos mascarar o que tivermos.
        return mascarar_documento_fiscal(decrypted)


    if 'emitente_cnpj' in df_display.columns:
        df_display['emitente_cnpj'] = df_display['emitente_cnpj'].apply(decrypt_and_mask)
    if 'destinatario_cnpj' in df_display.columns:
        df_display['destinatario_cnpj'] = df_display['destinatario_cnpj'].apply(decrypt_and_mask)
    
    return df_display

def descriptografar_dict_para_edicao(data: dict, cofre: Cofre) -> dict:
    """Descriptografa dados de um dicionário para edição (sem máscara)."""
    data_decrypted = data.copy()
    if not cofre.available:
        return data_decrypted
        
    # Lista de campos que podem conter CNPJ ou CPF
    campos_sensíveis = ['emitente_cnpj', 'destinatario_cnpj'] # Adicionar 'emitente_cpf', 'destinatario_cpf' se existirem
    
    for campo in campos_sensíveis:
        valor_criptografado = data_decrypted.get(campo)
        if valor_criptografado and isinstance(valor_criptografado, str):
            try:
                # Tenta descriptografar
                data_decrypted[campo] = cofre.decrypt_text(valor_criptografado)
            except Exception as e:
                # Se falhar (ex: chave errada ou dado corrompido), mantém o valor criptografado para o usuário ver
                print(f"Erro ao descriptografar {campo} para edição: {e}")
                data_decrypted[campo] = f"ERRO_CRIPTOGRAFIA: {valor_criptografado}"
    return data_decrypted

# --- Interface Principal ---

def ui_header():
    st.title("📄 Projeto Fiscal – Ingestão, OCR, XML, Validação & Análises")
    st.caption("I2A2 - PoC: processa XML/Imagens/PDFs, valida dados, permite revisão e análises com LLM.")

def ui_sidebar(orch: Orchestrator):
    st.sidebar.header("📤 Upload de Arquivos")
    uploaded_files = st.sidebar.file_uploader(
        "Selecione XML / PDF / Imagem", 
        type=["xml", "pdf", "jpg", "jpeg", "png", "tif", "tiff", "bmp"],
        accept_multiple_files=True # Permite múltiplos arquivos
    )
    origem = st.sidebar.text_input("Origem (rótulo livre)", value="upload_ui")

    if uploaded_files:
        if st.sidebar.button(f"Ingerir {len(uploaded_files)} Arquivo(s)"):
            all_success = True
            progress_bar = st.sidebar.progress(0, text="Iniciando ingestão...")
            
            for i, up in enumerate(uploaded_files):
                status_text = f"Processando: {up.name} ({i+1}/{len(uploaded_files)})..."
                progress_bar.progress((i + 1) / len(uploaded_files), text=status_text)
                
                try:
                    doc_id = orch.ingestir_arquivo(up.name, up.getvalue(), origem=origem)
                    doc_info = orch.db.get_documento(doc_id)
                    status = doc_info.get('status') if doc_info else 'desconhecido'
                    
                    if status in ('revisao_pendente', 'erro', 'quarentena'):
                        all_success = False
                        st.sidebar.warning(f"ID {doc_id} ('{up.name}'): Status **{status}**")
                    else:
                        st.sidebar.success(f"ID {doc_id} ('{up.name}'): Status **{status}**")
                
                except Exception as e:
                    all_success = False
                    st.sidebar.error(f"Falha em '{up.name}': {e}")

            progress_bar.empty()
            if all_success:
                st.sidebar.success("Todos os arquivos foram ingeridos com sucesso.")
            else:
                 st.sidebar.warning("Alguns arquivos falharam ou precisam de revisão.")
            st.rerun() # Recarrega a página para atualizar todas as abas

    st.sidebar.divider()
    st.sidebar.header("🧠 Configuração LLM")
    st.sidebar.caption(f"Status Atual: {st.session_state.llm_status_message}")

    providers = ["", "gemini", "openai", "openrouter"]
    selected_provider = st.sidebar.selectbox("Provedor LLM", options=providers, index=0, key="llm_provider")

    models: list[str] = []
    if selected_provider == "gemini": models = GEMINI_MODELS
    elif selected_provider == "openai": models = OPENAI_MODELS
    elif selected_provider == "openrouter": models = OPENROUTER_MODELS

    selected_model = st.sidebar.selectbox("Modelo", options=models, index=0 if models else -1, key="llm_model")
    api_key_input = st.sidebar.text_input("Chave API (opcional, usa var. ambiente se vazio)", type="password", key="llm_api_key")

    if st.sidebar.button("Aplicar Configuração LLM"):
        st.session_state.llm_instance = configure_llm(selected_provider, selected_model, api_key_input)
        st.cache_resource.clear() # Limpa o cache para recriar o Orchestrator com o novo LLM
        st.rerun()

def ui_tabs(orch: Orchestrator, db: BancoDeDados, cofre: Cofre):
    tab_list = [
        "📚 Documentos",       # Tela 1 (com mascaramento)
        "🧐 Revisão Pendente", # Tela 5 (implementada)
        "🧾 Itens & Impostos", # Consulta
        "🤖 Perguntas (LLM)",  # Tela 3 (com filtros)
        "📊 Métricas",       # Tela 4 (implementada)
        "⚙️ Administração",    # Tela 7 (implementada)
        "📝 Logs",
        "🧠 Memória LLM",
    ]
    tabs = st.tabs(tab_list)

    # --- TELA 1: Documentos (com Mascaramento) ---
    with tabs[0]:
        st.subheader("Documentos Processados")
        status_filter_options = ["Todos", "processado", "revisado", "revisao_pendente", "quarentena", "erro"]
        selected_status = st.selectbox("Filtrar por Status:", status_filter_options, key="doc_status_filter")

        where_clause = ""
        if selected_status != "Todos":
            where_clause = f"status = '{selected_status}'"

        try:
            df_docs = db.query_table("documentos", where=where_clause or None)
            
            # --- Aplica Descriptografia e Mascaramento ---
            df_display = descriptografar_e_mascarar_df(df_docs, cofre)
            # ---------------------------------------------
            
            st.dataframe(df_display, use_container_width=True, height=420)
            
        except Exception as e:
            st.error(f"Erro na consulta de documentos: {e}")
            st.exception(e)

        st.markdown("---")
        st.subheader("Ações")
        col1, col2, col3 = st.columns([1,1,2])
        with col1:
            doc_id_acao = st.number_input("ID do Documento para Ações:", min_value=0, step=1, key="doc_id_acao_geral")
        with col2:
            st.write("") # Espaçador
            st.write("") # Espaçador
            if st.button("🔁 Revalidar Documento", key="btn_revalidar_geral", disabled=(doc_id_acao == 0)):
                with st.spinner(f"Revalidando ID {doc_id_acao}..."):
                    out = orch.revalidar_documento(int(doc_id_acao))
                    if out.get("ok"):
                        st.success(out.get("mensagem"))
                        st.rerun()
                    else:
                        st.warning(out.get("mensagem"))
        with col3:
            st.caption("Selecione um ID da tabela acima e clique em 'Revalidar' para reprocessar as regras de validação.")

    # --- TELA 5: Revisão Pendente (Implementada) ---
    with tabs[1]:
        st.subheader("Documentos Pendentes de Revisão (Tela 5)")
        try:
            # Filtros da Tela de Revisão
            col_f1, col_f2, col_f3 = st.columns(3)
            with col_f1: tipo_filtro = st.selectbox("Filtrar Tipo:", ["Todos", "NFe", "NFCe", "CTe", "pdf", "jpg", "png"], key="rev_tipo")
            with col_f2: uf_filtro = st.text_input("Filtrar UF (ex: SP):", key="rev_uf")
            # Status é implicitamente "revisao_pendente" nesta aba
            
            where_rev = "status = 'revisao_pendente'"
            if tipo_filtro != "Todos": where_rev += f" AND tipo = '{tipo_filtro}'"
            if uf_filtro: where_rev += f" AND uf = '{uf_filtro.upper()}'"

            df_pendentes = db.query_table("documentos", where=where_rev)
            
            if df_pendentes.empty:
                st.info("🎉 Nenhum documento pendente de revisão com os filtros atuais.")
            else:
                # Exibe lista com mascaramento
                df_pendentes_display = descriptografar_e_mascarar_df(df_pendentes, cofre)
                st.dataframe(df_pendentes_display, use_container_width=True, height=250)

                doc_ids_pendentes = df_pendentes['id'].tolist()
                st.session_state.doc_id_revisao = st.selectbox(
                    "Selecione o ID do Documento para Revisar:",
                    options=[0] + doc_ids_pendentes, # Adiciona 0 como "Nenhum"
                    index=0,
                    key="select_doc_revisao"
                )

                if st.session_state.doc_id_revisao > 0:
                    doc_id_rev = int(st.session_state.doc_id_revisao)
                    st.markdown(f"--- \n#### 📝 Editando Documento ID: {doc_id_rev}")

                    doc_data = db.get_documento(doc_id_rev)
                    items_data = db.query_table("itens", where=f"documento_id = {doc_id_rev}")
                    # TODO: Carregar impostos se a edição for necessária

                    if not doc_data:
                        st.error("Documento não encontrado (pode ter sido processado).")
                    else:
                        # --- Descriptografa para Edição (SEM MÁSCARA) ---
                        doc_data_decrypted = descriptografar_dict_para_edicao(doc_data, cofre)
                        editable_doc_fields = {k: v for k, v in doc_data_decrypted.items() if k not in ['id', 'hash', 'caminho_arquivo', 'data_upload']}
                        
                        # Abas de Edição
                        rev_tab1, rev_tab2, rev_tab3 = st.tabs(["Cabeçalho", "Itens do Documento", "Recomendações (LLM)"])

                        with rev_tab1:
                            st.markdown("**Dados do Cabeçalho:**")
                            st.session_state.edited_doc_data = st.data_editor(
                                pd.DataFrame([editable_doc_fields]),
                                use_container_width=True,
                                num_rows="fixed", # Impede adicionar novas linhas
                                key=f"editor_doc_{doc_id_rev}"
                            )
                        with rev_tab2:
                            st.markdown("**Itens do Documento:**")
                            item_cols_to_edit = ['descricao', 'ncm', 'cfop', 'quantidade', 'unidade', 'valor_unitario', 'valor_total', 'codigo_produto']
                            # Garante que colunas existam mesmo se 'items_data' estiver vazio
                            if items_data.empty:
                                items_data_display = pd.DataFrame(columns=item_cols_to_edit)
                            else:
                                # Garante que apenas colunas existentes sejam selecionadas
                                cols_existentes = [col for col in item_cols_to_edit if col in items_data.columns]
                                items_data_display = items_data[cols_existentes]

                            st.session_state.edited_items_data = st.data_editor(
                                items_data_display,
                                use_container_width=True,
                                num_rows="dynamic", # Permite adicionar/remover/editar
                                key=f"editor_items_{doc_id_rev}"
                            )
                        with rev_tab3:
                            st.info("💡 Sugestões automáticas do Agente (LLM)")
                            if st.button("Gerar Sugestões de Correção (LLM)"):
                                if orch.analitico and st.session_state.llm_instance:
                                    with st.spinner("Analisando inconsistências..."):
                                        # Monta o prompt para sugestão
                                        prompt_sugestao = f"O documento ID {doc_id_rev} está em revisão (motivo: {doc_data.get('motivo_rejeicao')}). Analise os dados do cabeçalho {doc_data_decrypted} e itens {items_data.to_dict('records')} e sugira correções fiscais (ex: CFOP, NCM, UF) ou de OCR."
                                        # Chama o orquestrador (modo padrão, com geração de código)
                                        sugestao_out = orch.responder_pergunta(prompt_sugestao, scope_filters={}, safe_mode=False)
                                        st.markdown(sugestao_out.get("texto", "Não foi possível gerar sugestões."))
                                else:
                                    st.warning("O LLM não está configurado na sidebar.")

                        st.markdown("---")
                        col_r1, col_r2, col_r3 = st.columns(3)
                        
                        # Botão Salvar (Código Completo Restaurado)
                        if col_r1.button("💾 Salvar Correções e Marcar como Revisado", key=f"save_rev_{doc_id_rev}", type="primary"):
                            try:
                                # --- Salvar Cabeçalho ---
                                if not st.session_state.edited_doc_data.empty:
                                    updated_doc_fields = st.session_state.edited_doc_data.iloc[0].to_dict()
                                    
                                    # --- CRIPTOGRAFAR NOVAMENTE ANTES DE SALVAR ---
                                    campos_sensíveis = ['emitente_cnpj', 'destinatario_cnpj'] # Adicionar cpf se houver
                                    for campo in campos_sensíveis:
                                        if campo in updated_doc_fields and updated_doc_fields[campo]:
                                            # Apenas criptografa se o cofre estiver ativo
                                            if cofre.available:
                                                updated_doc_fields[campo] = cofre.encrypt_text(str(updated_doc_fields[campo]))
                                    # -----------------------------------------

                                    # Registrar alterações na tabela 'revisoes'
                                    for key, new_value in updated_doc_fields.items():
                                        old_value = doc_data.get(key) # Valor (potencialmente cripto) do DB
                                        if str(old_value) != str(new_value):
                                            db.inserir_revisao(
                                                documento_id=doc_id_rev, campo=f"documento.{key}",
                                                valor_anterior=str(old_value), valor_corrigido=str(new_value),
                                                usuario="revisor_ui" # TODO: Trocar por usuário logado
                                            )
                                    # Atualizar no banco
                                    db.atualizar_documento_campos(doc_id_rev, **updated_doc_fields)

                                # --- Salvar Itens ---
                                # Lógica de deletar e reinserir (simplificada)
                                # Primeiro, pega os IDs dos itens antigos para deletar impostos
                                old_item_ids = tuple(items_data['id'].unique().tolist())
                                
                                # Deleta itens antigos
                                db.conn.execute("DELETE FROM itens WHERE documento_id = ?", (doc_id_rev,))
                                
                                # Deleta impostos associados aos itens antigos (se houver)
                                if old_item_ids:
                                    id_placeholder = ', '.join('?' for _ in old_item_ids)
                                    db.conn.execute(f"DELETE FROM impostos WHERE item_id IN ({id_placeholder})", old_item_ids)
                                
                                db.conn.commit()

                                # Insere os itens editados
                                for index, row in st.session_state.edited_items_data.iterrows():
                                    item_dict = row.to_dict()
                                    item_dict_clean = {k: v for k, v in item_dict.items() if pd.notna(v)}
                                    if item_dict_clean: # Não insere linhas vazias
                                        new_item_id = db.inserir_item(documento_id=doc_id_rev, **item_dict_clean)
                                        # Registrar como revisão
                                        db.inserir_revisao(
                                            documento_id=doc_id_rev, campo=f"item[{index}]",
                                            valor_anterior="(item recriado)", valor_corrigido=str(item_dict_clean),
                                            usuario="revisor_ui"
                                        )
                                        # TODO: Adicionar lógica para salvar impostos associados se editados

                                # --- Atualizar Status ---
                                db.atualizar_documento_campo(doc_id_rev, "status", "revisado")
                                db.atualizar_documento_campo(doc_id_rev, "motivo_rejeicao", "Corrigido manually") # Atualiza motivo
                                db.log("revisao_concluida", "revisor_ui", f"doc_id={doc_id_rev} marcado como revisado.")
                                st.success(f"Documento ID {doc_id_rev} atualizado e marcado como 'revisado'.")
                                st.session_state.doc_id_revisao = 0; st.rerun()

                            except Exception as e_save:
                                st.error(f"Erro ao salvar revisões: {e_save}"); st.exception(e_save)
                        
                        if col_r2.button("✅ Aprovar (Marcar como Processado)", key=f"approve_{doc_id_rev}"):
                             db.atualizar_documento_campo(doc_id_rev, "status", "processado")
                             db.atualizar_documento_campo(doc_id_rev, "motivo_rejeicao", "Aprovado manualmente")
                             db.log("revisao_aprovada", "revisor_ui", f"doc_id={doc_id_rev} marcado como processado.")
                             st.success(f"Documento ID {doc_id_rev} marcado como 'processado'.")
                             st.session_state.doc_id_revisao = 0; st.rerun()

                        # --- Botão Reprocessar (Tela 5) ---
                        if col_r3.button("🔁 Reprocessar (Re-Extrair)", key=f"reprocess_{doc_id_rev}", help="Apaga dados extraídos e re-executa o pipeline de ingestão."):
                            with st.spinner(f"Reprocessando documento ID {doc_id_rev}..."):
                                try:
                                    # --- CORREÇÃO APLICADA: Chama a função reprocessar ---
                                    out = orch.reprocessar_documento(int(doc_id_rev))
                                    if out.get("ok"):
                                        st.success(out.get("mensagem"))
                                    else:
                                        st.error(out.get("mensagem"))
                                    st.session_state.doc_id_revisao = 0
                                    st.rerun()
                                except Exception as e_reproc:
                                    st.error(f"Falha ao acionar reprocessamento: {e_reproc}")
                                    st.exception(e_reproc)

        except Exception as e:
            st.error(f"Erro ao carregar documentos pendentes: {e}"); st.exception(e)

    # --- TELA 2: Itens & Impostos ---
    with tabs[2]:
        st.subheader("Consultar Itens & Impostos por Documento")
        doc_id_q_itens = st.number_input("Documento ID:", min_value=0, step=1, key="doc_id_q_itens")
        colA, colB = st.columns(2)

        if st.button("Consultar Detalhes", key="btn_consultar_itens", disabled=(doc_id_q_itens == 0)):
            try:
                doc_header = db.get_documento(int(doc_id_q_itens))
                if doc_header:
                    st.write(f"**Detalhes para Documento ID: {doc_id_q_itens} (Status: {doc_header.get('status')})**")
                    itens = db.query_table("itens", where=f"documento_id = {int(doc_id_q_itens)}")
                    impostos = pd.DataFrame()
                    with colA:
                        st.markdown("**Itens**")
                        if not itens.empty:
                            st.dataframe(itens, use_container_width=True, height=360)
                            item_ids = tuple(itens["id"].unique().tolist())
                            item_ids_sql = ', '.join(map(str, item_ids))
                            impostos = db.query_table("impostos", where=f"item_id IN ({item_ids_sql})")
                        else: st.info("Nenhum item encontrado.")
                    with colB:
                        st.markdown("**Impostos**")
                        if not impostos.empty:
                            st.dataframe(impostos, use_container_width=True, height=360)
                        elif not itens.empty:
                             st.info("Nenhum imposto associado aos itens deste documento.")
                        else:
                             st.info("Consulte os itens primeiro.")
                else: st.warning(f"Documento com ID {doc_id_q_itens} não encontrado.")
            except Exception as e: st.error(f"Erro ao consultar: {e}"); st.exception(e)

    # --- TELA 3: Perguntas (LLM) (com Filtros e Toggles) ---
    with tabs[3]:
        st.subheader("Perguntas Analíticas (LLM → Sandbox) (Tela 3)")
        
        # Filtros de Escopo
        st.markdown("**Filtros de Escopo (Opcional):**")
        col_f_llm1, col_f_llm2 = st.columns(2)
        with col_f_llm1:
            uf_escopo = st.text_input("Filtrar por UF (ex: SP, RJ):", key="llm_scope_uf")
        with col_f_llm2:
            tipo_escopo = st.multiselect("Filtrar por Tipo:", ["NFe", "NFCe", "CTe", "pdf", "png", "jpg"], key="llm_scope_tipo")

        # Toggles de Controle
        col_t_llm1, col_t_llm2 = st.columns(2)
        with col_t_llm1:
            safe_mode = st.toggle("Modo Seguro (sem geração de código)", value=False, key="llm_safe_mode", help="Força o uso de ferramentas pré-definidas (se implementadas). Desabilita a geração de código Python.")
        with col_t_llm2:
            show_code = st.toggle("Mostrar Código Gerado", value=True, key="llm_show_code")
        
        st.markdown("---")

        if orch.analitico and st.session_state.llm_instance:
            st.success(f"LLM Ativo: {st.session_state.llm_status_message}")
            pergunta = st.text_area("Sua Pergunta:", height=100, placeholder="Ex: Qual o valor total por UF dos documentos processados?")

            if st.button("Executar Análise", key="btn_executar_llm", disabled=not pergunta.strip()):
                with st.spinner("O Agente Analítico está pensando..."):
                    try:
                        # --- CORREÇÃO APLICADA: Passa filtros para o Orchestrator ---
                        scope = {
                            "uf": uf_escopo if uf_escopo else None,
                            "tipo": tipo_escopo if tipo_escopo else None
                        }
                        out = orch.responder_pergunta(pergunta, scope_filters=scope, safe_mode=safe_mode)
                        # -----------------------------------------------------------
                        
                        st.info(f"Análise concluída em {out.get('duracao_s', 0):.2f}s (Agente: {out.get('agent_name', 'N/A')})")
                        st.markdown("**Resposta:**"); st.markdown(out.get("texto", "*Nenhum texto retornado.*"))

                        tabela = out.get("tabela")
                        if isinstance(tabela, pd.DataFrame) and not tabela.empty:
                            st.markdown("**Tabela de Dados:**"); st.dataframe(tabela, use_container_width=True, height=360)

                        figs = out.get("figuras") or []
                        if figs:
                            st.markdown("**Gráfico(s):**")
                            for i, f in enumerate(figs):
                                try:
                                    import plotly.graph_objects as go; import matplotlib.figure
                                    if isinstance(f, go.Figure): st.plotly_chart(f, use_container_width=True)
                                    elif isinstance(f, matplotlib.figure.Figure): st.pyplot(f)
                                    else: st.warning(f"Tipo de figura não suportado: {type(f)}")
                                except ImportError: st.warning("Libs gráficas não instaladas.")
                                except Exception as e_fig: st.error(f"Erro ao exibir figura: {e_fig}")

                        if show_code: # Usa o toggle
                            with st.expander("Ver Código Executado"):
                                st.code(out.get("code", "# Nenhum código disponível"), language="python")

                    except Exception as e:
                        st.error(f"Falha ao responder: {e}"); st.code(traceback.format_exc(), language="python")
        else:
            st.warning(f"LLM não está ativo. Status: {st.session_state.llm_status_message}")

    # --- TELA 6: Métricas (Implementada com dados reais) ---
    with tabs[4]:
        st.subheader("Métricas e Monitoramento (Tela 4)")
        
        df_metricas = pd.DataFrame() # Inicializa
        try:
            # --- CORREÇÃO APLICADA: Lê dados reais da tabela 'metricas' ---
            df_metricas_raw = db.query_table("metricas")

            # Filtros
            tipos_no_db = df_metricas_raw['tipo_documento'].unique().tolist() if not df_metricas_raw.empty else []
            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1: date_range = st.date_input("Período", (pd.Timestamp.now() - pd.DateOffset(days=30), pd.Timestamp.now()), key="metric_date")
            with col_m2: doc_type = st.selectbox("Tipo de Documento", ["Todos"] + tipos_no_db, key="metric_doctype")
            with col_m3: status_met = st.selectbox("Status", ["Todos", "processado", "erro", "revisao_pendente", "revisado", "quarentena"], key="metric_status")
            
            # TODO: Aplicar filtros ao df_metricas
            df_metricas = df_metricas_raw.copy()
            # (Lógica de filtro de data, tipo e status seria aplicada aqui)
            
            if df_metricas.empty:
                st.info("Nenhuma métrica registrada no banco de dados (ou nos filtros selecionados).")
            else:
                # Calcula KPIs Reais
                acuracia_media = df_metricas['acuracia_media'].mean() * 100
                taxa_revisao = df_metricas['taxa_revisao'].mean() * 100
                tempo_medio = df_metricas['tempo_medio'].mean()
                total_docs = len(df_metricas)
                taxa_erro = df_metricas['taxa_erro'].mean() * 100

                # KPIs (Cards)
                col_k1, col_k2, col_k3, col_k4, col_k5 = st.columns(5)
                col_k1.metric("Acurácia Média (Conf.)", f"{acuracia_media:.1f}%")
                col_k2.metric("Taxa de Revisão", f"{taxa_revisao:.1f}%", help="Percentual de documentos que exigiram revisão humana.")
                col_k3.metric("Tempo Médio Proc.", f"{tempo_medio:.2f}s")
                col_k4.metric("Total de Eventos", f"{total_docs}", help="Total de eventos de processamento registrados.")
                col_k5.metric("Taxa de Erro", f"{taxa_erro:.1f}%", help="% de eventos de processamento que resultaram em erro.")

                st.markdown("---")
                
                col_g1, col_g2 = st.columns(2)
                with col_g1:
                    st.markdown("**Acurácia (Confiança) Média por Tipo**")
                    df_tipo_acuracia = df_metricas.groupby('tipo_documento')['acuracia_media'].mean()
                    st.bar_chart(df_tipo_acuracia)
                    
                with col_g2:
                    st.markdown("**Taxa de Erro por Tipo**")
                    df_tipo_erro = df_metricas.groupby('tipo_documento')['taxa_erro'].mean() * 100
                    st.bar_chart(df_tipo_erro)

            st.markdown("---")
            # --- IMPLEMENTAÇÃO: Insights Cognitivos (Tela 6) ---
            st.subheader("Insights Cognitivos (LLM)")
            if st.button("Gerar Insights Automáticos sobre Métricas"):
                if orch.analitico and st.session_state.llm_instance:
                    if df_metricas.empty:
                        st.warning("Não há métricas para analisar.")
                    else:
                        with st.spinner("Analisando métricas..."):
                            # Prepara os dados para o prompt
                            kpis_principais = {
                                "acuracia_media": acuracia_media, "taxa_revisao": taxa_revisao,
                                "tempo_medio": tempo_medio, "taxa_erro": taxa_erro, "total_eventos": total_docs
                            }
                            # Converte dataframes em strings (ou JSON) para o prompt
                            dados_tipo_acuracia = df_metricas.groupby('tipo_documento')['acuracia_media'].mean().to_json()
                            dados_tipo_erro = df_metricas.groupby('tipo_documento')['taxa_erro'].mean().to_json()
                            
                            prompt_insights = f"""
                            Analise os seguintes KPIs de um sistema de processamento de documentos:
                            KPIs Principais: {kpis_principais}
                            Acurácia (Confiança) por Tipo: {dados_tipo_acuracia}
                            Taxa de Erro por Tipo: {dados_tipo_erro}
                            
                            Com base nesses dados, forneça 2 a 3 insights acionáveis em português. 
                            Foque em:
                            1. Qual tipo de documento está performando melhor ou pior (em acurácia e erro)?
                            2. Qual é a relação entre a taxa de revisão e a acurácia?
                            3. Há algum ponto crítico óbvio que a equipe de operações deveria investigar?
                            
                            Seja sucinto e direto ao ponto.
                            """
                            # Chama o orquestrador (modo padrão, sem filtros de escopo, sem modo seguro)
                            insights_out = orch.responder_pergunta(prompt_insights, scope_filters={}, safe_mode=False)
                            st.markdown(insights_out.get("texto", "Não foi possível gerar insights."))
                else:
                    st.warning("O LLM não está configurado na sidebar. Ative-o para gerar insights.")
            
            st.markdown("---")
            st.subheader("Documentos com Baixa Confiança (< 70%)")
            # Esta tabela permanece, pois busca em 'extracoes'
            df_baixa_conf = db.query_table("extracoes", where="confianca_media < 0.7")
            if not df_baixa_conf.empty:
                st.dataframe(df_baixa_conf, use_container_width=True, height=200)
            else:
                st.info("Nenhum documento com confiança inferior a 70% encontrado.")
                
        except Exception as e_metric:
            st.error(f"Erro ao carregar métricas: {e_metric}")
            st.exception(e_metric)

    # --- TELA 7: Administração (Implementada) ---
    with tabs[5]:
        st.subheader("Administração e Segurança (Tela 7)")
        
        # NOTE: Esta é uma implementação de PoC sem autenticação real.
        # Em produção, esta tela inteira deveria ser protegida por login.
        
        admin_tab1, admin_tab2 = st.tabs(["Gerenciar Usuários", "Criar Novo Usuário"])
        
        with admin_tab1:
            st.markdown("**Usuários Cadastrados**")
            try:
                df_usuarios = db.query_table("usuarios")
                # Não exibir o hash da senha
                colunas_display = ['id', 'nome', 'email', 'perfil']
                st.dataframe(df_usuarios[colunas_display], use_container_width=True)
                
                st.markdown("**Deletar Usuário**")
                col_del1, col_del2 = st.columns([1, 3])
                with col_del1:
                    user_id_to_delete = st.number_input("ID do Usuário para Deletar:", min_value=1, step=1, key="user_delete_id")
                with col_del2:
                    st.write("") # Espaçador
                    if st.button("Deletar Usuário", key=f"delete_user_{user_id_to_delete}", disabled=(user_id_to_delete==0)):
                        try:
                            # Adiciona um método simples de deleção ao DB (idealmente estaria em banco_de_dados.py)
                            db.conn.execute("DELETE FROM usuarios WHERE id = ?", (user_id_to_delete,))
                            db.conn.commit()
                            st.success(f"Usuário ID {user_id_to_delete} deletado.")
                            st.rerun()
                        except Exception as e_del:
                            st.error(f"Erro ao deletar usuário: {e_del}")

            except Exception as e_admin_load:
                st.error(f"Erro ao carregar usuários: {e_admin_load}")

        with admin_tab2:
            st.markdown("**Criar Novo Usuário**")
            with st.form("form_novo_usuario"):
                nome = st.text_input("Nome")
                email = st.text_input("Email")
                perfil = st.selectbox("Perfil", ["operador", "conferente", "admin"])
                senha = st.text_input("Senha", type="password")
                senha_confirma = st.text_input("Confirmar Senha", type="password")
                
                submitted = st.form_submit_button("Criar Usuário")
                
                if submitted:
                    if not nome or not email or not perfil or not senha:
                        st.warning("Todos os campos são obrigatórios.")
                    elif senha != senha_confirma:
                        st.error("As senhas não conferem.")
                    else:
                        try:
                            # Hashing simples (SHA256) para a PoC (Não use em produção!)
                            senha_hash = sha256_text(senha)
                            # (Idealmente, inserir_usuario estaria em banco_de_dados.py)
                            db.inserir_usuario(
                                nome=nome,
                                email=email,
                                perfil=perfil,
                                senha_hash=senha_hash
                            )
                            st.success(f"Usuário '{nome}' ({email}) criado com perfil '{perfil}'.")
                            db.log("criacao_usuario", "admin_ui", f"Usuário {email} criado.")
                        except Exception as e_create:
                            st.error(f"Erro ao criar usuário: {e_create}")

    # --- Logs ---
    with tabs[6]:
        st.subheader("Logs do Sistema")
        try:
            limit = st.number_input("Número de logs recentes para exibir:", min_value=10, max_value=1000, value=100, step=10, key="log_limit")
            logs_df = db.query_table("logs")
            st.dataframe(logs_df.sort_values("id", ascending=False).head(limit), use_container_width=True, height=420)
        except Exception as e: st.error(f"Erro ao carregar logs: {e}")

    # --- Memória LLM ---
    with tabs[7]:
        st.subheader("Histórico de Perguntas (Memória LLM)")
        try:
            mem = db.query_table("memoria")
            st.dataframe(mem.sort_values("id", ascending=False), use_container_width=True, height=420)
        except Exception as e: st.error(f"Erro ao carregar memória: {e}")


# --- Função Principal ---
def main():
    # Pega serviços básicos (DB, Memoria, Cofre, Validador) do cache
    # Esta função agora cacheia todos os serviços essenciais e pré-configurados
    db, memoria, cofre, validador = get_core_services()

    # --- CORREÇÃO: Mover a lógica do st.toast para fora da função cacheada ---
    if 'toast_exibido' not in st.session_state:
        st.session_state.toast_exibido = False
        
    if not st.session_state.toast_exibido:
        if not CRYPTO_OK:
            st.toast("Biblioteca 'cryptography' não encontrada. Criptografia desativada.", icon="⚠️")
            st.session_state.toast_exibido = True # Exibe só uma vez
        elif not cofre.available:
             st.toast("Chave APP_SECRET_KEY não definida. Criptografia desativada.", icon="⚠️")
             st.session_state.toast_exibido = True # Exibe só uma vez
    # -------------------------------------------------------------------------

    # O LLM é configurado dinamicamente via sidebar e armazenado no session_state
    # Cria o orchestrator com o LLM do estado da sessão (pode ser None)
    # E passa o cofre e o validador já instanciados
    orch = Orchestrator(
        db=db,
        validador=validador,
        memoria=memoria,
        llm=st.session_state.llm_instance,
        cofre=cofre # Passa o Cofre para o Orchestrator
    )

    # Renderiza UI
    ui_header()
    ui_sidebar(orch) # Passa o orchestrator atualizado para a sidebar
    ui_tabs(orch, db, cofre) # Passa orchestrator, db, e cofre para as abas


if __name__ == "__main__":
    main()