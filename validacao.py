# validacao.py

from __future__ import annotations
from typing import Optional, Dict, Any, TYPE_CHECKING, Set
import re
import yaml # Importado para ler o arquivo de regras
from pathlib import Path # Importado para lidar com o caminho do arquivo
import logging # Importado para logar erros

if TYPE_CHECKING:
    from banco_de_dados import BancoDeDados
    # Adiciona Cofre ao type checking para melhor análise estática
    from seguranca import Cofre

# Importação relativa funciona se os arquivos estiverem na mesma estrutura de diretório
try:
    from banco_de_dados import BancoDeDados
    from seguranca import Cofre, CRYPTO_OK, carregar_chave_do_env # Importa Cofre e status
    import pandas as pd # Importa pandas aqui para uso nas validações
except ImportError as e:
    logging.error(f"Erro ao importar módulos necessários (BancoDeDados, seguranca, pandas): {e}")
    # Placeholders se executado isoladamente ou erro de importação
    BancoDeDados = type('BancoDeDados', (object,), {
        "get_documento": lambda s, id: {},
        "query_table": lambda s, t, **kwargs: pd.DataFrame(),
        "atualizar_documento_campo": lambda s, id, k, v: None,
        "log": lambda s, *args, **kwargs: None,
    })
    CRYPTO_OK = False
    Cofre = type('Cofre', (object,), {
        "__init__": lambda s, key=None: None, "available": False,
        "encrypt_text": lambda s, t: t, "decrypt_text": lambda s, t: t
    })
    carregar_chave_do_env = lambda var_name="APP_SECRET_KEY": None
    # Mock do pandas se a importação falhar
    class DataFrameMock:
        def empty(self): return True
        def fillna(self, val): return self
        def sum(self): return 0.0
        def iterrows(self): return iter([])
        def get(self, key, default=None): return default if key not in self.__dict__ else self.__dict__[key]
    pd = type('pandas', (object,), {"DataFrame": DataFrameMock})


log = logging.getLogger("projeto_fiscal.validacao") # Logger específico do módulo
# Garante que o logger tenha um handler se executado isoladamente
if not log.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    log.addHandler(handler)
    log.setLevel(logging.INFO)


# --- Funções Auxiliares ---

def _only_digits(s: Optional[str]) -> str:
    """Remove todos os caracteres não numéricos de uma string."""
    return re.sub(r"\D+", "", s or "")


def _valida_cnpj(cnpj: Optional[str]) -> bool:
    """
    Validação de CNPJ (dígito verificador). Recebe APENAS os dígitos.
    """
    c = _only_digits(cnpj) # Garante que só temos dígitos
    if len(c) != 14 or len(set(c)) == 1: # CNPJ inválido se todos os dígitos forem iguais
        return False

    try:
        # Cálculo do primeiro dígito verificador
        soma = sum(int(c[i]) * (5 - i if i < 4 else 13 - i) for i in range(12))
        dv1_calc = 11 - (soma % 11)
        if dv1_calc >= 10: dv1_calc = 0

        # Cálculo do segundo dígito verificador
        # Usa os 12 primeiros dígitos + dv1 calculado
        soma = sum(int(digit) * weight for digit, weight in zip(c[:12] + str(dv1_calc), [6,5,4,3,2,9,8,7,6,5,4,3,2]))
        dv2_calc = 11 - (soma % 11)
        if dv2_calc >= 10: dv2_calc = 0

        # Verifica se os dígitos calculados conferem com os dígitos reais (últimos 2)
        return c[12] == str(dv1_calc) and c[13] == str(dv2_calc)
    except Exception as e:
        # Qualquer erro no cálculo (ex: conversão int) indica um formato inválido
        log.error(f"Erro interno ao calcular DV do CNPJ '{cnpj}': {e}")
        return False


# --- Carregamento das Regras Fiscais ---

REGRAS_FISCAIS_PATH = Path("regras_fiscais.yaml") # Define o caminho do arquivo

def _carregar_regras_fiscais(path: Path = REGRAS_FISCAIS_PATH) -> Dict[str, Any]:
    """Lê e faz o parse do arquivo YAML de regras fiscais."""
    if not path.exists():
        log.warning(f"Arquivo de regras fiscais '{path}' não encontrado. Usando regras padrão.")
        return {} # Retorna dicionário vazio, forçando fallbacks
    try:
        with open(path, 'r', encoding='utf-8') as f:
            regras = yaml.safe_load(f)
            log.info(f"Regras fiscais carregadas com sucesso de '{path}'.")
            # Garante que é um dicionário
            return regras if isinstance(regras, dict) else {}
    except yaml.YAMLError as e_yaml:
        log.error(f"Erro de sintaxe ao parsear '{path}': {e_yaml}. Usando regras padrão.")
        return {}
    except Exception as e:
        log.error(f"Erro inesperado ao carregar '{path}': {e}. Usando regras padrão.")
        return {} # Retorna dicionário vazio em caso de erro


# --- Classe ValidadorFiscal ---

class ValidadorFiscal:
    """
    Aplica regras de validação aos documentos fiscais, utilizando configurações
    carregadas de um arquivo YAML e descriptografando dados sensíveis se necessário.

    Regras Atuais:
    - Valida dígito verificador de CNPJ (emitente/destinatário) após descriptografia.
    - Compara soma dos itens com valor total do documento (usando tolerância do YAML).
    - Valida CFOP, CST_ICMS, CSOSN_ICMS dos itens/impostos contra listas do YAML.
    - Atualiza status para "revisao_pendente" se encontrar inconsistências.
    """

    # Aceita a instância do Cofre na inicialização
    def __init__(self, cofre: Optional[Cofre] = None, regras_path: Path = REGRAS_FISCAIS_PATH):
        """Inicializa o validador carregando as regras fiscais e configurando o Cofre."""
        self.regras = _carregar_regras_fiscais(regras_path)

        # Carrega tolerância com fallback
        self.tolerancia_valor = self.regras.get('tolerancias', {}).get('total_documento', 0.05)

        # Carrega códigos válidos como conjuntos (sets) para lookup eficiente
        self.cfops_validos: Set[str] = set(self.regras.get('cfop', {}).keys())
        self.cst_icms_validos: Set[str] = set(self.regras.get('cst_icms', {}).keys())
        self.csosn_icms_validos: Set[str] = set(self.regras.get('csosn_icms', {}).keys())

        # Configura o Cofre
        if cofre and CRYPTO_OK:
            self.cofre = cofre
            log.info(f"ValidadorFiscal usando instância Cofre fornecida (Crypto: {'on' if self.cofre.available else 'off - chave?'}).")
        else:
            self.cofre = Cofre(key=None) # Cria um cofre dummy
            log.warning("ValidadorFiscal operando SEM CRIPTOGRAFIA (Cofre não fornecido ou 'cryptography' ausente).")

        log.info(f"ValidadorFiscal inicializado. Tolerância: {self.tolerancia_valor:.2f}. "
                 f"CFOPs: {len(self.cfops_validos)}, CSTs: {len(self.cst_icms_validos)}, CSOSNs: {len(self.csosn_icms_validos)}")

    def validar_documento(self, *, doc_id: int | None = None, doc: Dict[str, Any] | None = None,
                          db: BancoDeDados, force_revalidation: bool = False, **kwargs) -> None:
        """
        Executa as validações no documento especificado, descriptografando CNPJs e validando códigos.
        """
        if doc_id is None and doc is None:
            raise ValueError("Informe doc_id ou doc para validação.")

        if doc is None:
            doc = db.get_documento(int(doc_id))
            if not doc:
                log.error(f"Documento {doc_id} não encontrado para validação.")
                return

        current_doc_id = int(doc.get("id", doc_id))
        if not current_doc_id:
             log.error(f"Não foi possível obter um ID válido para o documento durante a validação.")
             return

        status_atual = doc.get("status", "")

        if not force_revalidation and status_atual not in ("", "processando", "quarentena", "revisao_pendente"):
             log.debug(f"Validação pulada para doc_id {current_doc_id} (status: {status_atual}, force=False)")
             return

        log.info(f"Iniciando validação para doc_id {current_doc_id} (Status atual: {status_atual}). Crypto: {'on' if self.cofre.available else 'off'}")
        inconsistencias: list[str] = []
        itens_df = pd.DataFrame() # Inicializa DataFrame vazio
        impostos_df = pd.DataFrame() # Inicializa DataFrame vazio

        # --- Regra 1: Validação de CNPJ(s) com Descriptografia ---
        emit_cnpj_db = doc.get("emitente_cnpj") or ""
        dest_cnpj_db = doc.get("destinatario_cnpj") or ""
        emit_cnpj_decrypted = emit_cnpj_db
        dest_cnpj_decrypted = dest_cnpj_db

        if self.cofre.available:
            try:
                if emit_cnpj_db: emit_cnpj_decrypted = self.cofre.decrypt_text(emit_cnpj_db)
            except Exception as e_dec_emit:
                log.warning(f"Doc {current_doc_id}: Falha ao descriptografar emitente_cnpj: {e_dec_emit}")
                emit_cnpj_decrypted = emit_cnpj_db # Mantém original se falhar

            try:
                if dest_cnpj_db: dest_cnpj_decrypted = self.cofre.decrypt_text(dest_cnpj_db)
            except Exception as e_dec_dest:
                log.warning(f"Doc {current_doc_id}: Falha ao descriptografar destinatario_cnpj: {e_dec_dest}")
                dest_cnpj_decrypted = dest_cnpj_db # Mantém original se falhar

        # Valida os CNPJs (descriptografados ou não)
        if emit_cnpj_decrypted and not _valida_cnpj(emit_cnpj_decrypted):
            inconsistencias.append(f"CNPJ do emitente inválido.") # Não mostra valor cripto no log
        if dest_cnpj_decrypted and len(_only_digits(dest_cnpj_decrypted)) == 14 and not _valida_cnpj(dest_cnpj_decrypted):
            inconsistencias.append(f"CNPJ do destinatário inválido.")


        # --- Carrega Itens e Impostos para as próximas validações ---
        try:
            itens_df = db.query_table("itens", where=f"documento_id = {current_doc_id}")
            if not itens_df.empty:
                item_ids = tuple(itens_df["id"].unique().tolist())
                item_ids_sql = ', '.join(map(str, item_ids))
                impostos_df = db.query_table("impostos", where=f"item_id IN ({item_ids_sql})")
        except Exception as e_load:
            log.error(f"Erro ao carregar itens/impostos para validação do doc_id {current_doc_id}: {e_load}")
            inconsistencias.append(f"Erro interno ao carregar detalhes para validação: {e_load}")


        # --- Regra 2: Comparação de Totais ---
        if not itens_df.empty:
            try:
                soma_itens = float(itens_df["valor_total"].fillna(0).sum())
                total_doc = float(doc.get("valor_total") or 0.0)
                if total_doc > 0:
                    diferenca = abs(soma_itens - total_doc)
                    if diferenca > self.tolerancia_valor:
                        inconsistencias.append(
                            f"Inconsistência de totais: Soma Itens={soma_itens:.2f} difere de "
                            f"Total Doc={total_doc:.2f} em {diferenca:.2f} (Tol: {self.tolerancia_valor:.2f})."
                        )
            except Exception as e_tot:
                log.error(f"Erro ao validar totais para doc_id {current_doc_id}: {e_tot}")
                inconsistencias.append(f"Erro interno ao verificar totais: {e_tot}")

        # --- Regra 3: Validação de Códigos (CFOP, CST/CSOSN) ---
        if not itens_df.empty:
            # Validação de CFOP
            if self.cfops_validos: # Só valida se a lista foi carregada
                for index, item in itens_df.iterrows():
                    # Usa .get() para evitar KeyError se a coluna não existir por algum motivo
                    cfop_item = str(item.get('cfop', '')).strip() if pd.notna(item.get('cfop')) else None
                    if cfop_item and cfop_item not in self.cfops_validos:
                         inconsistencias.append(f"Item ID {item.get('id', '?')} (linha {index+1}): CFOP '{cfop_item}' inválido.")

            # Validação de CST/CSOSN (somente para ICMS por enquanto)
            if not impostos_df.empty and (self.cst_icms_validos or self.csosn_icms_validos):
                impostos_icms_df = impostos_df[impostos_df['tipo_imposto'] == 'ICMS']
                for index, imposto in impostos_icms_df.iterrows():
                    cst_imposto = str(imposto.get('cst', '')).strip() if pd.notna(imposto.get('cst')) else None
                    if cst_imposto:
                        # Verifica se é um CST válido OU um CSOSN válido
                        is_cst_valid = cst_imposto in self.cst_icms_validos
                        is_csosn_valid = cst_imposto in self.csosn_icms_validos
                        # Se nenhuma das listas de códigos válidos estiver vazia E o código não estiver em nenhuma delas
                        if not (is_cst_valid or is_csosn_valid) and (self.cst_icms_validos or self.csosn_icms_validos) :
                             # Tenta encontrar o item_id correspondente para log mais útil
                             item_id_imposto = imposto.get('item_id', '?')
                             inconsistencias.append(f"Imposto ID {imposto.get('id', '?')} (Item ID {item_id_imposto}): Código ICMS (CST/CSOSN) '{cst_imposto}' inválido.")


        # --- Atualização do Status Final ---
        novo_status = ""
        motivo_log = ""
        if inconsistencias:
            novo_status = "revisao_pendente"
            motivo_log = "; ".join(inconsistencias)
            db.atualizar_documento_campo(current_doc_id, "status", novo_status)
            db.atualizar_documento_campo(current_doc_id, "motivo_rejeicao", motivo_log[:255]) # Limita tamanho
            db.log("validacao_inconsistente", "sistema", f"doc_id={current_doc_id}|motivos={motivo_log}")
            log.warning(f"Documento {current_doc_id} marcado para revisão. Motivos: {motivo_log}")
        else:
            if status_atual in ("", "processando", "quarentena", "revisao_pendente"):
                 novo_status = "processado"
                 db.atualizar_documento_campo(current_doc_id, "status", novo_status)
                 db.atualizar_documento_campo(current_doc_id, "motivo_rejeicao", None) # Limpa rejeição anterior
                 db.log("validacao_ok", "sistema", f"doc_id={current_doc_id}|status_anterior={status_atual}|novo_status={novo_status}")
                 log.info(f"Documento {current_doc_id} validado com sucesso. Status atualizado para '{novo_status}'.")
            else:
                 novo_status = status_atual # Mantém o status
                 db.log("revalidacao_ok", "sistema", f"doc_id={current_doc_id}|status={novo_status}")
                 log.info(f"Revalidação de doc_id {current_doc_id} concluída sem novas inconsistências (Status: {novo_status}).")