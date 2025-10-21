# validacao.py

from __future__ import annotations
from typing import Optional, Dict, Any
import re
import yaml # Importado para ler o arquivo de regras
from pathlib import Path # Importado para lidar com o caminho do arquivo
import logging # Importado para logar erros de carregamento

# Importação relativa funciona se os arquivos estiverem na mesma estrutura de diretório
try:
    from banco_de_dados import BancoDeDados
except ImportError:
    # Placeholder se executado isoladamente ou erro de importação
    BancoDeDados = type('BancoDeDados', (object,), {
        "get_documento": lambda s, id: {},
        "query_table": lambda s, t, **kwargs: pd.DataFrame(),
        "atualizar_documento_campo": lambda s, id, k, v: None,
        "log": lambda s, *args, **kwargs: None,
    })
    import pandas as pd # Necessário para o placeholder

log = logging.getLogger("projeto_fiscal.validacao") # Logger específico do módulo

# --- Funções Auxiliares ---

def _only_digits(s: Optional[str]) -> str:
    """Remove todos os caracteres não numéricos de uma string."""
    return re.sub(r"\D+", "", s or "")


def _valida_cnpj(cnpj: Optional[str]) -> bool:
    """
    Validação de CNPJ (dígito verificador).
    """
    c = _only_digits(cnpj)
    if len(c) != 14 or len(set(c)) == 1: # CNPJ inválido se todos os dígitos forem iguais
        return False

    try:
        # Cálculo do primeiro dígito verificador
        soma = sum(int(c[i]) * (5 - i if i < 4 else 13 - i) for i in range(12))
        dv1_calc = 11 - (soma % 11)
        if dv1_calc >= 10: dv1_calc = 0

        # Cálculo do segundo dígito verificador
        soma = sum(int(c[i]) * (6 - i if i < 5 else 14 - i) for i in range(13))
        dv2_calc = 11 - (soma % 11)
        if dv2_calc >= 10: dv2_calc = 0

        # Verifica se os dígitos calculados conferem
        return c[12] == str(dv1_calc) and c[13] == str(dv2_calc)
    except Exception:
        # Qualquer erro no cálculo indica um formato inválido
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
            return regras if isinstance(regras, dict) else {}
    except Exception as e:
        log.error(f"Erro ao carregar ou parsear '{path}': {e}. Usando regras padrão.")
        return {} # Retorna dicionário vazio em caso de erro


# --- Classe ValidadorFiscal ---

class ValidadorFiscal:
    """
    Aplica regras de validação aos documentos fiscais, utilizando configurações
    carregadas de um arquivo YAML.

    Regras Atuais:
    - Valida dígito verificador de CNPJ (emitente/destinatário).
    - Compara soma dos itens com valor total do documento (usando tolerância do YAML).
    - Atualiza status para "revisao_pendente" se encontrar inconsistências.
    """

    def __init__(self, regras_path: Path = REGRAS_FISCAIS_PATH):
        """Inicializa o validador carregando as regras fiscais."""
        self.regras = _carregar_regras_fiscais(regras_path)
        # Define um valor padrão seguro caso o carregamento falhe
        self.tolerancia_valor = self.regras.get('tolerancias', {}).get('total_documento', 0.05)
        log.info(f"ValidadorFiscal inicializado. Tolerância de valor: {self.tolerancia_valor}")

    def validar_documento(self, *, doc_id: int | None = None, doc: Dict[str, Any] | None = None,
                          db: BancoDeDados, force_revalidation: bool = False, **kwargs) -> None:
        """
        Executa as validações no documento especificado.

        Args:
            doc_id: ID do documento a ser validado (se 'doc' não for fornecido).
            doc: Dicionário contendo os dados do documento (se já carregado).
            db: Instância do BancoDeDados para consulta e atualização.
            force_revalidation: Se True, executa a validação mesmo que o status não seja 'processando'.
        """
        if doc_id is None and doc is None:
            raise ValueError("Informe doc_id ou doc para validação.")

        # Carrega o documento do banco se não foi fornecido
        if doc is None:
            doc = db.get_documento(int(doc_id))
            if not doc:
                # Log em vez de levantar exceção para não quebrar o fluxo do Orchestrator
                log.error(f"Documento {doc_id} não encontrado para validação.")
                return # Interrompe a validação se o documento não existe

        # Garante que temos o ID correto
        current_doc_id = int(doc.get("id", doc_id))
        if not current_doc_id:
             log.error(f"Não foi possível obter um ID válido para o documento durante a validação.")
             return

        status_atual = doc.get("status", "")

        # Evita revalidar documentos já finalizados, a menos que forçado
        if not force_revalidation and status_atual not in ("", "processando", "quarentena", "revisao_pendente"):
             log.debug(f"Validação pulada para doc_id {current_doc_id} (status: {status_atual}, force_revalidation=False)")
             return

        log.info(f"Iniciando validação para doc_id {current_doc_id} (Status atual: {status_atual}).")
        inconsistencias: list[str] = []

        # --- Regra 1: Validação de CNPJ(s) ---
        emit_cnpj = doc.get("emitente_cnpj") or ""
        dest_cnpj = doc.get("destinatario_cnpj") or "" # Assume que CPF não precisa validar aqui

        if emit_cnpj and not _valida_cnpj(emit_cnpj):
            inconsistencias.append(f"CNPJ do emitente inválido ({emit_cnpj}).")
        if dest_cnpj and not _valida_cnpj(dest_cnpj):
            # Valida apenas se for CNPJ (14 dígitos), ignora CPFs (11 dígitos)
            if len(_only_digits(dest_cnpj)) == 14:
                 inconsistencias.append(f"CNPJ do destinatário inválido ({dest_cnpj}).")
            # else: log.debug(f"Ignorando validação de dígito para possível CPF: {dest_cnpj}") # Opcional

        # --- Regra 2: Comparação de Totais (Soma Itens vs Total Documento) ---
        try:
            # Import local para evitar dependência global se pandas não for essencial em todo lugar
            import pandas as pd
            itens_df = db.query_table("itens", where=f"documento_id = {current_doc_id}")
            if not itens_df.empty:
                soma_itens = float(itens_df["valor_total"].fillna(0).sum())
                total_doc = float(doc.get("valor_total") or 0.0)

                # Compara apenas se o total do documento for positivo
                if total_doc > 0:
                    diferenca = abs(soma_itens - total_doc)
                    # Usa a tolerância carregada do YAML (com fallback)
                    if diferenca > self.tolerancia_valor:
                        inconsistencias.append(
                            f"Inconsistência de totais: Soma Itens={soma_itens:.2f} difere de "
                            f"Total Doc={total_doc:.2f} em {diferenca:.2f} (Tolerância: {self.tolerancia_valor:.2f})."
                        )
                # else: log.debug(f"Total do documento {current_doc_id} é zero ou nulo. Pulando comparação de totais.") # Opcional
            # else: log.debug(f"Documento {current_doc_id} não possui itens. Pulando comparação de totais.") # Opcional
        except ImportError:
             log.warning("Biblioteca Pandas não encontrada. Validação de totais pulada.")
        except Exception as e:
            # Loga o erro mas não impede outras validações
            log.error(f"Erro ao validar totais para doc_id {current_doc_id}: {e}")
            inconsistencias.append(f"Erro interno ao verificar totais: {e}")

        # --- Adicionar Mais Regras Aqui (usando self.regras) ---
        # Exemplo: Validar CFOP contra lista do YAML
        # cfops_validos = self.regras.get('cfop', {}).keys()
        # for index, item in itens_df.iterrows():
        #     if item['cfop'] not in cfops_validos:
        #         inconsistencias.append(f"Item {index+1} (ID:{item['id']}) possui CFOP inválido: {item['cfop']}")

        # --- Atualização do Status Final ---
        novo_status = ""
        motivo_log = ""
        if inconsistencias:
            novo_status = "revisao_pendente"
            motivo_log = "; ".join(inconsistencias)
            db.atualizar_documento_campo(current_doc_id, "status", novo_status)
            # Guarda o motivo detalhado no log, pode ser útil guardar no documento também
            db.atualizar_documento_campo(current_doc_id, "motivo_rejeicao", motivo_log[:255]) # Limita tamanho
            db.log("validacao_inconsistente", "sistema", f"doc_id={current_doc_id}|motivos={motivo_log}")
            log.warning(f"Documento {current_doc_id} marcado para revisão. Motivos: {motivo_log}")
        else:
            # Se passou em todas as validações E estava em um estado inicial ou pendente, marca como 'processado'.
            # Se já estava 'revisado', mantém 'revisado'.
            if status_atual in ("", "processando", "quarentena", "revisao_pendente"):
                 novo_status = "processado"
                 db.atualizar_documento_campo(current_doc_id, "status", novo_status)
                 db.atualizar_documento_campo(current_doc_id, "motivo_rejeicao", None) # Limpa rejeição anterior
                 db.log("validacao_ok", "sistema", f"doc_id={current_doc_id}|status_anterior={status_atual}|novo_status={novo_status}")
                 log.info(f"Documento {current_doc_id} validado com sucesso. Status atualizado para '{novo_status}'.")
            else:
                 # Se já estava 'processado' ou 'revisado' e passou na revalidação, apenas loga
                 novo_status = status_atual # Mantém o status
                 db.log("revalidacao_ok", "sistema", f"doc_id={current_doc_id}|status={novo_status}")
                 log.info(f"Revalidação de doc_id {current_doc_id} concluída sem novas inconsistências (Status: {novo_status}).")