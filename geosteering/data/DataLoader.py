#%% DataLoader.py
"""
DataLoader module para o pacote geosteering.
Responsável por carregar dados binários e metadados textuais, convertendo-os em arrays NumPy.
"""

import numpy as np  # Biblioteca fundamental para arrays e operações numéricas
import os           # Biblioteca para manipulação de caminhos e arquivos

def read_metadata_file(file_path):
    """
    Lê um arquivo de texto contendo metadados espaço-delimitados,
    converte cada token em int, float ou mantém como string.

    Parâmetros:
      file_path (str): Caminho para o arquivo de metadados.

    Retorna:
      List[List[Union[int, float, str]]]: Lista de linhas, onde cada linha
      é uma lista de valores convertidos.
    """
    metadata = []  # Inicializa lista que armazenará as linhas convertidas
    # Verifica se o arquivo existe antes de tentar abrir
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Arquivo de metadados não encontrado: {file_path}")
    # Abre o arquivo em modo leitura de texto
    with open(file_path, "r") as f:
        # Itera por cada linha do arquivo
        for line in f:
            # Remove espaços em branco iniciais/finais e divide por whitespace
            tokens = line.strip().split()
            converted = []  # Lista temporária para os valores convertidos da linha atual
            # Processa cada token individualmente
            for token in tokens:
                try:
                    # Tenta converter para inteiro
                    val = int(token)
                except ValueError:
                    try:
                        # Se falhar, tenta converter para float
                        val = float(token)
                    except ValueError:
                        # Se ainda falhar, mantém como string
                        val = token
                converted.append(val)
            metadata.append(converted)
    return metadata


def load_binary_data(file_path, dtype=None):
    """
    Lê um arquivo binário .dat com dtype estruturado (int32 + floats),
    converte para array NumPy bidimensional.

    Parâmetros:
      file_path (str): Caminho para o arquivo .dat.
      dtype (np.dtype, opcional): Descrição do tipo de dados para np.fromfile.
        Se None, usa dtype padrão: col1=int32, col2..col12=float64.

    Retorna:
      np.ndarray: Array de forma (n_rows, n_cols) contendo os dados brutos.
    """
    # Gera dtype padrão se não fornecido
    if dtype is None:
        # Primeiro campo inteiro 32 bits, demais 11 campos float64
        dtype = [("col1", np.int32)] + [
            (f"col{i}", np.float64) for i in range(2, 13)
        ]
    # Verifica existência do arquivo
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Arquivo binário não encontrado: {file_path}")
    # Lê o arquivo binário usando o dtype estruturado
    raw = np.fromfile(file_path, dtype=dtype)
    # Converte de array estruturado para array NumPy regular
    arr = np.array(raw.tolist())
    return arr
