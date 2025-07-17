#%% DatasetSplitter.py
"""
DatasetSplitter module para o pacote geosteering.
Divide os dados 3D em treino, teste e hold‑out, com suporte a shuffle e filtros.
"""

import numpy as np  # Biblioteca para operações de array

def split_dataset(arr, metadata, cfg):
    """
    Gera splits para treino, teste e hold‑out a partir de arr 2D e metadados.

    Parâmetros:
      arr (np.ndarray): Array 2D de shape (n_rows, n_cols) lido via DataLoader.
      metadata (List[List]): Saída de read_metadata_file, onde metadata[0]=[nt,nf,nm], metadata[3]=nmeds.
      cfg (dict): Dicionário de configurações contendo chaves como:
        - TRAIN_SPLIT (float): fração para treino (ex: 0.8).
        - USE_SHUFFLE_SPLIT (bool)
        - NUM_HOLDOUT_MODELS (int)
        - USE_RESISTIVITY_FILTER (bool)
        - FILTER_H_RES_MIN/MAX, FILTER_V_RES_MIN/MAX (floats)

    Retorna:
      Tupla: (x_train, y_train, x_test, y_test, x_hold, y_hold, holdout_indices)
    """
    # Extrai parâmetros dos metadados
    nt, nf, nm = metadata[0]      # número de ângulos, frequências e medidas
    nmeds = metadata[3][0]        # número de timesteps por modelo

    # Seleciona colunas de features e targets conforme convenção
    x_full = arr[:, [3, 6, 7, 10, 11]]  # [z, ReHxx, ImHxx, ReHzz, ImHzz]
    y_full = arr[:, [4, 5]]             # [rho_h, rho_v]

    # Calcula número de modelos
    nmodels = int(arr.shape[0] / nmeds)

    # Reformata para 3D: (nmodels, timesteps, channels)
    x3d = x_full.reshape(nmodels, nmeds, 5)
    y3d = y_full.reshape(nmodels, nmeds, 2)

    # Shuffle se configurado
    if cfg.get("USE_SHUFFLE_SPLIT", True):
        idx = np.arange(nmodels)
        np.random.shuffle(idx)
        x3d = x3d[idx]
        y3d = y3d[idx]

    # Divide treino/teste
    split_idx = int(cfg.get("TRAIN_SPLIT", 0.8) * nmodels)
    x_train = x3d[:split_idx]
    y_train = y3d[:split_idx]
    x_test  = x3d[split_idx:]
    y_test  = y3d[split_idx:]

    # Seleção de modelos de hold‑out
    if cfg.get("USE_RESISTIVITY_FILTER", False):
        # Se houver filtro, aplique critérios médios de resistividade (não implementado aqui)
        valid_indices = list(range(x_test.shape[0]))
    else:
        valid_indices = list(range(x_test.shape[0]))

    # Sorteio de índices para holdout
    NUM_HO = cfg.get("NUM_HOLDOUT_MODELS", 1)
    holdout_indices = np.random.choice(valid_indices, size=NUM_HO, replace=False)

    # Extrai holdout e retira do conjunto de teste
    x_hold = x_test[holdout_indices]
    y_hold = y_test[holdout_indices]
    x_test = np.delete(x_test, holdout_indices, axis=0)
    y_test = np.delete(y_test, holdout_indices, axis=0)

    return x_train, y_train, x_test, y_test, x_hold, y_hold, holdout_indices
