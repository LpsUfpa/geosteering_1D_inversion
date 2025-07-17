#%% CouplingRemover.py
"""
CouplingRemover module para o pacote geosteering.
Remove o acoplamento mútuo das componentes reais de Hxx e Hzz nos dados.
"""

import numpy as np  # Biblioteca para operações numéricas, incluindo pi

def remove_coupling(arr, L=1.0):
    """
    Remove efeitos de acoplamento mútuo das colunas:
      • arr[:, 6] (Re{Hxx})
      • arr[:,10] (Re{Hzz})

    Parâmetros:
      arr (np.ndarray): Array de dados de forma (n_rows, n_cols).
      L (float): Distância transmissor‑receptor (padrão = 1.0).

    Retorna:
      np.ndarray: Cópia de `arr` com acoplamento removido.
    """
    # Constantes de acoplamento calculadas a partir de L
    ACp = -1.0 / (4.0 * np.pi * L**3)  # Coeficiente para Hxx
    ACx =  1.0 / (2.0 * np.pi * L**3)  # Coeficiente para Hzz

    # Cria cópia para não modificar o array original
    arr_mod = arr.copy()
    # Subtrai ACp da coluna 6
    arr_mod[:, 6] = arr_mod[:, 6] - ACp
    # Subtrai ACx da coluna 10
    arr_mod[:, 10] = arr_mod[:, 10] - ACx

    return arr_mod
