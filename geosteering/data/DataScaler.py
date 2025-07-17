#%% DataScaler.py
"""
DataScaler module para o pacote geosteering.
Fornece funções para ajustar e aplicar escalonadores (scalers) de features.
"""

import numpy as np               # Biblioteca numérica
import joblib                    # Serialização de objetos Python
from sklearn.preprocessing import (
    StandardScaler,              # z-score padrão
    RobustScaler,                # baseada em mediana/IQR
    QuantileTransformer,         # quantis → uniforme
    PowerTransformer             # Yeo‑Johnson / Box‑Cox
)

def fit_feature_scaler(X, method, scaler_path=None, symlog_thresh=1e-3):
    """
    Ajusta um escalonador às features X.

    Parâmetros:
      X (np.ndarray): Array de forma (n_models, n_timesteps, n_features).
      method (str): Tipo de scaler: 'standard','robust','quantile','power','symlog'.
      scaler_path (str|None): Caminho para salvar o objeto via joblib.
      symlog_thresh (float): Limiar para transformação symlog (apenas se method='symlog').

    Retorna:
      object: Objeto scaler ajustado ou tupla (RobustScaler, symlog_thresh).
    """
    # Achata X de (NM, T, F) → (NM*T, F) para ajustar scaler
    flat = X.reshape(-1, X.shape[-1])

    # Seleção do scaler conforme método
    if method == "standard":
        scaler = StandardScaler().fit(flat)
    elif method == "robust":
        scaler = RobustScaler().fit(flat)
    elif method == "quantile":
        scaler = QuantileTransformer(output_distribution="uniform").fit(flat)
    elif method == "power":
        scaler = PowerTransformer(method="yeo-johnson").fit(flat)
    elif method == "symlog":
        # Ajusta RobustScaler antes da transformação simétrica log
        rs = RobustScaler().fit(flat)
        scaler = (rs, symlog_thresh)
    else:
        raise ValueError(f"Scaler desconhecido: {method}")

    # Persiste o scaler, se solicitado
    if scaler_path is not None:
        joblib.dump(scaler, scaler_path)

    return scaler


def apply_feature_scaler(X, scaler, method):
    """
    Aplica o scaler ajustado a X.

    Parâmetros:
      X (np.ndarray): Array de forma (n_models, n_timesteps, n_features).
      scaler (object): Objeto retornado por fit_feature_scaler.
      method (str): Mesmo método usado em fit_feature_scaler.

    Retorna:
      np.ndarray: X escalonado, mesma forma de entrada.
    """
    flat = X.reshape(-1, X.shape[-1])  # Achata para aplicar transform

    if method in ("standard", "robust", "quantile", "power"):
        scaled = scaler.transform(flat)
    elif method == "symlog":
        rs, thresh = scaler
        tmp = rs.transform(flat)
        # symlog: sign(x) * log10(1 + |x|/thresh)
        scaled = np.sign(tmp) * np.log10(1 + np.abs(tmp) / thresh)
    else:
        raise ValueError(f"Scaler desconhecido: {method}")

    # Retorna à forma original
    return scaled.reshape(X.shape)
