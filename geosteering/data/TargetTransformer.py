#%% TargetTransformer.py
"""
TargetTransformer module para o pacote geosteering.
Define transformações e inversões de escala para os targets (resistividades).
"""

import numpy as np  # Biblioteca para operações matemáticas

def transform_targets(targets, method):
    """
    Transforma os targets brutos em escala apropriada para treino.

    Parâmetros:
      targets (np.ndarray): Array (n_models, n_timesteps, 2) com resistividades.
      method (str): 'log','log10','db','sqrt','pow13','inv','zscore','minmax','real'.

    Retorna:
      Tuple[np.ndarray, dict]: (targets_transformados, parâmetros de inversão)
    """
    ts = method.lower()
    params = {}

    if ts == "log":
        out = np.log(targets)
    elif ts == "log10":
        out = np.log10(targets)
    elif ts == "db":
        out = 20.0 * np.log10(targets)
    elif ts == "sqrt":
        out = np.sqrt(targets)
    elif ts == "pow13":
        out = np.cbrt(targets)
    elif ts == "inv":
        out = 1.0 / targets
    elif ts == "zscore":
        mean = targets.mean(axis=(0,1))
        std  = targets.std(axis=(0,1))
        out = (targets - mean) / std
        params = {"mean": mean, "std": std}
    elif ts == "minmax":
        mn = targets.min(axis=(0,1))
        mx = targets.max(axis=(0,1))
        out = (targets - mn) / (mx - mn)
        params = {"min": mn, "max": mx}
    elif ts == "real":
        out = targets.copy()
    else:
        raise ValueError(f"Transformação desconhecida: {method}")

    return out, params


def inverse_transform_targets(scaled, method, params=None):
    """
    Reverte a transformação aplicada aos targets.

    Parâmetros:
      scaled (np.ndarray): Targets em escala transformada.
      method (str): Mesmo método usado em transform_targets.
      params (dict|None): Parâmetros gerados por transform_targets (para 'zscore' e 'minmax').

    Retorna:
      np.ndarray: Targets na escala original (Ω·m).
    """
    ts = method.lower()

    if ts == "log":
        return np.exp(scaled)
    elif ts == "log10":
        return 10 ** scaled
    elif ts == "db":
        return 10 ** (scaled / 20.0)
    elif ts == "sqrt":
        return np.square(scaled)
    elif ts == "pow13":
        return np.power(scaled, 3)
    elif ts == "inv":
        return 1.0 / scaled
    elif ts == "zscore":
        return scaled * params["std"] + params["mean"]
    elif ts == "minmax":
        rng = params["max"] - params["min"]
        return scaled * rng + params["min"]
    elif ts == "real":
        return scaled
    else:
        raise ValueError(f"Inversão desconhecida: {method}")
