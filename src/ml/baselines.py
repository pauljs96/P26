from __future__ import annotations
import pandas as pd


def _series(history: pd.DataFrame, y_col: str = "Demanda_Unid") -> pd.Series:
    """
    Convierte un DataFrame con columnas ['Mes', y_col] en una serie temporal ordenada.
    Asume frecuencia mensual y que la serie ya está completa (meses con 0 incluidos).
    """
    s = (
        history
        .sort_values("Mes")
        .set_index("Mes")[y_col]
        .astype(float)
    )
    return s


def naive_last(history: pd.DataFrame, y_col: str = "Demanda_Unid") -> float:
    """
    Baseline Naive (último valor):
    y_hat(t+1) = y(t)
    """
    s = _series(history, y_col)
    if s.empty:
        return 0.0
    return float(s.iloc[-1])


def seasonal_naive_12(history: pd.DataFrame, y_col: str = "Demanda_Unid") -> float:
    """
    Baseline Naive Estacional (t-12):
    y_hat(t+1) = y(t+1-12)
    Si no hay 12 meses, cae al naive_last.
    """
    s = _series(history, y_col)
    if len(s) < 12:
        return float(s.iloc[-1]) if len(s) else 0.0
    return float(s.iloc[-12])


def moving_average(history: pd.DataFrame, window: int = 3, y_col: str = "Demanda_Unid") -> float:
    """
    Baseline Media Móvil:
    y_hat(t+1) = promedio de los últimos `window` meses
    """
    s = _series(history, y_col)
    if s.empty:
        return 0.0
    w = max(1, int(window))
    return float(s.iloc[-w:].mean())
