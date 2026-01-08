from __future__ import annotations
import pandas as pd
import numpy as np
from dataclasses import dataclass

from src.ml.baselines import naive_last, seasonal_naive_12, moving_average


@dataclass
class BacktestResult:
    predictions: pd.DataFrame  # detalle por mes
    metrics: pd.DataFrame      # resumen de métricas


def _safe_mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-9) -> float:
    """
    MAPE clásico explota con ceros. Usamos versión segura:
    ignora puntos donde y_true==0 (o usa eps). Aquí usamos eps para evitar inf.
    """
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-9) -> float:
    denom = np.maximum((np.abs(y_true) + np.abs(y_pred)), eps)
    return float(np.mean(2.0 * np.abs(y_true - y_pred) / denom) * 100.0)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def backtest_baselines_1step(
    history: pd.DataFrame,
    y_col: str = "Demanda_Unid",
    test_months: int = 12,
    ma_window: int = 3,
) -> BacktestResult:
    """
    Backtest 1-step (t+1) para baselines.

    history: DataFrame con columnas ["Mes", y_col] ordenable por Mes y completo (meses con 0).
    test_months: cuántos puntos finales evaluar (por defecto 12 meses).
    Retorna:
      - predictions: filas con Mes_target (mes real), y_true, yhat por baseline
      - metrics: resumen MAE/RMSE/sMAPE/MAPE_safe por baseline
    """
    if history is None or history.empty:
        empty_pred = pd.DataFrame(columns=["Mes_target", "y_true", "Naive", "Seasonal12", f"MA{ma_window}"])
        empty_met = pd.DataFrame(columns=["Modelo", "MAE", "RMSE", "sMAPE_%", "MAPE_safe_%", "N"])
        return BacktestResult(empty_pred, empty_met)

    h = history.copy()
    h["Mes"] = pd.to_datetime(h["Mes"]).dt.to_period("M").dt.to_timestamp()
    h = h.sort_values("Mes").reset_index(drop=True)
    h[y_col] = pd.to_numeric(h[y_col], errors="coerce").fillna(0.0).astype(float)

    n = len(h)
    if n < 3:
        empty_pred = pd.DataFrame(columns=["Mes_target", "y_true", "Naive", "Seasonal12", f"MA{ma_window}"])
        empty_met = pd.DataFrame(columns=["Modelo", "MAE", "RMSE", "sMAPE_%", "MAPE_safe_%", "N"])
        return BacktestResult(empty_pred, empty_met)

    # Queremos predecir y(t) usando historia hasta t-1 (1-step).
    # Por eso empezamos en i=1. El target es i.
    # Tomamos los últimos `test_months` targets.
    start_target = max(1, n - test_months)
    rows = []

    for i in range(start_target, n):
        train = h.iloc[:i]         # hasta t-1
        target_mes = h.loc[i, "Mes"]
        y_true = float(h.loc[i, y_col])

        yhat_naive = naive_last(train, y_col=y_col)
        yhat_seas = seasonal_naive_12(train, y_col=y_col)
        yhat_ma = moving_average(train, window=ma_window, y_col=y_col)

        rows.append({
            "Mes_target": target_mes,
            "y_true": y_true,
            "Naive": yhat_naive,
            "Seasonal12": yhat_seas,
            f"MA{ma_window}": yhat_ma,
        })

    pred = pd.DataFrame(rows)

    # Métricas por modelo
    metrics_rows = []
    for col in ["Naive", "Seasonal12", f"MA{ma_window}"]:
        y_t = pred["y_true"].to_numpy(dtype=float)
        y_p = pred[col].to_numpy(dtype=float)
        metrics_rows.append({
            "Modelo": col,
            "MAE": mae(y_t, y_p),
            "RMSE": rmse(y_t, y_p),
            "sMAPE_%": smape(y_t, y_p),
            "MAPE_safe_%": _safe_mape(y_t, y_p),
            "N": int(len(pred)),
        })

    metrics_df = pd.DataFrame(metrics_rows).sort_values("MAE").reset_index(drop=True)
    return BacktestResult(pred, metrics_df)
