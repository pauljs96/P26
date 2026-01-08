from __future__ import annotations
import pandas as pd
import numpy as np
from dataclasses import dataclass

from src.ml.ets_model import ETSForecaster
from src.ml.backtest import mae, rmse, smape, _safe_mape  # reutilizamos métricas


@dataclass
class ETSBacktestResult:
    predictions: pd.DataFrame
    metrics: pd.DataFrame


def backtest_ets_1step(
    history: pd.DataFrame,
    y_col: str = "Demanda_Unid",
    test_months: int = 12,
    ets: ETSForecaster | None = None,
) -> ETSBacktestResult:
    """
    Backtest 1-step (t+1) para ETS (Holt-Winters).
    Igual esquema que baselines:
      - para target i, entrena con 0..i-1 y predice y(i)
    """
    if history is None or history.empty:
        empty_pred = pd.DataFrame(columns=["Mes_target", "y_true", "ETS"])
        empty_met = pd.DataFrame(columns=["Modelo", "MAE", "RMSE", "sMAPE_%", "MAPE_safe_%", "N"])
        return ETSBacktestResult(empty_pred, empty_met)

    ets = ets or ETSForecaster()

    h = history.copy()
    h["Mes"] = pd.to_datetime(h["Mes"]).dt.to_period("M").dt.to_timestamp()
    h = h.sort_values("Mes").reset_index(drop=True)
    h[y_col] = pd.to_numeric(h[y_col], errors="coerce").fillna(0.0).astype(float)

    n = len(h)
    if n < 3:
        empty_pred = pd.DataFrame(columns=["Mes_target", "y_true", "ETS"])
        empty_met = pd.DataFrame(columns=["Modelo", "MAE", "RMSE", "sMAPE_%", "MAPE_safe_%", "N"])
        return ETSBacktestResult(empty_pred, empty_met)

    start_target = max(1, n - test_months)
    rows = []

    for i in range(start_target, n):
        train = h.iloc[:i]
        target_mes = h.loc[i, "Mes"]
        y_true = float(h.loc[i, y_col])

        yhat = ets.forecast_1step(train, y_col=y_col)

        rows.append({
            "Mes_target": target_mes,
            "y_true": y_true,
            "ETS": float(yhat),
        })

    pred = pd.DataFrame(rows)

    y_t = pred["y_true"].to_numpy(dtype=float)
    y_p = pred["ETS"].to_numpy(dtype=float)

    metrics_df = pd.DataFrame([{
        "Modelo": "ETS(Holt-Winters)",
        "MAE": mae(y_t, y_p),
        "RMSE": rmse(y_t, y_p),
        "sMAPE_%": smape(y_t, y_p),
        "MAPE_safe_%": _safe_mape(y_t, y_p),
        "N": int(len(pred)),
    }])

    return ETSBacktestResult(pred, metrics_df)
