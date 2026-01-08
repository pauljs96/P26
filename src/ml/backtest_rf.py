from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
import numpy as np

from src.ml.backtest import mae, rmse, smape, _safe_mape
from src.ml.rf_model import RFForecaster


@dataclass
class RFBacktestResult:
    predictions: pd.DataFrame
    metrics: pd.DataFrame


def backtest_rf_1step(
    history: pd.DataFrame,
    y_col: str = "Demanda_Unid",
    test_months: int = 12,
    rf: RFForecaster | None = None,
) -> RFBacktestResult:
    """
    Backtest 1-step (t+1) para RF:
      - para target i, entrena con 0..i-1 y predice y(i)
    """
    if history is None or history.empty:
        empty_pred = pd.DataFrame(columns=["Mes_target", "y_true", "RF"])
        empty_met = pd.DataFrame(columns=["Modelo", "MAE", "RMSE", "sMAPE_%", "MAPE_safe_%", "N"])
        return RFBacktestResult(empty_pred, empty_met)

    rf = rf or RFForecaster()

    h = history.copy()
    h["Mes"] = pd.to_datetime(h["Mes"]).dt.to_period("M").dt.to_timestamp()
    h = h.sort_values("Mes").reset_index(drop=True)
    h[y_col] = pd.to_numeric(h[y_col], errors="coerce").fillna(0.0).astype(float)

    n = len(h)
    if n < 3:
        empty_pred = pd.DataFrame(columns=["Mes_target", "y_true", "RF"])
        empty_met = pd.DataFrame(columns=["Modelo", "MAE", "RMSE", "sMAPE_%", "MAPE_safe_%", "N"])
        return RFBacktestResult(empty_pred, empty_met)

    start_target = max(1, n - int(test_months))
    rows = []

    for i in range(start_target, n):
        train = h.iloc[:i].copy()
        target_mes = h.loc[i, "Mes"]
        y_true = float(h.loc[i, y_col])

        yhat = float(rf.forecast_1step(train, y_col=y_col))

        rows.append({
            "Mes_target": target_mes,
            "y_true": y_true,
            "RF": yhat,
        })

    pred = pd.DataFrame(rows)

    y_t = pred["y_true"].to_numpy(dtype=float)
    y_p = pred["RF"].to_numpy(dtype=float)

    metrics_df = pd.DataFrame([{
        "Modelo": "RandomForest",
        "MAE": mae(y_t, y_p),
        "RMSE": rmse(y_t, y_p),
        "sMAPE_%": smape(y_t, y_p),
        "MAPE_safe_%": _safe_mape(y_t, y_p),
        "N": int(len(pred)),
    }])

    return RFBacktestResult(pred, metrics_df)
