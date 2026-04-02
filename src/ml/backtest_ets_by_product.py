"""Backtest de ETS por PRODUCTO (no global).

Similar a backtest_rf_by_product pero con ETS, para comparación justa.
"""

from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
import numpy as np

from src.ml.backtest import mae, rmse, smape, _safe_mape
from src.ml.ets_model_contextual import ETSForecasterContextual


@dataclass
class ETSByProductBacktestResult:
    predictions: pd.DataFrame  # Mes_target, Producto_id, y_true, y_pred
    metrics: pd.DataFrame      # Por producto + global
    aggregated_pred: pd.DataFrame  # Mes_target, y_true_agg, y_pred_agg


def backtest_ets_by_product_1step(
    history: pd.DataFrame,
    y_col: str = "Demanda_Unid",
    test_months: int = 12,
) -> ETSByProductBacktestResult:
    """
    Backtest 1-step POR PRODUCTO con ETS.
    
    - Agrupa historia por Producto_id
    - Entrena ETS Contextual separadamente para cada producto
    - Predice demanda de cada producto
    - Agrega demanda total
    - Calcula métricas globales
    
    Args:
        history: DataFrame con columnas [Mes, Producto_id, Demanda_Unid, ...]
        y_col: Columna de demanda (default "Demanda_Unid")
        test_months: Cuántos meses finales evaluar (default 12)
    
    Returns:
        ETSByProductBacktestResult con predicciones y métricas
    """
    if history is None or history.empty:
        empty_pred = pd.DataFrame(columns=["Mes_target", "Producto_id", "y_true", "y_pred"])
        empty_met = pd.DataFrame(columns=["Producto_id", "MAE", "RMSE", "sMAPE_%", "MAPE_safe_%", "N"])
        empty_agg = pd.DataFrame(columns=["Mes_target", "y_true_agg", "y_pred_agg"])
        return ETSByProductBacktestResult(empty_pred, empty_met, empty_agg)

    h = history.copy()
    h["Mes"] = pd.to_datetime(h["Mes"]).dt.to_period("M").dt.to_timestamp() if isinstance(h["Mes"].iloc[0], str) else h["Mes"]
    h = h.sort_values(["Mes"]).reset_index(drop=True)
    h[y_col] = pd.to_numeric(h[y_col], errors="coerce").fillna(0.0).astype(float)
    
    # Validar que tenemos columna Producto_id
    if "Producto_id" not in h.columns:
        h["Producto_id"] = "ALL"

    # Determinar test range
    n_meses = h["Mes"].nunique()
    start_idx = max(0, n_meses - test_months)
    unique_meses = sorted(h["Mes"].unique())
    test_meses = unique_meses[start_idx:]

    all_predictions = []
    metrics_per_product = []

    # Por cada producto
    for producto_id in h["Producto_id"].unique():
        prod_history = h[h["Producto_id"] == producto_id].copy()
        
        if prod_history.empty or len(prod_history) < 3:
            continue

        prod_history = prod_history.sort_values("Mes").reset_index(drop=True)

        # Para este producto, hacer backtest mes a mes
        for target_mes in test_meses:
            # Historia hasta este mes (sin incluir)
            train = prod_history[prod_history["Mes"] < target_mes].copy()
            
            if train.empty or len(train) < 2:
                continue

            # Valor real en este mes
            target_data = prod_history[prod_history["Mes"] == target_mes]
            if target_data.empty:
                continue

            y_true = float(target_data[y_col].sum())
            
            # Entrenar y predecir con ETS
            ets = ETSForecasterContextual(min_obs=6)
            
            try:
                y_pred = float(ets.forecast_1step(train, y_col=y_col))
            except Exception:
                y_pred = 0.0

            all_predictions.append({
                "Mes_target": target_mes,
                "Producto_id": producto_id,
                "y_true": y_true,
                "y_pred": y_pred,
            })

    # DataFrames de predicciones
    pred_df = pd.DataFrame(all_predictions)

    if pred_df.empty:
        empty_pred = pd.DataFrame(columns=["Mes_target", "Producto_id", "y_true", "y_pred"])
        empty_met = pd.DataFrame(columns=["Producto_id", "MAE", "RMSE", "sMAPE_%", "MAPE_safe_%", "N"])
        empty_agg = pd.DataFrame(columns=["Mes_target", "y_true_agg", "y_pred_agg"])
        return ETSByProductBacktestResult(empty_pred, empty_met, empty_agg)

    # Métricas por producto
    for producto_id in pred_df["Producto_id"].unique():
        prod_pred = pred_df[pred_df["Producto_id"] == producto_id]
        
        y_t = prod_pred["y_true"].to_numpy(dtype=float)
        y_p = prod_pred["y_pred"].to_numpy(dtype=float)

        if len(y_t) > 0:
            metrics_per_product.append({
                "Producto_id": producto_id,
                "MAE": mae(y_t, y_p),
                "RMSE": rmse(y_t, y_p),
                "sMAPE_%": smape(y_t, y_p),
                "MAPE_safe_%": _safe_mape(y_t, y_p),
                "N": len(y_t),
            })

    metrics_df = pd.DataFrame(metrics_per_product)

    # Agregado: agrupar por Mes_target y sumar predicciones
    aggregated = pred_df.groupby("Mes_target").agg({
        "y_true": "sum",
        "y_pred": "sum",
    }).reset_index()
    aggregated.columns = ["Mes_target", "y_true_agg", "y_pred_agg"]

    # Agregar métrica global
    y_t_agg = aggregated["y_true_agg"].to_numpy(dtype=float)
    y_p_agg = aggregated["y_pred_agg"].to_numpy(dtype=float)

    if len(y_t_agg) > 0:
        global_metric = pd.DataFrame([{
            "Producto_id": "GLOBAL",
            "MAE": mae(y_t_agg, y_p_agg),
            "RMSE": rmse(y_t_agg, y_p_agg),
            "sMAPE_%": smape(y_t_agg, y_p_agg),
            "MAPE_safe_%": _safe_mape(y_t_agg, y_p_agg),
            "N": len(y_t_agg),
        }])
        metrics_df = pd.concat([metrics_df, global_metric], ignore_index=True)

    return ETSByProductBacktestResult(pred_df, metrics_df, aggregated)
