"""
ML Service - Lógica de pronóstico y simulación desacoplada de Streamlit.

Esto permite reutilizar en FastAPI backend (Fase 2).
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from scipy.stats import norm

from src.ml.baselines import naive_last, seasonal_naive_12, moving_average
from src.ml.backtest import backtest_baselines_1step
from src.ml.ets_model import ETSForecaster
from src.ml.backtest_ets import backtest_ets_1step
from src.ml.rf_model import RFForecaster
from src.ml.backtest_rf import backtest_rf_1step
from src.utils import config


def compare_models(
    history: pd.DataFrame,
    y_col: str = "Demanda_Unid",
    test_months: int = 12,
    ma_window: int = 3,
) -> dict:
    """
    Compara 3 modelos (Baselines, ETS, RF) y retorna ganador + métricas.
    
    Returns:
        {
            "winner": "ETS(Holt-Winters)",
            "winner_mae": 12.5,
            "metrics": {...}
        }
    """
    bt_base = backtest_baselines_1step(
        history, y_col=y_col, test_months=test_months, ma_window=ma_window
    )
    
    ets = ETSForecaster(
        seasonal_periods=12, trend="add", seasonal="add", 
        damped_trend=False, min_obs=24
    )
    bt_ets = backtest_ets_1step(
        history, y_col=y_col, test_months=test_months, ets=ets
    )
    
    rf = RFForecaster(n_estimators=400, min_obs=24, min_samples_leaf=1, random_state=42)
    bt_rf = backtest_rf_1step(
        history, y_col=y_col, test_months=test_months, rf=rf
    )
    
    # Unir métricas
    metrics_all = pd.concat(
        [bt_base.metrics, bt_ets.metrics, bt_rf.metrics], 
        ignore_index=True
    )
    metrics_all["MAE"] = pd.to_numeric(metrics_all["MAE"], errors="coerce")
    metrics_all = metrics_all.sort_values("MAE", ascending=True).reset_index(drop=True)
    
    if metrics_all.empty:
        return {
            "success": False,
            "error": "No se pudieron calcular métricas"
        }
    
    winner_row = metrics_all.iloc[0]
    return {
        "success": True,
        "winner": str(winner_row["Modelo"]),
        "winner_mae": float(winner_row["MAE"]),
        "metrics_df": metrics_all,
        "backtest_base": bt_base,
        "backtest_ets": bt_ets,
        "backtest_rf": bt_rf
    }


def forecast_next_month(
    history: pd.DataFrame,
    winner_model: str,
    ma_window: int = 3,
) -> float:
    """Pronostica t+1 con el modelo ganador"""
    
    if history.empty:
        return 0.0
    
    if winner_model == "Naive":
        return float(max(0.0, naive_last(history)))
    
    if winner_model == "Seasonal12":
        return float(max(0.0, seasonal_naive_12(history)))
    
    if winner_model in ("MA3", "MA6"):
        window = 3 if winner_model == "MA3" else 6
        return float(max(0.0, moving_average(history, window=window)))
    
    if winner_model == "ETS(Holt-Winters)":
        ets = ETSForecaster()
        return float(max(0.0, ets.forecast_1step(history)))
    
    if winner_model == "RandomForest":
        rf = RFForecaster()
        return float(max(0.0, rf.forecast_1step(history)))
    
    return float(max(0.0, naive_last(history)))


def calculate_production_quantity(
    forecast: float,
    stock_security: float,
    stock_actual: float,
) -> float:
    """
    Calcula cantidad de producción recomendada.
    
    Q = max(0, Forecast + SS - Stock_actual)
    """
    return float(max(0.0, forecast + stock_security - stock_actual))


def z_from_service_level(service_level: float) -> float:
    """Convierte SL (0-1) a Z (normal estándar)"""
    mapping = {
        0.85: 1.04,
        0.90: 1.28,
        0.95: 1.65,
        0.975: 1.96,
        0.99: 2.33
    }
    closest = min(mapping.keys(), key=lambda k: abs(k - service_level))
    return mapping[closest]


def z_from_cost_ratio(cost_stockout: float, cost_inv: float) -> float:
    """
    Convierte costos a Z-score usando método Newsvendor.
    
    Calcula Z = F^-1(cost_stockout / (cost_stockout + cost_inv))
    donde F^-1 es la inversa acumulada de la normal estándar.
    
    Args:
        cost_stockout: Costo unitario de quiebre
        cost_inv: Costo unitario de inventario
    
    Returns:
        float: Z-score óptimo según criterio costo-beneficio
    
    Example:
        Z = z_from_cost_ratio(10, 1)  # Si quiebre es 10x más caro
        # Retorna Z ≈ 1.34 (más conservador que Z=1.65)
    """
    # Evitar división por cero
    total_cost = float(cost_stockout + cost_inv)
    if total_cost <= 0:
        # Fallback a service level conservador (95%)
        return 1.65
    
    # Ratio de costo óptimo (Newsvendor)
    critical_ratio = float(cost_stockout) / total_cost
    
    # Asegurar que está en [0, 1] para ppf
    critical_ratio = np.clip(critical_ratio, 0.001, 0.999)
    
    # Inversa normal: Z tal que P(X <= Z) = critical_ratio
    z_score = float(norm.ppf(critical_ratio))
    
    return z_score


def calculate_safety_stock_newsvendor(
    cost_stockout: float,
    cost_inv: float,
    sigma: float,
    lead_time: int = 1
) -> float:
    """
    Calcula stock de seguridad óptimo usando Newsvendor Problem.
    
    SS = Z * sigma * sqrt(lead_time)
    donde Z se calcula dinámica según balance de costos.
    
    Args:
        cost_stockout: Costo unitario de quiebre
        cost_inv: Costo unitario de inventario
        sigma: Desviación estándar (MAE del pronóstico)
        lead_time: Lead time en períodos
    
    Returns:
        float: Stock de seguridad óptimo
    
    Example:
        SS = calculate_safety_stock_newsvendor(10, 1, 50, 1)
        # Si quiebre cuesta 10x más → más stock protector
    """
    z_dynamic = z_from_cost_ratio(cost_stockout, cost_inv)
    ss = float(z_dynamic * sigma * np.sqrt(float(lead_time)))
    return float(max(0.0, ss))


def service_level_by_abc(abc_class: str) -> float:
    """Retorna SL según ABC"""
    abc = (abc_class or "C").strip().upper()
    if abc == "A":
        return 0.95
    if abc == "B":
        return 0.90
    return 0.85


def build_abc_classification(demand_monthly: pd.DataFrame) -> pd.DataFrame:
    """
    Clasifica productos en ABC según demanda total.
    
    A: 80%, B: 95%, C: resto
    """
    d = demand_monthly.copy()
    d["Codigo"] = d["Codigo"].astype(str).str.strip()
    d["Demanda_Unid"] = pd.to_numeric(d["Demanda_Unid"], errors="coerce").fillna(0.0)
    
    tot = (
        d.groupby("Codigo", as_index=False)["Demanda_Unid"]
        .sum()
        .rename(columns={"Demanda_Unid": "Demanda_Total"})
    )
    
    tot = tot.sort_values("Demanda_Total", ascending=False).reset_index(drop=True)
    grand = float(tot["Demanda_Total"].sum()) if len(tot) else 0.0
    
    if grand <= 0:
        tot["Share"] = 0.0
        tot["CumShare"] = 0.0
        tot["ABC"] = "C"
        return tot
    
    tot["Share"] = tot["Demanda_Total"] / grand
    tot["CumShare"] = tot["Share"].cumsum()
    
    def _abc(cum):
        if cum <= 0.80:
            return "A"
        if cum <= 0.95:
            return "B"
        return "C"
    
    tot["ABC"] = tot["CumShare"].apply(_abc)
    return tot
