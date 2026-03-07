"""Dashboard Streamlit (MVP) - Proyecto de tesis.

Objetivo del dashboard en esta etapa:
- Cargar múltiples CSV (2021-2025)
- Ejecututar el pipeline de datos (carga -> limpieza -> reconciliación -> agregaciones)
- Para un producto seleccionado, mostrar por mes:
    1) Venta Tienda Sin Doc (Salida_unid)
    2) Salida por Consumo (Salida_unid)
    3) Guía de remisión - R (solo la parte "externa" neta calculada por reconciliación)
  y luego graficar la demanda total.

Esto ayuda a validar que la construcción de demanda mensual sea coherente.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
import plotly.express as px
import os
from dotenv import load_dotenv
import numpy as np
from scipy.stats import norm

# Cargar variables de entorno
load_dotenv()

from src.data.pipeline import DataPipeline
from src.utils import config
from src.ml.baselines import naive_last, seasonal_naive_12, moving_average
from src.ml.backtest import backtest_baselines_1step
from src.ml.ets_model import ETSForecaster
from src.ml.backtest_ets import backtest_ets_1step
from src.ml.rf_model import RFForecaster
from src.ml.backtest_rf import backtest_rf_1step
from src.db import get_db
from src.storage import get_storage_manager


# ==================== FUNCIONES DE PRESENTACIÓN VISUAL ====================

def display_prominent_chart(fig, title: str = "", description: str = ""):
    """Muestra una gráfica de forma destacada con título y descripción profesional."""
    if title:
        st.markdown(f"<h3 style='color: #1976D2; font-weight: 600; margin-top: 1em; margin-bottom: 0.5em;'>{title}</h3>", unsafe_allow_html=True)
    if description:
        st.markdown(f"<p style='color: #666; font-size: 0.95em; margin-bottom: 1em;'>{description}</p>", unsafe_allow_html=True)
    
    st.plotly_chart(fig, use_container_width=True, config={"responsive": True, "displayModeBar": True})


def display_metrics_row(metrics: list[dict], cols: int = 4):
    """
    Muestra KPIs en fila de forma elegante.
    
    Formato:
    metrics = [
        {"label": "Total", "value": 1000, "unit": "unid", "icon": "📦"},
        {"label": "Fill Rate", "value": 95.5, "unit": "%", "icon": "✅"},
    ]
    """
    cols_layout = st.columns(cols)
    for idx, metric in enumerate(metrics):
        if idx >= cols:
            break
        with cols_layout[idx % cols]:
            icon = metric.get("icon", "📊")
            label = metric.get("label", "")
            value = metric.get("value", 0)
            unit = metric.get("unit", "")
            
            st.metric(label=f"{icon} {label}", value=f"{value:,.1f} {unit}".strip())


def section_divider():
    """Crea un divisor visual profesional."""
    st.markdown("""
    <hr style='border: none; border-top: 2px solid #E0E0E0; margin: 2em 0;'>
    """, unsafe_allow_html=True)


def highlight_box(text: str, box_type: str = "info", icon: str = "ℹ️"):
    """Muestra un cuadro destacado de información."""
    if box_type == "success":
        bg_color = "#E8F5E9"
        border_color = "#4CAF50"
        default_icon = "✅"
    elif box_type == "warning":
        bg_color = "#FFF3E0"
        border_color = "#FF9800"
        default_icon = "⚠️"
    elif box_type == "danger":
        bg_color = "#FFEBEE"
        border_color = "#F44336"
        default_icon = "❌"
    else:  # info
        bg_color = "#E3F2FD"
        border_color = "#1976D2"
        default_icon = "ℹ️"
    
    icon = icon or default_icon
    st.markdown(f"""
    <div style='
        background-color: {bg_color};
        border-left: 5px solid {border_color};
        padding: 1em 1.2em;
        border-radius: 6px;
        margin: 1em 0;
    '>
        <span style='font-size: 1.2em; margin-right: 0.5em;'>{icon}</span>
        <span style='color: #333; font-size: 0.95em;'>{text}</span>
    </div>
    """, unsafe_allow_html=True)


#1. FUNCIONES AUXILIARES (Modular)

#A. Normalizacion y Construcción

def _normalize_text(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip()


def build_monthly_components(movements: pd.DataFrame, codigo: str) -> pd.DataFrame:
    """Construye la tabla mensual de componentes de demanda para un producto.

    Componentes:
    - Venta Tienda Sin Doc: sum(Salida_unid)
    - Salida por Consumo:   sum(Salida_unid)
    - Guía externa (YA reconciliada): sum(Guia_Salida_Externa_Unid) solo cuando Tipo_Guia == VENTA_EXTERNA
    """
    if movements is None or movements.empty:
        return pd.DataFrame(columns=["Mes", "Venta_Tienda", "Consumo", "Guia_Externa", "Demanda_Total"])

    df = movements[movements["Codigo"] == str(codigo)].copy()
    if df.empty:
        return pd.DataFrame(columns=["Mes", "Venta_Tienda", "Consumo", "Guia_Externa", "Demanda_Total"])

    # Normalizar texto (evita problemas por espacios)
    df["Documento"] = _normalize_text(df["Documento"])
    df["Numero"] = _normalize_text(df["Numero"])

    df["Mes"] = df["Fecha"].dt.to_period("M").dt.to_timestamp()

    # 1) Ventas
    venta = (
        df[df["Documento"] == config.DOC_VENTA_TIENDA]
        .groupby("Mes", as_index=False)["Salida_unid"]
        .sum()
        .rename(columns={"Salida_unid": "Venta_Tienda"})
    )

    # 2) Consumo
    consumo = (
        df[df["Documento"] == config.DOC_SALIDA_CONSUMO]
        .groupby("Mes", as_index=False)["Salida_unid"]
        .sum()
        .rename(columns={"Salida_unid": "Consumo"})
    )

    # 3) Guía externa (YA reconciliada por guide_reconciliation.py)
    if "Tipo_Guia" not in df.columns or "Guia_Salida_Externa_Unid" not in df.columns:
        guia_m = pd.DataFrame(columns=["Mes", "Guia_Externa"])
    else:
        guia_ext = df[df["Tipo_Guia"] == "VENTA_EXTERNA"].copy()
        if guia_ext.empty:
            guia_m = pd.DataFrame(columns=["Mes", "Guia_Externa"])
        else:
            guia_m = (
                guia_ext.groupby("Mes", as_index=False)["Guia_Salida_Externa_Unid"]
                .sum()
                .rename(columns={"Guia_Salida_Externa_Unid": "Guia_Externa"})
            )

    # Debug (solo dentro del tab de demanda, se controla desde el caller)
    # Unir y completar meses faltantes (base completa)
 
    min_mes = df["Mes"].min()
    max_mes = df["Mes"].max()
    months = pd.DataFrame({"Mes": pd.date_range(min_mes, max_mes, freq="MS")})

    out = (
        months.merge(venta, on="Mes", how="left")
        .merge(consumo, on="Mes", how="left")
        .merge(guia_m, on="Mes", how="left")
    )

    out["Venta_Tienda"] = out["Venta_Tienda"].fillna(0.0)
    out["Consumo"] = out["Consumo"].fillna(0.0)
    out["Guia_Externa"] = out["Guia_Externa"].fillna(0.0)

    out["Demanda_Total"] = out["Venta_Tienda"] + out["Consumo"] + out["Guia_Externa"]

    return out.sort_values("Mes").reset_index(drop=True)


def build_abc_from_demand(demand_monthly: pd.DataFrame) -> pd.DataFrame:
    """
    ABC por demanda total (unidades) en todo el horizonte.
    Regla típica:
      A: acumulado <= 80%
      B: acumulado <= 95%
      C: resto
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


#B. Comparación de Modelos

def compare_models_metrics(*metrics_dfs: pd.DataFrame, sort_by: str = "MAE") -> pd.DataFrame:
    """
    Une métricas de N modelos en una sola tabla y ordena por sort_by (default MAE).
    Cada metrics_df debe tener columnas: Modelo, MAE, RMSE, sMAPE_%, MAPE_safe_%, N
    """
    allm = pd.concat([d for d in metrics_dfs if d is not None and not d.empty], ignore_index=True)

    if allm.empty:
        return allm

    if sort_by not in allm.columns:
        sort_by = "MAE"

    # Asegurar numéricos
    for col in ["MAE", "RMSE", "sMAPE_%", "MAPE_safe_%"]:
        if col in allm.columns:
            allm[col] = pd.to_numeric(allm[col], errors="coerce")

    allm = allm.sort_values(sort_by, ascending=True).reset_index(drop=True)
    allm.insert(0, "Rank", range(1, len(allm) + 1))
    return allm



def select_winner_and_backtests_for_product(
    hist: pd.DataFrame,
    test_months: int,
    ma_window: int,
    ets_params: dict,
    rf_params: dict,
    sort_metric: str = "MAE",
):
    """
    Corre backtests (Baselines + ETS + RF) para un producto y retorna:
    - tabla comparativa de métricas
    - nombre del ganador
    - predicciones del ganador (para diagnóstico)
    - MAE del ganador (para SS)
    """
    bt_base = backtest_baselines_1step(hist, y_col="Demanda_Unid", test_months=int(test_months), ma_window=int(ma_window))

    ets = ETSForecaster(**ets_params)
    bt_ets = backtest_ets_1step(hist, y_col="Demanda_Unid", test_months=int(test_months), ets=ets)

    rf = RFForecaster(**rf_params)
    bt_rf = backtest_rf_1step(hist, y_col="Demanda_Unid", test_months=int(test_months), rf=rf)

    cmp = compare_models_metrics(bt_base.metrics, bt_ets.metrics, bt_rf.metrics, sort_by=sort_metric)
    if cmp.empty:
        return cmp, "N/A", pd.DataFrame(), np.nan

    winner = str(cmp.iloc[0]["Modelo"])
    winner_mae = float(cmp.iloc[0].get("MAE", np.nan))

    # predicciones del ganador (últimos meses backtest)
    if winner == "ETS(Holt-Winters)":
        pred_best = bt_ets.predictions[["Mes_target", "y_true", "ETS"]].rename(columns={"ETS": "y_pred"})
    elif winner == "RandomForest":
        pred_best = bt_rf.predictions[["Mes_target", "y_true", "RF"]].rename(columns={"RF": "y_pred"})
    else:
        # Baselines: columna con el mismo nombre del modelo
        pred_best = bt_base.predictions[["Mes_target", "y_true", winner]].rename(columns={winner: "y_pred"})

    return cmp, winner, pred_best, winner_mae




#C. Política de Inventario

def z_from_service_level(service_level: float) -> float:
    """Convierte nivel de servicio (0-1) a Z aproximado (normal estándar)."""
    # Valores típicos (suficiente para tesis y operación)
    mapping = {
        0.85: 1.04,
        0.90: 1.28,
        0.95: 1.65,
        0.975: 1.96,
        0.99: 2.33
    }
    # si no está exacto, aproximamos al más cercano
    closest = min(mapping.keys(), key=lambda k: abs(k - service_level))
    return mapping[closest]


def policy_service_level_by_abc(abc: str) -> float:
    """Define nivel de servicio por ABC (política)."""
    abc = (abc or "C").strip().upper()
    if abc == "A":
        return 0.95
    if abc == "B":
        return 0.90
    return 0.85



#D. Pronóstico y Stock de Seguridad

def forecast_next_month_with_winner(hist: pd.DataFrame, winner: str, ma_window: int, ets_params: dict, rf_params: dict) -> float:
    """Pronostica t+1 usando el modelo ganador."""
    if hist.empty:
        return 0.0

    # Baselines
    if winner == "Naive":
        yhat = naive_last(hist)
        return float(max(0.0, yhat))
    if winner == "Seasonal12":
        yhat = seasonal_naive_12(hist)
        return float(max(0.0, yhat))
    if winner in ("MA3", "MA6"):
        window = 3 if winner == "MA3" else 6
        yhat = moving_average(hist, window=window)
        return float(max(0.0, yhat))

    # ETS
    if winner == "ETS(Holt-Winters)":
        ets = ETSForecaster(**ets_params)
        # usamos el forecaster para 1-step (entrenando en todo el histórico)
        yhat = ets.forecast_1step(hist, y_col="Demanda_Unid")
        return float(max(0.0, yhat))

    # RF
    if winner == "RandomForest":
        rf = RFForecaster(**rf_params)
        yhat = rf.forecast_1step(hist, y_col="Demanda_Unid")
        return float(max(0.0, yhat))

    # fallback
    return float(max(0.0, naive_last(hist)))

#--------------------------------------

#2. SIMULACIÓN DE POLÍTICAS (Avanzado)

#A. Single Backtest (1 producto)

def simulate_policy_backtest_1step(
    hist: pd.DataFrame,
    stock_series: pd.DataFrame | None,
    winner: str,
    abc_class: str,
    lead_time: int = 1,
    eval_months: int = 12,
    ets_params: dict | None = None,
    rf_params: dict | None = None,
    sigma_fixed: float | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Simula política mensual (t -> t+1) con lead_time=1 por defecto.

    Supuestos (mensual):
    - En el mes t decides producción Q_t que llega al inicio de t+1 (si LT=1).
    - Se consume la demanda real D_{t+1}.
    - Stock se actualiza a fin de mes t+1.

    Parámetros:
    - hist: DataFrame con columnas ["Mes","Demanda_Unid"] ordenado mensual y completo (con ceros).
    - stock_series: opcional, DataFrame con stock mensual real ["Mes","Stock_Unid"] para inicializar Stock_0.
      Si None o vacío, inicializa Stock_0 = 0.
    - winner: modelo a usar para forecast t+1 (Naive / Seasonal12 / MA3 / MA6 / ETS(Holt-Winters) / RandomForest)
    - abc_class: A/B/C para determinar Z (por política)
    - eval_months: meses a simular al final del histórico

    Retorna:
    - df_sim con detalle por mes evaluado
    - kpis dict
    """
    ets_params = ets_params or dict(seasonal_periods=12, trend="add", seasonal="add", damped_trend=False, min_obs=24)
    rf_params = rf_params or dict(n_estimators=400, min_obs=24, min_samples_leaf=1, random_state=42)

    h = hist.copy()
    h["Mes"] = pd.to_datetime(h["Mes"]).dt.to_period("M").dt.to_timestamp()
    h["Demanda_Unid"] = pd.to_numeric(h["Demanda_Unid"], errors="coerce").fillna(0.0).astype(float)
    h = h.sort_values("Mes").reset_index(drop=True)

    if len(h) < (eval_months + 6):
        # evita evaluar con muy pocos datos
        eval_months = min(eval_months, max(1, len(h) - 2))

    # Ventana de simulación: últimos eval_months (prediciendo t+1)
    # Usamos índices para t en [start_idx .. end_idx-1], donde el target es t+1
    end_idx = len(h) - 1
    start_idx = max(0, end_idx - eval_months)

    # Stock inicial: stock real del mes anterior al primer target, si existe
    stock0 = 0.0
    if stock_series is not None and not stock_series.empty:
        s = stock_series.copy()
        s["Mes"] = pd.to_datetime(s["Mes"]).dt.to_period("M").dt.to_timestamp()
        s = s.sort_values("Mes")
        # Tomamos stock del mes start_idx (mes t) como stock disponible al cierre de ese mes
        # y asumimos que al inicio de t+1 ese stock está disponible.
        mes_t = h.loc[start_idx, "Mes"]
        srow = s[s["Mes"] == mes_t]
        if not srow.empty:
            stock0 = float(srow.iloc[-1]["Stock_Unid"])

    service_level = policy_service_level_by_abc(abc_class)
    z = z_from_service_level(service_level)

    # Modelo instanciado (ETS/RF se re-entrena cada paso con data hasta t)
    ets = ETSForecaster(**ets_params)
    rf = RFForecaster(**rf_params)

    rows = []
    stock_t = float(max(0.0, stock0))

    for t in range(start_idx, end_idx):
        # entrenamiento: hasta mes t (incluido)
        train = h.iloc[: t + 1].copy()

        # target real es el mes t+1
        mes_target = h.loc[t + 1, "Mes"]
        d_true = float(h.loc[t + 1, "Demanda_Unid"])

        # Forecast t+1
        if winner == "ETS(Holt-Winters)":
            yhat = float(max(0.0, ets.forecast_1step(train, y_col="Demanda_Unid")))
        elif winner == "RandomForest":
            yhat = float(max(0.0, rf.forecast_1step(train, y_col="Demanda_Unid")))
        elif winner == "Seasonal12":
            yhat = float(max(0.0, seasonal_naive_12(train)))
        elif winner == "MA6":
            yhat = float(max(0.0, moving_average(train, window=6)))
        elif winner == "MA3":
            yhat = float(max(0.0, moving_average(train, window=3)))
        else:  # "Naive"
            yhat = float(max(0.0, naive_last(train)))

        # σ proxy: usamos MAE del ganador estimado en training window reciente (simple)
        # Para no recalcular un backtest completo cada mes, usamos MAE fijo como "promedio absoluto" de residuo naive:
        # Mejor: usa MAE del ganador que ya calculaste en comparador por producto (si lo tienes en session_state).
        # Aquí: proxy conservador = mean(|Δ|) de últimos 6 meses
        if sigma_fixed is not None and np.isfinite(sigma_fixed):
            sigma = float(max(0.0, sigma_fixed))
        else:
            recent = train["Demanda_Unid"].tail(6).values
            if len(recent) >= 2:
                sigma = float(np.mean(np.abs(np.diff(recent))))
            else:
                sigma = 0.0


        ss = float(z * sigma * np.sqrt(float(lead_time)))

        # Producción recomendada (llega a t+1 si LT=1; para LT>1 simplificamos igual en esta versión)
        q_t = float(max(0.0, yhat + ss - stock_t))

        # Stock disponible para atender demanda en t+1
        stock_available = stock_t + q_t  # (LT=1)
        served = min(stock_available, d_true)
        lost = max(0.0, d_true - stock_available)
        stock_next = max(0.0, stock_available - d_true)

        rows.append({
            "Mes_target": mes_target,
            "Stock_inicio": stock_t,
            "Forecast": yhat,
            "Sigma_proxy": sigma,
            "SS": ss,
            "Produccion_Q": q_t,
            "Demanda_real": d_true,
            "Servido": served,
            "Faltante": lost,
            "Stock_fin": stock_next,
            "Quiebre": bool(lost > 0)
        })

        stock_t = stock_next

    df_sim = pd.DataFrame(rows)

    total_d = float(df_sim["Demanda_real"].sum()) if not df_sim.empty else 0.0
    total_served = float(df_sim["Servido"].sum()) if not df_sim.empty else 0.0
    total_lost = float(df_sim["Faltante"].sum()) if not df_sim.empty else 0.0

    kpis = {
        "Meses_evaluados": int(len(df_sim)),
        "Meses_con_quiebre": int(df_sim["Quiebre"].sum()) if not df_sim.empty else 0,
        "FillRate_%": (100.0 * total_served / total_d) if total_d > 0 else 0.0,
        "Unidades_faltantes": total_lost,
        "Inventario_promedio": float(df_sim["Stock_fin"].mean()) if not df_sim.empty else 0.0,
        "Produccion_total": float(df_sim["Produccion_Q"].sum()) if not df_sim.empty else 0.0,
    }
    return df_sim, kpis

#B. Comparativa Sin Sistema vs Con Sistema

def simulate_compare_policy_vs_baseline(
    hist: pd.DataFrame,
    stock_series: pd.DataFrame | None,
    abc_class: str,
    winner: str,
    eval_months: int = 12,
    lead_time: int = 1,
    # costos relativos (puedes ajustar en el dashboard)
    cost_stock_unit: float = 1.0,
    cost_stockout_unit: float = 5.0,
    # params modelos
    ets_params: dict | None = None,
    rf_params: dict | None = None,
    ma_window: int = 3,
    test_months_for_mae: int = 12,
) -> tuple[pd.DataFrame, dict]:
    """
    Compara:
      A) Sin sistema: Q_t = D_{t-1} (demanda real anterior)
      B) Con sistema: Q_t = max(0, forecast_{t} + SS_t - stock_t)
         donde SS_t = Z(ABC) * sigma, y sigma = MAE del ganador (estimado por backtest)

    Nota:
    - lead_time: versión base implementada con LT=1 (producción llega al inicio del mes target).
    """
    ets_params = ets_params or dict(seasonal_periods=12, trend="add", seasonal="add", damped_trend=False, min_obs=24)
    rf_params = rf_params or dict(n_estimators=400, min_obs=24, min_samples_leaf=1, random_state=42)

    h = hist.copy()
    h["Mes"] = pd.to_datetime(h["Mes"]).dt.to_period("M").dt.to_timestamp()
    h["Demanda_Unid"] = pd.to_numeric(h["Demanda_Unid"], errors="coerce").fillna(0.0).astype(float)
    h = h.sort_values("Mes").reset_index(drop=True)

    if len(h) < (eval_months + 6):
        eval_months = min(eval_months, max(1, len(h) - 2))

    end_idx = len(h) - 1
    start_idx = max(1, end_idx - eval_months)  # start en 1 para poder usar D_{t-1}

    # Stock inicial (mes start_idx-1)
    stock0 = 0.0
    if stock_series is not None and not stock_series.empty:
        s = stock_series.copy()
        s["Mes"] = pd.to_datetime(s["Mes"]).dt.to_period("M").dt.to_timestamp()
        s = s.sort_values("Mes")
        mes0 = h.loc[start_idx - 1, "Mes"]
        srow = s[s["Mes"] == mes0]
        if not srow.empty:
            stock0 = float(srow.iloc[-1]["Stock_Unid"])

    # Z por ABC
    service_level = policy_service_level_by_abc(abc_class)
    z = z_from_service_level(service_level)

    # sigma = MAE del ganador (estimado una vez en ventana final para eficiencia)
    # backtest en el tramo final de la serie (fair)
    bt_mae_months = min(test_months_for_mae, max(6, len(h) - 2))
    train_full = h.copy()

    mae_winner = np.nan
    # baselines + ETS + RF: calculamos MAE por ganador elegido
    bt_base = backtest_baselines_1step(train_full, y_col="Demanda_Unid", test_months=int(bt_mae_months), ma_window=int(ma_window))

    ets = ETSForecaster(**ets_params)
    bt_ets = backtest_ets_1step(train_full, y_col="Demanda_Unid", test_months=int(bt_mae_months), ets=ets)

    rf = RFForecaster(**rf_params)
    bt_rf = backtest_rf_1step(train_full, y_col="Demanda_Unid", test_months=int(bt_mae_months), rf=rf)

    # Unificamos métricas para extraer MAE del winner
    metrics_all = pd.concat([bt_base.metrics, bt_ets.metrics, bt_rf.metrics], ignore_index=True)
    metrics_all["MAE"] = pd.to_numeric(metrics_all["MAE"], errors="coerce")
    roww = metrics_all[metrics_all["Modelo"] == winner]
    if not roww.empty:
        mae_winner = float(roww.iloc[0]["MAE"])
    else:
        # fallback conservador
        mae_winner = float(metrics_all["MAE"].min()) if metrics_all["MAE"].notna().any() else 0.0

    sigma = float(max(0.0, mae_winner))
    
    # ==================== NEWSVENDOR: Stock de seguridad dinámico por costos ====================
    from src.services.ml_service import calculate_safety_stock_newsvendor
    ss_const = calculate_safety_stock_newsvendor(
        cost_stockout=cost_stockout_unit,
        cost_inv=cost_stock_unit,
        sigma=sigma,
        lead_time=lead_time
    )
    # ss_const ahora depende de la relación cost_stockout/cost_inv ✅

    # Modelos para forecast (re-entrena cada paso)
    ets_step = ETSForecaster(**ets_params)
    rf_step = RFForecaster(**rf_params)

    def forecast_1step(train: pd.DataFrame) -> float:
        if winner == "ETS(Holt-Winters)":
            return float(max(0.0, ets_step.forecast_1step(train, y_col="Demanda_Unid")))
        if winner == "RandomForest":
            return float(max(0.0, rf_step.forecast_1step(train, y_col="Demanda_Unid")))
        if winner == "Seasonal12":
            return float(max(0.0, seasonal_naive_12(train)))
        if winner == "MA6":
            return float(max(0.0, moving_average(train, window=6)))
        if winner == "MA3":
            return float(max(0.0, moving_average(train, window=3)))
        return float(max(0.0, naive_last(train)))

    # Simulación paralela
    stock_base = float(max(0.0, stock0))
    stock_sys = float(max(0.0, stock0))

    rows = []
    for t in range(start_idx, end_idx + 1):
        mes = h.loc[t, "Mes"]
        d_t = float(h.loc[t, "Demanda_Unid"])

        # --------- A) SIN SISTEMA (Q_t = D_{t-1}) ----------
        d_prev = float(h.loc[t - 1, "Demanda_Unid"])
        q_base = float(max(0.0, d_prev))

        stock_disp_base = stock_base + q_base
        served_base = min(stock_disp_base, d_t)
        lost_base = max(0.0, d_t - stock_disp_base)
        stock_end_base = max(0.0, stock_disp_base - d_t)

        cost_stock_base = stock_end_base * cost_stock_unit
        cost_lost_base = lost_base * cost_stockout_unit
        cost_total_base = cost_stock_base + cost_lost_base

        # --------- B) CON SISTEMA ----------
        train = h.iloc[:t].copy()  # hasta t-1 para pronosticar t (1-step)
        yhat = forecast_1step(train)

        q_sys = float(max(0.0, yhat + ss_const - stock_sys))

        stock_disp_sys = stock_sys + q_sys
        served_sys = min(stock_disp_sys, d_t)
        lost_sys = max(0.0, d_t - stock_disp_sys)
        stock_end_sys = max(0.0, stock_disp_sys - d_t)

        cost_stock_sys = stock_end_sys * cost_stock_unit
        cost_lost_sys = lost_sys * cost_stockout_unit
        cost_total_sys = cost_stock_sys + cost_lost_sys

        rows.append({
            "Mes": mes,
            "Demanda_real": d_t,

            "Base_Stock_ini": stock_base,
            "Base_Q": q_base,
            "Base_Stock_fin": stock_end_base,
            "Base_Faltante": lost_base,
            "Base_Costo_inv": cost_stock_base,
            "Base_Costo_quiebre": cost_lost_base,
            "Base_Costo_total": cost_total_base,

            "Sys_Stock_ini": stock_sys,
            "Sys_Forecast": yhat,
            "Sys_SS": ss_const,
            "Sys_Q": q_sys,
            "Sys_Stock_fin": stock_end_sys,
            "Sys_Faltante": lost_sys,
            "Sys_Costo_inv": cost_stock_sys,
            "Sys_Costo_quiebre": cost_lost_sys,
            "Sys_Costo_total": cost_total_sys,
        })

        stock_base = stock_end_base
        stock_sys = stock_end_sys

    df = pd.DataFrame(rows)

    def kpis(prefix: str) -> dict:
        total_d = float(df["Demanda_real"].sum())
        lost = float(df[f"{prefix}_Faltante"].sum())
        served = total_d - lost
        fill = (100.0 * served / total_d) if total_d > 0 else 0.0
        months_break = int((df[f"{prefix}_Faltante"] > 0).sum())
        return {
            "Meses_con_quiebre": months_break,
            "FillRate_%": fill,
            "Unidades_faltantes": lost,
            "Costo_total": float(df[f"{prefix}_Costo_total"].sum()),
            "Costo_quiebre": float(df[f"{prefix}_Costo_quiebre"].sum()),
            "Costo_inventario": float(df[f"{prefix}_Costo_inv"].sum()),
            "Stock_fin_prom": float(df[f"{prefix}_Stock_fin"].mean()),
        }

    k_base = kpis("Base")
    k_sys = kpis("Sys")

    summary = {
        "ABC": abc_class,
        "Winner": winner,
        "Sigma_MAE": sigma,
        "SS_const": ss_const,
        "Base": k_base,
        "Sistema": k_sys,
        "Ahorro_CostoTotal": k_base["Costo_total"] - k_sys["Costo_total"],
        "Mejora_FillRate_pp": k_sys["FillRate_%"] - k_base["FillRate_%"],
        "Reduccion_Faltantes": k_base["Unidades_faltantes"] - k_sys["Unidades_faltantes"],
    }
    
    # ==================== INFORMACIÓN DEL PERÍODO EVALUADO ====================
    period_info = {
        "start_date": df.iloc[0]["Mes"] if not df.empty else None,
        "end_date": df.iloc[-1]["Mes"] if not df.empty else None,
        "num_months": len(df),
        "num_rows_evaluated": len(df),
        "costo_base_manual_sum": float(df["Base_Costo_total"].sum()) if not df.empty else 0.0,
        "costo_sys_manual_sum": float(df["Sys_Costo_total"].sum()) if not df.empty else 0.0,
    }
    
    return df, summary, period_info

#C. Portafolio ABC A (Masivo)
@st.cache_data(show_spinner=False)
def run_portfolio_cost_comparison_abcA(
    demand_monthly: pd.DataFrame,
    stock_monthly: pd.DataFrame,
    abc_df: pd.DataFrame,
    eval_months: int,
    cost_stock_unit: float,
    cost_stockout_unit: float,
    # modelo ganador para todos (simplifica) o "AUTO" para escoger por producto
    winner_mode: str = "AUTO",
    ma_window: int = 3,
    test_months_for_mae: int = 12,
    max_products: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Recorre SOLO ABC A, corre simulate_compare_policy_vs_baseline por producto,
    y agrega resultados portafolio.

    Retorna:
    - resumen_portafolio: 1 fila con costos totales, fill rate, etc.
    - detalle_por_producto: tabla con ahorro por SKU
    """
    dm = demand_monthly.copy()
    dm["Codigo"] = dm["Codigo"].astype(str).str.strip()
    dm["Mes"] = pd.to_datetime(dm["Mes"]).dt.to_period("M").dt.to_timestamp()
    dm = dm.sort_values(["Codigo", "Mes"])
    dm["Demanda_Unid"] = pd.to_numeric(dm["Demanda_Unid"], errors="coerce").fillna(0.0).astype(float)

    stkm = stock_monthly.copy() if stock_monthly is not None else pd.DataFrame()
    if not stkm.empty:
        stkm["Codigo"] = stkm["Codigo"].astype(str).str.strip()
        stkm["Mes"] = pd.to_datetime(stkm["Mes"]).dt.to_period("M").dt.to_timestamp()
        stkm = stkm.sort_values(["Codigo", "Mes"])

    # SOLO ABC A
    abcA = abc_df.copy()
    abcA["Codigo"] = abcA["Codigo"].astype(str).str.strip()
    abcA = abcA[abcA["ABC"] == "A"].copy()
    abcA = abcA.sort_values("Demanda_Total", ascending=False)

    codigos = abcA["Codigo"].tolist()
    if max_products is not None:
        codigos = codigos[: int(max_products)]

    rows = []

    # agregados portafolio
    sum_demand = 0.0
    sum_lost_base = 0.0
    sum_lost_sys = 0.0
    sum_cost_base = 0.0
    sum_cost_sys = 0.0
    sum_cost_inv_base = 0.0
    sum_cost_inv_sys = 0.0
    sum_cost_break_base = 0.0
    sum_cost_break_sys = 0.0

    # params modelos (consistentes con tu simulación)
    ets_params = dict(seasonal_periods=12, trend="add", seasonal="add", damped_trend=False, min_obs=24)
    rf_params = dict(n_estimators=400, min_obs=24, min_samples_leaf=1, random_state=42)

    for cod in codigos:
        hist = dm[dm["Codigo"] == str(cod)][["Mes", "Demanda_Unid"]].copy().sort_values("Mes")
        if hist.empty or len(hist) < 8:
            continue

        # stock serie producto (opcional)
        stock_p = pd.DataFrame()
        if not stkm.empty:
            stock_p = stkm[stkm["Codigo"] == str(cod)][["Mes", "Stock_Unid"]].copy().sort_values("Mes")

        # ABC class (aquí siempre será A, pero lo dejamos formal)
        abc_class = "A"

        # winner por producto o fijo
        if winner_mode == "AUTO":
            # elegir ganador por MAE (como ya haces en otros tabs)
            bt_base = backtest_baselines_1step(hist, y_col="Demanda_Unid", test_months=int(test_months_for_mae), ma_window=int(ma_window))
            ets = ETSForecaster(**ets_params)
            bt_ets = backtest_ets_1step(hist, y_col="Demanda_Unid", test_months=int(test_months_for_mae), ets=ets)
            rf = RFForecaster(**rf_params)
            bt_rf = backtest_rf_1step(hist, y_col="Demanda_Unid", test_months=int(test_months_for_mae), rf=rf)

            cmp = compare_models_metrics(bt_base.metrics, bt_ets.metrics, bt_rf.metrics, sort_by="MAE")
            if cmp.empty:
                continue
            winner = str(cmp.iloc[0]["Modelo"])
        else:
            winner = winner_mode  # e.g. "ETS(Holt-Winters)"

        # correr tu simulación existente
        df_cmp, s, period_info = simulate_compare_policy_vs_baseline(
            hist=hist,
            stock_series=stock_p,
            abc_class=abc_class,
            winner=winner,
            eval_months=int(eval_months),
            cost_stock_unit=float(cost_stock_unit),
            cost_stockout_unit=float(cost_stockout_unit),
            ets_params=ets_params,
            rf_params=rf_params,
            ma_window=int(ma_window),
            test_months_for_mae=int(test_months_for_mae),
        )

        if df_cmp is None or df_cmp.empty:
            continue

        demand = float(df_cmp["Demanda_real"].sum())
        lost_b = float(df_cmp["Base_Faltante"].sum())
        lost_s = float(df_cmp["Sys_Faltante"].sum())

        cost_b = float(df_cmp["Base_Costo_total"].sum())
        cost_s = float(df_cmp["Sys_Costo_total"].sum())

        inv_b = float(df_cmp["Base_Costo_inv"].sum())
        inv_s = float(df_cmp["Sys_Costo_inv"].sum())

        brk_b = float(df_cmp["Base_Costo_quiebre"].sum())
        brk_s = float(df_cmp["Sys_Costo_quiebre"].sum())

        fill_b = (1.0 - (lost_b / demand)) * 100 if demand > 0 else 0.0
        fill_s = (1.0 - (lost_s / demand)) * 100 if demand > 0 else 0.0

        rows.append({
            "Codigo": str(cod),
            "Modelo_usado": winner,
            "Periodos_evaluados": period_info.get("num_months", 0),
            "Inicio": period_info.get("start_date").strftime("%Y-%m") if period_info.get("start_date") else "N/A",
            "Fin": period_info.get("end_date").strftime("%Y-%m") if period_info.get("end_date") else "N/A",
            "Demanda_total_eval": demand,
            "FillRate_Base_%": fill_b,
            "FillRate_Sistema_%": fill_s,
            "Faltante_Base": lost_b,
            "Faltante_Sistema": lost_s,
            "CostoTotal_Base": cost_b,
            "CostoTotal_Sistema": cost_s,
            "CostoInv_Base": inv_b,
            "CostoInv_Sistema": inv_s,
            "CostoQuiebre_Base": brk_b,
            "CostoQuiebre_Sistema": brk_s,
            "Ahorro": cost_b - cost_s,
        })

        sum_demand += demand
        sum_lost_base += lost_b
        sum_lost_sys += lost_s
        sum_cost_base += cost_b
        sum_cost_sys += cost_s
        sum_cost_inv_base += inv_b
        sum_cost_inv_sys += inv_s
        sum_cost_break_base += brk_b
        sum_cost_break_sys += brk_s

    detalle = pd.DataFrame(rows)
    if detalle.empty:
        return pd.DataFrame(), pd.DataFrame()

    fill_port_base = (1.0 - (sum_lost_base / sum_demand)) * 100 if sum_demand > 0 else 0.0
    fill_port_sys = (1.0 - (sum_lost_sys / sum_demand)) * 100 if sum_demand > 0 else 0.0

    resumen = pd.DataFrame([{
        "ABC": "A",
        "N_productos": int(detalle["Codigo"].nunique()),
        "Demanda_total_eval": sum_demand,
        "FillRate_Base_%": fill_port_base,
        "FillRate_Sistema_%": fill_port_sys,
        "Faltante_Base": sum_lost_base,
        "Faltante_Sistema": sum_lost_sys,
        "CostoTotal_Base": sum_cost_base,
        "CostoTotal_Sistema": sum_cost_sys,
        "Ahorro_total": (sum_cost_base - sum_cost_sys),
        "CostoInv_Base": sum_cost_inv_base,
        "CostoInv_Sistema": sum_cost_inv_sys,
        "CostoQuiebre_Base": sum_cost_break_base,
        "CostoQuiebre_Sistema": sum_cost_break_sys,
    }])

    detalle = detalle.sort_values("Ahorro", ascending=False).reset_index(drop=True)
    return resumen, detalle



@st.cache_data(show_spinner=False)
def run_portfolio_comparison(
    demand_monthly: pd.DataFrame,
    sort_metric: str,
    test_months: int,
    ma_window: int,
    ets_params: dict,
    max_products: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Corre comparación por SKU y devuelve:
      - per_sku: ganador y métricas por producto
      - summary_wins: conteo ganadores por ABC
      - summary_errors: promedio de error por modelo (normal y ponderado)
    """
    dm = demand_monthly.copy()
    dm["Codigo"] = dm["Codigo"].astype(str).str.strip()
    dm["Mes"] = pd.to_datetime(dm["Mes"]).dt.to_period("M").dt.to_timestamp()
    dm = dm.sort_values(["Codigo", "Mes"])
    dm["Demanda_Unid"] = pd.to_numeric(dm["Demanda_Unid"], errors="coerce").fillna(0.0).astype(float)

    abc = build_abc_from_demand(dm)[["Codigo", "ABC", "Demanda_Total"]].copy()

    # limitar productos si se quiere (por performance)
    codigos = abc["Codigo"].tolist()
    if max_products is not None:
        codigos = codigos[: int(max_products)]

    ets = ETSForecaster(**ets_params)

    rows = []
    for cod in codigos:
        hist = dm[dm["Codigo"] == str(cod)][["Mes", "Demanda_Unid"]].copy()
        if hist.empty:
            continue

        bt_base = backtest_baselines_1step(
            hist, y_col="Demanda_Unid", test_months=test_months, ma_window=int(ma_window)
        )
        bt_ets = backtest_ets_1step(
            hist, y_col="Demanda_Unid", test_months=test_months, ets=ets
        )

        rf = RFForecaster(n_estimators=400, min_obs=24, min_samples_leaf=1, random_state=42)
        bt_rf = backtest_rf_1step(
            hist, y_col="Demanda_Unid", test_months=test_months, rf=rf
        )

        cmp = compare_models_metrics(bt_base.metrics, bt_ets.metrics, bt_rf.metrics, sort_by=sort_metric)

        if cmp.empty:
            continue

        winner = str(cmp.iloc[0]["Modelo"])
        # Guardamos métricas del ganador
        win_row = cmp.iloc[0].to_dict()
        win_row.update({"Codigo": cod, "Winner": winner})
        rows.append(win_row)

    per_sku = pd.DataFrame(rows)
    if per_sku.empty:
        return per_sku, pd.DataFrame(), pd.DataFrame()

    # unir ABC
    per_sku = per_sku.merge(abc, on="Codigo", how="left")

    # Conteo de ganadores por ABC
    summary_wins = (
        per_sku.groupby(["ABC", "Winner"], as_index=False)
               .size()
               .rename(columns={"size": "N_Productos"})
               .sort_values(["ABC", "N_Productos"], ascending=[True, False])
    )

    # Errores promedio por modelo (normal y ponderado por demanda total)
    # Nota: per_sku contiene solo el ganador por SKU; para error por modelo “global” completo
    # deberíamos guardar métricas de TODOS los modelos por SKU. Para tesis suele bastar:
    # - promedio del error del ganador, y
    # - distribución de ganadores por ABC.
    # Si quieres error por modelo completo, te lo armo después.
    per_sku["Weight"] = per_sku["Demanda_Total"].fillna(0.0).astype(float)

    def wavg(x, w):
        den = float(w.sum())
        if den <= 0:
            return float(x.mean())
        return float((x * w).sum() / den)

    summary_errors = (
        per_sku.groupby(["ABC"], as_index=False)
               .apply(lambda g: pd.Series({
                   "N_Productos": int(len(g)),
                   f"{sort_metric}_Promedio": float(g[sort_metric].mean()),
                   f"{sort_metric}_Ponderado": wavg(g[sort_metric], g["Weight"]),
               }))
               .reset_index(drop=True)
    )

    return per_sku, summary_wins, summary_errors




class EmptyTab:
    """Context manager que no renderiza nada - usado para tabs de admin-only cuando el usuario es viewer"""
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        pass


class Dashboard:
    def _ensure_project_created(self):
        """Auto-crear proyecto 'Default' si el usuario no tiene ninguno"""
        user_id = st.session_state.get("user_id")
        if not user_id or user_id == "demo-user-id":
            st.session_state.current_project_id = "demo-project"
            return
        
        try:
            db = get_db()
            # Obtener proyectos del usuario
            projects = db.get_projects(user_id)
            
            if not projects:
                # Crear proyecto Default automáticamente
                result = db.create_project(user_id, "Default", "Proyecto de planificación por defecto")
                if result["success"]:
                    st.session_state.current_project_id = result["project_id"]
                else:
                    st.warning(f"⚠️ No se pudo crear proyecto: {result['error']}")
                    st.session_state.current_project_id = "default"
            else:
                # Usar primer proyecto existente
                st.session_state.current_project_id = projects[0]["id"]
        except Exception as e:
            st.session_state.current_project_id = "demo-project"

    def _check_authentication(self) -> bool:
        """
        Verifica autenticación. Retorna True si usuario está autenticado.
        Si no, muestra pantalla de login/registro.
        """
        # Verificar si ya está autenticado
        if st.session_state.get("authenticated", False):
            # Auto-crear proyecto por defecto si no existe
            self._ensure_project_created()
            return True
        
        # Mostrar formulario de autenticación
        st.title("🔐 Sistema de Planificación")
        st.write("Inicia sesión para continuar")
        
        st.subheader("🔐 Iniciar Sesión")
        email = st.text_input("Email:", placeholder="usuario@empresa.com", key="login_email_v2")
        password = st.text_input("Contraseña:", type="password", key="login_password_v2")
        
        if st.button("Entrar", type="primary", use_container_width=True, key="login_btn"):
            if not email or not password:
                st.error("Por favor completa todos los campos")
            else:
                # Intentar Supabase primero
                try:
                    db = get_db()
                    result = db.login_user(email, password)
                    if result["success"]:
                        # Obtener info completa del usuario (org_id, is_admin)
                        user_info = db.get_user(result["user_id"])
                        
                        st.session_state.authenticated = True
                        st.session_state.user_id = result["user_id"]
                        st.session_state.email = result["email"]
                        st.session_state.organization_id = user_info.get("organization_id") if user_info else None
                        st.session_state.is_admin = user_info.get("is_admin", False) if user_info else False
                        
                        # Obtener nombre de organización
                        if st.session_state.organization_id:
                            org = db.get_organization(st.session_state.organization_id)
                            st.session_state.organization_name = org.get("nombre") if org else "Unknown"
                        
                        # CLEAR CACHE para evitar conflictos multi-usuario
                        st.cache_data.clear()
                        st.cache_resource.clear()
                        
                        st.rerun()
                    else:
                        st.error(f"Error: {result['error']}")
                except Exception as e:
                    # Demo mode fallback
                    st.session_state.authenticated = True
                    st.session_state.user_id = "demo-user-id"
                    st.session_state.email = email
                    st.session_state.company = "Demo Company"
                    st.session_state.organization_id = "demo-org-id"
                    st.session_state.is_admin = True
                    st.session_state.organization_name = "Demo Organization"
                    
                    # CLEAR CACHE for demo mode
                    st.cache_data.clear()
                    st.cache_resource.clear()
                    
                    st.success("✅ Modo Demo: Sesión iniciada (datos no persistentes)")
                    st.info("💡 Para usar BD real, configura SUPABASE_URL y SUPABASE_KEY en .env")
                    st.rerun()
        
        st.divider()
        st.info("ℹ️ Para crear una nueva cuenta, contacta al administrador de tu organización.")
        
        return False

    def render(self):
        
        # ==================== INICIALIZAR COSTOS EN SESSION_STATE ====================
        # (Evita desincronización entre tabs: asegura que existan defaults aunque
        # Comparativa Retrospectiva no se haya abierto todavía)
        if "sync_cost_stock_unit" not in st.session_state:
            st.session_state.sync_cost_stock_unit = 1.0  # Cost de mantener 1 unidad en stock
        if "sync_cost_stockout_unit" not in st.session_state:
            st.session_state.sync_cost_stockout_unit = 5.0  # Cost de quiebre de 1 unidad

        # ========================================
        # FUNCIÓN CACHEADA: Crear gráfico demo
        # ========================================
        @st.cache_data(ttl=3600)
        def crear_grafico_demo():
            """Genera el gráfico demo de predicción (se cachea para evitar regenerar en cada rerun)"""
            import numpy as np
            
            # Datos sintéticos de demanda histórica (24 meses)
            meses_demo = pd.date_range(start="2023-01", periods=24, freq="MS")
            demanda_historica = np.array([
                120, 135, 145, 155, 140, 130,
                150, 165, 175, 160, 140, 135,
                125, 140, 150, 165, 155, 145,
                160, 175, 185, 170, 150, 145
            ])
            
            demo_df = pd.DataFrame({
                "Mes": meses_demo,
                "Demanda": demanda_historica,
                "Tipo": ["Real"] * 24
            })
            
            forecast_valor = int(np.mean(demanda_historica[-6:]))
            next_mes_demo = meses_demo[-1] + pd.DateOffset(months=1)
            
            fig_demo = px.line(
                demo_df,
                x="Mes", y="Demanda",
                title="<b>📈 Visualización: Demanda Histórica y Pronóstico Inteligente</b>",
                markers=True,
                line_shape="spline",
                height=400
            )
            
            # Agregar área sombreada bajo la línea histórica (efecto visual premium)
            fig_demo.add_scatter(
                x=demo_df["Mes"].tolist() + demo_df["Mes"].tolist()[::-1],
                y=demo_df["Demanda"].tolist() + [0] * len(demo_df),
                fill="tozeroy",
                fillcolor="rgba(25, 118, 210, 0.1)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                showlegend=False
            )
            
            # Pronóstico con estrella roja impactante
            fig_demo.add_scatter(
                x=[next_mes_demo],
                y=[forecast_valor],
                mode="markers+text",
                name="",  # Sin leyenda (es obvio que es el pronóstico)
                marker=dict(size=18, color="#FF6B6B", symbol="star", line=dict(color="white", width=2)),
                text=[f"<b>{forecast_valor} unid.</b>"],
                textposition="top center",
                hovertemplate="<b>Pronóstico:</b> %{y} unidades<extra></extra>",
                showlegend=False
            )
            
            # Estilo premium del gráfico
            fig_demo.update_layout(
                hovermode="x unified",
                template="plotly_white",
                yaxis_title="<b>Unidades de Demanda</b>",
                xaxis_title="<b>Período (Mes)</b>",
                showlegend=False,
                paper_bgcolor="rgba(240, 245, 250, 0.5)",
                plot_bgcolor="white",
                font=dict(family="Arial, sans-serif", size=11, color="#333"),
                title_font=dict(size=14, color="#1565C0"),
                margin=dict(l=50, r=50, t=60, b=50)
            )
            
            # Mejorar línea histórica (más gruesa y con gradiente de color)
            fig_demo.update_traces(
                line=dict(color="#1976D2", width=3),
                marker=dict(size=6, color="#1565C0"),
                selector=dict(mode="lines+markers")
            )
            
            return fig_demo

        # ========================================
        # FUNCIÓN CACHEADA: Componentes por producto
        # ========================================
        @st.cache_data(ttl=3600)
        def get_demanda_components(prod_sel):
            """Cachea los componentes de demanda por producto"""
            return build_monthly_components(res_movements, prod_sel)

        # ========================================
        # FUNCIÓN CACHEADA: Backtests del comparador
        # ========================================
        @st.cache_data(ttl=3600)
        def get_comparador_backtests(prod_sel, test_months: int, sort_metric: str):
            """Ejecuta y cachea todos los backtests (Baselines, ETS, RF) para un producto"""
            dm = res_demand.copy()
            dm["Codigo"] = dm["Codigo"].astype(str).str.strip()
            hist_cmp = dm[dm["Codigo"] == str(prod_sel)][["Mes", "Demanda_Unid"]].copy().sort_values("Mes")
            
            if hist_cmp.empty:
                return None, None, None, None, None
            
            # Auto-optimizar MA (3 vs 6)
            bt_ma3 = backtest_baselines_1step(hist_cmp, y_col="Demanda_Unid", test_months=int(test_months), ma_window=3)
            bt_ma6 = backtest_baselines_1step(hist_cmp, y_col="Demanda_Unid", test_months=int(test_months), ma_window=6)
            
            mae_ma3 = float(bt_ma3.metrics.iloc[0]["MAE"]) if not bt_ma3.metrics.empty else float("inf")
            mae_ma6 = float(bt_ma6.metrics.iloc[0]["MAE"]) if not bt_ma6.metrics.empty else float("inf")
            ma_window_cmp = 3 if mae_ma3 < mae_ma6 else 6
            bt_base_cmp = bt_ma3 if ma_window_cmp == 3 else bt_ma6
            
            # ETS
            ets = ETSForecaster(seasonal_periods=12, trend="add", seasonal="add", damped_trend=False, min_obs=24)
            bt_ets_cmp = backtest_ets_1step(hist_cmp, y_col="Demanda_Unid", test_months=int(test_months), ets=ets)
            
            # RF
            rf = RFForecaster(n_estimators=400, min_obs=24, min_samples_leaf=1, random_state=42)
            bt_rf_cmp = backtest_rf_1step(hist_cmp, y_col="Demanda_Unid", test_months=int(test_months), rf=rf)
            
            # Comparar
            cmp = compare_models_metrics(bt_base_cmp.metrics, bt_ets_cmp.metrics, bt_rf_cmp.metrics, sort_by=sort_metric)
            
            return (bt_base_cmp, bt_ets_cmp, bt_rf_cmp, cmp, {"ma_window": ma_window_cmp, "mae_ma3": mae_ma3, "mae_ma6": mae_ma6})

        st.set_page_config(
            page_title="Predicast - Sistema de Planificación",
            layout="wide",
            initial_sidebar_state="expanded",
            menu_items={"About": "Sistema avanzado de planificación de demanda y pronósticos"}
        )
        
        # ==================== ESTILOS CSS PROFESIONALES ====================
        st.markdown("""
        <style>
        /* Paleta de colores profesionales */
        :root {
            --primary: #0D47A1;     /* Azul marino corporativo */
            --secondary: #1976D2;   /* Azul profesional */
            --accent: #4CAF50;      /* Verde éxito */
            --warning: #FF9800;     /* Naranja advertencia */
            --danger: #F44336;      /* Rojo crítico */
            --dark: #263238;        /* Gris oscuro */
            --light: #ECEFF1;       /* Gris claro */
        }
        
        /* Configuración general */
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #FAFAFA;
            margin: 0;
            padding: 0;
        }
        
        .main {
            padding-top: 0 !important;
        }
        
        /* Títulos principales */
        h1 {
            color: #0D47A1;
            font-weight: 700;
            font-size: 2.5em;
            margin-top: 0 !important;
            margin-bottom: 0.3em;
            letter-spacing: -0.5px;
        }
        
        /* Subtítulos */
        h2 {
            color: #1976D2;
            font-weight: 600;
            font-size: 1.8em;
            margin-top: 1.5em;
            margin-bottom: 0.8em;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 0.5em;
        }
        
        h3 {
            color: #263238;
            font-weight: 600;
            font-size: 1.3em;
        }
        
        /* Métricas desatacadas */
        [data-testid="metric-container"] {
            background: linear-gradient(135deg, #FFFFFF 0%, #F5F5F5 100%);
            border: 2px solid #E0E0E0;
            border-radius: 12px;
            padding: 1.5em;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
            transition: all 0.3s ease;
        }
        
        [data-testid="metric-container"]:hover {
            box-shadow: 0 4px 16px rgba(13, 71, 161, 0.15);
            border-color: #1976D2;
            transform: translateY(-2px);
        }
        
        /* Plotly charts con sombra */
        .plotly-graph-div {
            border-radius: 10px;
            box-shadow: 0 3px 12px rgba(0, 0, 0, 0.1);
            background: white;
            padding: 1em;
            margin: 1em 0;
        }
        
        /* Tabs profesionales */
        .stTabs [role="tablist"] {
            background-color: #F5F5F5;
            border-radius: 8px;
            border-bottom: 3px solid #0D47A1;
            padding: 0.5em;
        }
        
        .stTabs [role="tab"][aria-selected="true"] {
            background-color: #0D47A1;
            color: white;
            border-radius: 6px;
            font-weight: 600;
        }
        
        .stTabs [role="tab"][aria-selected="false"] {
            color: #263238;
        }
        
        /* Botones estilizados */
        .stButton > button {
            background-color: #1976D2;
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.75em 1.5em;
            font-weight: 600;
            font-size: 0.95em;
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            background-color: #0D47A1;
            box-shadow: 0 4px 12px rgba(13, 71, 161, 0.3);
            transform: translateY(-2px);
        }
        
        .stButton > button[kind="primary"] {
            background-color: #4CAF50;
        }
        
        .stButton > button[kind="primary"]:hover {
            background-color: #388E3C;
        }
        
        /* Input fields */
        .stTextInput input, .stNumberInput input, .stSelectbox select {
            border: 2px solid #E0E0E0;
            border-radius: 6px;
            color: #263238;
        }
        
        .stTextInput input:focus, .stNumberInput input:focus {
            border-color: #1976D2;
            box-shadow: 0 0 0 3px rgba(25, 118, 210, 0.1);
        }
        
        /* Info boxes */
        .stInfo {
            background-color: #E3F2FD;
            border-left: 5px solid #1976D2;
            border-radius: 6px;
        }
        
        .stSuccess {
            background-color: #E8F5E9;
            border-left: 5px solid #4CAF50;
            border-radius: 6px;
        }
        
        .stWarning {
            background-color: #FFF3E0;
            border-left: 5px solid #FF9800;
            border-radius: 6px;
        }
        
        .stError {
            background-color: #FFEBEE;
            border-left: 5px solid #F44336;
            border-radius: 6px;
        }
        
        /* Sidebar */
        .stSidebar {
            background-color: #FAFAFA;
            border-right: 2px solid #E0E0E0;
        }
        
        /* Dividers */
        hr {
            border-top: 2px solid #E0E0E0;
            margin: 1.5em 0;
        }
        
        /* Dataframe styling */
        .stDataFrame, [data-testid="stDataFrame"] {
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        }
        
        /* Expander styling */
        .stExpander {
            border: 1px solid #E0E0E0;
            border-radius: 8px;
        }
        
        </style>
        """, unsafe_allow_html=True)
        
        # ==================== AUTENTICACIÓN ====================
        if not self._check_authentication():
            return  # Muestra login screen y retorna
        
        # ==================== DASHBOARD PRINCIPAL ====================
        # (Título movido al sidebar para ganar espacio)

        # Información de usuario y organización en sidebar
        # Agregar título compacto al inicio del sidebar
        st.sidebar.markdown("""
        <div style='text-align: center; padding: 0.5em 0;'>
            <h2 style='color: #0D47A1; font-size: 1.5em; font-weight: 800; margin: 0;'>🔮 PREDICAST</h2>
            <p style='color: #666; font-size: 0.75em; margin: 0.3em 0 0;'>Predicción de Demanda</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.sidebar.divider()
        st.sidebar.write("**👤 Información de Sesión**")
        st.sidebar.write(f"Email: {st.session_state.email}")
        
        org_name = st.session_state.get("organization_name", "N/A")
        st.sidebar.write(f"🏢 Org: {org_name}")
        
        if st.session_state.get("is_admin"):
            st.sidebar.write("👑 **Rol:** Admin")
        else:
            st.sidebar.write("👤 **Rol:** Viewer")
        
        st.sidebar.divider()
        
        # DEBUG SUPERADMIN
        is_superadmin_check = _is_superadmin()
        with st.sidebar.expander("🔧 DEBUG (QUITAR DESPUÉS)", expanded=False):
            st.write(f"**Email:** {st.session_state.get('email', 'N/A').lower()}")
            try:
                sa_emails = st.secrets.get("SUPERADMIN_EMAILS", "")
            except:
                import os
                sa_emails = os.getenv("SUPERADMIN_EMAILS", "")
            st.write(f"**SUPERADMIN_EMAILS config:** `{sa_emails}`")
            st.write(f"**¿Es superadmin?:** {is_superadmin_check}")
        
        st.sidebar.divider()
        if st.sidebar.button("🚪 Cerrar Sesión", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.user_id = None
            st.session_state.email = None
            st.session_state.organization_id = None
            st.session_state.is_admin = False
            st.session_state.organization_name = None
            
            # CLEAR CACHE on logout
            st.cache_data.clear()
            st.cache_resource.clear()
            
            st.success("Sesión cerrada. Recargando...")
            st.rerun()
        st.sidebar.divider()

        # ==================== DATA LOADING: CACHE FIRST ====================
        org_id = st.session_state.get("organization_id")
        user_id = st.session_state.get("user_id")
        is_admin = st.session_state.get("is_admin", False)
        
        # Obtener DB
        db = None
        try:
            db = get_db()
        except Exception as e:
            db = None
        
        # Verificar si hay data cacheada
        from src.services.cache_service import check_and_load_org_cache, save_org_cache
        
        has_cache, cached_data = False, None
        if db and org_id:
            has_cache, cached_data = check_and_load_org_cache(db, org_id)
        
        if has_cache and cached_data:
            # ==================== CARGAR DESDE CACHE ====================
            st.sidebar.success("✅ **Datos Cacheados**")
            st.sidebar.info(f"📅 Actualizado: {cached_data.get('updated_at', 'N/A')[:10]}")
            st.sidebar.write(f"📄 CSVs: {cached_data.get('csv_files_count', 0)}")
            
            # Usar datos del cache
            res_movements = cached_data.get("movements")
            res_demand = cached_data.get("demand_monthly")
            res_stock = cached_data.get("stock_monthly")
            
            # Guardar en session_state para que las tabs puedan acceder
            st.session_state.pipeline_movements = res_movements
            st.session_state.pipeline_demand = res_demand
            st.session_state.pipeline_stock = res_stock
            
            st.sidebar.write("✨ Los datos están listos para análisis")
            
            # === MOSTRAR KPIs COMPACTOS EN SIDEBAR ===
            st.sidebar.divider()
            with st.sidebar.expander("📊 Resumen de Datos", expanded=False):
                dm_kpi = res_demand.copy()
                dm_kpi["Codigo"] = dm_kpi["Codigo"].astype(str).str.strip()
                abc_kpi = build_abc_from_demand(dm_kpi)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("📦 Productos", res_demand["Codigo"].nunique())
                    st.metric("🔴 Clase A", len(abc_kpi[abc_kpi["ABC"] == "A"]))
                with col2:
                    st.metric("📅 Meses", len(res_demand["Mes"].unique()))
                    st.metric("🟡 Clase B", len(abc_kpi[abc_kpi["ABC"] == "B"]))
                
                col3, col4 = st.columns(2)
                with col3:
                    st.metric("📋 Movimientos", len(res_movements))
                    st.metric("🟢 Clase C", len(abc_kpi[abc_kpi["ABC"] == "C"]))
                with col4:
                    min_mes = res_demand["Mes"].min()
                    max_mes = res_demand["Mes"].max()
                    period_str = f"{min_mes.strftime('%Y-%m')}\n{max_mes.strftime('%Y-%m')}"
                    st.caption(f"Período:\n{period_str}")
        
        else:
            # ==================== NO HAY CACHE ====================
            st.sidebar.warning("⚠️ **Sin Datos Cacheados**")
            
            if is_admin:
                # Admin puede subir
                st.sidebar.header("📤 Subir Datos")
                files = st.sidebar.file_uploader(
                    "Sube CSV (2021–2025)",
                    type=["csv"],
                    accept_multiple_files=True
                )
                
                if not files:
                    st.info("👆 Admin: Sube los CSV para procesar")
                    return
                
                # ==================== S3 UPLOAD ====================
                storage = get_storage_manager()
                project_id = st.session_state.get("current_project_id")
                
                with st.spinner("📤 Procesando archivos..."):
                    saved_files = []
                    for file in files:
                        try:
                            file_contents = file.read()
                            file.seek(0)
                            
                            result = storage.upload_file_bytes(
                                file_contents,
                                file.name,
                                user_id=user_id,
                                project_id=project_id
                            )
                            
                            if result["success"]:
                                if db and result.get("s3_url"):
                                    try:
                                        save_result = db.save_upload(
                                            user_id=user_id,
                                            project_id=project_id,
                                            filename=file.name,
                                            s3_path=result.get("s3_url"),
                                            file_size=len(file_contents),
                                            organization_id=org_id  # NUEVO: guardar org_id
                                        )
                                        if save_result.get("success"):
                                            st.success(f"✅ {file.name} - Guardado")
                                    except Exception as db_error:
                                        pass
                                saved_files.append(file)
                            else:
                                st.warning(f"⚠️ {file.name}: {result.get('error', 'Error desconocido')}")
                        except Exception as e:
                            st.error(f"❌ {file.name}: {str(e)}")
                
                if not saved_files:
                    st.error("No se han podido procesar los archivos")
                    return
                
                # ==================== PIPELINE ====================
                pipeline = DataPipeline()
                
                with st.spinner("⚙️ Ejecutando pipeline de datos..."):
                    res = pipeline.run(saved_files)
                
                if res.movements.empty:
                    st.error("No se detectaron columnas mínimas o la data quedó vacía tras limpieza.")
                    return
                
                # ==================== GUARDAR EN CACHE ====================
                st.info("💾 Guardando datos en cache...")
                if db:
                    cache_saved = save_org_cache(
                        db=db,
                        org_id=org_id,
                        movements=res.movements,
                        demand_monthly=res.demand_monthly,
                        stock_monthly=res.stock_monthly,
                        processed_by=user_id,
                        csv_files_count=len(saved_files)
                    )
                    
                    if cache_saved:
                        st.success("✅ Datos guardados en cache")
                        st.balloons()
                    else:
                        st.warning("⚠️ Error saving cache (pero data está lista para análisis)")
                
                # Usar los datos procesados
                res_movements = res.movements
                res_demand = res.demand_monthly
                res_stock = res.stock_monthly
                
                st.session_state.pipeline_movements = res_movements
                st.session_state.pipeline_demand = res_demand
                st.session_state.pipeline_stock = res_stock
            
            else:
                # Viewer esperando
                st.warning("⏳ Los datos aún no han sido cargados")
                st.info("Por favor espera a que el administrador cargue los archivos CSV")
                return
        
        # ==================== CONTINUAR CON ANÁLISIS ====================
        # En este punto, tenemos data (ya sea de cache o recién procesada)
        res_movements = st.session_state.get("pipeline_movements")
        res_demand = st.session_state.get("pipeline_demand")
        res_stock = st.session_state.get("pipeline_stock")

        if res_movements is None or res_demand is None or res_stock is None:
            st.error("❌ No hay data disponible para análisis")
            return

        # --- ABC (una vez) ---
        dm_abc = res_demand.copy()
        dm_abc["Codigo"] = dm_abc["Codigo"].astype(str).str.strip()
        abc_df = build_abc_from_demand(dm_abc)  # columnas: Codigo, Demanda_Total, Share, CumShare, ABC

        # === FILTROS EN SIDEBAR (compacto) ===
        st.sidebar.divider()
        with st.sidebar.expander("🔍 **Filtros de Producto**", expanded=False):
            col_abc, col_prod = st.columns(2)
            
            with col_abc:
                # 1) Filtro ABC
                abc_options = ["A", "B", "C", "Todos"]
                abc_sel = st.selectbox("Categoría ABC", options=abc_options, index=0, key="sidebar_abc")



            with col_prod:
                # 2) Productos filtrados por ABC
                if abc_sel == "Todos":
                    productos = sorted(res_movements["Codigo"].dropna().astype(str).str.strip().unique().tolist())
                else:
                    productos = (
                        abc_df[abc_df["ABC"] == abc_sel]["Codigo"]
                        .dropna()
                        .astype(str)
                        .str.strip()
                        .unique()
                        .tolist()
                    )
                    productos = sorted(productos)

                # 3) Select producto (con fallback)
                if not productos:
                    st.warning("No hay productos en esa categoría ABC.")
                    prod_sel = None
                else:
                    prod_sel = st.selectbox("Producto (Código)", options=productos, key="sidebar_prod")

        # === BOTÓN FLOTANTE VISUAL EN SIDEBAR ===
        st.sidebar.markdown("""
        <style>
        .floating-filter-badge {
            position: fixed;
            bottom: 30px;
            right: 20px;
            background: linear-gradient(135deg, #1976D2 0%, #1565C0 100%);
            color: white;
            border-radius: 50%;
            width: 60px;
            height: 60px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 24px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            cursor: pointer;
            z-index: 1000;
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0%, 100% { box-shadow: 0 4px 12px rgba(0,0,0,0.15); }
            50% { box-shadow: 0 4px 20px rgba(25, 118, 210, 0.4); }
        }
        .floating-filter-badge:hover {
            transform: scale(1.1);
        }
        </style>
        <div class='floating-filter-badge' title='Filtros disponibles en el sidebar'>🔍</div>
        """, unsafe_allow_html=True)
        
        # === CSS PARA MEJORAR DISTRIBUCIÓN DE TABS ===
        st.markdown("""
        <style>
        /* Hacer que las tabs ocupen más espacio horizontal */
        .stTabs [role="tablist"] {
            display: flex;
            justify-content: flex-start;
            gap: 1px;
            width: 100%;
            flex-wrap: wrap;
        }
        .stTabs [role="tab"] {
            flex: 1;
            min-width: 150px;
            padding: 10px 15px !important;
            text-align: center;
        }
        </style>
        """, unsafe_allow_html=True)

        # ==================== VERIFICAR SUPERADMIN ====================
        def _is_superadmin() -> bool:
            """Verifica si usuario actual es superadmin"""
            import os
            
            # Intentar leer de st.secrets primero (Streamlit Cloud)
            # Si no existe, leer de os.getenv (local)
            try:
                superadmin_emails_str = st.secrets.get("SUPERADMIN_EMAILS", "")
            except:
                superadmin_emails_str = os.getenv("SUPERADMIN_EMAILS", "")
            
            superadmin_emails = [e.strip().lower() for e in superadmin_emails_str.split(",") if e.strip()]
            current_email = st.session_state.get("email", "").lower()
            
            # DEBUG
            print(f"[DEBUG SUPERADMIN] Emails configurados: {superadmin_emails}")
            print(f"[DEBUG SUPERADMIN] Email actual: {current_email}")
            print(f"[DEBUG SUPERADMIN] ¿Es superadmin?: {current_email in superadmin_emails}")
            
            return current_email in superadmin_emails
        
        is_superadmin = _is_superadmin()
        
        # ==================== TABS DINÁMICAS ====================
        # Construir tabs según permisos del usuario
        tabs_config = [
            ("🏠 Dashboard", "dashboard"),
            ("📊 Análisis Individual", "individual"),
            ("📊 Análisis de Grupo", "grupal"),
        ]
        
        if is_admin:
            tabs_config.append(("⚙️ Panel Admin", "admin"))
        
        if is_superadmin:
            tabs_config.append(("🏆 Superadmin", "superadmin"))
        
        tabs_labels = [t[0] for t in tabs_config]
        tabs_keys = [t[1] for t in tabs_config]
        
        tabs = st.tabs(tabs_labels)
        tabs_dict = {key: tab for key, tab in zip(tabs_keys, tabs)}
        
        tab_dashboard = tabs_dict["dashboard"]
        tab_individual = tabs_dict["individual"]
        tab_grupal = tabs_dict["grupal"]
        tab_admin = tabs_dict.get("admin", EmptyTab())
        tab_superadmin = tabs_dict.get("superadmin", None)
        
        # Crear subtabs dentro de Análisis Individual (para TODOS)
        with tab_individual:
            tab_demanda, tab_stock_diag, Tab_Comparativa, tab_reco = st.tabs([
                "🧩 Demanda y Componentes",
                "🏢 Stock y Diagnóstico",
                "🏆 Comparador de Modelos (Baselines vs ETS vs RF)",
                "🎯 Recomendación Individual",
            ])
        
        # Crear subtabs dentro de Análisis de Grupo (para TODOS)
        with tab_grupal:
            ResumenComparativa, ComparaRetroEntreSistema, Reco_Masiva = st.tabs([
                "📊 Resumen Comparativa Global",
                "📉 Comparativa Retrospectiva",
                "📑 Recomendación Masiva",
            ])
        
        # Renderizar admin panel (solo si es admin)
        if is_admin:
            with tab_admin:
                from src.ui.admin_panel import AdminPanel
                admin = AdminPanel(get_db())
                admin.render()
        
        # Renderizar superadmin panel (solo si es superadmin)
        if is_superadmin and tab_superadmin:
            with tab_superadmin:
                from src.ui.superadmin_panel import SuperAdminPanel
                superadmin = SuperAdminPanel(get_db())
                superadmin.render()

        # ==========================================================
        # TAB 0: DASHBOARD (LANDING PAGE)
        # ==========================================================
        with tab_dashboard:
            # Header compacto
            st.markdown("""
            <div style='text-align: center; margin-bottom: 1em; background: linear-gradient(135deg, #1976D2 0%, #1565C0 100%); padding: 2em; border-radius: 10px; box-shadow: 0 4px 15px rgba(25, 118, 210, 0.3);'>
                <h1 style='color: white; font-size: 2.5em; margin: 0; font-weight: bold;'>🚀 Inteligencia en Planificación de Demanda</h1>
                <p style='color: rgba(255, 255, 255, 0.9); font-size: 1.1em; margin-top: 0.5em; margin-bottom: 0;'>Pronósticos precisos + Recomendaciones inteligentes = Máxima eficiencia</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Descripción colapsable con botones interactivos
            with st.expander("**🎯 ¿Qué encontrarás aquí? / Funciones disponibles**", expanded=False):
                
                
                # ========== FILA 1: ANÁLISIS INDIVIDUAL ==========
                st.markdown("##### 📊 Análisis Individual (por producto)")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown("""
                    <div style="
                        display: flex;
                        flex-direction: column;
                    ">
                        <h4 style="margin-top: 0; margin-bottom: 0.3em; font-size: 0.95em;">📈 Demanda y Componentes</h4>
                        <div style="
                            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                            color: white;
                            padding: 10px;
                            border-radius: 8px;
                            text-align: center;
                            margin-bottom: 10px;
                            font-weight: 600;
                            font-size: 0.9em;
                        ">
                        📌 Análisis Individual → 📈 Demanda
                        </div>
                        <p style="margin: 0; font-size: 0.9em; color: #555;">Visualiza desglose de demanda: venta, consumo y guía externa.</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown("""
                    <div style="
                        display: flex;
                        flex-direction: column;
                    ">
                        <h4 style="margin-top: 0; margin-bottom: 0.3em; font-size: 0.95em;">🏢 Stock y Diagnóstico</h4>
                        <div style="
                            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                            color: white;
                            padding: 10px;
                            border-radius: 8px;
                            text-align: center;
                            margin-bottom: 10px;
                            font-weight: 600;
                            font-size: 0.9em;
                        ">
                        📌 Análisis Individual → 🏢 Stock
                        </div>
                        <p style="margin: 0; font-size: 0.9em; color: #555;">Analiza niveles de stock histórico y diagnóstico actual.</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown("""
                    <div style="
                        display: flex;
                        flex-direction: column;
                    ">
                        <h4 style="margin-top: 0; margin-bottom: 0.3em; font-size: 0.95em;">🏆 Comparador de Modelos</h4>
                        <div style="
                            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
                            color: white;
                            padding: 10px;
                            border-radius: 8px;
                            text-align: center;
                            margin-bottom: 10px;
                            font-weight: 600;
                            font-size: 0.9em;
                        ">
                        📌 Análisis Individual → 🏆 Comparador
                        </div>
                        <p style="margin: 0; font-size: 0.9em; color: #555;">Compara Baselines vs ETS vs Random Forest.</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    st.markdown("""
                    <div style="
                        display: flex;
                        flex-direction: column;
                    ">
                        <h4 style="margin-top: 0; margin-bottom: 0.3em; font-size: 0.95em;">🎯 Recomendación Individual</h4>
                        <div style="
                            background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
                            color: white;
                            padding: 10px;
                            border-radius: 8px;
                            text-align: center;
                            margin-bottom: 10px;
                            font-weight: 600;
                            font-size: 0.9em;
                        ">
                        📌 Análisis Individual → 🎯 Recomendación
                        </div>
                        <p style="margin: 0; font-size: 0.9em; color: #555;">Obtén cantidad exacta a producir el próximo mes.</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.divider()
                
                # ========== FILA 2: ANÁLISIS DE GRUPO ==========
                st.markdown("##### 📊 Análisis de Grupo (múltiples productos)")
                col5, col6, col7 = st.columns(3)
                
                with col5:
                    st.markdown("""
                    <div style="
                        display: flex;
                        flex-direction: column;
                    ">
                        <h4 style="margin-top: 0; margin-bottom: 0.3em; font-size: 0.95em;">📊 Resumen Comparativa</h4>
                        <div style="
                            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
                            color: white;
                            padding: 10px;
                            border-radius: 8px;
                            text-align: center;
                            margin-bottom: 10px;
                            font-weight: 600;
                            font-size: 0.9em;
                        ">
                        📌 Análisis de Grupo → 📊 Resumen
                        </div>
                        <p style="margin: 0; font-size: 0.9em; color: #555;">Comparar rendimiento de todos los productos globalmente.</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col6:
                    st.markdown("""
                    <div style="
                        display: flex;
                        flex-direction: column;
                    ">
                        <h4 style="margin-top: 0; margin-bottom: 0.3em; font-size: 0.95em;">📉 Comparativa Retrospectiva</h4>
                        <div style="
                            background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
                            color: #333;
                            padding: 10px;
                            border-radius: 8px;
                            text-align: center;
                            margin-bottom: 10px;
                            font-weight: 600;
                            font-size: 0.9em;
                        ">
                        📌 Análisis de Grupo → 📉 Comparativa
                        </div>
                        <p style="margin: 0; font-size: 0.9em; color: #555;">Compara costos: sin sistema vs con sistema.</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col7:
                    st.markdown("""
                    <div style="
                        display: flex;
                        flex-direction: column;
                    ">
                        <h4 style="margin-top: 0; margin-bottom: 0.3em; font-size: 0.95em;">📑 Recomendación Masiva</h4>
                        <div style="
                            background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
                            color: #333;
                            padding: 10px;
                            border-radius: 8px;
                            text-align: center;
                            margin-bottom: 10px;
                            font-weight: 600;
                            font-size: 0.9em;
                        ">
                        📌 Análisis de Grupo → 📑 Masiva
                        </div>
                        <p style="margin: 0; font-size: 0.9em; color: #555;">Obtén recomendaciones para todos los productos.</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Gráfico Demo compacto
            
            fig_demo = crear_grafico_demo()
            st.plotly_chart(fig_demo, use_container_width=True)
            
            # Resumen del objetivo del sistema
            st.markdown("""
            <div style="
                background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                padding: 15px;
                border-radius: 10px;
                border-left: 4px solid #1976D2;
                margin-top: 15px;
            ">
                <p style="margin: 0; font-size: 0.95em; color: #333; line-height: 1.6;">
                    <strong>🎯 Objetivo del Sistema:</strong> Optimizar la planificación de producción mediante pronósticos precisos de demanda, 
                    análisis de stock y recomendaciones inteligentes para minimizar costos y maximizar eficiencia operativa.
                </p>
            </div>
            """, unsafe_allow_html=True)

        # TAB 1: DEMANDA Y COMPONENTES
        # ==========================================================
        with tab_demanda:
         
            comp = get_demanda_components(prod_sel)

            # ==================== RESUMEN EJECUTIVO ====================
            st.markdown("### 📊 Resumen de Demanda")
            
            if not comp.empty:
                # Calcular KPIs
                demanda_promedio = float(comp["Demanda_Total"].mean())
                demanda_max = float(comp["Demanda_Total"].max())
                demanda_min = float(comp["Demanda_Total"].min())
                demanda_reciente = float(comp.iloc[-1]["Demanda_Total"]) if len(comp) > 0 else 0
                demanda_anterior = float(comp.iloc[-2]["Demanda_Total"]) if len(comp) > 1 else demanda_reciente
                trend = ((demanda_reciente - demanda_anterior) / demanda_anterior * 100) if demanda_anterior != 0 else 0
                volatilidad = float(comp["Demanda_Total"].std())
                
                # Mostrar KPIs en columnas
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "📈 Promedio",
                        f"{demanda_promedio:,.0f} unid",
                        f"{volatilidad:,.0f} (σ)"
                    )
                
                with col2:
                    st.metric(
                        "🔴 Máximo",
                        f"{demanda_max:,.0f} unid",
                        f"+{((demanda_max - demanda_promedio)/demanda_promedio*100):.1f}%"
                    )
                
                with col3:
                    st.metric(
                        "🟢 Mínimo",
                        f"{demanda_min:,.0f} unid",
                        f"{((demanda_min - demanda_promedio)/demanda_promedio*100):.1f}%"
                    )
                
                with col4:
                    trend_emoji = "📈" if trend > 0 else "📉" if trend < 0 else "➡️"
                    st.metric(
                        f"{trend_emoji} Tendencia Reciente",
                        f"{abs(trend):.1f}%",
                        f"Mes anterior: {demanda_anterior:,.0f} → {demanda_reciente:,.0f}"
                    )
                
                # Explicación visual
                st.markdown("""
                <div style='background: linear-gradient(135deg, #e3f2fd 0%, #f5f5f5 100%); padding: 14px; border-left: 4px solid #1976d2; border-radius: 8px; margin-bottom: 1em;'>
                    <p style='margin: 0; color: #333; font-size: 0.9em;'>
                        <strong>💡 Qué observar:</strong><br>
                        • <strong>Promedio:</strong> Nivel normal de demanda. Usar como referencia para planificación.<br>
                        • <strong>Volatilidad (σ):</strong> Desviación estándar. Valores altos = demanda impredecible.<br>
                        • <strong>Tendencia:</strong> Si es positiva, la demanda está creciendo. Planes de producción deben ajustarse.<br>
                        • <strong>Máximo vs Mínimo:</strong> Si la diferencia es grande, necesitas stock de seguridad mayor.
                    </p>
                </div>
                """, unsafe_allow_html=True)

            # Gráfico de demanda total (ancho completo)
            fig_total = px.line(
                comp, x="Mes", y="Demanda_Total", markers=True,
                title=f"Demanda total histórica - Producto {prod_sel}",
                height=400
            )
            fig_total.add_hline(y=demanda_promedio, line_dash="dash", line_color="green", 
                               annotation_text=f"Promedio: {demanda_promedio:,.0f}", annotation_position="right")
            st.plotly_chart(fig_total, use_container_width=True)

            # ==================== ANÁLISIS DE COMPONENTES ====================
            st.markdown("### 🧩 Desglose de Componentes")
            
            if not comp.empty and "Venta_unid" in comp.columns:
                col_comp1, col_comp2 = st.columns(2)
                
                with col_comp1:
                    # Suma de componentes
                    venta_total = comp["Venta_unid"].sum() if "Venta_unid" in comp.columns else 0
                    consumo_total = comp["Consumo_unid"].sum() if "Consumo_unid" in comp.columns else 0
                    guia_total = comp["Guia_Salida_Externa_Unid"].sum() if "Guia_Salida_Externa_Unid" in comp.columns else 0
                    
                    total = venta_total + consumo_total + guia_total
                    
                    if total > 0:
                        # Gráfico de pastel
                        component_data = {
                            "Componente": ["Venta", "Consumo", "Guía Externa"],
                            "Unidades": [venta_total, consumo_total, guia_total]
                        }
                        fig_pie = px.pie(
                            component_data, 
                            values="Unidades", 
                            names="Componente",
                            title="Proporción de Componentes"
                        )
                        st.plotly_chart(fig_pie, use_container_width=True)
                
                with col_comp2:
                    st.markdown("""
                    <div style='background: #fff9e6; padding: 14px; border-left: 4px solid #ff9800; border-radius: 8px; height: 100%;'>
                        <p style='margin: 0; color: #333; font-size: 0.9em; font-weight: 600;'>📌 Significado de Componentes:</p>
                        <ul style='margin: 0.5em 0; padding-left: 20px; color: #555; font-size: 0.85em;'>
                            <li><strong>Venta:</strong> Demanda de clientes externos</li>
                            <li><strong>Consumo:</strong> Uso interno en operaciones</li>
                            <li><strong>Guía Externa:</strong> Transferencias a otras bodegas</li>
                        </ul>
                        <p style='margin: 0.5em 0; color: #666; font-size: 0.85em;'>
                            Si <strong>Venta es baja</strong> pero <strong>Consumo es alto</strong>, hay ineficiencia operativa.<br>
                            Si <strong>Guía Externa es significativa</strong>, hay movimiento entre sedes.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

            # Tabla de componentes colapsada
            with st.expander("🧩 Detalles: Componentes de demanda por mes", expanded=False):
                st.dataframe(comp, use_container_width=True, height=380)

            st.divider()

            st.subheader("🧾 Diagnóstico: Guías de remisión")
            guia = res_movements[res_movements["Documento"].astype(str).str.strip() == config.GUIDE_DOC].copy()
            if guia.empty:
                st.info("No se encontraron guías de remisión en los archivos cargados.")
            else:
                with st.expander("🔎 Muestra de guías (filas)", expanded=False):
                    cols = [
                        "Fecha", "Codigo", "Bodega", "Documento", "Numero",
                        "Entrada_unid", "Salida_unid", "Tipo_Guia", "Guia_Salida_Externa_Unid"
                    ]
                    st.dataframe(guia[cols].sort_values("Fecha").head(300), use_container_width=True)

        # ==========================================================
        # TAB: COMPARADOR DE MODELOS (CONSOLIDADO)
        # ==========================================================
        with Tab_Comparativa:
            st.subheader("🏆 Comparador de Modelos: Baselines vs ETS vs RF")

            dm = res_demand.copy()
            dm["Codigo"] = dm["Codigo"].astype(str).str.strip()
            hist_cmp = dm[dm["Codigo"] == str(prod_sel)][["Mes", "Demanda_Unid"]].copy().sort_values("Mes")

            if hist_cmp.empty:
                st.info("No hay serie mensual para este producto.")
            else:
                # ==================== EXPLICACIÓN GENERAL ====================
                st.markdown("""
                <div style='background: linear-gradient(135deg, #e8f5e9 0%, #f1f8e9 100%); padding: 14px; border-left: 5px solid #4caf50; border-radius: 8px; margin-bottom: 1.5em;'>
                    <p style='margin: 0; color: #333; font-size: 0.9em;'>
                        <strong>📖 Cómo funciona este análisis:</strong><br>
                        El sistema prueba 3 tipos diferentes de modelos de pronóstico en los últimos meses de datos y mide qué tan precisos fueron.
                        Luego elige el que menor error cometió. Es como una "competencia" de precisión.
                    </p>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("### ⚙️ Parámetros de Evaluación")
                c1, c2, c3 = st.columns([1, 1, 1.5])
                
                with c1:
                    # Auto-calcular 25% escalable
                    auto_test_months = max(6, int(len(hist_cmp) * 0.25))
                    test_months_cmp = auto_test_months
                    st.metric("Meses a evaluar", f"{test_months_cmp} (25% de {len(hist_cmp)})")
                
                with c2:
                    st.metric("Ventana MA", "Auto-optimizada")
                
                with c3:
                    metric_to_sort = st.selectbox(
                        "Criterio ganador",
                        options=["MAE", "RMSE", "sMAPE_%", "MAPE_safe_%"],
                        index=0,
                        key="cmp_sort_metric"
                    )

                st.divider()

                with st.spinner("🔄 Cargando resultados de backtests cacheados..."):
                    # Usar función cacheada para backtests
                    bt_base_cmp, bt_ets_cmp, bt_rf_cmp, cmp, bt_info = get_comparador_backtests(
                        prod_sel, 
                        test_months=int(test_months_cmp),
                        sort_metric=metric_to_sort
                    )
                
                if bt_base_cmp is None:
                    st.info("No hay serie mensual para ejecutar backtests.")
                else:
                    ma_window_cmp = bt_info["ma_window"]
                    mae_ma3 = bt_info["mae_ma3"]
                    mae_ma6 = bt_info["mae_ma6"]

                    # ========== RESULTADO PRINCIPAL ==========
                    winner = str(cmp.iloc[0]["Modelo"]) if not cmp.empty else "N/A"
                    
                    # Mostrar MA seleccionado
                    st.markdown(f"**✅ Ventana MA seleccionada:** MA{ma_window_cmp} (MAE: {min(mae_ma3, mae_ma6):.2f}) | MA3 MAE: {mae_ma3:.2f} | MA6 MAE: {mae_ma6:.2f}")
                    
                    # Destacar ganador visualmente
                    st.markdown(f"## 🥇 **Ganador: {winner}**")
                    
                    # ==================== EXPLICACIÓN DE MÉTRICAS ====================
                    with st.expander("📊 Entiende las Métricas", expanded=True):
                        col_exp1, col_exp2 = st.columns(2)
                        
                        with col_exp1:
                            st.markdown("""
                            **📈 MAE (Error Absoluto Medio)**
                            - **Qué es:** Promedio de errores en unidades (ej: 100 unidades de error)
                            - **Ejemplo:** Si MAE=100, el modelo se equivoca ±100 unidades en promedio
                            - **Relevancia:** ⭐⭐⭐⭐⭐ CRÍTICA - Usa esta métrica por defecto
                            - **Interpretación:** Números BAJOS = Buenos pronósticos
                            
                            **📊 RMSE (Raíz Error Cuadrático Medio)**
                            - **Qué es:** Similar a MAE pero penaliza más los errores grandes
                            - **Ejemplo:** RMSE=150 vs MAE=100 = hay algunos errores MUY grandes
                            - **Relevancia:** ⭐⭐⭐⭐ Buena para detectar outliers
                            - **Interpretación:** Si RMSE >> MAE, hay variabilidad en errores
                            """, unsafe_allow_html=True)
                        
                        with col_exp2:
                            st.markdown("""
                            **📉 sMAPE (Error Porcentual Simétrico)**
                            - **Qué es:** Error como porcentaje (0-100%) de la demanda real
                            - **Ejemplo:** sMAPE=10% = pronóstico fue ~10% diferente a lo real
                            - **Relevancia:** ⭐⭐⭐ Buena para ver % de error relativo
                            - **Interpretación:** <15% es EXCELENTE, 15-30% es BUENO, >30% es POBRE
                            
                            **🎯 MAPE_safe (Alternativa segura)**
                            - **Qué es:** Versión mejorada de sMAPE sin problemas con valores bajos
                            - **Ejemplo:** Cuando demanda es muy baja, MAPE_safe es más confiable
                            - **Relevancia:** ⭐⭐ Usar solo para demandas muy variables/bajas
                            - **Interpretación:** Similar a sMAPE pero más preciso
                            """, unsafe_allow_html=True)
                    
                    # Tabla de resultados con formato mejorado
                    st.markdown("### 🏆 Resultados de Comparison")
                    st.dataframe(cmp, use_container_width=True)
                    
                    # Explicación de la tabla
                    st.markdown("""
                    <div style='background: #f3e5f5; padding: 14px; border-left: 4px solid #9c27b0; border-radius: 8px; margin-top: 1em;'>
                        <p style='margin: 0; color: #333; font-size: 0.85em;'>
                            <strong>📋 Cómo leer la tabla:</strong><br>
                            • <strong>Rank:</strong> Posición (1 = mejor)<br>
                            • <strong>Modelo:</strong> Nombre del método de pronóstico<br>
                            • <strong>MAE:</strong> Error promedio en unidades - BUSCA EL MÁS BAJO<br>
                            • <strong>RMSE:</strong> Error más sensible a valores atípicos<br>
                            • <strong>sMAPE_%:</strong> Error en porcentaje - ÚTIL PARA ENTENDER EL CONTEXTO<br>
                            • <strong>N:</strong> Número de predicciones evaluadas
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                    # Plot ganador vs real
                    if not cmp.empty:
                        if winner == "ETS(Holt-Winters)":
                            pred_best = bt_ets_cmp.predictions[["Mes_target", "y_true", "ETS"]].rename(columns={"ETS": "y_pred"})
                        elif winner == "RandomForest":
                            pred_best = bt_rf_cmp.predictions[["Mes_target", "y_true", "RF"]].rename(columns={"RF": "y_pred"})
                        else:
                            pred_best = bt_base_cmp.predictions[["Mes_target", "y_true", winner]].rename(columns={winner: "y_pred"})

                        fig_best = px.line(
                            pred_best, x="Mes_target", y=["y_true", "y_pred"], markers=True,
                            title=f"Ganador vs Real (Backtest) - {winner} - Producto {prod_sel}",
                            labels={"y_true": "Demanda Real", "y_pred": "Pronóstico"}
                        )
                        st.plotly_chart(fig_best, use_container_width=True)
                        
                        # Explicación del gráfico
                        st.markdown("""
                        <div style='background: #e3f2fd; padding: 14px; border-left: 4px solid #1976d2; border-radius: 8px;'>
                            <p style='margin: 0; color: #333; font-size: 0.85em;'>
                                <strong>📈 Qué ves en el gráfico:</strong><br>
                                • <strong>Línea azul:</strong> Demanda REAL que sucedió en los últimos meses (realidad)<br>
                                • <strong>Línea naranja:</strong> Lo que el modelo PREDIJO (pronóstico)<br>
                                • <strong>Qué es BUENO:</strong> Las líneas casi se superponen (predicción cerca de la realidad)<br>
                                • <strong>Qué es MALO:</strong> Grandes separaciones entre líneas (predicción muy diferente)
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

                    st.divider()

                    # ========== EXPANDIBLES PARA DETALLES AVANZADOS ==========
                    st.markdown("### 📊 Detalles por Modelo (Usuarios Avanzados)")

                    with st.expander("📈 Baselines - Detalles y predicciones"):
                        st.markdown("**ℹ️ Baselines = Métodos Simples (MA3, MA6, Seasonal, Naive)**")
                        col1, col2 = st.columns([1, 1])
                        with col1:
                            st.markdown("**Métricas Baselines**")
                            st.dataframe(bt_base_cmp.metrics, use_container_width=True)
                        with col2:
                            st.markdown("**Predicciones Baselines (últimos 12 meses)**")
                            st.dataframe(
                                bt_base_cmp.predictions.tail(min(12, len(bt_base_cmp.predictions))),
                                use_container_width=True,
                                height=300
                            )
                        
                        # Gráfico Baselines
                        plot = bt_base_cmp.predictions.copy()
                        plot_long = plot.melt(
                            id_vars=["Mes_target", "y_true"],
                            value_vars=[c for c in plot.columns if c not in ["Mes_target", "y_true"]],
                            var_name="Modelo",
                            value_name="y_pred"
                        )
                        fig_base = px.line(
                            plot_long, x="Mes_target", y="y_pred", color="Modelo", markers=True,
                            title=f"Predicciones Baselines - Producto {prod_sel}"
                        )
                        st.plotly_chart(fig_base, use_container_width=True)

                    with st.expander("🌀 ETS (Holt-Winters) - Detalles y predicciones"):
                        st.markdown("""
                        **ℹ️ ETS = Modelo de Tendencia & Estacionalidad (detecta ciclos)**
                        
                        • **Qué hace:** Detecta patrones cíclicos en tus datos (ej: picos de demanda cada mes)
                        • **Cuándo usar:** Productos con demanda estacional o tendencias claras
                        • **Ventaja:** Muy bueno para detectar cambios graduales y ciclos regulares
                        • **Desventaja:** Lento si demanda es muy irregular o tiene shocks
                        """)
                        
                        col1, col2 = st.columns([1, 1])
                        with col1:
                            st.markdown("**Métricas ETS**")
                            st.dataframe(bt_ets_cmp.metrics, use_container_width=True)
                        with col2:
                            st.markdown("**Predicciones ETS (últimos 12 meses)**")
                            st.dataframe(
                                bt_ets_cmp.predictions.tail(min(12, len(bt_ets_cmp.predictions))),
                                use_container_width=True,
                                height=300
                            )
                        
                        fig_ets = px.line(
                            bt_ets_cmp.predictions, x="Mes_target", y=["y_true", "ETS"], markers=True,
                            title=f"ETS vs Real (Backtest) - Producto {prod_sel}",
                            labels={"y_true": "Demanda Real", "ETS": "Predicción ETS"}
                        )
                        st.plotly_chart(fig_ets, use_container_width=True)

                    with st.expander("🤖 Random Forest (RF) - Detalles y predicciones"):
                        st.markdown("""
                        **ℹ️ Random Forest = Aprendizaje Automático Avanzado (múltiples decisiones)**
                        
                        • **Qué hace:** Combinación de 100+ árboles de decisión que votan por la mejor predicción
                        • **Cuándo usar:** Datos complejos con múltiples patrones o relaciones ocultas
                        • **Ventaja:** Muy flexible, captura patrones complejos automáticamente
                        • **Desventaja:** Puede ser lento, más "caja negra" (difícil de interpretar)
                        """)
                        
                        col1, col2 = st.columns([1, 1])
                        with col1:
                            st.markdown("**Métricas RF**")
                            st.dataframe(bt_rf_cmp.metrics, use_container_width=True)
                        with col2:
                            st.markdown("**Predicciones RF (últimos 12 meses)**")
                            st.dataframe(
                                bt_rf_cmp.predictions.tail(min(12, len(bt_rf_cmp.predictions))),
                                use_container_width=True,
                                height=300
                            )
                        
                        fig_rf = px.line(
                            bt_rf_cmp.predictions, x="Mes_target", y=["y_true", "RF"], markers=True,
                            title=f"RF vs Real (Backtest) - Producto {prod_sel}",
                            labels={"y_true": "Demanda Real", "RF": "Predicción RF"}
                        )
                        st.plotly_chart(fig_rf, use_container_width=True)

                    st.divider()

                    # ========== RECOMENDACIONES FINALES ==========
                    st.markdown("### 🎯 Recomendaciones de Uso")
                    
                    if not cmp.empty:
                        winner_mae = cmp.iloc[0].get("MAE", 0)
                        winner_smape = cmp.iloc[0].get("sMAPE_%", 0)
                        
                        # Generar recomendación basada en métricas
                        recommendation = ""
                        if winner == "RandomForest":
                            recommendation = "✅ **Usar Random Forest:** Este producto tiene patrones complejos que los métodos simples no capturan bien. El ML avanzado es la mejor opción."
                        elif winner == "ETS(Holt-Winters)":
                            recommendation = "✅ **Usar ETS:** Este producto tiene ciclos estacionales claros. ETS detecta perfectamente estos patrones sin complejidad innecesaria."
                        else:
                            recommendation = f"✅ **Usar {winner}:** Este producto tiene demanda simple y predecible. Los métodos simples y rápidos son óptimos y eficientes."
                        
                        quality = ""
                        if winner_smape < 15:
                            quality = "🟢 **EXCELENTE:** Predicciones muy confiables (sMAPE < 15%)"
                        elif winner_smape < 30:
                            quality = "🟡 **BUENO:** Predicciones confiables pero revisar casos atípicos (sMAPE 15-30%)"
                        else:
                            quality = "🔴 **REVISAR:** Predicciones tienen margen de error significativo (sMAPE > 30%). Considerar manualmente"
                        
                        st.markdown(f"""
                        <div style='background: #fff3e0; padding: 14px; border-left: 4px solid #ff9800; border-radius: 8px; margin-top: 1em;'>
                            <p style='margin: 0.3em 0; color: #333;'>{recommendation}</p>
                            <p style='margin: 0.3em 0; color: #333;'>{quality}</p>
                            <p style='margin: 0.3em 0; color: #333; font-size: 0.9em;'>💡 <strong>Tip:</strong> Revisa el gráfico "Ganador vs Real". Si ves grandes separaciones, evalúa combinar este modelo con análisis manual para mayor seguridad.</p>
                        </div>
                        """, unsafe_allow_html=True)



        # ==========================================================
        # TAB 6: COMPARATIVA GLOBAL ETS VS BASELINES VS RF
        # ==========================================================
        with ResumenComparativa:
            # ==================== HEADER IMPACTANTE ====================
            st.markdown("""
            <div style='text-align: center; margin-bottom: 2em; background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 2.5em; border-radius: 12px; box-shadow: 0 4px 15px rgba(245, 87, 108, 0.3);'>
                <h1 style='color: white; font-size: 2.2em; margin: 0; font-weight: bold;'>🏆 Análisis de Mejor Modelo por Categoría</h1>
                <p style='color: rgba(255, 255, 255, 0.9); font-size: 1em; margin-top: 0.8em; margin-bottom: 0;'>¿Cuál es el mejor método de pronóstico para tus productos? Baselines vs ETS vs Random Forest</p>
            </div>
            """, unsafe_allow_html=True)

            # ==================== EXPLICACIÓN AMIGABLE ====================
            st.markdown("""
            <div style='background: linear-gradient(135deg, #e8f5e9 0%, #f1f8e9 100%); padding: 16px; border-left: 5px solid #4caf50; border-radius: 8px; margin-bottom: 2em;'>
                <h3 style='color: #2e7d32; margin-top: 0;'>📊 ¿Qué ves aquí?</h3>
                <p style='color: #333; margin: 0.5em 0;'><strong>Este análisis compara automáticamente 3 métodos de pronóstico</strong> en cada categoría ABC de tu portafolio:</p>
                <ul style='color: #333; margin: 0.5em 0; padding-left: 20px;'>
                    <li><strong>Baselines (MA3, MA6, Seasonal):</strong> Métodos simples y rápidos</li>
                    <li><strong>ETS (Holt-Winters):</strong> Captura tendencias y ciclos</li>
                    <li><strong>Random Forest:</strong> Aprendizaje automático avanzado</li>
                </ul>
                <p style='color: #333; margin: 0.5em 0;'>El sistema <strong>elige automáticamente el mejor</strong> para cada grupo ABC basado en precisión de errores.</p>
            </div>
            """, unsafe_allow_html=True)

            # ==================== PARÁMETROS (EXPANDIBLE) ====================
            with st.expander("⚙️ Parámetros de Análisis (avanzado)", expanded=False):
                col_p1, col_p2, col_p3 = st.columns([1, 1, 1.5])
                
                with col_p1:
                    sort_metric = st.selectbox(
                        "Métrica ganadora",
                        options=["MAE", "RMSE", "sMAPE_%", "MAPE_safe_%"],
                        index=0,
                        key="global_sort_metric",
                        help="Criterio usado para elegir el mejor modelo"
                    )
                
                with col_p2:
                    test_months_global = st.slider(
                        "Meses de backtest",
                        min_value=6, max_value=24, value=12, step=1,
                        key="global_test_months",
                        help="Últimos meses usados para evaluar cada modelo"
                    )
                
                with col_p3:
                    max_products = st.selectbox(
                        "Productos a analizar",
                        options=[20, 50, 100, 200, "Todos"],
                        index=0,
                        key="global_max_products",
                        help="Más productos = análisis más completo pero más lento"
                    )
                    max_products = None if max_products == "Todos" else int(max_products)
                
                st.info("✅ **Configuración automática:** Ventana MA optimizada (3 vs 6) + ETS con parámetros estándares")

            # ==================== BOTÓN EJECUTAR ====================
            run_btn = st.button("▶️ Ejecutar análisis global (puede tardar 1-2 min)", type="primary", use_container_width=True)

            if run_btn:
                ma_window_global = 3
                with st.spinner("⏳ Analizando todos los productos y comparando modelos... Por favor espera"):
                    per_sku, summary_wins, summary_errors = run_portfolio_comparison(
                        res_demand,
                        sort_metric=sort_metric,
                        test_months=int(test_months_global),
                        ma_window=int(ma_window_global),
                        ets_params=dict(seasonal_periods=12, trend="add", seasonal="add", damped_trend=False, min_obs=24),
                        max_products=max_products
                    )

                if per_sku.empty:
                    st.warning("❌ No se generaron resultados. Revisa los parámetros o la cantidad de datos.")
                else:
                    st.success(f"✅ Análisis completado: **{per_sku['Codigo'].nunique():,} productos** evaluados")
                    
                    # ==================== RESUMEN EJECUTIVO ====================
                    st.markdown("---")
                    st.markdown("### 🎯 Resumen Ejecutivo: Ganadores por Categoría ABC")
                    
                    # Tarjetas KPI de ganadores
                    if not summary_wins.empty:
                        cols_kpi = st.columns(3)
                        
                        for idx, row in summary_wins.iterrows():
                            with cols_kpi[idx % 3]:
                                # Extraer ABC correctamente del row (no del índice)
                                abc_class = row.get("ABC", "?")
                                modelo_ganador = row.get("Winner", "N/A")
                                cant_productos = row.get("N_Productos", 0)
                                
                                # Color según categoría ABC
                                if abc_class == "A":
                                    bg_color = "#ffebee"
                                    border_color = "#c62828"
                                    icon = "🔴"
                                elif abc_class == "B":
                                    bg_color = "#fff3e0"
                                    border_color = "#ef6c00"
                                    icon = "🟡"
                                else:
                                    bg_color = "#e8f5e9"
                                    border_color = "#2e7d32"
                                    icon = "🟢"
                                
                                st.markdown(f"""
                                <div style='
                                    background: {bg_color};
                                    border-left: 4px solid {border_color};
                                    padding: 16px;
                                    border-radius: 8px;
                                    text-align: center;
                                '>
                                    <p style='margin: 0; color: #666; font-size: 0.9em;'>{icon} <strong>Clase {abc_class}</strong></p>
                                    <h2 style='margin: 8px 0; color: {border_color}; font-size: 1.8em;'>{modelo_ganador}</h2>
                                    <p style='margin: 0; color: #888; font-size: 0.85em;'>Ganador en {int(cant_productos)} productos</p>
                                </div>
                                """, unsafe_allow_html=True)
                    
                    # ==================== COMPARATIVA DE ERRORES ====================
                    st.markdown("### 📉 Precisión del Ganador por Categoría")
                    
                    if not summary_errors.empty:
                        cols_err = st.columns(3)
                        
                        # Detectar el nombre de columnas de error (MAE_Promedio o MAE_promedio según versión)
                        error_cols = [c for c in summary_errors.columns if "Promedio" in c or "promedio" in c]
                        error_col_name = error_cols[0] if error_cols else None
                        pond_cols = [c for c in summary_errors.columns if "Ponderado" in c or "ponderado" in c]
                        pond_col_name = pond_cols[0] if pond_cols else None
                        
                        for idx, row in summary_errors.iterrows():
                            with cols_err[idx % 3]:
                                # Extraer ABC correctamente del row (no del índice)
                                abc_class = row.get("ABC", "?")
                                mae_prom = float(row.get(error_col_name, 0)) if error_col_name else 0
                                mae_pond = float(row.get(pond_col_name, 0)) if pond_col_name else 0
                                
                                # Determinar color de riesgo (menor error = mejor)
                                if mae_prom < 100:
                                    error_color = "#4caf50"
                                    error_status = "🟢 Excelente"
                                elif mae_prom < 300:
                                    error_color = "#ff9800"
                                    error_status = "🟡 Bueno"
                                else:
                                    error_color = "#f44336"
                                    error_status = "🔴 Revisar"
                                
                                st.markdown(f"""
                                <div style='
                                    background: #f9f9f9;
                                    border: 2px solid {error_color};
                                    padding: 14px;
                                    border-radius: 8px;
                                '>
                                    <p style='margin: 0; color: #666; font-size: 0.85em; font-weight: bold;'>Clase {abc_class} - Error Promedio</p>
                                    <h3 style='margin: 8px 0; color: {error_color};'>{mae_prom:.1f}</h3>
                                    <p style='margin: 4px 0; color: #888; font-size: 0.8em;'>{error_status}</p>
                                    <hr style='margin: 8px 0; border: none; border-top: 1px solid #ddd;'>
                                    <p style='margin: 4px 0; color: #888; font-size: 0.8em;'><strong>Ponderado:</strong> {mae_pond:.1f}</p>
                                </div>
                                """, unsafe_allow_html=True)
                    
                    st.divider()
                    
                    # ==================== TABLA DETALLADA ====================
                    st.markdown("### 📋 Detalle Completo: Todos los Productos")
                    
                    # Preparar tabla con formato
                    reco_display = per_sku.copy()
                    reco_display['Demanda_Total'] = reco_display['Demanda_Total'].apply(lambda x: f"{x:,.0f}")
                    
                    # Detectar columnas disponibles
                    cols_mostrar = ['Codigo', 'ABC', 'Winner', 'MAE', 'Demanda_Total', 'RMSE', 'sMAPE_%']
                    cols_disponibles = [c for c in cols_mostrar if c in reco_display.columns]
                    
                    st.dataframe(
                        reco_display[cols_disponibles].sort_values(['ABC', 'Demanda_Total'], ascending=[True, False]),
                        use_container_width=True,
                        height=450,
                        hide_index=True
                    )
                    
                    # ==================== INSIGHTS Y RECOMENDACIONES ====================
                    st.markdown("---")
                    st.markdown("### 💡 Insights y Recomendaciones")
                    
                    # Extraer insights de forma segura
                    if not summary_wins.empty and "Winner" in summary_wins.columns:
                        modelos_ganadores = summary_wins['Winner'].unique()
                        
                        insight_html = "<div style='background: #f5f5f5; padding: 16px; border-radius: 8px;'>"
                        
                        if any("ETS" in str(m) for m in modelos_ganadores):
                            insight_html += "<p style='margin: 0.5em 0;'>✅ <strong>ETS destaca en tu portafolio:</strong> Tus productos tienen tendencias y ciclos bien definidos. El método ETS captura bien estos patrones. Considera ver históricamente si hay cambios de producción/demanda que siguen ciclos.</p>"
                        
                        if any("RandomForest" in str(m) or "RF" in str(m) for m in modelos_ganadores):
                            insight_html += "<p style='margin: 0.5em 0;'>⚡ <strong>Random Forest gana en algunos casos:</strong> Algunos productos tienen patrones complejos que los métodos simples no capturan. Esto es normal en portafolios grandes y diversificados.</p>"
                        
                        if any(m in ["MA3", "MA6", "Seasonal12", "Naive"] for m in modelos_ganadores):
                            insight_html += "<p style='margin: 0.5em 0;'>📊 <strong>Métodos simples son efectivos:</strong> Algunos productos responden bien a modelos basados en promedios móviles. Esto indica demanda relativamente estable sin grandes sorpresas.</p>"
                        
                        insight_html += "</div>"
                        st.markdown(insight_html, unsafe_allow_html=True)


        # ==========================================================
        # TAB 7: STOCK + DIAGNÓSTICO
        # ==========================================================
        with tab_stock_diag:
            st.markdown("### 📦 Análisis de Stock")

            stock = res_stock
            if stock is None or stock.empty:
                st.warning("No se generó stock mensual (revisa columna Saldo_unid).")
            else:
                splot = stock[stock["Codigo"] == str(prod_sel)].copy()
                if splot.empty:
                    st.info("No hay stock mensual para ese producto.")
                else:
                    # ==================== RESUMEN EJECUTIVO STOCK ====================
                    if not splot.empty:
                        stock_promedio = float(splot["Stock_Unid"].mean())
                        stock_max = float(splot["Stock_Unid"].max())
                        stock_min = float(splot["Stock_Unid"].min())
                        stock_reciente = float(splot.iloc[-1]["Stock_Unid"])
                        stock_anterior = float(splot.iloc[-2]["Stock_Unid"]) if len(splot) > 1 else stock_reciente
                        cambio_stock = stock_reciente - stock_anterior
                        ocupacion_actual = (stock_reciente / stock_max * 100) if stock_max != 0 else 0
                        
                        # Calcular cobertura (días)
                        dm = res_demand.copy()
                        dm["Codigo"] = dm["Codigo"].astype(str).str.strip()
                        demanda_diaria = dm[dm["Codigo"] == str(prod_sel)]["Demanda_Unid"].sum() / len(dm) if not dm[dm["Codigo"] == str(prod_sel)].empty else 1
                        cobertura_dias = (stock_reciente / demanda_diaria) if demanda_diaria > 0 else 0
                        
                        # Mostrar KPIs
                        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                        
                        with col_s1:
                            st.metric(
                                "📈 Promedio",
                                f"{stock_promedio:,.0f} unid",
                                f"Rango: {stock_min:,.0f} a {stock_max:,.0f}"
                            )
                        
                        with col_s2:
                            cambio_emoji = "📈" if cambio_stock > 0 else "📉" if cambio_stock < 0 else "➡️"
                            st.metric(
                                f"{cambio_emoji} Stock Actual",
                                f"{stock_reciente:,.0f} unid",
                                f"{cambio_stock:+.0f} respecto mes anterior"
                            )
                        
                        with col_s3:
                            ocupacion_status = "🟢 Óptimo" if 30 <= ocupacion_actual <= 80 else "🟡 Revisar" if ocupacion_actual > 80 else "🔴 Bajo"
                            st.metric(
                                "🎯 Ocupación",
                                f"{ocupacion_actual:.1f}%",
                                ocupacion_status
                            )
                        
                        with col_s4:
                            cobertura_status = "🟢 Bien" if cobertura_dias >= 7 else "🟡 Justo" if cobertura_dias >= 3 else "🔴 Crítico"
                            st.metric(
                                "⏱️ Cobertura",
                                f"{cobertura_dias:.1f} días",
                                cobertura_status
                            )
                        
                        # Explicación visual
                        st.markdown("""
                        <div style='background: linear-gradient(135deg, #f3e5f5 0%, #f5f5f5 100%); padding: 14px; border-left: 4px solid #9c27b0; border-radius: 8px; margin-bottom: 1em;'>
                            <p style='margin: 0; color: #333; font-size: 0.9em;'>
                                <strong>💡 Qué observar:</strong><br>
                                • <strong>Stock Actual:</strong> Cantidad disponible ahora. Si baja mucho = riesgo de quiebre.<br>
                                • <strong>Ocupación:</strong> % del máximo histórico. 30-80% es ideal para no exceso ni falta.<br>
                                • <strong>Cobertura (días):</strong> Cuántos días de demanda cubre el stock actual. Mínimo recomendado: 7-10 días.<br>
                                • <strong>Tendencia:</strong> Si stock crece = producción > demanda. Si baja = demanda > producción.
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Gráfico con línea de referencia
                    fig_stock = px.line(
                        splot, x="Mes", y="Stock_Unid", markers=True,
                        title=f"Stock mensual histórico - Producto {prod_sel}",
                        height=400
                    )
                    fig_stock.add_hline(y=stock_promedio, line_dash="dash", line_color="orange", 
                                       annotation_text=f"Promedio: {stock_promedio:,.0f}", annotation_position="right")
                    st.plotly_chart(fig_stock, use_container_width=True)
                    
                    # Análisis de volatilidad
                    with st.expander("📊 Análisis Detallado de Stock", expanded=False):
                        col_a1, col_a2 = st.columns(2)
                        
                        with col_a1:
                            volatilidad_stock = float(splot["Stock_Unid"].std())
                            coeficiente_var = (volatilidad_stock / stock_promedio * 100) if stock_promedio > 0 else 0
                            
                            st.markdown(f"""
                            **📉 Volatilidad:**
                            - Desv. Estándar: {volatilidad_stock:,.0f} unid
                            - Coef. Variación: {coeficiente_var:.1f}%
                            
                            **Status:** {'🔴 Alta variabilidad' if coeficiente_var > 50 else '🟡 Moderada' if coeficiente_var > 25 else '🟢 Estable'}
                            """)
                        
                        with col_a2:
                            meses_bajo_promedio = len(splot[splot["Stock_Unid"] < stock_promedio])
                            pct_bajo = (meses_bajo_promedio / len(splot) * 100) if len(splot) > 0 else 0
                            
                            st.markdown(f"""
                            **📈 Distribución:**
                            - Meses bajo promedio: {meses_bajo_promedio} ({pct_bajo:.1f}%)
                            - Stock máximo: {stock_max:,.0f} unid
                            - Stock mínimo: {stock_min:,.0f} unid
                            - Rango total: {stock_max - stock_min:,.0f} unid
                            """)

            st.divider()

            # (Diagnóstico de guías de remisión movido a tab Demanda y Componentes)

        # ==========================================================
        # TAB 8: RECOMENDACIÓN DE PRODUCCIÓN
        # ==========================================================

        with tab_reco:
            # Data mensual para el producto
            dm = res_demand.copy()
            dm["Codigo"] = dm["Codigo"].astype(str).str.strip()
            hist = dm[dm["Codigo"] == str(prod_sel)][["Mes", "Demanda_Unid"]].copy().sort_values("Mes")

            # Calcular el mes predicho (t+1) con formato legible
            if not hist.empty:
                last_mes = hist.iloc[-1]["Mes"]
                next_mes = last_mes + pd.DateOffset(months=1)
                months_es = {
                    1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril",
                    5: "Mayo", 6: "Junio", 7: "Julio", 8: "Agosto",
                    9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre"
                }
                mes_nombre = months_es.get(next_mes.month, "")
                predicted_month_str = f"{mes_nombre} {next_mes.year}" if mes_nombre else "Mes siguiente"
                st.subheader(f"🧾 Recomendación de producción - {predicted_month_str}")
            else:
                st.subheader("🧾 Recomendación de producción")

            if hist.empty:
                st.info("No hay serie mensual para este producto.")
            else:
                # Stock actual (último mes disponible)
                stock = res_stock.copy() if res_stock is not None else pd.DataFrame()
                stock_actual = 0.0
                if not stock.empty:
                    stock["Codigo"] = stock["Codigo"].astype(str).str.strip()
                    splot = stock[stock["Codigo"] == str(prod_sel)].copy().sort_values("Mes")
                    if not splot.empty:
                        stock_actual = float(splot.iloc[-1]["Stock_Unid"])

                # ABC (calculado por demanda total)
                abc_df = build_abc_from_demand(dm)
                abc_row = abc_df[abc_df["Codigo"] == str(prod_sel)]
                abc_class = str(abc_row.iloc[0]["ABC"]) if not abc_row.empty else "C"

                # Parámetros de política
                lead_time = 1  # Parámetro operacional fijo
                service_level = policy_service_level_by_abc(abc_class)
                z = z_from_service_level(service_level)

                # Parámetros de evaluación para elegir ganador (automáticos para máxima comparabilidad)
                test_months = max(6, int(len(hist) * 0.25))
                
                # Auto-optimizar MA (3 vs 6) evaluando Baselines
                bt_ma3 = backtest_baselines_1step(hist, y_col="Demanda_Unid", test_months=int(test_months), ma_window=3)
                bt_ma6 = backtest_baselines_1step(hist, y_col="Demanda_Unid", test_months=int(test_months), ma_window=6)
                mae_ma3 = float(bt_ma3.metrics.iloc[0]["MAE"]) if not bt_ma3.metrics.empty else float("inf")
                mae_ma6 = float(bt_ma6.metrics.iloc[0]["MAE"]) if not bt_ma6.metrics.empty else float("inf")
                ma_window = 3 if mae_ma3 < mae_ma6 else 6

                ets_params = dict(seasonal_periods=12, trend="add", seasonal="add", damped_trend=False, min_obs=24)
                rf_params = dict(n_estimators=400, min_obs=24, min_samples_leaf=1, random_state=42)

                # Seleccionar ganador y extraer MAE del ganador
                cmp, winner, pred_best, winner_mae = select_winner_and_backtests_for_product(
                    hist=hist,
                    test_months=int(test_months),
                    ma_window=int(ma_window),
                    ets_params=ets_params,
                    rf_params=rf_params,
                    sort_metric="MAE",  # ganador por defecto MAE
                )





                if cmp.empty or winner == "N/A" or np.isnan(winner_mae):
                    st.warning("No se pudo seleccionar un modelo ganador (revisa longitud de serie).")
                else:
                    # Pronóstico t+1 con el ganador
                    yhat = forecast_next_month_with_winner(hist, winner, int(ma_window), ets_params, rf_params)

                    # Stock de seguridad: NEWSVENDOR dinámico por costos
                    sigma = float(max(0.0, winner_mae))
                    
                    # Obtener costos del session_state (sincronizados desde Comparativa Retrospectiva)
                    # Si no existen, usar defaults conservadores
                    cost_stock_unit_reco = st.session_state.get("sync_cost_stock_unit", 1.0)
                    cost_stockout_unit_reco = st.session_state.get("sync_cost_stockout_unit", 5.0)
                    
                    # ==================== NEWSVENDOR: dinámico por costos ====================
                    from src.services.ml_service import calculate_safety_stock_newsvendor
                    ss = calculate_safety_stock_newsvendor(
                        cost_stockout=cost_stockout_unit_reco,
                        cost_inv=cost_stock_unit_reco,
                        sigma=sigma,
                        lead_time=lead_time
                    )

                    # Producción recomendada
                    prod_reco = max(0.0, yhat + ss - float(stock_actual))
                    prod_reco_int = int(np.ceil(prod_reco))

                    # =================================================================
                    # DISEÑO AMIGABLE PARA CLIENTE (Sin términos técnicos)
                    # =================================================================
                    
                    st.markdown("---")
                    
                    # SECCIÓN PRINCIPAL: LO MÁS IMPORTANTE
                    st.markdown(f"""
                    <div style='
                        background: linear-gradient(135deg, #1976D2 0%, #1565C0 100%);
                        padding: 30px;
                        border-radius: 10px;
                        text-align: center;
                        color: white;
                        margin-bottom: 40px;
                    '>
                        <h2 style='margin: 0; font-size: 1.2em; opacity: 0.9;'>Cantidad a Producir en {predicted_month_str.upper()}</h2>
                        <h1 style='margin: 15px 0 0 0; font-size: 3.5em; font-weight: bold;'>{prod_reco_int:,.0f}</h1>
                        <p style='margin: 10px 0 0 0; font-size: 1.1em; opacity: 0.95;'>unidades</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # CONTEXTO: La información que el cliente necesita entender
                    st.markdown("### 📊 Contexto de la Recomendación")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "📈 Se espera vender",
                            f"{yhat:,.0f}",
                            delta="unidades el próximo mes"
                        )
                        st.caption("Pronóstico de demanda basado en histórico")
                    
                    with col2:
                        st.metric(
                            "📦 Stock actual",
                            f"{stock_actual:,.0f}",
                            delta="unidades disponibles"
                        )
                        st.caption("Inventario al final del mes actual")
                    
                    with col3:
                        st.metric(
                            "🛡️ Stock de seguridad",
                            f"{ss:,.0f}",
                            delta="unidades recomendadas"
                        )
                        st.caption("Colchón para posibles sorpresas")
                    
                    # EXPLICACIÓN NARRATIVA
                    st.markdown("### 💡 ¿Por qué esta cantidad?")
                    
                    explanation = f"""
                    <div style='
                        background-color: #F5F5F5;
                        padding: 20px;
                        border-left: 4px solid #1976D2;
                        border-radius: 5px;
                        line-height: 1.8;
                    '>
                    
                    <p><strong>Demanda esperada:</strong> {yhat:,.0f} unidades</p>
                    <p style='margin-top: 12px;'><strong>+ Stock de seguridad:</strong> {ss:,.0f} unidades 
                    <br><span style='font-size: 0.95em; color: #666;'>(Protección contra cambios inesperados en la demanda)</span></p>
                    
                    <p style='margin-top: 12px;'><strong>- Stock disponible:</strong> {stock_actual:,.0f} unidades
                    <br><span style='font-size: 0.95em; color: #666;'>(Inventario que ya tienes)</span></p>
                    
                    <p style='margin-top: 20px; padding-top: 15px; border-top: 1px solid #DDD;'>
                    <strong style='font-size: 1.1em; color: #1565C0;'>= {prod_reco_int:,.0f} unidades a producir</strong>
                    </p>
                    </div>
                    """
                    st.markdown(explanation, unsafe_allow_html=True)
                    
                    # INFORMACIÓN TÉCNICA ADICIONAL (desplegable)
                    with st.expander("📋 Detalles técnicos (Variables y cálculos)", expanded=False):
                        st.markdown("#### 📖 Significado de cada variable:")
                        
                        # Obtener costos si existen
                        cost_stock_reco_show = st.session_state.get("sync_cost_stock_unit", None)
                        cost_stockout_reco_show = st.session_state.get("sync_cost_stockout_unit", None)
                        
                        if cost_stock_reco_show is not None and cost_stockout_reco_show is not None:
                            # Mostrar método Newsvendor
                            ratio_newsvendor = cost_stockout_reco_show / (cost_stockout_reco_show + cost_stock_reco_show)
                            st.info("🎯 **Método: Newsvendor (costo-óptimo)**\n\nEl stock de seguridad se calcula balanceando costos de inventario vs quiebres de stock.")
                            
                            sigma_show = float(max(0.0, winner_mae))
                            z_dynamic_show = float(norm.ppf(np.clip(ratio_newsvendor, 0.001, 0.999)))
                            
                            st.markdown(f"""
                            **Producto:** {prod_sel} 
                            > El código/nombre del artículo que estás analizando
                            
                            **ABC:** {abc_class}
                            > Clasificación de importancia basada en demanda total. A=Críticos, B=Importantes, C=Bajos
                            
                            **Modelo ganador:** {winner}
                            > El método de pronóstico que mejor predice tu demanda histórica
                            
                            **Lead time:** {lead_time} mes
                            > Tiempo que tarda la producción desde que la ordenas
                            
                            **Error promedio (MAE):** {sigma_show:,.2f}
                            > Cuánto se desvía el modelo en promedio
                            
                            **Costos de la política:**
                            > - Costo mantener inventario: {cost_stock_reco_show:.2f} por unidad
                            > - Costo quiebre/venta perdida: {cost_stockout_reco_show:.2f} por unidad
                            > - Ratio crítico: {ratio_newsvendor:.2%}
                            
                            **Factor de seguridad dinámico (Z):** ≈{z_dynamic_show:.2f}
                            > Se calcula automáticamente del balance de costos. Mayor cost_stockout → mayor Z
                            
                            **Inversión en stock de seguridad:** {ss:,.0f}
                            > Unidades extra según el balance óptimo de costos
                            
                            **Producción recomendada:** {prod_reco_int:,.0f}
                            > = Demanda esperada ({yhat:,.0f}) + Stock de seguridad ({ss:,.0f}) - Stock actual
                            """)
                        else:
                            # Mostrar método legacy (service level)
                            st.info("ℹ️ **Método: Service Level (basado en ABC)**\n\nEjecuta 'Comparativa Retrospectiva' para activar método Newsvendor optimizado por costos.")
                            
                            sigma_show_legacy = float(max(0.0, winner_mae))
                            st.markdown(f"""
                            **Producto:** {prod_sel} 
                            > El código/nombre del artículo que estás analizando
                            
                            **ABC:** {abc_class}
                            > Clasificación de importancia basada en demanda total. A=Críticos (95%), B=Importantes (90%), C=Bajos (85%)
                            
                            **Modelo ganador:** {winner}
                            > El método de pronóstico que mejor predice tu demanda histórica
                            
                            **Lead time:** {lead_time} mes
                            > Tiempo que tarda la producción desde que la ordenas
                            
                            **Nivel de servicio (por ABC):** {int(service_level*100)}%
                            > % de veces que logras satisfacer la demanda
                            
                            **Factor de seguridad (Z):** {z:.2f}
                            > Cuántas desviaciones estándar añades al pronóstico (fijo por ABC)
                            
                            **Error promedio (MAE):** {sigma_show_legacy:,.2f}
                            > Cuánto se desvía el modelo en promedio
                            
                            **Inversión en stock de seguridad:** {ss:,.0f}
                            > Unidades extra según el nivel de servicio del ABC
                            
                            **Producción recomendada:** {prod_reco_int:,.0f}
                            > = Demanda esperada ({yhat:,.0f}) + Stock de seguridad ({ss:,.0f}) - Stock actual
                            """)
                        
                        st.divider()
                        st.markdown("**Comparación detallada de modelos (backtest):")
                        st.dataframe(cmp, use_container_width=True)
                    
                    with st.expander("📈 Validación del modelo (gráfico)", expanded=False):
                        fig = px.line(pred_best, x="Mes_target", y=["y_true", "y_pred"], 
                                    markers=True,
                                    title=f"Precisión histórica: {winner}",
                                    labels={"y_true": "Demanda Real", "y_pred": "Predicción"})
                        fig.update_layout(hovermode="x unified", template="plotly_white")
                        st.plotly_chart(fig, use_container_width=True)


        # ==========================================================
        # TAB 9: RECOMENDACIÓN MASIVA
        # ==========================================================
        with Reco_Masiva:
            st.subheader("📋 Recomendación masiva (según ABC seleccionado)")

            dm = res_demand.copy()
            dm["Codigo"] = dm["Codigo"].astype(str).str.strip()
            dm = dm.sort_values(["Codigo", "Mes"])

                # --- Productos a evaluar según ABC seleccionado ---
            if abc_sel == "Todos":
                codigos_eval = abc_df["Codigo"].dropna().astype(str).str.strip().unique().tolist()
            else:
                codigos_eval = abc_df[abc_df["ABC"] == abc_sel]["Codigo"].dropna().astype(str).str.strip().unique().tolist()

            codigos_eval = sorted(codigos_eval)

            if not codigos_eval:
                st.info("No hay productos para la categoría ABC seleccionada.")
            else:
                c1, c2, c3 = st.columns([1, 1, 2])

                with c1:
                    lead_time = 1  # Parámetro operacional fijo
                    st.metric("Lead time (meses)", lead_time)

                with c2:
                        # Calcular 25% del máximo histórico disponible
                    max_hist_months = res_demand.groupby('Codigo')['Mes'].count().max() if not res_demand.empty else 24
                    test_months = max(6, int(max_hist_months * 0.25))
                    st.info(f"✅ Usando **{test_months} meses** para backtest (25% del histórico máximo)")

                with c3:
                    max_products = st.selectbox(
                        "Cantidad de productos a procesar (performance)",
                        options=[20, 50, 100, 200, "Todos"],
                        index=1,
                        key="mass_max"
                    )
                    max_products = None if max_products == "Todos" else int(max_products)

                run_btn = st.button("▶️ Generar recomendación masiva", type="primary", key="run_mass")

                if run_btn:
                    with st.spinner("Calculando recomendaciones (puede tardar según la cantidad de productos)..."):

                            # Limitar por performance (prioriza los de mayor demanda total)
                        abc_work = abc_df.copy()
                        abc_work["Codigo"] = abc_work["Codigo"].astype(str).str.strip()

                        if abc_sel != "Todos":
                            abc_work = abc_work[abc_work["ABC"] == abc_sel].copy()

                        abc_work = abc_work.sort_values("Demanda_Total", ascending=False)
                        codigos = abc_work["Codigo"].tolist()
                        if max_products is not None:
                            codigos = codigos[:max_products]

                            # ETS y RF params (estables)
                        ets_params = dict(seasonal_periods=12, trend="add", seasonal="add", damped_trend=False, min_obs=24)
                        rf_params = dict(n_estimators=400, min_obs=24, min_samples_leaf=1, random_state=42)

                            # Obtener costos del session_state (sincronizados desde Comparativa Retrospectiva)
                        cost_stock_unit_mass = st.session_state.get("sync_cost_stock_unit", 1.0)
                        cost_stockout_unit_mass = st.session_state.get("sync_cost_stockout_unit", 5.0)

                            # Stock mensual
                        stock = res_stock.copy() if res_stock is not None else pd.DataFrame()
                        if not stock.empty:
                            stock["Codigo"] = stock["Codigo"].astype(str).str.strip()
                            stock = stock.sort_values(["Codigo", "Mes"])

                        rows = []
                        for cod in codigos:
                            hist = dm[dm["Codigo"] == str(cod)][["Mes", "Demanda_Unid"]].copy().sort_values("Mes")
                            if hist.empty:
                                continue

                                # Elegir ganador por MAE (baselines + ETS + RF) y extraer MAE del ganador
                                # IMPORTANTE: Optimizar MA_WINDOW (3 vs 6) igual que en Análisis Individual
                            bt_ma3 = backtest_baselines_1step(hist, y_col="Demanda_Unid", test_months=int(test_months), ma_window=3)
                            bt_ma6 = backtest_baselines_1step(hist, y_col="Demanda_Unid", test_months=int(test_months), ma_window=6)
                            
                            mae_ma3 = float(bt_ma3.metrics.iloc[0]["MAE"]) if not bt_ma3.metrics.empty else float("inf")
                            mae_ma6 = float(bt_ma6.metrics.iloc[0]["MAE"]) if not bt_ma6.metrics.empty else float("inf")
                            ma_window = 3 if mae_ma3 < mae_ma6 else 6
                            bt_base = bt_ma3 if ma_window == 3 else bt_ma6
                            
                            ets = ETSForecaster(**ets_params)
                            bt_ets = backtest_ets_1step(hist, y_col="Demanda_Unid", test_months=int(test_months), ets=ets)
                            rf = RFForecaster(**rf_params)
                            bt_rf = backtest_rf_1step(hist, y_col="Demanda_Unid", test_months=int(test_months), rf=rf)

                            cmp = compare_models_metrics(bt_base.metrics, bt_ets.metrics, bt_rf.metrics, sort_by="MAE")
                            if cmp.empty:
                                continue

                            winner = str(cmp.iloc[0]["Modelo"])
                            mae_win = float(pd.to_numeric(cmp.iloc[0].get("MAE", np.nan), errors="coerce"))

                                # Forecast t+1 con el ganador
                            if winner == "ETS(Holt-Winters)":
                                yhat = float(max(0.0, ets.forecast_1step(hist, y_col="Demanda_Unid")))
                            elif winner == "RandomForest":
                                yhat = float(max(0.0, rf.forecast_1step(hist, y_col="Demanda_Unid")))
                            elif winner == "Seasonal12":
                                yhat = float(max(0.0, seasonal_naive_12(hist)))
                            elif winner in ("MA3", "MA6"):
                                w = 3 if winner == "MA3" else 6
                                yhat = float(max(0.0, moving_average(hist, window=w)))
                            else:  # Naive fallback
                                yhat = float(max(0.0, naive_last(hist)))

                                # Stock actual (último)
                            stock_actual = 0.0
                            if not stock.empty:
                                splot = stock[stock["Codigo"] == str(cod)]
                                if not splot.empty:
                                    stock_actual = float(splot.iloc[-1]["Stock_Unid"])

                                # ABC + política Z (CRÍTICO: calcular z para cada producto como en Análisis Individual)
                            row_abc = abc_work[abc_work["Codigo"] == str(cod)]
                            abc_class = str(row_abc.iloc[0]["ABC"]) if not row_abc.empty else "C"
                            demanda_total = float(row_abc.iloc[0]["Demanda_Total"]) if not row_abc.empty else 0.0
                            
                            # Calcular Z dinámicamente según la clase ABC (esto faltaba y causaba NameError)
                            service_level_prod = policy_service_level_by_abc(abc_class)
                            z_prod = z_from_service_level(service_level_prod)

                                # SS con NEWSVENDOR dinámico (igual que Análisis Individual)
                            sigma = float(max(0.0, mae_win if np.isfinite(mae_win) else 0.0))
                            
                            # Importar función Newsvendor (para consistencia)
                            from src.services.ml_service import calculate_safety_stock_newsvendor
                            ss = calculate_safety_stock_newsvendor(
                                cost_stockout=cost_stockout_unit_mass,
                                cost_inv=cost_stock_unit_mass,
                                sigma=sigma,
                                lead_time=lead_time
                            )

                                # Producción recomendada
                            prod_reco = max(0.0, yhat + ss - stock_actual)
                            prod_reco_int = int(np.ceil(prod_reco))

                            rows.append({
                                "Codigo": str(cod),
                                "ABC": abc_class,
                                "Modelo_Ganador": winner,
                                "Forecast_t+1": yhat,
                                "MAE_ganador": sigma,
                                "Z": z_prod,
                                "SS": ss,
                                "Stock_Actual": stock_actual,
                                "Produccion_Recomendada": prod_reco_int,
                                "RIESGO_QUIEBRE": bool(stock_actual < ss),
                                "DEMANDA_TOTAL_HIST": demanda_total,
                            })

                        reco_df = pd.DataFrame(rows)

                    if reco_df.empty:
                        st.warning("No se generaron recomendaciones (revisa parámetros / data).")
                    else:
                        st.success(f"✅ Recomendación generada para {reco_df['Codigo'].nunique():,} productos.")

                            # Orden sugerido: primero riesgo quiebre, luego producción recomendada, luego demanda total
                        reco_df = reco_df.sort_values(
                            ["RIESGO_QUIEBRE", "Produccion_Recomendada", "DEMANDA_TOTAL_HIST"],
                            ascending=[False, False, False]
                        ).reset_index(drop=True)

                            # KPIs rápidos
                        k1, k2, k3 = st.columns(3)
                        # ==================== RESUMEN EJECUTIVO ====================
                        # Calcular mes siguiente para contexto
                        if not dm.empty:
                            last_mes = pd.to_datetime(dm["Mes"]).max()
                            next_mes = last_mes + pd.DateOffset(months=1)
                            months_es = {
                                1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril",
                                5: "Mayo", 6: "Junio", 7: "Julio", 8: "Agosto",
                                9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre"
                            }
                            mes_nombre = months_es.get(next_mes.month, "próximo mes")
                            next_mes_str = f"{mes_nombre} {next_mes.year}"
                        else:
                            next_mes_str = "próximo mes"
                        
                        st.markdown(f"### 📅 Recomendaciones para: **{next_mes_str}**")
                        st.markdown(f"Basadas en pronósticos automáticos (modelo ganador por producto) y niveles de stock actual")
                        
                        st.divider()
                        
                        # ==================== TARJETAS DE RESUMEN GLOBAL ====================
                        st.markdown("#### 📊 Resumen Global")
                        r1, r2, r3, r4 = st.columns(4)
                        
                        with r1:
                            st.markdown(f"""
                            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                        padding: 20px; border-radius: 10px; text-align: center; 
                                        box-shadow: 0 4px 12px rgba(0,0,0,0.1);'>
                                <p style='margin: 0; font-size: 0.85em; color: rgba(255,255,255,0.8);'>Productos Analizados</p>
                                <p style='margin: 10px 0 0 0; font-size: 2.5em; font-weight: bold; color: white;'>{reco_df['Codigo'].nunique():,}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with r2:
                            riesgo_count = int(reco_df['RIESGO_QUIEBRE'].sum())
                            color_riesgo = "#ef4444" if riesgo_count > 0 else "#10b981"
                            st.markdown(f"""
                            <div style='background: linear-gradient(135deg, {color_riesgo} 0%, {color_riesgo}CC 100%); 
                                        padding: 20px; border-radius: 10px; text-align: center; 
                                        box-shadow: 0 4px 12px rgba(0,0,0,0.1);'>
                                <p style='margin: 0; font-size: 0.85em; color: rgba(255,255,255,0.8);'>⚠️ Con Riesgo de Quiebre</p>
                                <p style='margin: 10px 0 0 0; font-size: 2.5em; font-weight: bold; color: white;'>{riesgo_count}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with r3:
                            prod_total = int(reco_df['Produccion_Recomendada'].sum())
                            st.markdown(f"""
                            <div style='background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%); 
                                        padding: 20px; border-radius: 10px; text-align: center; 
                                        box-shadow: 0 4px 12px rgba(0,0,0,0.1);'>
                                <p style='margin: 0; font-size: 0.85em; color: rgba(255,255,255,0.8);'>Producción Total Sugerida</p>
                                <p style='margin: 10px 0 0 0; font-size: 2.5em; font-weight: bold; color: white;'>{prod_total:,}</p>
                                <p style='margin: 5px 0 0 0; font-size: 0.75em; color: rgba(255,255,255,0.7);'>unidades</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with r4:
                            promedio_prod = int(reco_df['Produccion_Recomendada'].mean())
                            st.markdown(f"""
                            <div style='background: linear-gradient(135deg, #06b6d4 0%, #0891b2 100%); 
                                        padding: 20px; border-radius: 10px; text-align: center; 
                                        box-shadow: 0 4px 12px rgba(0,0,0,0.1);'>
                                <p style='margin: 0; font-size: 0.85em; color: rgba(255,255,255,0.8);'>Producción Promedio</p>
                                <p style='margin: 10px 0 0 0; font-size: 2.5em; font-weight: bold; color: white;'>{promedio_prod:,}</p>
                                <p style='margin: 5px 0 0 0; font-size: 0.75em; color: rgba(255,255,255,0.7);'>por producto</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        st.divider()
                        
                        # ==================== TOP PRODUCTOS PRIORITARIOS ====================
                        st.markdown("#### 🎯 Top Productos Prioritarios (Ordena por: Riesgo → Producción → Demanda)")
                        
                        # Top 10 prioritarios
                        top_n = min(10, len(reco_df))
                        reco_top = reco_df.head(top_n).copy()
                        
                        # Convertir booleano a string para Plotly
                        reco_top['Estado_Riesgo'] = reco_top['RIESGO_QUIEBRE'].apply(
                            lambda x: '🚨 RIESGO' if x else '✅ Seguro'
                        )
                        
                        col_viz1, col_viz2 = st.columns(2)
                        
                        with col_viz1:
                            # Gráfico: Top productos por Producción Recomendada
                            reco_top_sorted = reco_top.sort_values('Produccion_Recomendada', ascending=True).reset_index(drop=True)
                            
                            # Debug: verificar datos antes de plotear
                            if reco_top_sorted.empty:
                                st.warning("Sin datos para graficar producción recomendada")
                            else:
                                try:
                                    fig_prod = px.bar(
                                        reco_top_sorted,
                                        x='Produccion_Recomendada',
                                        y='Codigo',
                                        color='Estado_Riesgo',
                                        orientation='h',
                                        color_discrete_map={'🚨 RIESGO': '#ef4444', '✅ Seguro': '#10b981'},
                                        title=f'Top {top_n}: Producción Recomendada',
                                        labels={'Produccion_Recomendada': 'Unidades', 'Codigo': 'Producto', 'Estado_Riesgo': 'Estado'}
                                    )
                                    fig_prod.update_layout(height=450, font=dict(size=11), showlegend=True)
                                    st.plotly_chart(fig_prod, use_container_width=True)
                                except Exception as e:
                                    st.error(f"Error al generar gráfico de producción: {str(e)}")
                                    st.write("Datos disponibles:", reco_top_sorted.columns.tolist())
                                    st.write("Primeras filas:", reco_top_sorted.head())
                        
                        with col_viz2:
                            # Gráfico: Stock Actual vs SS (Stock de Seguridad)
                            reco_top_viz = reco_top[['Codigo', 'Stock_Actual', 'SS']].copy().reset_index(drop=True)
                            
                            if reco_top_viz.empty:
                                st.warning("Sin datos para graficar stock")
                            else:
                                try:
                                    fig_stock_ss = px.bar(
                                        reco_top_viz,
                                        x='Codigo',
                                        y=['Stock_Actual', 'SS'],
                                        barmode='group',
                                        title=f'Top {top_n}: Stock Actual vs Stock de Seguridad',
                                        labels={'value': 'Unidades', 'variable': 'Tipo'},
                                        color_discrete_map={'Stock_Actual': '#06b6d4', 'SS': '#f59e0b'}
                                    )
                                    fig_stock_ss.update_layout(height=450, font=dict(size=11))
                                    st.plotly_chart(fig_stock_ss, use_container_width=True)
                                except Exception as e:
                                    st.error(f"Error al generar gráfico de stock: {str(e)}")
                            
                            # Explicación para el cliente (fuera del try/except)
                            st.markdown("""
                            <div style='background: #f0f7ff; padding: 12px; border-left: 4px solid #06b6d4; border-radius: 4px; font-size: 0.9em; margin-top: 10px;'>
                            <strong>💡 Interpretación Clave:</strong><br>
                            <strong style='color: #06b6d4;'>Cyan (Stock Actual):</strong> Inventario que tienes HOY<br>
                            <strong style='color: #f59e0b;'>Naranja (Stock de Seguridad):</strong> Mínimo recomendado para protegerte de quiebres<br><br>
                            <strong>Qué significa:</strong><br>
                            ✅ Si <strong>Cyan > Naranja:</strong> Tienes colchón, sin riesgo inmediato<br>
                            🚨 Si <strong>Cyan < Naranja:</strong> Estás bajo el nivel seguro - necesitas producción URGENTE<br>
                            ⚠️ Si <strong>Cyan ≈ Naranja:</strong> Punto crítico, pequeños cambios en demanda causan quiebre
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # ==================== TABLA DETALLADA CON EXPLICACIONES ====================
                        st.markdown("#### 📋 Tabla Detallada: Todos los Productos")
                        st.markdown("**Ordenamiento:** 🚨 Productos en riesgo primero → Mayor producción recomendada → Mayor demanda histórica")
                        
                        # Preparar tabla para display (con formato)
                        reco_display = reco_df.copy()
                        reco_display['Estado'] = reco_display['RIESGO_QUIEBRE'].apply(
                            lambda x: '🚨 RIESGO' if x else '✅ Seguro'
                        )
                        reco_display['Produccion_Recomendada'] = reco_display['Produccion_Recomendada'].apply(
                            lambda x: f"{int(x):,} unid."
                        )
                        reco_display['Forecast_t+1'] = reco_display['Forecast_t+1'].apply(
                            lambda x: f"{int(x):,.0f}"
                        )
                        reco_display['Stock_Actual'] = reco_display['Stock_Actual'].apply(
                            lambda x: f"{int(x):,.0f}"
                        )
                        reco_display['SS'] = reco_display['SS'].apply(
                            lambda x: f"{x:.0f}"
                        )
                        
                        cols_display = ['Codigo', 'ABC', 'Estado', 'Modelo_Ganador', 'Forecast_t+1', 
                                       'Stock_Actual', 'SS', 'MAE_ganador', 'Z', 'Produccion_Recomendada']
                        
                        st.dataframe(reco_display[cols_display], use_container_width=True, height=520)
                        
                        st.divider()
                        
                        # ==================== EXPLICACIÓN DE LAS RECOMENDACIONES ====================
                        with st.expander("💡 ¿Cómo se calculan estas recomendaciones?", expanded=False):
                            st.markdown(f"""
                            **Variables clave:**
                            - **Forecast (t+1)**: Predicción de demanda para {next_mes_str} usando el modelo ganador
                            - **Stock Actual**: Inventario disponible hoy (último mes registrado)
                            - **Stock de Seguridad (SS)**: Protección contra variaciones = Z × MAE × √(Lead Time)
                              - Z: Factor según clasificación ABC (A=1.65, B=1.28, C=1.04)
                              - MAE: Error promedio del modelo ganador
                            - **Producción Recomendada**: = Forecast + SS - Stock Actual
                            
                            **¿Por qué algunos productos tienen "RIESGO"?**
                            - Cuando Stock Actual < SS, significa que HOY ya hay riesgo de quiebre
                            - Estos productos necesitan producción INMEDIATA (prioritarios)
                            
                            **¿Por qué el modelo ganador es diferente por producto?**
                            - Cada serie de demanda es única
                            - El sistema elige entre Baselines (MA3, MA6, Seasonal), ETS y RandomForest
                            - Se selecciona por MAE más bajo en los últimos {test_months} meses de histórico
                            """
                            )
                        
                        # ==================== DESCARGAR RECOMENDACIONES ====================
                        with st.expander("⬇️ Descargar Recomendaciones (CSV / Tabla Completa)", expanded=False):
                            csv = reco_df.to_csv(index=False).encode("utf-8-sig")
                            st.download_button(
                                "📥 Descargar recomendaciones_masivas.csv",
                                data=csv,
                                file_name=f"recomendaciones_masivas_{abc_sel}_{next_mes_str.replace(' ', '_')}.csv",
                                mime="text/csv",
                                key="download_reco_masiva"
                            )
                            
                            st.markdown("**Columnas en el archivo CSV:**")
                            st.markdown("""
                            - **Codigo**: Identificador del producto
                            - **ABC**: Clasificación (A/B/C)
                            - **Modelo_Ganador**: Qué modelo de pronóstico se usó
                            - **Forecast_t+1**: Predicción de demanda para el próximo mes
                            - **MAE_ganador**: Error promedio del modelo elegido
                            - **Z**: Factor de seguridad (según ABC)
                            - **SS**: Stock de Seguridad calculado
                            - **Stock_Actual**: Inventario hoy
                            - **Produccion_Recomendada**: **← ESTO ES LO QUE DEBES PRODUCIR**
                            - **RIESGO_QUIEBRE**: Si hay riesgo actual (True/False)
                            - **DEMANDA_TOTAL_HIST**: Demanda total histórica
                            """)



        # ==========================================================
        # TAB 10: COMPARATIVA RETROSPECTIVA SIN SISTEMA VS CON SISTEMA
        # ==========================================================
        with ComparaRetroEntreSistema:
            st.subheader("⚖️ Comparativa de Costos: Sin sistema vs Con sistema")
            
            st.markdown("Compara el impacto económico de implementar el sistema inteligente. Analiza tanto producto individual como portafolio completo.")
            
            # ==================== INPUTS DE COSTO COMPARTIDOS (LAZY LOADING) ====================
            st.markdown("### 💰 Parámetros de Costo (compartidos para ambos análisis)")
            cost_col1, cost_col2 = st.columns(2)
            
            with cost_col1:
                cost_stock_unit_shared = st.number_input("Costo inventario por unidad (proxy)", min_value=0.0, value=1.0, step=0.5, key="shared_cinv")
            with cost_col2:
                cost_stockout_unit_shared = st.number_input("Costo quiebre por unidad (proxy)", min_value=0.0, value=5.0, step=0.5, key="shared_cbrk")
            
            # Detectar cambio global en costos
            prev_cost_stock_shared = st.session_state.get("prev_cost_stock_shared", 1.0)
            prev_cost_stockout_shared = st.session_state.get("prev_cost_stockout_shared", 5.0)
            
            valores_cambiaron_shared = (cost_stock_unit_shared != prev_cost_stock_shared or 
                                       cost_stockout_unit_shared != prev_cost_stockout_shared)
            
            if valores_cambiaron_shared:
                if (st.session_state.get("comparativa_individual_cmp") is not None or 
                    st.session_state.get("portafolio_abc_a_resumen") is not None):
                    st.warning("⚠️ Detectamos cambio en costos. Haz clic en 'Ejecutar' en cualquiera de los análisis para actualizar resultados con los nuevos valores.")
            
            st.divider()
            
            # ========== ANÁLISIS 1: POR PRODUCTO (EXPANDER) ==========
            with st.expander("📊 Análisis Individual (Por Producto)", expanded=False):
                st.markdown("**Análisis:** Compara cuánto hubieras gastado sin sistema (produciendo lo vendido anteriormente) vs con sistema (inteligencia + stock de seguridad).")
                if prod_sel is not None:
                    st.info(f"📊 Comparando: **Producto {prod_sel}** (según filtro seleccionado)")
                else:
                    st.warning("⚠️ Selecciona un producto en los Filtros de Producto del sidebar")

                dm = res_demand.copy()
                dm["Codigo"] = dm["Codigo"].astype(str).str.strip()
                hist = dm[dm["Codigo"] == str(prod_sel)][["Mes", "Demanda_Unid"]].copy().sort_values("Mes")

                    # stock mensual producto
                stock_p = pd.DataFrame()
                if res_stock is not None and not res_stock.empty:
                    stock_p = res_stock.copy()
                    stock_p["Codigo"] = stock_p["Codigo"].astype(str).str.strip()
                    stock_p = stock_p[stock_p["Codigo"] == str(prod_sel)][["Mes", "Stock_Unid"]].copy().sort_values("Mes")

                row = abc_df[abc_df["Codigo"] == str(prod_sel)]
                abc_class = str(row.iloc[0]["ABC"]) if not row.empty else "C"

                # Usar costos compartidos (NO inputs individuales)
                cost_stock_unit = cost_stock_unit_shared
                cost_stockout_unit = cost_stockout_unit_shared
                
                run_cmp = st.button("▶️ Ejecutar análisis por producto", type="primary", key="run_cmp_prod")

                if run_cmp and not hist.empty:
                    with st.spinner("Calculando parámetros automáticos y comparativa..."):
                        # ==================== AHORA SÍ CALCULAMOS (DESPUÉS DE CLICK) ====================
                        eval_months = max(6, int(len(hist) * 0.25))
                        
                        # AUTO-optimizar MA (3 vs 6) para elegir la mejor ventana
                        ets_params_auto = dict(seasonal_periods=12, trend="add", seasonal="add", damped_trend=False, min_obs=24)
                        rf_params_auto = dict(n_estimators=400, min_obs=24, min_samples_leaf=1, random_state=42)
                        
                        bt_ma3_auto = backtest_baselines_1step(hist, y_col="Demanda_Unid", test_months=int(eval_months), ma_window=3)
                        bt_ma6_auto = backtest_baselines_1step(hist, y_col="Demanda_Unid", test_months=int(eval_months), ma_window=6)
                        mae_ma3_auto = float(bt_ma3_auto.metrics.iloc[0]["MAE"]) if not bt_ma3_auto.metrics.empty else float("inf")
                        mae_ma6_auto = float(bt_ma6_auto.metrics.iloc[0]["MAE"]) if not bt_ma6_auto.metrics.empty else float("inf")
                        ma_window_opt = 3 if mae_ma3_auto < mae_ma6_auto else 6
                        bt_base_auto = bt_ma3_auto if ma_window_opt == 3 else bt_ma6_auto
                        
                        # Luego ETS y RF
                        ets_auto = ETSForecaster(**ets_params_auto)
                        bt_ets_auto = backtest_ets_1step(hist, y_col="Demanda_Unid", test_months=int(eval_months), ets=ets_auto)
                        rf_auto = RFForecaster(**rf_params_auto)
                        bt_rf_auto = backtest_rf_1step(hist, y_col="Demanda_Unid", test_months=int(eval_months), rf=rf_auto)
                        
                        cmp_auto = compare_models_metrics(bt_base_auto.metrics, bt_ets_auto.metrics, bt_rf_auto.metrics, sort_by="MAE")
                        winner = str(cmp_auto.iloc[0]["Modelo"]) if not cmp_auto.empty else "ETS(Holt-Winters)"
                        
                        # Mostrar parámetros sincronizados
                        col_info1, col_info2, col_info3 = st.columns(3)
                        with col_info1:
                            st.metric("📅 Meses evaluados (AUTO)", f"{eval_months}", delta=f"{eval_months + 1} filas (incluye mes base)")
                        with col_info2:
                            st.metric("🏆 Modelo ganador (AUTO)", winner.split("(")[0].strip())
                        with col_info3:
                            st.metric("📊 Producto", str(prod_sel))
                    
                        df_cmp, s, period_info = simulate_compare_policy_vs_baseline(
                            hist=hist,
                            stock_series=stock_p,
                            abc_class=abc_class,
                            winner=winner,
                            eval_months=int(eval_months),
                            cost_stock_unit=float(cost_stock_unit),
                            cost_stockout_unit=float(cost_stockout_unit),
                            ma_window=int(ma_window_opt),
                            test_months_for_mae=int(eval_months),
                        )
                        
                        # 💾 Guardar en session_state para persistencia
                        st.session_state.comparativa_individual_cmp = df_cmp
                        st.session_state.comparativa_individual_summary = s
                        st.session_state.comparativa_individual_prod = str(prod_sel)
                        st.session_state.comparativa_individual_period = period_info
                        
                        # Guardar costos compartidos para detectar cambios próximos
                        st.session_state.prev_cost_stock_shared = cost_stock_unit_shared
                        st.session_state.prev_cost_stockout_shared = cost_stockout_unit_shared
                        
                        # 💾 Guardar parámetros sincronizados en session_state
                        st.session_state.sync_eval_months = eval_months
                        st.session_state.sync_winner_model = winner
                        st.session_state.sync_ma_window = ma_window_opt
                        st.session_state.sync_product_code = str(prod_sel)
                        st.session_state.sync_cost_stock_unit = cost_stock_unit
                        st.session_state.sync_cost_stockout_unit = cost_stockout_unit

                # 📊 Mostrar resultados guardados (si existen)
                if st.session_state.get("comparativa_individual_cmp") is not None:
                    df_cmp = st.session_state.comparativa_individual_cmp
                    s = st.session_state.comparativa_individual_summary
                    period_info = st.session_state.get("comparativa_individual_period", {})
                    
                    # Extraer nombre del usuario (de email)
                    user_email = st.session_state.get("email", "Estimado usuario")
                    user_name = user_email.split("@")[0].replace(".", " ").title() if "@" in user_email else "Estimado usuario"
                    
                    # ==================== RESUMEN EJECUTIVO (AMIGABLE PARA CLIENTE) ====================
                    with st.container(border=True):
                        st.markdown(f"### 📊 Resumen Ejecutivo para {user_name}")
                        st.markdown(f"**¿Qué significa este análisis?**")
                        st.markdown(f"Hemos simulado cómo habría funcionado el producto **{str(prod_sel)}** durante los últimos **{period_info.get('num_months', 0)} meses** con dos estrategias diferentes:")
                        st.markdown(f"- **❌ Sin el sistema:** Produciendo solo lo que se vendió el mes anterior (método reactivo)")
                        st.markdown(f"- **✅ Con el sistema:** Usando pronósticos inteligentes + stock de seguridad (método proactivo)")
                        st.markdown(f"**Los resultados muestran:** Implementar el sistema inteligente habría generado un **ahorro de {s['Ahorro_CostoTotal']:,.0f} unidades monetarias** mientras se mejora el servicio (menos quiebres de stock).")
                    
                    # ==================== INFORMACIÓN DEL PERÍODO EVALUADO ====================
                    st.markdown("### 📅 Período Evaluado + Modelo Usado")
                    col_period1, col_period2, col_period3, col_period4, col_period5 = st.columns(5)
                    
                    if period_info:
                        start_date = period_info.get("start_date")
                        end_date = period_info.get("end_date")
                        num_months = period_info.get("num_months", 0)
                        
                        with col_period1:
                            start_str = start_date.strftime("%Y-%m") if start_date else "N/A"
                            st.metric("📍 Inicio", start_str)
                        
                        with col_period2:
                            end_str = end_date.strftime("%Y-%m") if end_date else "N/A"
                            st.metric("📍 Fin", end_str)
                        
                        with col_period3:
                            st.metric("📊 Meses evaluados", num_months)
                            st.caption("(+ 1 fila base inicial)")
                        
                        with col_period4:
                            st.metric("✅ Filas resultado", len(df_cmp), delta=f"{num_months} evaluados + 1 base")
                        
                        with col_period5:
                            model_display = s.get("Winner", "N/A").split("(")[0].strip()
                            st.metric("🏆 Modelo (costos)", model_display)
                    
                    # ==================== MÉTRICAS PRINCIPALES (TARJETAS VISUALES) ====================
                    st.markdown("### 💰 Impacto Financiero y Operativo")
                    
                    ahorro_total = float(s['Ahorro_CostoTotal'])
                    mejora_fillrate = float(s['Mejora_FillRate_pp'])
                    reduccion_faltantes = float(s['Reduccion_Faltantes'])
                    fill_rate_base = float(s['Base']['FillRate_%'])
                    fill_rate_sys = float(s['Sistema']['FillRate_%'])
                    
                    # Tarjetas principales - Ahorro, Fill Rate, Quiebres
                    col_card1, col_card2, col_card3 = st.columns(3, gap="medium")
                    
                    with col_card1:
                        st.markdown(f"""
                        <div style='background: linear-gradient(135deg, #10B981 0%, #059669 100%); padding: 20px; border-radius: 10px; text-align: center; box-shadow: 0 4px 12px rgba(0,0,0,0.1);'>
                            <p style='margin: 0; font-size: 0.85em; color: rgba(255,255,255,0.8);'>Ahorro Total</p>
                            <p style='margin: 5px 0 0 0; font-size: 2em; font-weight: bold; color: white;'>{ahorro_total:,.0f}</p>
                            <p style='margin: 5px 0 0 0; font-size: 0.75em; color: rgba(255,255,255,0.7);'>unidades monetarias</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col_card2:
                        st.markdown(f"""
                        <div style='background: linear-gradient(135deg, #06B6D4 0%, #0891B2 100%); padding: 20px; border-radius: 10px; text-align: center; box-shadow: 0 4px 12px rgba(0,0,0,0.1);'>
                            <p style='margin: 0; font-size: 0.85em; color: rgba(255,255,255,0.8);'>Mejora Fill Rate</p>
                            <p style='margin: 5px 0 0 0; font-size: 2em; font-weight: bold; color: white;'>+{mejora_fillrate:.1f}%</p>
                            <p style='margin: 5px 0 0 0; font-size: 0.75em; color: rgba(255,255,255,0.7);'>{fill_rate_base:.0f}% → {fill_rate_sys:.0f}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col_card3:
                        st.markdown(f"""
                        <div style='background: linear-gradient(135deg, #F59E0B 0%, #D97706 100%); padding: 20px; border-radius: 10px; text-align: center; box-shadow: 0 4px 12px rgba(0,0,0,0.1);'>
                            <p style='margin: 0; font-size: 0.85em; color: rgba(255,255,255,0.8);'>Quiebres Evitados</p>
                            <p style='margin: 5px 0 0 0; font-size: 2em; font-weight: bold; color: white;'>{reduccion_faltantes:,.0f}</p>
                            <p style='margin: 5px 0 0 0; font-size: 0.75em; color: rgba(255,255,255,0.7);'>menos faltantes</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    
                    st.divider()
                    
                    # ==================== GRÁFICOS PRINCIPALES (DESTACADOS) ====================
                    st.markdown("### 📊 Gráficos de Impacto")
                    
                    col_g1, col_g2 = st.columns(2, gap="medium")
                    
                    with col_g1:
                        fig_cost = px.line(df_cmp, x="Mes", y=["Base_Costo_total", "Sys_Costo_total"], markers=True,
                                        title="Costo total mensual: Baseline vs Sistema")
                        fig_cost.update_traces(line=dict(width=3))
                        st.plotly_chart(fig_cost, use_container_width=True)
                    
                    with col_g2:
                        fig_lost = px.bar(df_cmp, x="Mes", y=["Base_Faltante", "Sys_Faltante"], barmode="group",
                                        title="Quiebres por mes: Baseline vs Sistema")
                        st.plotly_chart(fig_lost, use_container_width=True)
                    
                    st.divider()
                    
                    # ==================== INFORMACIÓN DETALLADA (COLAPSADA) ====================
                    with st.expander("📈 ¿Qué significa esto para tu negocio? (Explicación detallada)", expanded=False):
                        st.markdown(f"**{user_name},** durante el período de **{period_info.get('num_months', 0)} meses**, tu estrategia anterior (producir lo que se vendió antes) habría costado aproximadamente **{s['Base']['Costo_total']:,.0f}** en inventario y quiebres.")
                        st.markdown(f"Si hubiera implementado el sistema inteligente desde entonces, el costo habría sido **{s['Sistema']['Costo_total']:,.0f}**, lo que representa un **ahorro de {ahorro_total:,.0f}**.")
                        st.markdown("**Análisis de impacto operativo:**")
                        
                        # Lógica adaptativa para Fill Rate
                        if fill_rate_sys >= fill_rate_base:
                            st.markdown(f"✅ **Fill Rate:** Mejora de **{fill_rate_base:.1f}%** a **{fill_rate_sys:.1f}%** (ganancia de {fill_rate_sys - fill_rate_base:.1f}%)")
                        else:
                            cambio_ff = fill_rate_base - fill_rate_sys
                            st.markdown(f"📊 **Fill Rate:** {fill_rate_base:.1f}% (sin sistema) vs {fill_rate_sys:.1f}% (con sistema)")
                            st.markdown(f"   → Reducción de {cambio_ff:.1f}%, pero compensada por **ahorros de ${ahorro_total:,.0f}**")
                        
                        # Lógica adaptativa para Faltantes
                        if reduccion_faltantes >= 0:
                            st.markdown(f"✅ **Quiebres evitados:** {reduccion_faltantes:,.0f} unidades menos faltantes")
                        else:
                            faltantes_adicionales = abs(reduccion_faltantes)
                            st.markdown(f"📊 **Quiebres:** Se podrían generar {faltantes_adicionales:,.0f} unidades más faltantes")
                            st.markdown(f"   → Trade-off aceptable: más quiebres pero **${ahorro_total:,.0f} en ahorros**")
                        
                        st.markdown("---")
                        st.markdown("**Conclusión:**")
                        if ahorro_total > 0:
                            st.markdown(f"🎯 El sistema inteligente genera **${ahorro_total:,.0f} en ahorros** durante {period_info.get('num_months', 0)} meses.")
                            if fill_rate_sys >= fill_rate_base:
                                st.markdown("✨ **Beneficio doble:** Ahorras dinero Y mejoras la disponibilidad para clientes.")
                            else:
                                st.markdown("⚖️ **Balance económico:** Ahorras dinero con un trade-off aceptable en disponibilidad.")
                        else:
                            st.markdown("ℹ️ En este caso, el sistema simplemente optimiza los costos.")
                    
                    with st.expander("🔍 Validación técnica y detalles de costos", expanded=False):
                        st.markdown("#### Suma Manual de Costos (Validación)")
                        col_cost1, col_cost2, col_cost3 = st.columns(3)
                        
                        with col_cost1:
                            costo_base_suma = float(df_cmp["Base_Costo_total"].sum())
                            st.metric("Base_Costo_total (suma manual)", f"{costo_base_suma:,.1f}", 
                                      delta=f"KPI: {s['Base']['Costo_total']:,.1f}")
                        
                        with col_cost2:
                            costo_sys_suma = float(df_cmp["Sys_Costo_total"].sum())
                            st.metric("Sys_Costo_total (suma manual)", f"{costo_sys_suma:,.1f}",
                                      delta=f"KPI: {s['Sistema']['Costo_total']:,.1f}")
                        
                        with col_cost3:
                            ahorro_suma = costo_base_suma - costo_sys_suma
                            st.metric("Ahorro (calc manual)", f"{ahorro_suma:,.1f}",
                                      delta=f"KPI: {s['Ahorro_CostoTotal']:,.1f}")
                        
                        st.divider()
                        st.markdown("#### Tabla Detallada Mes a Mes")
                        st.dataframe(df_cmp, use_container_width=True, height=420)
                    
                    # Explicación de columnas
                    with st.expander("📋 Significado de las columnas en la tabla comparativa", expanded=False):
                        st.markdown("""
                        | Columna | Significado |
                        |---------|------------|
                        | **Mes** | Mes del período evaluado |
                        | **Demanda_real** | Unidades que los clientes solicitaron ese mes |
                        | **Base_stock_ini** | Stock inicial disponible al inicio del mes (método baseline) |
                        | **Base_Q** | Producción recomendada mes a mes (método baseline/reactivo) |
                        | **Base_Stock_Fin** | Stock final después de atender demanda (método baseline) |
                        | **Base_Faltante** | Unidades NO satisfechas en baseline (quiebre) |
                        | **Base_Costo_inv** | Costo de mantener inventario en baseline |
                        | **Base_Costo_stockout** | Costo de quiebres en baseline |
                        | **Base_Costo_total** | Costo total (inventario + quiebres) en baseline |
                        | **Sys_Stock_ini** | Stock inicial disponible al inicio del mes (sistema inteligente) |
                        | **Sys_Forecast** | Pronóstico de demanda para el siguiente mes |
                        | **Sys_SS** | Stock de seguridad calculado por la política inteligente |
                        | **Sys_Q** | Producción recomendada por la política inteligente |
                        | **Sys_Stock_fin** | Stock final después de atender demanda (sistema inteligente) |
                        | **Sys_Faltante** | Unidades NO satisfechas con sistema (quiebre reducido) |
                        | **Sys_Costo_inv** | Costo de inventario con sistema inteligente |
                        | **Sys_Costo_stockout** | Costo de quiebres con sistema (menor que baseline) |
                        | **Sys_Costo_total** | Costo total (inventario + quiebres) con sistema inteligente |
                        """)

            
            st.divider()
            
            # ========== ANÁLISIS 2: POR PORTAFOLIO ABC A (EXPANDER) ==========
            with st.expander("📦 Análisis de Portafolio ABC A", expanded=False):
                st.markdown("**Análisis:** Evaluación de costos para TODOS los productos ABC A. Simula cómo habría funcionado la política inteligente en el portafolio.")
                
                # Usar costos compartidos
                cost_stock_unit_port = cost_stock_unit_shared
                cost_stockout_unit_port = cost_stockout_unit_shared
                
                # ==================== SINCRONIZACIÓN DE PARÁMETROS ====================
                has_sync_params = st.session_state.get("sync_eval_months") is not None
                
                if has_sync_params:
                    eval_months_port = st.session_state.get("sync_eval_months")
                    ma_window_port = st.session_state.get("sync_ma_window", 3)
                    prod_individual = st.session_state.get("sync_product_code", "N/A")
                    
                    # Mostrar parámetros sincronizados (compacto)
                    st.info(f"🔗 **Parámetros sincronizados desde Análisis Individual**\n- Producto: {prod_individual} | Meses: {eval_months_port} | MA Window: MA{ma_window_port}")
                else:
                    # Si NO hay parámetros sincronizados, permitir que el usuario los defina
                    st.warning("⚠️ Primero ejecuta el Análisis Individual para sincronizar parámetros automáticamente.")
                    eval_months_port = st.slider("Meses a evaluar portafolio (últimos)", 6, 24, 12, 1, key="port_eval")
                    ma_window_port = st.selectbox("Ventana MA para Baselines", options=[3, 6], index=0, key="port_ma_window")
                
                winner_mode = "AUTO"  # Cada producto calcula su propio ganador

                max_products_port = st.selectbox(
                    "Cantidad de productos ABC A a procesar (performance)",
                    options=[20, 50, 100, 200, "Todos"],
                    index=1,
                    key="port_max"
                )
                max_products_port = None if max_products_port == "Todos" else int(max_products_port)

                run_port = st.button("▶️ Ejecutar análisis de portafolio", type="primary", key="run_port_abcA")

            if run_port:
                with st.spinner("Calculando portafolio ABC A (puede tardar)..."):
                    resumenA, detalleA = run_portfolio_cost_comparison_abcA(
                        demand_monthly=res_demand,
                        stock_monthly=res_stock,
                        abc_df=abc_df,   # ya existe arriba en tu render
                        eval_months=int(eval_months_port),
                        cost_stock_unit=float(cost_stock_unit_port),
                        cost_stockout_unit=float(cost_stockout_unit_port),
                        winner_mode=str(winner_mode),
                        ma_window=int(ma_window_port),  # ← Usar la ventana sincronizada/seleccionada
                        test_months_for_mae=int(eval_months_port),  # ← Usar eval_months para consistencia
                        max_products=max_products_port,
                    )

                    if not resumenA.empty:
                        # 💾 Guardar en session_state para persistencia
                        st.session_state.portafolio_abc_a_resumen = resumenA
                        st.session_state.portafolio_abc_a_detalle = detalleA
                    else:
                        st.warning("No se generó portafolio (revisa si hay productos ABC A con historia suficiente).")

            # 📊 Mostrar resultados guardados (si existen)
            if st.session_state.get("portafolio_abc_a_resumen") is not None:
                resumenA = st.session_state.portafolio_abc_a_resumen
                detalleA = st.session_state.portafolio_abc_a_detalle
                
                # ==================== RESUMEN EJECUTIVO PORTAFOLIO ====================
                with st.container(border=True):
                    st.markdown(f"### 📊 Análisis de Portafolio ABC A")
                    st.markdown(f"**¿Qué es ABC A?** Son tus productos más importantes - los que generan el 80% de tu demanda. Esta sección simula el impacto de implementar el sistema inteligente en TODOS estos productos simultáneamente.")
                    st.markdown(f"**Período analizado:** {eval_months_port} meses (consistente con tu análisis individual)")
                    st.markdown(f"**Cobertura:** {detalleA['Codigo'].nunique():,} productos de clase ABC A")
                
                # ==================== IMPACTO FINANCIERO DEL PORTAFOLIO - TARJETAS VISUALES ====================
                st.markdown("### 💰 Impacto Financiero en el Portafolio ABC A")
                
                ahorro_total_port = float(resumenA.iloc[0]['Ahorro_total'])
                fillrate_base_port = float(resumenA.iloc[0]['FillRate_Base_%'])
                fillrate_sys_port = float(resumenA.iloc[0]['FillRate_Sistema_%'])
                mejora_fillrate_port = fillrate_sys_port - fillrate_base_port
                costo_base_total_port = float(resumenA.iloc[0]['CostoTotal_Base'])
                costo_sys_total_port = float(resumenA.iloc[0]['CostoTotal_Sistema'])
                
                # Tarjetas principales del Portafolio
                col_pcard1, col_pcard2, col_pcard3 = st.columns(3, gap="medium")
                
                with col_pcard1:
                    ahorro_color = "#10B981" if ahorro_total_port > 0 else "#EF4444"
                    st.markdown(f"""
                    <div style='background: linear-gradient(135deg, {ahorro_color} 0%, {ahorro_color}CC 100%); padding: 20px; border-radius: 10px; text-align: center; box-shadow: 0 4px 12px rgba(0,0,0,0.1);'>
                        <p style='margin: 0; font-size: 0.85em; color: rgba(255,255,255,0.8);'>Ahorro Total Portafolio</p>
                        <p style='margin: 5px 0 0 0; font-size: 2em; font-weight: bold; color: white;'>{ahorro_total_port:,.0f}</p>
                        <p style='margin: 5px 0 0 0; font-size: 0.75em; color: rgba(255,255,255,0.7);'>unidades monetarias</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_pcard2:
                    st.markdown(f"""
                    <div style='background: linear-gradient(135deg, #06B6D4 0%, #0891B2 100%); padding: 20px; border-radius: 10px; text-align: center; box-shadow: 0 4px 12px rgba(0,0,0,0.1);'>
                        <p style='margin: 0; font-size: 0.85em; color: rgba(255,255,255,0.8);'>Mejora Fill Rate</p>
                        <p style='margin: 5px 0 0 0; font-size: 2em; font-weight: bold; color: white;'>+{mejora_fillrate_port:.1f}%</p>
                        <p style='margin: 5px 0 0 0; font-size: 0.75em; color: rgba(255,255,255,0.7);'>{fillrate_base_port:.0f}% → {fillrate_sys_port:.0f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_pcard3:
                    st.markdown(f"""
                    <div style='background: linear-gradient(135deg, #8B5CF6 0%, #7C3AED 100%); padding: 20px; border-radius: 10px; text-align: center; box-shadow: 0 4px 12px rgba(0,0,0,0.1);'>
                        <p style='margin: 0; font-size: 0.85em; color: rgba(255,255,255,0.8);'>Productos ABC A</p>
                        <p style='margin: 5px 0 0 0; font-size: 2em; font-weight: bold; color: white;'>{detalleA['Codigo'].nunique():,}</p>
                        <p style='margin: 5px 0 0 0; font-size: 0.75em; color: rgba(255,255,255,0.7);'>evaluados en portafolio</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.divider()
                
                # ==================== GRÁFICOS PRINCIPALES (DESTACADOS) ====================
                st.markdown("### 📊 Visualización: Impacto en Costos y Disponibilidad")
                
                col_pg1, col_pg2 = st.columns(2, gap="medium")
                
                with col_pg1:
                    fig_cost_port = px.bar(
                        resumenA,
                        x=["CostoTotal_Base", "CostoTotal_Sistema"],
                        title="Costo Total Portafolio: Baseline vs Sistema",
                        labels={"value": "Costo Total", "variable": "Estrategia"}
                    )
                    fig_cost_port.update_layout(height=450, font=dict(size=12))
                    st.plotly_chart(fig_cost_port, use_container_width=True)
                
                with col_pg2:
                    fig_fillrate_port = px.bar(
                        resumenA,
                        x=["FillRate_Base_%", "FillRate_Sistema_%"],
                        title="Fill Rate Portafolio: Baseline vs Sistema",
                        labels={"value": "Fill Rate (%)", "variable": "Estrategia"}
                    )
                    fig_fillrate_port.update_layout(height=450, font=dict(size=12))
                    st.plotly_chart(fig_fillrate_port, use_container_width=True)
                
                st.divider()
                
                # ==================== TOP OPORTUNIDADES POR PRODUCTO ====================
                st.markdown("### 💡 Top Productos: Mayor Potencial de Ahorro")
                
                # Calcular ahorro si no existe en detalleA
                if 'Ahorro_total' not in detalleA.columns:
                    detalleA['Ahorro_total'] = detalleA.get('CostoTotal_Base', 0) - detalleA.get('CostoTotal_Sistema', 0)
                
                # Obtener los top 30 productos (o todos si hay menos), ordenados de mayor a menor
                num_products_to_show = min(30, len(detalleA))
                detalleA_sorted = detalleA.nlargest(num_products_to_show, 'Ahorro_total')
                # Ordenar ascendente para que Plotly dibuje de mayor a menor (mayor en la parte superior)
                detalleA_sorted = detalleA_sorted.sort_values('Ahorro_total', ascending=True)
                
                fig_oportunidades = px.bar(
                    detalleA_sorted,
                    x='Ahorro_total',
                    y='Codigo',
                    orientation='h',
                    title=f'Top {num_products_to_show} productos: Ahorro potencial',
                    labels={'Ahorro_total': 'Ahorro Unitario', 'Codigo': 'Producto'}
                )
                fig_oportunidades.update_layout(height=max(550, num_products_to_show * 20), font=dict(size=11))
                st.plotly_chart(fig_oportunidades, use_container_width=True)
                
                st.divider()
                
                # ==================== INFORMACIÓN DETALLADA (COLAPSADA) ====================
                with st.expander("📈 Análisis detallado de tu portafolio ABC A", expanded=False):
                    st.markdown(f"**¿Qué ves aquí?** Analizando TODOS tus productos ABC A ({detalleA['Codigo'].nunique():,} productos) durante {eval_months_port} meses:")
                    st.markdown(f"- **Sin el sistema:** Costo total de **{costo_base_total_port:,.0f}** (método reactivo)")
                    st.markdown(f"- **Con el sistema:** Costo total de **{costo_sys_total_port:,.0f}** (método inteligente)")
                    st.markdown(f"")
                    st.markdown(f"**Resultado neto:** Ahorraría **{ahorro_total_port:,.0f} unidades monetarias** en {eval_months_port} meses")
                    st.markdown(f"**Proyección anual:** {ahorro_total_port * (12/eval_months_port):,.0f} unidades (extrapolado a 12 meses)")
                    st.markdown(f"")
                    
                    # Lógica adaptativa para Fill Rate del Portafolio
                    if fillrate_sys_port >= fillrate_base_port:
                        cambio_ff_port = fillrate_sys_port - fillrate_base_port
                        st.markdown(f"✅ **Fill Rate:** Mejora de **{fillrate_base_port:.1f}%** a **{fillrate_sys_port:.1f}%** (ganancia de {cambio_ff_port:.1f}%)")
                        st.markdown(f"✨ **En conclusión:** Implementar el sistema en tu portafolio ABC A te permite **reducir costos Y mejorar servicio simultáneamente**")
                    else:
                        cambio_ff_port = fillrate_base_port - fillrate_sys_port
                        st.markdown(f"📊 **Fill Rate:** {fillrate_base_port:.1f}% (sin sistema) vs {fillrate_sys_port:.1f}% (con sistema)")
                        st.markdown(f"   → Reducción de {cambio_ff_port:.1f}%, pero compensada por **ahorros de ${ahorro_total_port:,.0f}**")
                        st.markdown(f"⚖️ **En conclusión:** Implementar el sistema en tu portafolio ABC A genera **ahorros significativos** con un trade-off aceptable en disponibilidad")
                
                with st.expander("📊 Resumen y tablas detalladas del portafolio", expanded=False):
                    st.markdown("#### Resumen Agregado del Portafolio")
                    st.dataframe(resumenA, use_container_width=True)
                    
                    st.divider()
                    
                    st.markdown("#### Top 30 productos por mayor ahorro")
                    st.markdown("*Estos productos son los que más beneficio tendrían con el sistema inteligente*")
                    st.dataframe(detalleA.head(30), use_container_width=True, height=420)
                
                # ==================== EXPORTACIÓN ====================
                with st.expander("⬇️ Exportar detalle completo del portafolio (CSV)", expanded=False):
                    csv_det = detalleA.to_csv(index=False).encode("utf-8-sig")
                    st.download_button(
                        "📥 Descargar detalle_portafolio_ABC_A.csv",
                        data=csv_det,
                        file_name="detalle_portafolio_ABC_A.csv",
                        mime="text/csv",
                        key="dl_detalle_portafolio_A"
                    )


