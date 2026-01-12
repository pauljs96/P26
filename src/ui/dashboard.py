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

from src.data.pipeline import DataPipeline
from src.utils import config
from src.ml.baselines import naive_last, seasonal_naive_12, moving_average
from src.ml.backtest import backtest_baselines_1step
from src.ml.ets_model import ETSForecaster
from src.ml.backtest_ets import backtest_ets_1step
import numpy as np
from src.ml.rf_model import RFForecaster
from src.ml.backtest_rf import backtest_rf_1step



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
    ss_const = float(z * sigma * np.sqrt(float(lead_time)))  # SS fijo para la simulación (simple y estable)

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
    return df, summary


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
        df_cmp, s = simulate_compare_policy_vs_baseline(
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


class Dashboard:
    def render(self):
        st.set_page_config(page_title="Planificación - MVP", layout="wide")
        st.title("📦 Sistema de Planificación (MVP)")

        st.write(
            "Sube tus archivos CSV (2021–2025). "
            "Luego selecciona un producto para ver demanda, pronósticos y diagnósticos."
        )

        st.sidebar.header("Datos")
        files = st.sidebar.file_uploader(
            "Sube CSV (2021–2025)",
            type=["csv"],
            accept_multiple_files=True
        )

        if not files:
            st.info("👆 Sube los CSV para comenzar.")
            return

        pipeline = DataPipeline()

        with st.spinner("Procesando archivos..."):
            res = pipeline.run(files)

        if res.movements.empty:
            st.error("No se detectaron columnas mínimas o la data quedó vacía tras limpieza.")
            st.stop()


        # --- ABC (una vez) ---
        dm_abc = res.demand_monthly.copy()
        dm_abc["Codigo"] = dm_abc["Codigo"].astype(str).str.strip()
        abc_df = build_abc_from_demand(dm_abc)  # columnas: Codigo, Demanda_Total, Share, CumShare, ABC


        st.sidebar.header("Filtros")

        # 1) Filtro ABC
        abc_options = ["A", "B", "C", "Todos"]
        abc_sel = st.sidebar.selectbox("Categoría ABC", options=abc_options, index=0)

        # 2) Productos filtrados por ABC
        if abc_sel == "Todos":
            productos = sorted(res.movements["Codigo"].dropna().astype(str).str.strip().unique().tolist())
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
            st.sidebar.warning("No hay productos en esa categoría ABC.")
            prod_sel = None
        else:
            prod_sel = st.sidebar.selectbox("Producto (Código)", options=productos)


        # Defaults para evitar UnboundLocalError entre tabs / reruns
        if "ets_test_months" not in st.session_state:
            st.session_state["ets_test_months"] = 12
        if "global_test_months" not in st.session_state:
            st.session_state["global_test_months"] = 12

        # ------------------------------
        # TABS
        # ------------------------------
        tab_demanda, tab_baselines, tab_ets,tab_ml,Tab_Comparativa,ResumenComparativa ,tab_stock_diag,tab_reco,Reco_Masiva,Valida_Retro,ComparaRetroEntreSistema = st.tabs([
            "🧩 Demanda y Componentes",
            "🔮 Baselines y Backtest",
            "📈 Holt–Winters (ETS)",
            "🤖 Random Forest (RF)",
            "🏆 Comparativa ETS vs Baselines vs RF",
            "📊 Resumen Comparativa",
            "🏢 Stock y Diagnóstico",
            "🔄 Recomendación de Producción",
            "📑 Recomendación Masiva",
            "✅ Validación Retrospectiva",
            "📉 Comparativa Retrospectiva entre Sistemas",
        ])

        # ==========================================================
        # TAB 1: DEMANDA Y COMPONENTES
        # ==========================================================
        with tab_demanda:
            st.subheader("🧩 Componentes de demanda por mes (producto seleccionado)")
            comp = build_monthly_components(res.movements, prod_sel)

            cA, cB = st.columns([1, 1])
            with cA:
                st.dataframe(comp, use_container_width=True, height=380)

            with cB:
                fig_total = px.line(
                    comp, x="Mes", y="Demanda_Total", markers=True,
                    title=f"Demanda total (suma de componentes) - Producto {prod_sel}"
                )
                st.plotly_chart(fig_total, use_container_width=True)

            st.subheader("📊 Componentes (Venta / Consumo / Guía externa)")
            comp_long = comp.melt(
                id_vars=["Mes"],
                value_vars=["Venta_Tienda", "Consumo", "Guia_Externa"],
                var_name="Componente",
                value_name="Unidades"
            )
            fig_comp = px.line(
                comp_long, x="Mes", y="Unidades", color="Componente", markers=True,
                title=f"Componentes de demanda - Producto {prod_sel}"
            )
            st.plotly_chart(fig_comp, use_container_width=True)

            with st.expander("🔍 Debug: detalle de guías externas por mes", expanded=False):
                dfp = res.movements[res.movements["Codigo"] == str(prod_sel)].copy()
                if not dfp.empty:
                    dfp["Documento"] = _normalize_text(dfp["Documento"])
                    dfp["Numero"] = _normalize_text(dfp["Numero"])
                    dfp["Mes"] = dfp["Fecha"].dt.to_period("M").dt.to_timestamp()
                    det = dfp[(dfp["Documento"] == config.GUIDE_DOC)].copy()
                    if not det.empty:
                        det = det[["Fecha", "Mes", "Numero", "Bodega", "Entrada_unid", "Salida_unid", "Tipo_Guia", "Guia_Salida_Externa_Unid"]]
                        det = det.sort_values(["Mes", "Fecha", "Numero"])
                        st.dataframe(det, use_container_width=True, height=350)
                    else:
                        st.info("No hay guías para este producto.")
                else:
                    st.info("No hay movimientos para este producto.")

        # ==========================================================
        # TAB 2: BASELINES Y BACKTEST
        # ==========================================================
        with tab_baselines:
            st.subheader("🔮 Pronóstico baseline (t+1) - Demanda mensual")

            dm = res.demand_monthly.copy()
            dm["Codigo"] = dm["Codigo"].astype(str).str.strip()
            hist = dm[dm["Codigo"] == str(prod_sel)].copy().sort_values("Mes")

            if hist.empty:
                st.info("No hay serie mensual para este producto.")
            else:
                last_mes = hist["Mes"].max()
                next_mes = last_mes + pd.offsets.MonthBegin(1)

                model_name = st.selectbox(
                    "Modelo baseline",
                    options=[
                        "Naive (último mes)",
                        "Naive estacional (t-12)",
                        "Media móvil (3)"
                    ],
                    index=0
                )

                if model_name == "Naive (último mes)":
                    yhat = naive_last(hist)
                elif model_name == "Naive estacional (t-12)":
                    yhat = seasonal_naive_12(hist)
                else:
                    yhat = moving_average(hist, window=3)

                c1, c2, c3 = st.columns(3)
                c1.metric("Último mes", last_mes.strftime("%Y-%m"))
                c2.metric("Mes pronosticado", next_mes.strftime("%Y-%m"))
                c3.metric("Pronóstico (unid)", f"{yhat:,.0f}")

                plot_df = hist[["Mes", "Demanda_Unid"]].rename(columns={"Demanda_Unid": "Demanda"})
                forecast_row = pd.DataFrame({"Mes": [next_mes], "Demanda": [yhat]})
                plot_all = pd.concat([plot_df, forecast_row], ignore_index=True)

                fig = px.line(
                    plot_all, x="Mes", y="Demanda", markers=True,
                    title=f"Demanda histórica + pronóstico baseline (Producto {prod_sel})"
                )
                st.plotly_chart(fig, use_container_width=True)

            st.divider()

            st.subheader("🧪 Backtest de baselines (t+1)")
            dm = res.demand_monthly.copy()
            dm["Codigo"] = dm["Codigo"].astype(str).str.strip()
            hist = dm[dm["Codigo"] == str(prod_sel)].copy().sort_values("Mes")

            if hist.empty:
                st.info("No hay serie mensual para este producto.")
            else:
                test_months = st.slider("Meses a evaluar (últimos)", min_value=6, max_value=24, value=12, step=1)
                ma_window = st.selectbox("Ventana Media Móvil", options=[3, 6], index=0)

                bt = backtest_baselines_1step(
                    hist, y_col="Demanda_Unid",
                    test_months=test_months,
                    ma_window=int(ma_window)
                )

                c1, c2 = st.columns([1, 1])
                with c1:
                    st.markdown("**Métricas (menor = mejor)**")
                    st.dataframe(bt.metrics, use_container_width=True)

                with c2:
                    st.markdown("**Predicciones (últimos meses)**")
                    st.dataframe(bt.predictions.tail(min(24, len(bt.predictions))),
                                 use_container_width=True, height=320)

                plot = bt.predictions.copy()
                plot_long = plot.melt(
                    id_vars=["Mes_target", "y_true"],
                    value_vars=[c for c in plot.columns if c not in ["Mes_target", "y_true"]],
                    var_name="Modelo",
                    value_name="y_pred"
                )
                fig_bt = px.line(
                    plot_long, x="Mes_target", y="y_pred", color="Modelo", markers=True,
                    title=f"Backtest (predicción) - Producto {prod_sel}"
                )
                st.plotly_chart(fig_bt, use_container_width=True)

                fig_true = px.line(
                    plot, x="Mes_target", y="y_true", markers=True,
                    title=f"Backtest (real) - Producto {prod_sel}"
                )
                st.plotly_chart(fig_true, use_container_width=True)

        # ==========================================================
        # TAB 3: ETS (HOLT–WINTERS)
        # ==========================================================
        with tab_ets:
            st.subheader("📈 ETS (Holt–Winters) (Backtest t+1)")

            dm = res.demand_monthly.copy()
            dm["Codigo"] = dm["Codigo"].astype(str).str.strip()
            hist = dm[dm["Codigo"] == str(prod_sel)].copy().sort_values("Mes")

            if hist.empty:
                st.info("No hay serie mensual para este producto.")
            else:
                test_months_ets = st.slider(
                    "Meses a evaluar ETS (últimos)",
                    min_value=6, max_value=24, value=12, step=1,
                    key="ets_test"
                )
                ets = ETSForecaster(
                    seasonal_periods=12,
                    trend="add",
                    seasonal="add",
                    damped_trend=False,
                    min_obs=24
                )

                bt_ets = backtest_ets_1step(
                    hist,
                    y_col="Demanda_Unid",
                    test_months=test_months_ets,
                    ets=ets
                )



                st.markdown("**Métricas ETS**")
                st.dataframe(bt_ets.metrics, use_container_width=True)

                st.markdown("**Predicciones ETS (detalle)**")
                st.dataframe(bt_ets.predictions.tail(min(24, len(bt_ets.predictions))),
                             use_container_width=True, height=280)

                fig_ets = px.line(
                    bt_ets.predictions, x="Mes_target", y=["y_true", "ETS"], markers=True,
                    title=f"ETS vs Real (Backtest) - Producto {prod_sel}"
                )
                st.plotly_chart(fig_ets, use_container_width=True)

        # ==========================================================
        # TAB 4: ML-RANDOM FOREST
        # ==========================================================

        with tab_ml:
            st.subheader("🤖 Random Forest (Backtest t+1)")

            dm = res.demand_monthly.copy()
            dm["Codigo"] = dm["Codigo"].astype(str).str.strip()
            hist = (dm[dm["Codigo"] == str(prod_sel)][["Mes", "Demanda_Unid"]].copy().sort_values("Mes"))


            if hist.empty:
                st.info("No hay serie mensual para este producto.")
            else:
                test_months_rf = st.slider(
                    "Meses a evaluar RF (últimos)",
                    min_value=6, max_value=24, value=12, step=1,
                    key="rf_test_months"
                )

                # Parámetros RF (mínimos y estables)
                n_estimators = st.slider("Árboles (n_estimators)", 200, 800, 400, 50)
                min_obs = st.slider("Mínimo de meses para entrenar", 12, 36, 24, 1)
                min_leaf = st.slider("min_samples_leaf", 1, 10, 1, 1)

                rf = RFForecaster(
                    n_estimators=int(n_estimators),
                    min_obs=int(min_obs),
                    min_samples_leaf=int(min_leaf),
                    random_state=42
                )

                bt_rf = backtest_rf_1step(hist, y_col="Demanda_Unid", test_months=int(test_months_rf), rf=rf)

                st.markdown("**Métricas RF**")
                st.dataframe(bt_rf.metrics, use_container_width=True)

                st.markdown("**Predicciones RF (detalle)**")
                st.dataframe(bt_rf.predictions.tail(min(24, len(bt_rf.predictions))), use_container_width=True, height=280)

                fig_rf = px.line(
                    bt_rf.predictions, x="Mes_target", y=["y_true", "RF"], markers=True,
                    title=f"RF vs Real (Backtest) - Producto {prod_sel}"
                )
                st.plotly_chart(fig_rf, use_container_width=True)



        # ==========================================================
        # TAB 5: COMPARATIVA ETS VS BASELINES VS RF
        # ==========================================================
        with Tab_Comparativa:
            st.subheader("🏆 Comparador final: Baselines vs ETS vs RF (Backtest t+1)")

            dm = res.demand_monthly.copy()
            dm["Codigo"] = dm["Codigo"].astype(str).str.strip()
            hist_cmp = dm[dm["Codigo"] == str(prod_sel)][["Mes", "Demanda_Unid"]].copy().sort_values("Mes")

            if hist_cmp.empty:
                st.info("No hay serie mensual para este producto.")
            else:
                # Parámetros de evaluación (tab independiente)
                test_months_cmp = st.slider(
                    "Meses a evaluar (últimos)",
                    min_value=6, max_value=24, value=int(st.session_state.get("ets_test_months", 12)), step=1,
                    key="cmp_test_months"
                )
                st.session_state["ets_test_months"] = int(test_months_cmp)  # mantiene consistencia

                ma_window_cmp = st.selectbox("Ventana MA", options=[3, 6], index=0, key="cmp_ma_window")

                # métrica para ordenar (default MAE)
                metric_to_sort = st.selectbox(
                    "Ordenar ganador por",
                    options=["MAE", "RMSE", "sMAPE_%", "MAPE_safe_%"],
                    index=0,  # MAE
                    key="cmp_sort_metric"
                )

                # 1) Baselines
                bt_base_cmp = backtest_baselines_1step(
                    hist_cmp,
                    y_col="Demanda_Unid",
                    test_months=int(test_months_cmp),
                    ma_window=int(ma_window_cmp)
                )

                # 2) ETS
                ets = ETSForecaster(seasonal_periods=12, trend="add", seasonal="add", damped_trend=False, min_obs=24)
                bt_ets_cmp = backtest_ets_1step(
                    hist_cmp,
                    y_col="Demanda_Unid",
                    test_months=int(test_months_cmp),
                    ets=ets
                )

                # 3) RF
                rf = RFForecaster(n_estimators=400, min_obs=24, min_samples_leaf=1, random_state=42)
                bt_rf_cmp = backtest_rf_1step(
                    hist_cmp,
                    y_col="Demanda_Unid",
                    test_months=int(test_months_cmp),
                    rf=rf
                )

                # Unir métricas
                cmp = compare_models_metrics(bt_base_cmp.metrics, bt_ets_cmp.metrics, bt_rf_cmp.metrics, sort_by=metric_to_sort)

                winner = str(cmp.iloc[0]["Modelo"]) if not cmp.empty else "N/A"
                st.metric("Modelo ganador (default MAE)", winner)
                st.dataframe(cmp, use_container_width=True)

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
                        title=f"Ganador vs Real (Backtest) - {winner} - Producto {prod_sel}"
                    )
                    st.plotly_chart(fig_best, use_container_width=True)



        # ==========================================================
        # TAB 6: COMPARATIVA GLOBAL ETS VS BASELINES VS RF
        # ==========================================================
        with ResumenComparativa:

            st.divider()
            st.subheader("🌍 Comparación global + ABC (todos los productos)")

            sort_metric = st.selectbox(
                "Métrica para elegir ganador",
                options=["MAE", "RMSE", "sMAPE_%", "MAPE_safe_%"],
                index=0,
                key="global_sort_metric"
            )

            test_months_global = st.slider(
                "Backtest (últimos meses)",
                min_value=6, max_value=24, value=12, step=1,
                key="global_test_months"
            )

            ma_window_global = st.selectbox("Media móvil (ventana)", options=[3, 6], index=0, key="global_ma_window")

            max_products = st.selectbox(
                "Cantidad de productos a evaluar (performance)",
                options=[50, 100, 200, "Todos"],
                index=1,
                key="global_max_products"
            )
            max_products = None if max_products == "Todos" else int(max_products)

            ets_params = dict(seasonal_periods=12, trend="add", seasonal="add", damped_trend=False, min_obs=24)

            run_btn = st.button("▶️ Ejecutar comparación global", type="primary")

            if run_btn:
                with st.spinner("Corriendo comparación global (puede tardar según la cantidad de productos)..."):
                    per_sku, summary_wins, summary_errors = run_portfolio_comparison(
                        res.demand_monthly,
                        sort_metric=sort_metric,
                        test_months=int(test_months_global),
                        ma_window=int(ma_window_global),
                        ets_params=ets_params,
                        max_products=max_products
                    )

                if per_sku.empty:
                    st.warning("No se generaron resultados (revisa data/parametros).")
                else:
                    st.success(f"✅ Resultados generados para {per_sku['Codigo'].nunique():,} productos.")

                    c1, c2 = st.columns([1, 1])

                    with c1:
                        st.markdown("**Ganadores por ABC**")
                        st.dataframe(summary_wins, use_container_width=True, height=320)

                    with c2:
                        st.markdown("**Error del ganador (promedio y ponderado por demanda)**")
                        st.dataframe(summary_errors, use_container_width=True, height=320)

                    st.markdown("**Detalle por producto (ganador + ABC + métricas)**")
                    st.dataframe(
                        per_sku.sort_values(["ABC", "Demanda_Total"], ascending=[True, False]),
                        use_container_width=True,
                        height=420
                    )


        # ==========================================================
        # TAB 7: STOCK + DIAGNÓSTICO
        # ==========================================================
        with tab_stock_diag:
            st.subheader("🏢 Stock mensual del producto (empresa consolidada)")

            stock = res.stock_monthly
            if stock is None or stock.empty:
                st.warning("No se generó stock mensual (revisa columna Saldo_unid).")
            else:
                splot = stock[stock["Codigo"] == str(prod_sel)].copy()
                if splot.empty:
                    st.info("No hay stock mensual para ese producto.")
                else:
                    fig_stock = px.line(
                        splot, x="Mes", y="Stock_Unid", markers=True,
                        title=f"Stock mensual (Saldo_unid consolidado) - Producto {prod_sel}"
                    )
                    st.plotly_chart(fig_stock, use_container_width=True)

            st.divider()

            st.subheader("🧾 Diagnóstico: Guías de remisión")
            guia = res.movements[res.movements["Documento"].astype(str).str.strip() == config.GUIDE_DOC].copy()
            if guia.empty:
                st.info("No se encontraron guías de remisión en los archivos cargados.")
                return

            with st.expander("🔎 Muestra de guías (filas)", expanded=False):
                cols = [
                    "Fecha", "Codigo", "Bodega", "Documento", "Numero",
                    "Entrada_unid", "Salida_unid", "Tipo_Guia", "Guia_Salida_Externa_Unid"
                ]
                st.dataframe(guia[cols].sort_values("Fecha").head(300), use_container_width=True)


        # ==========================================================
        # TAB 8: RECOMENDACIÓN DE PRODUCCIÓN
        # ==========================================================

        with tab_reco:
            st.subheader("🧾 Recomendación de producción (t+1)")

            # Data mensual para el producto
            dm = res.demand_monthly.copy()
            dm["Codigo"] = dm["Codigo"].astype(str).str.strip()
            hist = dm[dm["Codigo"] == str(prod_sel)][["Mes", "Demanda_Unid"]].copy().sort_values("Mes")

            if hist.empty:
                st.info("No hay serie mensual para este producto.")
            else:
                # Stock actual (último mes disponible)
                stock = res.stock_monthly.copy() if res.stock_monthly is not None else pd.DataFrame()
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
                lead_time = st.selectbox("Lead time (meses)", options=[1, 2, 3], index=0)
                service_level = policy_service_level_by_abc(abc_class)
                z = z_from_service_level(service_level)

                st.caption(f"ABC del producto: **{abc_class}** → Nivel de servicio por política: **{int(service_level*100)}%** (Z≈{z})")

                # Parámetros de evaluación para elegir ganador
                test_months = st.slider("Backtest para elegir ganador (últimos meses)", 6, 24, 12, 1, key="reco_test_months")
                ma_window = st.selectbox("Ventana MA (baselines)", options=[3, 6], index=0, key="reco_ma_window")

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


                st.markdown("### 🗓️ Línea de tiempo de la decisión de producción")

                st.markdown(f"""
                **Producto:** `{prod_sel}`  

                | Etapa | Mes | Qué sucede |
                |---|---|---|
                | 📊 Datos históricos | **{hist['Mes'].max().strftime('%Y-%m')}** | Último mes cerrado usado para el modelo |
                | 🏭 Producción | **Mes intermedio** | Se ejecuta la producción recomendada |
                | 📦 Demanda objetivo | **Mes siguiente** | Se atiende la demanda pronosticada |
                """)

                st.info(
                    "ℹ️ La producción recomendada **no se realiza en un solo día**. "
                    "Corresponde a una **planificación mensual agregada**, que puede ejecutarse "
                    "de forma distribuida durante el mes previo al mes de demanda."
                )





                if cmp.empty or winner == "N/A" or np.isnan(winner_mae):
                    st.warning("No se pudo seleccionar un modelo ganador (revisa longitud de serie).")
                else:
                    # Pronóstico t+1 con el ganador
                    yhat = forecast_next_month_with_winner(hist, winner, int(ma_window), ets_params, rf_params)

                    # Stock de seguridad usando MAE como proxy (σ ≈ MAE)
                    sigma = float(max(0.0, winner_mae))
                    ss = float(z * sigma * np.sqrt(float(lead_time)))

                    # Producción recomendada
                    prod_reco = max(0.0, yhat + ss - float(stock_actual))
                    prod_reco_int = int(np.ceil(prod_reco))

                    # Mostrar resultados
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Modelo ganador (MAE)", winner)
                    c2.metric("Forecast t+1 (unid)", f"{yhat:,.0f}")
                    c3.metric("MAE (proxy σ)", f"{sigma:,.2f}")
                    c4.metric("Stock actual (unid)", f"{stock_actual:,.0f}")

                    c5, c6, c7 = st.columns(3)
                    c5.metric("Z", f"{z}")
                    c6.metric("Stock seguridad (SS)", f"{ss:,.0f}")
                    c7.metric("✅ Producción recomendada", f"{prod_reco_int:,.0f}")

                    st.markdown("### Detalle cálculo")
                    det = pd.DataFrame([{
                        "Producto": str(prod_sel),
                        "ABC": abc_class,
                        "LeadTime_meses": int(lead_time),
                        "ServiceLevel_%": int(service_level * 100),
                        "Z": z,
                        "Forecast_t+1": yhat,
                        "MAE_ganador": sigma,
                        "Stock_Seguridad": ss,
                        "Stock_Actual": stock_actual,
                        "Produccion_Recomendada": prod_reco_int
                    }])
                    st.dataframe(det, use_container_width=True)

                    with st.expander("📌 Backtest: métricas comparadas (para justificar ganador)", expanded=False):
                        st.dataframe(cmp, use_container_width=True)

                    with st.expander("📈 Backtest ganador: Real vs Predicho", expanded=False):
                        fig = px.line(pred_best, x="Mes_target", y=["y_true", "y_pred"], markers=True,
                                    title=f"Ganador vs Real - {winner} (Backtest)")
                        st.plotly_chart(fig, use_container_width=True)


        # ==========================================================
        # TAB 9: RECOMENDACIÓN MASIVA
        # ==========================================================


        with Reco_Masiva:
            st.subheader("📋 Recomendación masiva (según ABC seleccionado)")

            dm = res.demand_monthly.copy()
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
                    lead_time = st.selectbox("Lead time (meses)", options=[1, 2, 3], index=0, key="mass_lt")

                with c2:
                    test_months = st.slider("Backtest (últimos meses)", 6, 24, 12, 1, key="mass_test")

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

                        # Stock mensual
                        stock = res.stock_monthly.copy() if res.stock_monthly is not None else pd.DataFrame()
                        if not stock.empty:
                            stock["Codigo"] = stock["Codigo"].astype(str).str.strip()
                            stock = stock.sort_values(["Codigo", "Mes"])

                        rows = []
                        for cod in codigos:
                            hist = dm[dm["Codigo"] == str(cod)][["Mes", "Demanda_Unid"]].copy().sort_values("Mes")
                            if hist.empty:
                                continue

                            # Elegir ganador por MAE (baselines + ETS + RF) y extraer MAE del ganador
                            # OJO: usamos ma_window fijo (puedes exponerlo si quieres)
                            ma_window = 3
                            bt_base = backtest_baselines_1step(hist, y_col="Demanda_Unid", test_months=int(test_months), ma_window=int(ma_window))
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

                            # ABC + política Z
                            row_abc = abc_work[abc_work["Codigo"] == str(cod)]
                            abc_class = str(row_abc.iloc[0]["ABC"]) if not row_abc.empty else "C"
                            demanda_total = float(row_abc.iloc[0]["Demanda_Total"]) if not row_abc.empty else 0.0

                            service_level = policy_service_level_by_abc(abc_class)
                            z = z_from_service_level(service_level)

                            # SS con MAE como proxy σ
                            sigma = float(max(0.0, mae_win if np.isfinite(mae_win) else 0.0))
                            ss = float(z * sigma * np.sqrt(float(lead_time)))

                            # Producción recomendada
                            prod_reco = max(0.0, yhat + ss - stock_actual)
                            prod_reco_int = int(np.ceil(prod_reco))

                            rows.append({
                                "Codigo": str(cod),
                                "ABC": abc_class,
                                "Modelo_Ganador": winner,
                                "Forecast_t+1": yhat,
                                "MAE_ganador": sigma,
                                "Z": z,
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
                        k1.metric("Productos evaluados", f"{reco_df['Codigo'].nunique():,}")
                        k2.metric("Con riesgo quiebre", f"{int(reco_df['RIESGO_QUIEBRE'].sum()):,}")
                        k3.metric("Producción total sugerida", f"{int(reco_df['Produccion_Recomendada'].sum()):,}")

                        st.dataframe(reco_df, use_container_width=True, height=520)

                        with st.expander("⬇️ Descargar tabla (CSV)", expanded=False):
                            csv = reco_df.to_csv(index=False).encode("utf-8")
                            st.download_button("Descargar recomendaciones.csv", csv, file_name="recomendaciones.csv", mime="text/csv")


        # ==========================================================
        # TAB 10: VALIDACIÓN RETROSPECTIVA DE LA POLÍTICA
        # ==========================================================


        with Valida_Retro:
            st.subheader("🧪 Validación retrospectiva de la política (simulación)")

            dm = res.demand_monthly.copy()
            dm["Codigo"] = dm["Codigo"].astype(str).str.strip()
            hist = dm[dm["Codigo"] == str(prod_sel)][["Mes", "Demanda_Unid"]].copy().sort_values("Mes")

            if hist.empty:
                st.info("No hay serie mensual para este producto.")
            else:
                # Stock mensual del producto (empresa)
                stock_p = pd.DataFrame()
                if res.stock_monthly is not None and not res.stock_monthly.empty:
                    stock_p = res.stock_monthly.copy()
                    stock_p["Codigo"] = stock_p["Codigo"].astype(str).str.strip()
                    stock_p = stock_p[stock_p["Codigo"] == str(prod_sel)][["Mes", "Stock_Unid"]].copy().sort_values("Mes")

                # ABC del producto
                row = abc_df[abc_df["Codigo"] == str(prod_sel)]
                abc_class = str(row.iloc[0]["ABC"]) if not row.empty else "C"

                eval_months = st.slider("Meses a simular (últimos)", 6, 24, 12, 1, key="sim_eval")
                test_months = st.slider("Meses para elegir ganador (backtest)", 6, 24, 12, 1, key="sim_bt")
                ma_window = st.selectbox("Ventana media móvil (baselines)", options=[3, 6], index=0, key="sim_ma")
                lead_time = st.selectbox("Lead time (meses)", options=[1], index=0)

                run_sim = st.button("▶️ Ejecutar simulación (ganador automático por MAE)", type="primary", key="run_sim")

                if run_sim:
                    with st.spinner("Calculando ganador por MAE y simulando política..."):
                        # 1) Backtests para elegir ganador por MAE
                        bt_base = backtest_baselines_1step(hist, y_col="Demanda_Unid", test_months=int(test_months), ma_window=int(ma_window))

                        ets_params = dict(seasonal_periods=12, trend="add", seasonal="add", damped_trend=False, min_obs=24)
                        ets = ETSForecaster(**ets_params)
                        bt_ets = backtest_ets_1step(hist, y_col="Demanda_Unid", test_months=int(test_months), ets=ets)

                        rf_params = dict(n_estimators=400, min_obs=24, min_samples_leaf=1, random_state=42)
                        rf = RFForecaster(**rf_params)
                        bt_rf = backtest_rf_1step(hist, y_col="Demanda_Unid", test_months=int(test_months), rf=rf)

                        # 2) Unir métricas y escoger ganador por MAE
                        cmp = compare_models_metrics(bt_base.metrics, bt_ets.metrics, bt_rf.metrics, sort_by="MAE")
                        if cmp.empty:
                            st.warning("No se pudo determinar ganador (métricas vacías).")
                            st.stop()

                        winner = str(cmp.iloc[0]["Modelo"])
                        mae_win = float(pd.to_numeric(cmp.iloc[0].get("MAE", np.nan), errors="coerce"))

                        st.success(f"Ganador por MAE: **{winner}**  |  MAE: **{mae_win:.3f}**")

                        # 3) Simular política usando winner y sigma_fixed = MAE ganador
                        df_sim, kpis = simulate_policy_backtest_1step(
                            hist=hist,
                            stock_series=stock_p,
                            winner=winner,
                            abc_class=abc_class,
                            lead_time=int(lead_time),
                            eval_months=int(eval_months),
                            ets_params=ets_params,
                            rf_params=rf_params,
                            sigma_fixed=mae_win,   # 👈 CLAVE
                        )

                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Meses con quiebre", f"{kpis['Meses_con_quiebre']}/{kpis['Meses_evaluados']}")
                    c2.metric("Fill Rate", f"{kpis['FillRate_%']:.1f}%")
                    c3.metric("Unidades faltantes", f"{kpis['Unidades_faltantes']:,.0f}")
                    c4.metric("Inventario promedio", f"{kpis['Inventario_promedio']:,.0f}")

                    st.dataframe(df_sim, use_container_width=True, height=420)

                    fig_stock = px.line(df_sim, x="Mes_target", y="Stock_fin", markers=True, title="Stock fin mensual (simulado)")
                    st.plotly_chart(fig_stock, use_container_width=True)

                    fig_lost = px.bar(df_sim, x="Mes_target", y="Faltante", title="Unidades faltantes por mes (quiebres)")
                    st.plotly_chart(fig_lost, use_container_width=True)



        # ==========================================================
        # TAB 11: COMPARATIVA RETROSPECTIVA SIN SISTEMA VS CON SISTEMA
        # ==========================================================

        with ComparaRetroEntreSistema:
            st.subheader("⚖️ Comparativa retrospectiva: Sin sistema vs Con sistema (costos)")

            dm = res.demand_monthly.copy()
            dm["Codigo"] = dm["Codigo"].astype(str).str.strip()
            hist = dm[dm["Codigo"] == str(prod_sel)][["Mes", "Demanda_Unid"]].copy().sort_values("Mes")

            # stock mensual producto
            stock_p = pd.DataFrame()
            if res.stock_monthly is not None and not res.stock_monthly.empty:
                stock_p = res.stock_monthly.copy()
                stock_p["Codigo"] = stock_p["Codigo"].astype(str).str.strip()
                stock_p = stock_p[stock_p["Codigo"] == str(prod_sel)][["Mes", "Stock_Unid"]].copy().sort_values("Mes")

            row = abc_df[abc_df["Codigo"] == str(prod_sel)]
            abc_class = str(row.iloc[0]["ABC"]) if not row.empty else "C"

            winner = st.session_state.get("winner_model", "ETS(Holt-Winters)")  # si guardas winner, sino pon uno fijo

            eval_months = st.slider("Meses a evaluar (últimos)", 6, 24, 12, 1, key="cmp_eval")
            cost_stock_unit = st.number_input("Costo inventario por unidad (proxy)", min_value=0.0, value=1.0, step=0.5)
            cost_stockout_unit = st.number_input("Costo quiebre por unidad (proxy)", min_value=0.0, value=5.0, step=0.5)

            run_cmp = st.button("▶️ Ejecutar comparativa", type="primary", key="run_cmp")

            if run_cmp and not hist.empty:
                df_cmp, s = simulate_compare_policy_vs_baseline(
                    hist=hist,
                    stock_series=stock_p,
                    abc_class=abc_class,
                    winner=winner,
                    eval_months=int(eval_months),
                    cost_stock_unit=float(cost_stock_unit),
                    cost_stockout_unit=float(cost_stockout_unit),
                    ma_window=3,
                    test_months_for_mae=12,
                )

                c1, c2, c3 = st.columns(3)
                c1.metric("Ahorro costo total", f"{s['Ahorro_CostoTotal']:,.1f}")
                c2.metric("Mejora Fill Rate (pp)", f"{s['Mejora_FillRate_pp']:.1f}")
                c3.metric("Reducción faltantes", f"{s['Reduccion_Faltantes']:,.0f}")

                st.dataframe(df_cmp, use_container_width=True, height=420)

                fig_cost = px.line(df_cmp, x="Mes", y=["Base_Costo_total", "Sys_Costo_total"], markers=True,
                                title="Costo total mensual: Baseline vs Sistema")
                st.plotly_chart(fig_cost, use_container_width=True)

                fig_lost = px.bar(df_cmp, x="Mes", y=["Base_Faltante", "Sys_Faltante"], barmode="group",
                                title="Faltantes (quiebre) por mes: Baseline vs Sistema")
                st.plotly_chart(fig_lost, use_container_width=True)



            st.divider()
            st.subheader("📦 Portafolio: Comparativa costos SOLO ABC A (agregado)")

            eval_months_port = st.slider("Meses a evaluar portafolio (últimos)", 6, 24, 12, 1, key="port_eval")
            cost_stock_unit_port = st.number_input("Costo inventario por unidad (proxy) - Portafolio", min_value=0.0, value=1.0, step=0.5, key="port_cinv")
            cost_stockout_unit_port = st.number_input("Costo quiebre por unidad (proxy) - Portafolio", min_value=0.0, value=5.0, step=0.5, key="port_cbrk")

            winner_mode = st.selectbox(
                "Modelo en portafolio",
                options=["AUTO", "ETS(Holt-Winters)", "RandomForest", "Naive", "Seasonal12", "MA3", "MA6"],
                index=0,
                key="port_winner_mode"
            )

            max_products_port = st.selectbox(
                "Cantidad de productos ABC A a procesar (performance)",
                options=[20, 50, 100, 200, "Todos"],
                index=1,
                key="port_max"
            )
            max_products_port = None if max_products_port == "Todos" else int(max_products_port)

            run_port = st.button("▶️ Ejecutar portafolio ABC A", type="primary", key="run_port_abcA")

            if run_port:
                with st.spinner("Calculando portafolio ABC A (puede tardar)..."):
                    resumenA, detalleA = run_portfolio_cost_comparison_abcA(
                        demand_monthly=res.demand_monthly,
                        stock_monthly=res.stock_monthly,
                        abc_df=abc_df,   # ya existe arriba en tu render
                        eval_months=int(eval_months_port),
                        cost_stock_unit=float(cost_stock_unit_port),
                        cost_stockout_unit=float(cost_stockout_unit_port),
                        winner_mode=str(winner_mode),
                        ma_window=3,
                        test_months_for_mae=12,
                        max_products=max_products_port,
                    )

                if resumenA.empty:
                    st.warning("No se generó portafolio (revisa si hay productos ABC A con historia suficiente).")
                else:
                    st.success("✅ Portafolio ABC A generado.")
                    st.dataframe(resumenA, use_container_width=True)

                    k1, k2, k3 = st.columns(3)
                    k1.metric("Ahorro total", f"{float(resumenA.iloc[0]['Ahorro_total']):,.1f}")
                    k2.metric("FillRate base", f"{float(resumenA.iloc[0]['FillRate_Base_%']):.1f}%")
                    k3.metric("FillRate sistema", f"{float(resumenA.iloc[0]['FillRate_Sistema_%']):.1f}%")

                    st.markdown("### 🔝 Top productos (mayor ahorro)")
                    st.dataframe(detalleA.head(30), use_container_width=True, height=420)

                    with st.expander("⬇️ Exportar detalle portafolio (CSV)", expanded=False):
                        csv_det = detalleA.to_csv(index=False).encode("utf-8-sig")  # utf-8-sig para Excel
                        st.download_button(
                            "Descargar detalle_portafolio_ABC_A.csv",
                            data=csv_det,
                            file_name="detalle_portafolio_ABC_A.csv",
                            mime="text/csv",
                            key="dl_detalle_portafolio_A"
                        )

                    fig_cost_port = px.bar(
                        resumenA,
                        x=["CostoTotal_Base", "CostoTotal_Sistema"],
                        title="Costo total portafolio ABC A: Base vs Sistema"
                    )
                    st.plotly_chart(fig_cost_port, use_container_width=True)


