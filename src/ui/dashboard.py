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
    months = pd.DataFrame({"Mes": pd.date_range("2021-01-01", "2025-05-01", freq="MS")})

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


class Dashboard:
    def render(self):
        st.set_page_config(page_title="Planificación - MVP", layout="wide")
        st.title("📦 Sistema de Planificación (MVP)")

        st.write(
            "Sube tus archivos CSV (2021–2025). "
            "Luego selecciona un producto para ver demanda, pronósticos y diagnósticos."
        )

        files = st.file_uploader(
            "Sube uno o varios CSV",
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

        # Sidebar filtros
        st.sidebar.header("Filtros")
        productos = sorted(res.movements["Codigo"].dropna().unique().tolist())
        prod_sel = st.sidebar.selectbox("Producto (Código)", options=productos)

        # Defaults para evitar UnboundLocalError entre tabs / reruns
        if "ets_test_months" not in st.session_state:
            st.session_state["ets_test_months"] = 12
        if "global_test_months" not in st.session_state:
            st.session_state["global_test_months"] = 12

        # ------------------------------
        # TABS
        # ------------------------------
        tab_demanda, tab_baselines, tab_ets,tab_ml,Tab_Comparativa,ResumenComparativa ,tab_stock_diag = st.tabs([
            "🧩 Demanda y Componentes",
            "🔮 Baselines y Backtest",
            "📈 Holt–Winters (ETS)",
            "🤖 Random Forest (RF)",
            "🏆 Comparativa ETS vs Baselines vs RF",
            "📊 Resumen Comparativa",
            "🏢 Stock y Diagnóstico"
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
        # TAB 6: STOCK + DIAGNÓSTICO
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
