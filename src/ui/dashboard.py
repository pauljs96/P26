"""Dashboard Streamlit (MVP) - Proyecto de tesis.

Objetivo del dashboard en esta etapa:
- Cargar múltiples CSV (2021-2025)
- Ejecutar el pipeline de datos (carga -> limpieza -> reconciliación -> agregaciones)
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


def _normalize_text(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip()


def build_monthly_components(movements: pd.DataFrame, codigo: str) -> pd.DataFrame:
    """Construye la tabla mensual de componentes de demanda para un producto.

    Componentes:
    - Venta Tienda Sin Doc: sum(Salida_unid)
    - Salida por Consumo:   sum(Salida_unid)
    - Guía externa neta:    max(0, sum(Salida_unid) - sum(Entrada_unid)) por (Documento, Numero, Codigo)
      asignada al mes de la última fecha del grupo.
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
    # Solo cuenta cuando Tipo_Guia == "VENTA_EXTERNA"
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

    with st.expander("🔍 Debug: detalle de guías externas por mes", expanded=False):
        det = df[(df["Documento"] == config.GUIDE_DOC)].copy()
        det = det[["Fecha", "Mes", "Numero", "Bodega", "Entrada_unid", "Salida_unid", "Tipo_Guia", "Guia_Salida_Externa_Unid"]]
        det = det.sort_values(["Mes", "Fecha", "Numero"])
        st.dataframe(det, use_container_width=True, height=350)



    # Unir y completar meses faltantes
    # Base de meses existentes para el producto
   #### months = pd.DataFrame({"Mes": sorted(df["Mes"].unique())})
    # Base completa de meses (para mostrar meses sin movimiento como 0)
    months = pd.DataFrame({
        "Mes": pd.date_range("2021-01-01", "2025-05-01", freq="MS")
    })


    out = months.merge(venta, on="Mes", how="left")                 .merge(consumo, on="Mes", how="left")                 .merge(guia_m, on="Mes", how="left")

    out["Venta_Tienda"] = out["Venta_Tienda"].fillna(0.0)
    out["Consumo"] = out["Consumo"].fillna(0.0)
    out["Guia_Externa"] = out["Guia_Externa"].fillna(0.0)

    out["Demanda_Total"] = out["Venta_Tienda"] + out["Consumo"] + out["Guia_Externa"]

    return out.sort_values("Mes").reset_index(drop=True)



class Dashboard:
    def render(self):
        st.set_page_config(page_title="Planificación - MVP", layout="wide")
        st.title("📦 Sistema de Planificación (MVP)")

        st.write(
            "Sube tus archivos CSV (2021–2025). "
            "Luego selecciona un producto para ver la demanda mensual descompuesta."
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

     

        # ------------------------------
        # Tabla de componentes por mes
        # ------------------------------
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

        # Gráfico comparativo de componentes
        st.subheader("📊 Componentes (Venta / Consumo / Guía externa)")
        comp_long = comp.melt(id_vars=["Mes"], value_vars=["Venta_Tienda", "Consumo", "Guia_Externa"],
                              var_name="Componente", value_name="Unidades")
        fig_comp = px.line(comp_long, x="Mes", y="Unidades", color="Componente", markers=True,
                           title=f"Componentes de demanda - Producto {prod_sel}")
        st.plotly_chart(fig_comp, use_container_width=True)



        # ------------------------------
        # STOCK mensual por producto (empresa consolidada)
        # ------------------------------
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



        # ------------------------------
        # Resumen de guías reconciliadas (diagnóstico)
        # ------------------------------
        st.subheader("🧾 Diagnóstico: Guías de remisión")
        guia = res.movements[res.movements["Documento"].astype(str).str.strip() == config.GUIDE_DOC].copy()
        if guia.empty:
            st.info("No se encontraron guías de remisión en los archivos cargados.")
            return


        with st.expander("🔎 Muestra de guías (filas)", expanded=False):
            cols = ["Fecha", "Codigo", "Bodega", "Documento", "Numero", "Entrada_unid", "Salida_unid", "Tipo_Guia", "Guia_Salida_Externa_Unid"]
            st.dataframe(guia[cols].sort_values("Fecha").head(300), use_container_width=True)
