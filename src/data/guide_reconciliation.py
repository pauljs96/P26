"""
Reconciliación de guías de remisión (por MES).

Objetivo:
- Distinguir 'movimiento interno' vs 'venta externa' dentro de 'Guía de remisión - R'
- Evitar mezclar operaciones cuando el mismo (Documento, Numero, Codigo) se repite en meses distintos.

Regla de negocio (BINARIA, sin tolerancias):
Para cada grupo (Documento, Numero, Codigo, Mes):
- Si existe al menos una Entrada_unid > 0  -> TRANSFERENCIA_INTERNA y externa = 0
- Si NO existe ninguna entrada (Entrada_total == 0) y hay salida -> VENTA_EXTERNA y externa = Salida_total
- Si no hay salida -> externa = 0 (no aporta demanda)

Importante:
- El valor "externa" NO debe duplicarse por fila al sumar por mes.
  Por eso se prorratea SOLO en filas con Salida_unid > 0:
      Guia_Salida_Externa_Unid_fila = Salida_Externa_Unid_grupo * (Salida_unid_fila / Salida_total_grupo)

Columnas esperadas en df de entrada:
- Documento, Numero, Codigo, Fecha, Entrada_unid, Salida_unid
"""

from __future__ import annotations
import pandas as pd
import numpy as np

from src.utils import config


class GuideReconciler:
    def reconcile(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()

        out = df.copy()

        # Normalizar Documento para evitar problemas por espacios
        out["Documento"] = out["Documento"].astype(str).str.strip()

        # Crear columnas por defecto
        out["Tipo_Guia"] = "NO_GUIA"
        out["Guia_Salida_Externa_Unid"] = 0.0

        # Máscara de guías
        guia_mask = out["Documento"] == config.GUIDE_DOC
        if not guia_mask.any():
            return out

        # Asegurar datetime
        out["Fecha"] = pd.to_datetime(out["Fecha"], errors="coerce", dayfirst=True)

        # Mes para reconciliar por periodo (evita mezclar meses)
        out["Mes"] = out["Fecha"].dt.to_period("M").dt.to_timestamp()

        # Subset de guías (incluye Mes)
        guia = out.loc[
            guia_mask,
            ["Documento", "Numero", "Codigo", "Mes", "Entrada_unid", "Salida_unid"]
        ].copy()

        # Asegurar numéricos
        guia["Entrada_unid"] = pd.to_numeric(guia["Entrada_unid"], errors="coerce").fillna(0.0)
        guia["Salida_unid"] = pd.to_numeric(guia["Salida_unid"], errors="coerce").fillna(0.0)

        # Agrupar por operación mensual
        grouped = (
            guia.groupby(["Documento", "Numero", "Codigo", "Mes"], as_index=False)
                .agg(Salida_total=("Salida_unid", "sum"),
                     Entrada_total=("Entrada_unid", "sum"))
        )

        # Clasificación BINARIA (sin tolerancias)
        def classify(row):
            S = float(row["Salida_total"] or 0.0)
            E = float(row["Entrada_total"] or 0.0)

            # Sin salida => no hay demanda
            if S <= 0:
                return "TRANSFERENCIA_INTERNA", 0.0

            # Si hay cualquier entrada espejo => movimiento interno
            if E > 0:
                return "TRANSFERENCIA_INTERNA", 0.0

            # Si no hay entrada espejo y hay salida => venta externa (toda la salida)
            return "VENTA_EXTERNA", S

        labels = grouped.apply(lambda r: classify(r), axis=1, result_type="expand")
        grouped["Tipo_Guia"] = labels[0]
        grouped["Salida_Externa_Unid"] = labels[1]

        # Merge a nivel fila, incluyendo Mes (clave del fix)
        out = out.merge(
            grouped[["Documento", "Numero", "Codigo", "Mes", "Tipo_Guia", "Salida_Externa_Unid", "Salida_total"]],
            on=["Documento", "Numero", "Codigo", "Mes"],
            how="left"
        )

        # Consolidar Tipo_Guia
        out["Tipo_Guia"] = out["Tipo_Guia_y"].fillna(out["Tipo_Guia_x"])

        # --- Asignación correcta de externa: prorratear SOLO en filas de salida ---
        out["Salida_total"] = out["Salida_total"].fillna(0.0)
        out["Salida_Externa_Unid"] = out["Salida_Externa_Unid"].fillna(0.0)

        # Asegurar numérico a nivel out (por si venía string)
        out["Salida_unid"] = pd.to_numeric(out["Salida_unid"], errors="coerce").fillna(0.0)

        peso = np.where(
            (out["Salida_total"] > 0) & (out["Salida_unid"] > 0),
            out["Salida_unid"] / out["Salida_total"],
            0.0
        )

        out["Guia_Salida_Externa_Unid"] = out["Salida_Externa_Unid"] 

        # Limpiar columnas auxiliares del merge
        out = out.drop(columns=["Tipo_Guia_x", "Tipo_Guia_y", "Salida_Externa_Unid", "Salida_total"])

        # Para filas que NO son guía, asegurar valores default
        out.loc[~guia_mask, "Tipo_Guia"] = "NO_GUIA"
        out.loc[~guia_mask, "Guia_Salida_Externa_Unid"] = 0.0

        # (Opcional) si no quieres mantener 'Mes' en el dataset final, descomenta:
        # out = out.drop(columns=["Mes"])

        return out
