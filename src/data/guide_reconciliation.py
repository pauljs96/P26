"""
Reconciliación de datos v4 (pass-through).

En Dataset v4:
- Tipo_movimiento ya está definido ('Venta' o 'Producción')
- No hay guías de remisión ni distinción interna/externa
- Stock ya es coherente

Por lo tanto, esta clase simplemente retorna el df sin cambios.
No se realiza ninguna transformación.
"""

from __future__ import annotations
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class GuideReconciler:
    def reconcile(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pass-through: retorna el dataframe sin cambios.
        
        Para v4, la reconciliación no es necesaria.
        """
        if df is None or df.empty:
            logger.warning("GuideReconciler: DataFrame vacío recibido")
            return pd.DataFrame()

        logger.info(f"GuideReconciler (pass-through): {len(df)} filas, sin transformaciones")
        return df
        out["Fecha"] = pd.to_datetime(out["Fecha"], format='%Y-%m-%d', errors='coerce')
        
        # Si hay NaNs, intentar con formato europeo DD/MM/YYYY
        nan_mask = out["Fecha"].isna()
        if nan_mask.any():
            out.loc[nan_mask, "Fecha"] = pd.to_datetime(
                out.loc[nan_mask, "Fecha"], 
                format='%d/%m/%Y', 
                errors='coerce'
            )

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
