"""Construcción de stock mensual por bodega.

Definición:
- Stock mensual (producto-bodega-mes) = último Saldo_unid registrado en el mes
  (por fecha; si hay empates misma fecha, usamos el último por orden de aparición).

Esto refleja el stock al cierre del mes en cada bodega.
"""

from __future__ import annotations
import pandas as pd


class StockBuilder:
    def build_monthly(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()

        d = df.copy()
        d["Mes"] = d["Fecha"].dt.to_period("M").dt.to_timestamp()

        # Orden por fecha y por índice (para "último del día/mes")
        d["__row"] = range(len(d))
        d = d.sort_values(["Codigo", "Bodega", "Mes", "Fecha", "__row"])

        # Tomar último saldo por mes
        last = (
            d.groupby(["Codigo", "Bodega", "Mes"], as_index=False)
             .tail(1)[["Codigo", "Bodega", "Mes", "Saldo_unid"]]
             .rename(columns={"Saldo_unid": "Stock_Unid"})
             .reset_index(drop=True)
        )

        return last.sort_values(["Codigo", "Bodega", "Mes"])
