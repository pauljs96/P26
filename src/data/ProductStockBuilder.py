"""Construcción de stock mensual para Dataset v4.

Stock mensual por PRODUCTO se calcula tomando el último Stock_posterior
del mes para cada producto.

Entrada: DataFrame v4 con Fecha, Producto_id, Stock_posterior
Salida: DataFrame con (Producto_id, Año, Mes, Stock_final)
"""

from __future__ import annotations
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class ProductStockBuilder:
    """Stock mensual por PRODUCTO agrupado a nivel empresa."""

    def build_monthly(self, df: pd.DataFrame) -> pd.DataFrame:
        """Construye tabla de stock mensual.
        
        Para cada (Producto_id, Año-Mes), toma el último Stock_posterior del mes.
        """
        if df is None or df.empty:
            logger.warning("ProductStockBuilder: DataFrame vacío")
            return pd.DataFrame(columns=["Producto_id", "Año", "Mes", "Stock_final"])

        d = df.copy()

        # Asegurar por lo menos Fecha como datetime
        if "Fecha" in d.columns:
            d["Fecha"] = pd.to_datetime(d["Fecha"], errors="coerce")
        
        d = d.dropna(subset=["Fecha"])

        # Normalizar
        d["Producto_id"] = d["Producto_id"].astype(str).str.strip()
        d["Stock_posterior"] = pd.to_numeric(d["Stock_posterior"], errors="coerce").fillna(0.0)

        # Extraer año y mes
        d["Año"] = d["Fecha"].dt.year
        d["Mes"] = d["Fecha"].dt.month

        # Ordenar por fecha para tomar el último valor del mes
        d = d.sort_values(["Producto_id", "Año", "Mes", "Fecha"])

        # Tomar último stock del mes para cada producto
        monthly = (
            d.groupby(["Producto_id", "Año", "Mes"], as_index=False)
             .tail(1)[["Producto_id", "Año", "Mes", "Stock_posterior"]]
             .rename(columns={"Stock_posterior": "Stock_final"})
             .reset_index(drop=True)
        )

        logger.info(
            f"ProductStockBuilder: {len(monthly)} filas mensuales generadas "
            f"({monthly['Producto_id'].nunique()} productos)"
        )

        return monthly.sort_values(["Producto_id", "Año", "Mes"]).reset_index(drop=True)
