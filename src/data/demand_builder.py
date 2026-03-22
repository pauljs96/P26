"""Construcción de demanda mensual para Dataset v4.

Demanda mensual se calcula como la suma de unidades vendidas (Tipo_movimiento='Venta')
agrupadas por (Producto_id, Año-Mes).

Columnas de entrada requeridas:
- Fecha, Producto_id, Tipo_movimiento, Cantidad

Columnas de salida:
- Producto_id, Año, Mes, Cantidad_total (demanda)
"""

from __future__ import annotations
import pandas as pd
import logging

logger = logging.getLogger(__name__)

from src.utils.config import MOVEMENT_SALE


class DemandBuilder:
    def build_monthly(self, df: pd.DataFrame) -> pd.DataFrame:
        """Construye tabla de demanda mensual desde transacciones diarias.
        
        Entrada: DataFrame v4 con Fecha, Producto_id, Tipo_movimiento, Cantidad
        Salida: DataFrame con (Producto_id, Año, Mes, Cantidad_total)
        """
        if df is None or df.empty:
            logger.warning("DemandBuilder: DataFrame vacío")
            return pd.DataFrame(columns=["Producto_id", "Año", "Mes", "Cantidad_total"])

        d = df.copy()

        # Filtrar solo ventas (demanda)
        d = d[d["Tipo_movimiento"] == MOVEMENT_SALE].copy()
        
        if d.empty:
            logger.warning("DemandBuilder: No hay movimientos de tipo 'Venta'")
            return pd.DataFrame(columns=["Producto_id", "Año", "Mes", "Cantidad_total"])

        # Extraer año y mes
        d["Año"] = d["Fecha"].dt.year
        d["Mes"] = d["Fecha"].dt.month

        # Normalizar columnas de texto
        d["Producto_id"] = d["Producto_id"].astype(str).str.strip()

        # Agregar por (Producto_id, Año, Mes)
        monthly = (
            d.groupby(["Producto_id", "Año", "Mes"], as_index=False)["Cantidad"]
             .sum()
             .rename(columns={"Cantidad": "Cantidad_total"})
        )

        # Asegurar que Cantidad_total sea siempre positivo (es demanda)
        monthly["Cantidad_total"] = monthly["Cantidad_total"].abs()

        logger.info(
            f"DemandBuilder: {len(monthly)} filas mensuales generadas "
            f"({monthly['Producto_id'].nunique()} productos)"
        )

        return monthly.sort_values(["Producto_id", "Año", "Mes"]).reset_index(drop=True)
