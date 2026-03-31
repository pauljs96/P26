"""Análisis de ganancias y márgenes por producto y período.

Requiere columnas de Data.csv:
- Producto_id, Fecha, Cantidad, Precio_unitario, Costo_unitario, Valor_total, Tipo_movimiento
"""

from __future__ import annotations
import pandas as pd
import logging

logger = logging.getLogger(__name__)

from src.utils.config import MOVEMENT_SALE


class ProfitBuilder:
    def build_product_profit_monthly(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ganancia mensual por producto.
        
        Formulae:
        - Ganancia = Valor_total - (Costo_unitario * Cantidad)
        - Margen_pct = (Ganancia / Valor_total) * 100
        
        Entrada: DataFrame con Producto_id, Fecha, Cantidad, Precio_unitario, Costo_unitario, Valor_total, Tipo_movimiento
        Salida: DataFrame con (Producto_id, Año, Mes, Cantidad_total, Valor_total, Costo_total, Ganancia, Margen_pct)
        """
        if df is None or df.empty:
            logger.warning("ProfitBuilder: DataFrame vacío")
            return pd.DataFrame()

        # Verificar columnas requeridas
        required = ["Producto_id", "Fecha", "Cantidad", "Tipo_movimiento"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            logger.warning(f"ProfitBuilder: Faltan columnas {missing}")
            return pd.DataFrame()

        # Solo ventas
        d = df[df["Tipo_movimiento"] == MOVEMENT_SALE].copy()
        
        if d.empty:
            return pd.DataFrame()

        d["Año"] = d["Fecha"].dt.year
        d["Mes"] = d["Fecha"].dt.month

        # Normalizar numéricas
        d["Cantidad"] = pd.to_numeric(d["Cantidad"], errors="coerce").fillna(0)
        
        if "Valor_total" not in d.columns:
            d["Valor_total"] = 0
        else:
            d["Valor_total"] = pd.to_numeric(d["Valor_total"], errors="coerce").fillna(0)

        if "Costo_unitario" not in d.columns:
            d["Costo_unitario"] = 0
        else:
            d["Costo_unitario"] = pd.to_numeric(d["Costo_unitario"], errors="coerce").fillna(0)

        if "Precio_unitario" not in d.columns:
            d["Precio_unitario"] = 0
        else:
            d["Precio_unitario"] = pd.to_numeric(d["Precio_unitario"], errors="coerce").fillna(0)

        # Agregar por (Producto_id, Año, Mes)
        profit = d.groupby(["Producto_id", "Año", "Mes"], as_index=False).agg({
            "Cantidad": "sum",
            "Valor_total": "sum",
            "Costo_unitario": "mean",  # Costo promedio del producto ese mes
            "Precio_unitario": "mean"   # Precio promedio ese mes
        }).rename(columns={"Cantidad": "Cantidad_total"})

        # Calcular costo total = Cantidad * Costo_unitario promedio
        profit["Costo_total"] = profit["Cantidad_total"] * profit["Costo_unitario"]

        # Ganancia bruta
        profit["Ganancia"] = profit["Valor_total"] - profit["Costo_total"]

        # Margen %
        profit["Margen_pct"] = (
            (profit["Ganancia"] / (profit["Valor_total"] + 0.001)) * 100
        ).round(2)

        logger.info(
            f"ProfitBuilder: {len(profit)} registros, "
            f"Ganancia total: ${profit['Ganancia'].sum():.2f}, "
            f"Margen promedio: {profit['Margen_pct'].mean():.2f}%"
        )

        return profit.sort_values(["Producto_id", "Año", "Mes"]).reset_index(drop=True)

    def build_top_profitable_products(self, df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
        """Top N productos más rentables.
        
        Entrada: DataFrame con Producto_id, Cantidad, Precio_unitario, Costo_unitario, Valor_total, Tipo_movimiento
        Salida: DataFrame con (Producto_id, Cantidad_total, Valor_total, Costo_total, Ganancia, Margen_pct)
        """
        if df is None or df.empty:
            return pd.DataFrame()

        d = df[df["Tipo_movimiento"] == MOVEMENT_SALE].copy()
        
        if d.empty:
            return pd.DataFrame()

        d["Cantidad"] = pd.to_numeric(d["Cantidad"], errors="coerce").fillna(0)
        d["Valor_total"] = pd.to_numeric(d.get("Valor_total", 0), errors="coerce").fillna(0)
        d["Costo_unitario"] = pd.to_numeric(d.get("Costo_unitario", 0), errors="coerce").fillna(0)

        # Agregar por producto
        prod = d.groupby("Producto_id", as_index=False).agg({
            "Cantidad": "sum",
            "Valor_total": "sum",
            "Costo_unitario": "mean"
        }).rename(columns={"Cantidad": "Cantidad_total"})

        prod["Costo_total"] = prod["Cantidad_total"] * prod["Costo_unitario"]
        prod["Ganancia"] = prod["Valor_total"] - prod["Costo_total"]
        prod["Margen_pct"] = (
            (prod["Ganancia"] / (prod["Valor_total"] + 0.001)) * 100
        ).round(2)

        logger.info(f"ProfitBuilder.build_top_profitable_products: Top {top_n}")

        return prod.nlargest(top_n, "Ganancia").reset_index(drop=True)

    def build_channel_profit_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Análisis de rentabilidad por canal.
        
        Entrada: DataFrame con Canal_venta, Cantidad, Valor_total, Costo_unitario, Tipo_movimiento
        Salida: DataFrame con (Canal_venta, Cantidad_total, Valor_total, Costo_total, Ganancia, Margen_pct)
        """
        if df is None or df.empty or "Canal_venta" not in df.columns:
            logger.warning("ProfitBuilder: No hay columna 'Canal_venta'")
            return pd.DataFrame()

        d = df[df["Tipo_movimiento"] == MOVEMENT_SALE].copy()
        
        if d.empty:
            return pd.DataFrame()

        d["Cantidad"] = pd.to_numeric(d["Cantidad"], errors="coerce").fillna(0)
        d["Valor_total"] = pd.to_numeric(d.get("Valor_total", 0), errors="coerce").fillna(0)
        d["Costo_unitario"] = pd.to_numeric(d.get("Costo_unitario", 0), errors="coerce").fillna(0)

        # Agregar por canal
        channel = d.groupby("Canal_venta", as_index=False).agg({
            "Cantidad": "sum",
            "Valor_total": "sum",
            "Costo_unitario": "mean"
        }).rename(columns={"Cantidad": "Cantidad_total"})

        channel["Costo_total"] = channel["Cantidad_total"] * channel["Costo_unitario"]
        channel["Ganancia"] = channel["Valor_total"] - channel["Costo_total"]
        channel["Margen_pct"] = (
            (channel["Ganancia"] / (channel["Valor_total"] + 0.001)) * 100
        ).round(2)

        logger.info(f"ProfitBuilder.build_channel_profit_analysis: {len(channel)} canales")

        return channel.sort_values("Ganancia", ascending=False).reset_index(drop=True)
