"""Limpieza y normalización para Dataset v4.

Convierte el DataFrame v4 a esquema estable:
- Validación de columnas requeridas
- Conversión de tipos (Fecha, cantidad, stock)
- Validación de coherencia de stock
- Filtrado de registros inválidos
"""

from __future__ import annotations
import pandas as pd
import numpy as np
import logging

from src.utils.config import REQUIRED_COLUMNS, OPTIONAL_COLUMNS, STOCK_TOLERANCE

logger = logging.getLogger(__name__)


def _to_numeric(series: pd.Series) -> pd.Series:
    """Convierte serie a numérico, limpiando espacios y decimales."""
    s = series.astype(str).str.replace(" ", "", regex=False)
    s = s.str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")


class DataCleaner:
    def clean(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """Limpia y normaliza DataFrame v4."""
        
        if df_raw is None or df_raw.empty:
            logger.error("❌ DataCleaner.clean: DataFrame vacío o None")
            return pd.DataFrame()

        df = df_raw.copy()
        logger.info(f"📊 DataCleaner recibió: {len(df)} filas, {len(df.columns)} columnas")
        
        # 1. Validar columnas requeridas
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            error_msg = (
                f"COLUMNAS REQUERIDAS FALTANTES: {missing}\n"
                f"Disponibles: {list(df.columns)}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        # 2. Convertir tipos
        df["Fecha"] = pd.to_datetime(df["Fecha"], format="%Y-%m-%d", errors="coerce")
        
        # Si hay NaNs en Fecha, intentar formato alternativo
        nan_mask = df["Fecha"].isna()
        if nan_mask.any():
            df.loc[nan_mask, "Fecha"] = pd.to_datetime(
                df.loc[nan_mask, "Fecha"],
                format="%d/%m/%Y",
                errors="coerce"
            )
        
        # Cantidad y Stock como float
        df["Cantidad"] = _to_numeric(df["Cantidad"])
        df["Stock_anterior"] = _to_numeric(df["Stock_anterior"])
        df["Stock_posterior"] = _to_numeric(df["Stock_posterior"])
        
        # Columnas opcionales: convertir si existen
        for col in ["Precio_unitario", "Descuento_pct", "Valor_total", "Costo_unitario"]:
            if col in df.columns:
                df[col] = _to_numeric(df[col])
        
        # 3. Validar coherencia de stock
        # Para Venta: Stock_anterior - Cantidad = Stock_posterior
        # Para Producción: Stock_anterior + Cantidad = Stock_posterior
        
        venta_mask = df["Tipo_movimiento"] == "Venta"
        prod_mask = df["Tipo_movimiento"] == "Producción"
        
        # Validar ventas
        if venta_mask.any():
            expected_stock = df.loc[venta_mask, "Stock_anterior"] - df.loc[venta_mask, "Cantidad"]
            actual_stock = df.loc[venta_mask, "Stock_posterior"]
            diff = abs((expected_stock - actual_stock) / (actual_stock + 1).abs())  # Evitar división por cero
            invalid_venta = diff > STOCK_TOLERANCE
            if invalid_venta.any():
                logger.warning(
                    f"⚠️ {invalid_venta.sum()} registros de Venta con incoherencia "
                    f"de stock (tolerancia={STOCK_TOLERANCE*100}%)"
                )
                df = df[~invalid_venta]
        
        # Validar producciones
        if prod_mask.any():
            expected_stock = df.loc[prod_mask, "Stock_anterior"] + df.loc[prod_mask, "Cantidad"]
            actual_stock = df.loc[prod_mask, "Stock_posterior"]
            diff = abs((expected_stock - actual_stock) / (actual_stock + 1).abs())
            invalid_prod = diff > STOCK_TOLERANCE
            if invalid_prod.any():
                logger.warning(
                    f"⚠️ {invalid_prod.sum()} registros de Producción con incoherencia "
                    f"de stock (tolerancia={STOCK_TOLERANCE*100}%)"
                )
                df = df[~invalid_prod]
        
        # 4. Filtrar registros inválidos
        df = df.dropna(subset=["Fecha", "Producto_id", "Tipo_movimiento", "Cantidad"])
        df = df[df["Producto_id"].astype(str).str.strip() != ""]
        df = df[df["Cantidad"].notna()]
        
        # 5. Espacios en strings
        str_cols = ["Producto_id", "Producto_nombre", "Tipo_movimiento"]
        for col in str_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
        
        final_count = len(df)
        pct = 100 * final_count / len(df_raw) if df_raw.shape[0] > 0 else 0
        
        logger.info(f"✓ DataCleaner: {final_count}/{len(df_raw)} filas ({pct:.1f}%)")
        logger.info(f"  Período: {df['Fecha'].min()} a {df['Fecha'].max()}")
        logger.info(f"  Productos: {df['Producto_id'].nunique()}")
        logger.info(f"  Ventas: {(df['Tipo_movimiento'] == 'Venta').sum()}")
        logger.info(f"  Producciones: {(df['Tipo_movimiento'] == 'Producción').sum()}")
        
        return df.reset_index(drop=True)
