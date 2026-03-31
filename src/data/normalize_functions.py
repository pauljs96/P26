"""Funciones de normalización unificadas para todos los formatos de CSV.

Compatible con:
- Dataset v4 (Inventario ML Completo) - formato actual
- Data.csv (nuevo formato con más dimensiones)

Centraliza transformaciones comunes como:
- Conversión de fechas (DD/MM/YYYY → YYYY-MM-DD)
- Mapeo de códigos de producto
- Estandarización de nombres de columnas
- Tipado de datos
"""

from __future__ import annotations
import pandas as pd
import logging
from datetime import datetime
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


# ============================================
# NORMALIZACIÓN DE FECHAS
# ============================================

def normalize_fecha(df: pd.DataFrame, fecha_col: str = "Fecha") -> pd.DataFrame:
    """
    Normaliza columna de fechas a YYYY-MM-DD.
    
    Detecta automáticamente:
    - DD/MM/YYYY (nuevo format Data.csv)
    - YYYY-MM-DD (v4 actual)
    - Otras variaciones
    """
    df = df.copy()
    
    if fecha_col not in df.columns:
        logger.warning(f"Columna '{fecha_col}' no encontrada. Columnas: {list(df.columns)}")
        return df
    
    # Muestra de la primera fecha para detección
    sample_fecha = str(df[fecha_col].iloc[0]) if len(df) > 0 else ""
    logger.info(f"Detectando formato de fecha. Muestra: {sample_fecha}")
    
    try:
        # Intentar DD/MM/YYYY primero (nuevo formato)
        if "/" in sample_fecha and len(sample_fecha) == 10:
            df[fecha_col] = pd.to_datetime(df[fecha_col], format="%d/%m/%Y", errors="coerce")
            logger.info("✓ Fecha detectada como DD/MM/YYYY")
        else:
            # Intentar YYYY-MM-DD (formato v4)
            df[fecha_col] = pd.to_datetime(df[fecha_col], errors="coerce")
            logger.info("✓ Fecha detectada como YYYY-MM-DD")
        
        # Verificar si hay nulls
        null_count = df[fecha_col].isna().sum()
        if null_count > 0:
            logger.warning(f"⚠ {null_count} fechas inválidas convertidas a NaT")
        
        return df
    except Exception as e:
        logger.error(f"Error normalizando fechas: {e}")
        return df


# ============================================
# NORMALIZACIÓN DE PRODUCTO_ID
# ============================================

def normalize_producto_codigo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detecta Producto_id o Producto_codigo y los normaliza a Producto_id.
    
    - Si existe Producto_codigo → rename a Producto_id
    - Si existe Producto_id → mantener igual
    - Limpia espacios en blanco
    """
    df = df.copy()
    
    # Detectar cuál columna existe
    if "Producto_codigo" in df.columns and "Producto_id" not in df.columns:
        logger.info("Detectado: Producto_codigo → renombrado a Producto_id")
        df = df.rename(columns={"Producto_codigo": "Producto_id"})
    
    # Limpiar espacios
    if "Producto_id" in df.columns:
        df["Producto_id"] = df["Producto_id"].astype(str).str.strip()
    
    return df


# ============================================
# NORMALIZACIÓN DE TIPO_MOVIMIENTO
# ============================================

def normalize_tipo_movimiento(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza valores de Tipo_movimiento:
    - "Venta" (mantiene igual)
    - "Producción" o "ProducciÃ³n" → "Producción"
    - Trimming de espacios
    """
    df = df.copy()
    
    if "Tipo_movimiento" not in df.columns:
        logger.warning("Columna Tipo_movimiento no encontrada")
        return df
    
    # Limpiar espacios y manejar encoding issues
    df["Tipo_movimiento"] = (
        df["Tipo_movimiento"]
        .astype(str)
        .str.strip()
        .str.replace("ProducciÃ³n", "Producción", case=False)
        .str.replace("Producción", "Producción", case=False)
    )
    
    # Log de valores únicos
    unique_vals = df["Tipo_movimiento"].unique()
    logger.info(f"Tipos de movimiento normalizados: {list(unique_vals)}")
    
    return df


# ============================================
# NORMALIZACIÓN DE CANTIDAD Y STOCK
# ============================================

def normalize_cantidades(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte Cantidad, Stock_anterior, Stock_posterior a int.
    Reemplaza valores inválidos con 0.
    """
    df = df.copy()
    
    cols_cantidad = ["Cantidad", "Stock_anterior", "Stock_posterior"]
    
    for col in cols_cantidad:
        if col not in df.columns:
            logger.warning(f"Columna {col} no encontrada")
            continue
        
        try:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
        except Exception as e:
            logger.error(f"Error normalizando {col}: {e}")
    
    return df


# ============================================
# NORMALIZACIÓN DE PRECIOS Y VALORES
# ============================================

def normalize_precios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte columnas de precio/valor/descuento a float.
    
    Columnas procesadas (si existen):
    - Precio_unitario
    - Costo_unitario
    - Valor_total
    - Descuento_pct
    """
    df = df.copy()
    
    cols_precio = ["Precio_unitario", "Costo_unitario", "Valor_total", "Descuento_pct"]
    
    for col in cols_precio:
        if col not in df.columns:
            continue
        
        try:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype(float)
        except Exception as e:
            logger.error(f"Error normalizando {col}: {e}")
    
    return df


# ============================================
# NORMALIZACIÓN COMPLETA
# ============================================

def normalize_dataframe(df: pd.DataFrame, format_hint: str = "auto") -> Tuple[pd.DataFrame, bool]:
    """
    Aplica todas las normalizaciones en orden correcto.
    
    Args:
        df: DataFrame a normalizar
        format_hint: "v4_legacy" | "data_new" | "auto"
    
    Returns:
        (df_normalizado, es_nuevo_formato)
    """
    df = df.copy()
    logger.info(f"=== Iniciando normalización (format_hint={format_hint}) ===")
    
    # 1. Detectar formato si es auto
    is_new_format = False
    if format_hint == "auto":
        is_new_format = "Producto_codigo" in df.columns
        format_hint = "data_new" if is_new_format else "v4_legacy"
        logger.info(f"Auto-detectado formato: {format_hint}")
    else:
        is_new_format = format_hint == "data_new"
    
    # 2. Normalizar productos
    logger.info("1. Normalizando Producto_id...")
    df = normalize_producto_codigo(df)
    
    # 3. Normalizar fechas
    logger.info("2. Normalizando fechas...")
    df = normalize_fecha(df)
    
    # 4. Normalizar tipo movimiento
    logger.info("3. Normalizando Tipo_movimiento...")
    df = normalize_tipo_movimiento(df)
    
    # 5. Normalizar cantidades
    logger.info("4. Normalizando cantidades...")
    df = normalize_cantidades(df)
    
    # 6. Normalizar precios (si existen)
    logger.info("5. Normalizando precios...")
    df = normalize_precios(df)
    
    # 7. Limpiar columnas vacías
    logger.info("6. Limpiando columnas vacías...")
    df = df.loc[:, [c for c in df.columns if str(c).strip() and not str(c).lower().startswith("unnamed")]]
    
    logger.info(f"✓ Normalización completada. Filas: {len(df)}, Columnas: {len(df.columns)}")
    
    return df, is_new_format


# ============================================
# MAPEO DE COLUMNAS
# ============================================

def get_column_mapping(is_new_format: bool) -> Dict[str, str]:
    """
    Retorna mapeo de columnas según el formato.
    Útil para procesamiento descendente.
    """
    if is_new_format:
        return {
            "Fecha": "Fecha",
            "Producto_id": "Producto_id",
            "Producto_nombre": "Producto_nombre",
            "Tipo_movimiento": "Tipo_movimiento",
            "Cantidad": "Cantidad",
            "Stock_anterior": "Stock_anterior",
            "Stock_posterior": "Stock_posterior",
            # Nuevas columnas
            "Empresa_cliente": "Empresa_cliente",
            "Departamento_cliente": "Departamento_cliente",
            "Canal_venta": "Canal_venta",
            "Punto_venta": "Punto_venta",
            "Precio_unitario": "Precio_unitario",
            "Costo_unitario": "Costo_unitario",
            "Valor_total": "Valor_total",
            "Descuento_pct": "Descuento_pct",
            "Campana": "Campana",
        }
    else:
        # v4 legacy
        return {
            "Fecha": "Fecha",
            "Producto_id": "Producto_id",
            "Producto_nombre": "Producto_nombre",
            "Tipo_movimiento": "Tipo_movimiento",
            "Cantidad": "Cantidad",
            "Stock_anterior": "Stock_anterior",
            "Stock_posterior": "Stock_posterior",
        }


# ============================================
# VALIDACIÓN POST-NORMALIZACIÓN
# ============================================

def validate_normalized_dataframe(df: pd.DataFrame, required_cols: list = None) -> Tuple[bool, str]:
    """
    Valida que el dataframe normalizado tenga las columnas requeridas.
    
    Returns:
        (is_valid, error_message)
    """
    if required_cols is None:
        required_cols = ["Fecha", "Producto_id", "Tipo_movimiento", "Cantidad", "Stock_anterior", "Stock_posterior"]
    
    missing = [col for col in required_cols if col not in df.columns]
    
    if missing:
        return False, f"Columnas requeridas faltantes: {missing}"
    
    # Validar Fecha
    if df["Fecha"].isna().any():
        count = df["Fecha"].isna().sum()
        return False, f"{count} fechas inválidas"
    
    # Validar Tipo_movimiento
    if df["Tipo_movimiento"].isna().any():
        return False, "Hay valores nulos en Tipo_movimiento"
    
    # Validar Cantidad
    if not pd.api.types.is_numeric_dtype(df["Cantidad"]):
        return False, "Cantidad no es numérico"
    
    return True, "✓ Validación exitosa"
