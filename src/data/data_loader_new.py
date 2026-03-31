"""Loader específico para el nuevo formato Data.csv.

Características:
- Separador: punto y coma (;)
- Encoding: UTF-8
- Formato fecha: DD/MM/YYYY
- Columnas extendidas: Cliente, Departamento, Canal, Precio, Costo, Campaña
- ~230K transacciones 2021-2025+
"""

from __future__ import annotations
import io
import pandas as pd
from typing import List, Optional
import logging

from src.utils.config import CSV_ENCODINGS
from src.data.normalize_functions import normalize_dataframe, validate_normalized_dataframe

logger = logging.getLogger(__name__)


class DataLoaderNew:
    """Loader para formato Data.csv (nuevo)."""
    
    # Columnas requeridas para nuevo formato
    REQUIRED_COLUMNS_NEW = [
        "Fecha",
        "Producto_codigo",  # Nota: será renombrado a Producto_id en normalización
        "Producto_nombre",
        "Tipo_movimiento",
        "Cantidad",
        "Stock_anterior",
        "Stock_posterior",
    ]
    
    # Columnas opcionales en nuevo formato
    OPTIONAL_COLUMNS_NEW = [
        "Empresa_cliente",
        "Departamento_cliente",
        "Canal_venta",
        "Punto_venta",
        "Precio_unitario",
        "Descuento_pct",
        "Valor_total",
        "Campana",
        "Costo_unitario",
    ]
    
    def __init__(self):
        pass
    
    def load_files(self, uploaded_files: List) -> pd.DataFrame:
        """Carga y concatena múltiples CSV del nuevo formato."""
        dfs = []
        for f in uploaded_files:
            df = self._load_single_file(f)
            if df is not None and not df.empty:
                df["__source_file"] = getattr(f, "name", "uploaded.csv")
                dfs.append(df)
        
        if not dfs:
            logger.warning("No se cargó ningún archivo exitosamente")
            return pd.DataFrame()
        
        result = pd.concat(dfs, ignore_index=True)
        logger.info(f"Total concatenado: {len(result)} filas de {len(dfs)} archivo(s)")
        return result
    
    def _load_single_file(self, uploaded_file) -> Optional[pd.DataFrame]:
        """Carga un CSV del nuevo formato desde upload Streamlit."""
        content = uploaded_file.getvalue()
        filename = getattr(uploaded_file, "name", "unknown")
        
        # Nuevo formato: siempre es UTF-8 con punto y coma
        logger.info(f"Cargando: {filename}")
        
        for enc in CSV_ENCODINGS:  # ["utf-8"]
            try:
                bio = io.BytesIO(content)
                df = pd.read_csv(
                    bio,
                    sep=";",  # Punto y coma para nuevo formato
                    encoding=enc,
                    dtype=str,  # Mantener como string para conversión manual
                    engine="python",
                )
                
                # Limpiar espacios en encabezado
                df.columns = df.columns.str.strip()
                
                # Eliminar columnas vacías
                df = df.loc[:, [
                    c for c in df.columns
                    if str(c).strip() and not str(c).lower().startswith("unnamed")
                ]]
                
                logger.info(f"✓ {filename}: {len(df)} filas, {len(df.columns)} columnas")
                logger.debug(f"  Columnas: {list(df.columns)}")
                
                # Validar columnas mínimas (antes de normalización)
                missing = [c for c in self.REQUIRED_COLUMNS_NEW if c not in df.columns]
                if missing:
                    # Podría ser v4 legacy, devolver None para que lo intente otro loader
                    logger.debug(f"Columnas nuevas esperadas no encontradas: {missing}")
                    return None
                
                return df.reset_index(drop=True)
                
            except Exception as e:
                logger.debug(f"Error con encoding={enc}: {e}")
                last_error = e
        
        logger.error(f"No se pudo leer '{filename}' como formato nuevo")
        return None
    
    @staticmethod
    def normalize(df: pd.DataFrame) -> pd.DataFrame:
        """Normaliza dataframe del nuevo formato.
        
        Aplica:
        - Normalización de fechas (DD/MM/YYYY → YYYY-MM-DD)
        - Renombrado de Producto_codigo → Producto_id
        - Tipado de datos
        - Limpieza
        """
        if df.empty:
            logger.warning("DataFrame vacío para normalizar")
            return df
        
        # Aplicar normalización unificada
        df_norm, is_new_format = normalize_dataframe(df, format_hint="data_new")
        
        # Validar
        is_valid, msg = validate_normalized_dataframe(df_norm)
        if not is_valid:
            logger.error(f"Validación fallida: {msg}")
            return pd.DataFrame()
        
        logger.info(msg)
        logger.info(f"Formato detectado como NUEVO: {is_new_format}")
        
        return df_norm
