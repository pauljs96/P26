"""Loader para formato v4 legacy (Inventario ML Completo).

Mantiene la interfaz y lógica del loader original.
Formato:
- Separador: coma
- Encoding: UTF-8
- Fecha: YYYY-MM-DD
- Columnas actuales: 7 requeridas
"""

from __future__ import annotations
import io
import pandas as pd
from typing import List, Optional
import logging

from src.utils.config import CSV_SEPARATORS, CSV_ENCODINGS, REQUIRED_COLUMNS
from src.data.normalize_functions import normalize_dataframe, validate_normalized_dataframe

logger = logging.getLogger(__name__)


class DataLoaderLegacy:
    """Loader para formato v4 legacy (actual)."""
    
    def __init__(self):
        pass
    
    def load_files(self, uploaded_files: List) -> pd.DataFrame:
        """Carga y concatena múltiples CSV v4 legacy."""
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
        """Carga un CSV v4 legacy desde upload Streamlit."""
        content = uploaded_file.getvalue()
        filename = getattr(uploaded_file, "name", "unknown")
        
        logger.info(f"Cargando: {filename}")
        
        # v4 siempre es UTF-8 con coma
        for enc in CSV_ENCODINGS:  # ["utf-8"]
            for sep in CSV_SEPARATORS:  # [","]
                try:
                    bio = io.BytesIO(content)
                    df = pd.read_csv(
                        bio,
                        sep=sep,
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
                    
                    # Validar que tenga columnas requeridas v4
                    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
                    if missing:
                        logger.debug(f"Columnas v4 requeridas no encontradas: {missing}")
                        continue  # Intentar siguiente combo sep/enc
                    
                    logger.info(f"✓ {filename}: {len(df)} filas, {len(df.columns)} columnas")
                    logger.debug(f"  Columnas: {list(df.columns)}")
                    
                    return df.reset_index(drop=True)
                    
                except Exception as e:
                    logger.debug(f"Error con enc={enc}, sep={sep}: {e}")
                    last_error = e
        
        logger.warning(f"No se pudo leer '{filename}' como formato v4 legacy")
        return None
    
    @staticmethod
    def normalize(df: pd.DataFrame) -> pd.DataFrame:
        """Normaliza dataframe del formato v4 legacy.
        
        Aplica:
        - Tipado de datos
        - Limpieza
        - Validación
        """
        if df.empty:
            logger.warning("DataFrame vacío para normalizar")
            return df
        
        # Aplicar normalización unificada
        df_norm, is_new_format = normalize_dataframe(df, format_hint="v4_legacy")
        
        # Validar
        is_valid, msg = validate_normalized_dataframe(df_norm, required_cols=REQUIRED_COLUMNS)
        if not is_valid:
            logger.error(f"Validación fallida: {msg}")
            return pd.DataFrame()
        
        logger.info(msg)
        logger.info(f"Formato detectado como NUEVO: {is_new_format}")
        
        return df_norm
