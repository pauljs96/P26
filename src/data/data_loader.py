"""Carga simple de CSV v4 (Inventario ML Completo).

Dataset v4 es un CSV limpio con:
- Encabezado en primera fila
- Separador: coma
- Encoding: UTF-8
- Sin filas de metadatos previas
"""

from __future__ import annotations
import io
import pandas as pd
from typing import List, Optional
import logging

from src.utils.config import CSV_SEPARATORS, CSV_ENCODINGS, REQUIRED_COLUMNS

logger = logging.getLogger(__name__)


class DataLoader:
    def __init__(self):
        pass

    def load_files(self, uploaded_files: List) -> pd.DataFrame:
        """Carga y concatena múltiples CSV de v4."""
        dfs = []
        for f in uploaded_files:
            df = self._load_single_file(f)
            df["__source_file"] = getattr(f, "name", "uploaded.csv")
            dfs.append(df)
        if not dfs:
            return pd.DataFrame()
        return pd.concat(dfs, ignore_index=True)

    def _load_single_file(self, uploaded_file) -> pd.DataFrame:
        """Carga un CSV v4 desde upload Streamlit."""
        content = uploaded_file.getvalue()
        filename = getattr(uploaded_file, "name", "unknown")

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

                    # Validar que tenga columnas requeridas
                    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
                    if missing:
                        raise ValueError(
                            f"Columnas requeridas faltantes: {missing}\n"
                            f"Columnas encontradas: {list(df.columns)}"
                        )

                    logger.info(
                        f"✓ {filename}: {len(df)} filas, "
                        f"{len(df.columns)} columnas"
                    )
                    return df.reset_index(drop=True)

                except Exception as e:
                    logger.error(f"Error con enc={enc}, sep={sep}: {e}")
                    last_error = e

        # Si llegamos aquí, fallamos
        raise RuntimeError(
            f"No se pudo leer '{filename}': {last_error}"
        )
