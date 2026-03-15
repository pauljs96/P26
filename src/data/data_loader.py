"""Carga robusta de CSV del ERP.

Problema típico: el CSV incluye una primera fila tipo "Periodo del;..." y
recién en la segunda fila están los encabezados reales.

Solución:
- Leemos sin encabezado (header=None).
- Buscamos una fila que contenga 'codigo/código', 'fecha' y 'documento'.
- Usamos esa fila como header y el resto como data.
- Desduplicamos nombres de columnas ('Entrada', 'Entrada__2', etc.)
"""

from __future__ import annotations
import io
import pandas as pd
from typing import List, Optional

from src.utils.config import CSV_SEPARATORS, CSV_ENCODINGS


class DataLoader:
    def __init__(self):
        pass

    def load_files(self, uploaded_files: List) -> pd.DataFrame:
        """Carga y concatena múltiples CSV (Streamlit uploader)."""
        dfs = []
        for f in uploaded_files:
            df = self._load_single_file(f)
            df["__source_file"] = getattr(f, "name", "uploaded.csv")
            dfs.append(df)
        if not dfs:
            return pd.DataFrame()
        return pd.concat(dfs, ignore_index=True)

    def _load_single_file(self, uploaded_file) -> pd.DataFrame:
        content = uploaded_file.getvalue()
        last_error: Optional[Exception] = None
        filename = getattr(uploaded_file, "name", "unknown")

        for enc in CSV_ENCODINGS:
            for sep in CSV_SEPARATORS:
                try:
                    bio = io.BytesIO(content)
                    df_raw = pd.read_csv(
                        bio,
                        sep=sep,
                        encoding=enc,
                        header=None,
                        dtype=str,
                        engine="python",
                    )

                    header_idx = self._detect_header_row(df_raw)
                    if header_idx is None:
                        continue

                    header = df_raw.iloc[header_idx].astype(str).str.strip().tolist()
                    df = df_raw.iloc[header_idx + 1 :].copy()
                    df.columns = header

                    df.columns = self._dedupe_columns(df.columns)

                    # Eliminar columnas vacías / Unnamed
                    df = df.loc[:, [
                        c for c in df.columns
                        if str(c).strip() and not str(c).lower().startswith("unnamed")
                    ]]

                    # Log exitoso
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.info(f"✓ {filename}: Cargado con sep='{sep}', encoding='{enc}', {len(df)} filas, {len(df.columns)} columnas")
                    
                    return df.reset_index(drop=True)

                except Exception as e:
                    last_error = e

        # Si llegamos aquí, fallamos
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"✗ {filename}: No se pudo leer")
        logger.error(f"  Último error: {last_error}")
        logger.error(f"  Tamaño archivo: {len(content)} bytes")
        logger.error(f"  Primeros 200 bytes: {content[:200]}")
        
        raise RuntimeError(
            f"No se pudo leer el CSV '{filename}'. "
            f"Último error: {last_error}"
        )

    def _detect_header_row(self, df: pd.DataFrame) -> Optional[int]:
        """Detecta fila de encabezados reales buscando columnas clave.
        
        Busca en las primeras 30 filas una que contenga:
        - codigo/código
        - fecha
        - documento
        """
        for i in range(min(30, len(df))):
            row_str = df.iloc[i].astype(str)
            # Buscar en toda la fila
            row_values = [str(cell).strip().lower() for cell in row_str.values]
            
            # Buscar patrones
            has_codigo = any(
                "codigo" in val or "código" in val 
                for val in row_values
            )
            has_fecha = any(
                "fecha" in val or "date" in val 
                for val in row_values
            )
            has_documento = any(
                "documento" in val or "doc" in val 
                for val in row_values
            )
            
            if has_codigo and has_fecha and has_documento:
                return i
        
        return None

    def _dedupe_columns(self, cols) -> list:
        """Renombra columnas duplicadas agregando sufijo __2, __3, ..."""
        seen = {}
        new_cols = []
        for c in cols:
            name = str(c).strip()
            if name in seen:
                seen[name] += 1
                new_cols.append(f"{name}__{seen[name]}")
            else:
                seen[name] = 1
                new_cols.append(name)
        return new_cols
