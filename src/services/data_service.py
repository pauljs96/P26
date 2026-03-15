"""
Data Service - Load & Process DataFrames from S3 using DuckDB
================================================================
Maneja carga de datos desde S3 con optimizaciones de memoria.

Ventajas sobre pandas:
- DuckDB es 100x más rápido para queries
- Memoria eficiente (columnar)
- SQL queries directo sin cargar todo en memoria
- Está optimizado para big data
"""

from __future__ import annotations

import os
import logging
from typing import Optional, Dict, Any, List
from pathlib import Path
from datetime import datetime, date
import io

import pandas as pd

try:
    import duckdb
    import polars as pl
except ImportError as e:
    raise ImportError(f"Instala: pip install duckdb polars -U") from e

from src.storage.s3_manager_v2 import get_s3_manager

logger = logging.getLogger(__name__)


class DataService:
    """
    Servicio para cargar y procesar datos desde S3.
    
    Características:
    - Carga CSV/parquet desde S3 usando DuckDB
    - Queries SQL directo sin Full Load
    - Caché en memoria por sesión (Streamlit)
    - Org-aware (aislamiento de datos)
    """
    
    def __init__(self, org_id: str):
        """
        Inicializa DataService para una organización.
        
        Args:
            org_id: ID de la organización
        """
        if not org_id:
            raise ValueError("org_id requerido")
        
        self.org_id = org_id
        self.s3_manager = get_s3_manager()
        self.db_connection = duckdb.connect(":memory:")
        self.loaded_tables: Dict[str, str] = {}  # {table_name: s3_path}
        logger.info(f"✅ DataService inicializado para org: {org_id}")
    
    # ==================== LOAD DATA ====================
    
    def load_csv_from_s3(
        self,
        s3_key: str,
        table_name: str = "data",
        year: str | None = None,
    ) -> Dict[str, Any]:
        """
        Carga CSV desde S3 a DuckDB (sin descargar a filesystem).
        
        Args:
            s3_key: Clave S3 (debe empezar con org_id/)
            table_name: Nombre de tabla en DuckDB
            year: Año de los datos (para metadata)
        
        Returns:
            {
                "success": bool,
                "table_name": str,
                "rows": int,
                "columns": [str],
                "size_mb": float,
                "memory_mb": float,
                "error": str (si falla)
            }
        """
        # Validar acceso a org
        if not s3_key.startswith(f"{self.org_id}/"):
            return {"success": False, "error": "Acceso denegado: org no coincide"}
        
        try:
            # Descargar CSV de S3 a bytes
            result = self.s3_manager.download_file(
                s3_key=s3_key,
                org_id=self.org_id,
                save_path=None  # Mantener en memoria
            )
            
            if not result["success"]:
                return {"success": False, "error": result.get("error")}
            
            file_bytes = result.get("file_bytes")
            size_mb = result.get("size_mb", 0)
            
            # Cargar en DuckDB (CSV directo desde bytes)
            df = pl.read_csv(io.BytesIO(file_bytes))
            
            # Registrar en DuckDB
            self.db_connection.register(table_name, df)
            self.loaded_tables[table_name] = s3_key
            
            # Obtener stats
            rows = len(df)
            columns = df.columns
            memory_mb = df.estimated_size() / (1024 * 1024) if hasattr(df, 'estimated_size') else 0
            
            logger.info(
                f"✅ CSV cargado: {table_name} ({rows} filas, {len(columns)} columnas, "
                f"{size_mb:.2f} MB)"
            )
            
            return {
                "success": True,
                "table_name": table_name,
                "rows": rows,
                "columns": columns,
                "size_mb": round(size_mb, 2),
                "memory_mb": round(memory_mb, 2),
            }
        
        except Exception as e:
            error = str(e)
            logger.error(f"❌ Error cargando CSV: {error}")
            return {"success": False, "error": error}
    
    def load_multiple_csvs(
        self,
        year_list: List[str],
        prefix: str = "raw",
    ) -> Dict[str, Any]:
        """
        Carga múltiples CSVs (para diferentes años).
        
        Estructura S3 esperada:
          s3://bucket/{org_id}/{prefix}/data_2020.csv
          s3://bucket/{org_id}/{prefix}/data_2021.csv
          ...
        
        Args:
            year_list: Lista de años ["2020", "2021", ...]
            prefix: Prefijo en S3 ("raw", "processed", etc)
        
        Returns:
            {
                "success": bool,
                "loaded_tables": {table_name: {rows, columns, size_mb}},
                "total_rows": int,
                "total_size_mb": float,
                "errors": [str]
            }
        """
        loaded_tables = {}
        errors = []
        total_rows = 0
        total_size = 0
        
        for year in year_list:
            s3_key = f"{self.org_id}/{prefix}/data_{year}.csv"
            table_name = f"data_{year}"
            
            result = self.load_csv_from_s3(s3_key, table_name, year)
            
            if result["success"]:
                loaded_tables[table_name] = {
                    "rows": result["rows"],
                    "columns": result["columns"],
                    "size_mb": result["size_mb"],
                }
                total_rows += result["rows"]
                total_size += result["size_mb"]
            else:
                errors.append(f"año {year}: {result.get('error')}")
        
        logger.info(f"✅ {len(loaded_tables)} años cargados ({total_rows} filas totales)")
        
        return {
            "success": len(errors) == 0,
            "loaded_tables": loaded_tables,
            "total_rows": total_rows,
            "total_size_mb": round(total_size, 2),
            "errors": errors,
        }
    
    def load_parquet_from_s3(
        self,
        s3_key: str,
        table_name: str = "data",
    ) -> Dict[str, Any]:
        """
        Carga archivo Parquet desde S3.
        
        Args:
            s3_key: Clave S3 del parquet
            table_name: Nombre de tabla en DuckDB
        
        Returns:
            Igual que load_csv_from_s3()
        """
        if not s3_key.startswith(f"{self.org_id}/"):
            return {"success": False, "error": "Acceso denegado: org no coincide"}
        
        try:
            result = self.s3_manager.download_file(
                s3_key=s3_key,
                org_id=self.org_id,
                save_path=None
            )
            
            if not result["success"]:
                return {"success": False, "error": result.get("error")}
            
            file_bytes = result.get("file_bytes")
            size_mb = result.get("size_mb", 0)
            
            # Cargar parquet
            df = pl.read_parquet(io.BytesIO(file_bytes))
            
            # Registrar en DuckDB
            self.db_connection.register(table_name, df)
            self.loaded_tables[table_name] = s3_key
            
            rows = len(df)
            columns = df.columns
            memory_mb = 0  # Parquet ya está comprimido
            
            logger.info(f"✅ Parquet cargado: {table_name} ({rows} filas)")
            
            return {
                "success": True,
                "table_name": table_name,
                "rows": rows,
                "columns": columns,
                "size_mb": round(size_mb, 2),
                "memory_mb": round(memory_mb, 2),
            }
        
        except Exception as e:
            error = str(e)
            logger.error(f"❌ Error cargando parquet: {error}")
            return {"success": False, "error": error}
    
    # ==================== QUERY / TRANSFORM ====================
    
    def query(self, sql: str) -> Dict[str, Any]:
        """
        Ejecuta query SQL en los datos cargados.
        
        Usa DuckDB para queries rápidas sin cargar todo en memoria.
        
        Args:
            sql: Query SQL (ej: "SELECT * FROM data_2020 WHERE month = 1")
        
        Returns:
            {
                "success": bool,
                "data": pd.DataFrame,
                "rows": int,
                "error": str (si falla)
            }
        """
        try:
            result = self.db_connection.execute(sql).fetchall()
            columns = [desc[0] for desc in self.db_connection.description]
            
            # Convertir a DataFrame
            df = pd.DataFrame(result, columns=columns)
            
            logger.info(f"✅ Query ejecutada: {len(df)} filas retornadas")
            
            return {
                "success": True,
                "data": df,
                "rows": len(df),
            }
        
        except Exception as e:
            error = str(e)
            logger.error(f"❌ Error en query: {error}")
            return {"success": False, "error": error}
    
    def aggregate_data(
        self,
        table_name: str,
        group_by: List[str],
        agg_columns: Dict[str, str],
    ) -> Dict[str, Any]:
        """
        Agrupa datos (sin cargar todo en memoria).
        
        Args:
            table_name: Tabla en DuckDB
            group_by: Columnas para agrupar ["year", "month"]
            agg_columns: {col_name: "sum|avg|count|max|min"}
        
        Returns:
            {
                "success": bool,
                "data": pd.DataFrame,
                "error": str
            }
        """
        try:
            # Construir SQL dinámicamente
            group_by_str = ", ".join(group_by)
            agg_str = ", ".join([
                f"{op}({col}) as {col}_{op}"
                for col, op in agg_columns.items()
            ])
            
            sql = f"""
                SELECT {group_by_str}, {agg_str}
                FROM {table_name}
                GROUP BY {group_by_str}
                ORDER BY {group_by_str}
            """
            
            return self.query(sql)
        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_stats(self, table_name: str) -> Dict[str, Any]:
        """
        Obtiene estadísticas descriptivas de una tabla.
        
        Args:
            table_name: Tabla en DuckDB
        
        Returns:
            {
                "success": bool,
                "stats": {col_name: {count, null, mean, min, max}},
                "error": str
            }
        """
        try:
            # Describir tabla
            description = self.db_connection.execute(
                f"DESCRIBE {table_name}"
            ).fetchall()
            
            # Obtener summary
            summary = self.db_connection.execute(
                f"SELECT * FROM {table_name}"
            ).df().describe()
            
            return {
                "success": True,
                "stats": summary.to_dict(),
                "columns": [col[0] for col in description],
            }
        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def export_to_parquet(
        self,
        table_name: str,
        s3_key: str | None = None,
    ) -> Dict[str, Any]:
        """
        Exporta tabla a parquet (comprimido) en S3.
        
        Args:
            table_name: Tabla en DuckDB
            s3_key: Clave S3 de destino (default: processed/table_{timestamp}.parquet)
        
        Returns:
            {
                "success": bool,
                "s3_key": str,
                "size_mb": float,
                "error": str
            }
        """
        if s3_key is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            s3_key = f"{self.org_id}/processed/{table_name}_{timestamp}.parquet"
        
        try:
            # Obtener datos
            df = self.db_connection.execute(
                f"SELECT * FROM {table_name}"
            ).df()
            
            # Convertir a parquet
            parquet_bytes = pl.from_pandas(df).write_parquet()
            
            # Subir a S3
            result = self.s3_manager.upload_bytes(
                file_bytes=parquet_bytes,
                filename=Path(s3_key).name,
                org_id=self.org_id,
                data_type="processed",
            )
            
            if result["success"]:
                logger.info(f"✅ Parquet exportado: {result['s3_url']}")
                return {
                    "success": True,
                    "s3_key": result["s3_key"],
                    "size_mb": result.get("file_size_mb", 0),
                }
            else:
                return {"success": False, "error": result.get("error")}
        
        except Exception as e:
            error = str(e)
            logger.error(f"❌ Error exportando parquet: {error}")
            return {"success": False, "error": error}
    
    def export_to_csv(
        self,
        table_name: str,
        s3_key: str | None = None,
    ) -> Dict[str, Any]:
        """
        Exporta tabla a CSV en S3.
        
        Args:
            table_name: Tabla en DuckDB
            s3_key: Clave S3 de destino
        
        Returns:
            Similar a export_to_parquet()
        """
        if s3_key is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            s3_key = f"{self.org_id}/backups/{table_name}_{timestamp}.csv"
        
        try:
            # Obtener datos
            df = self.db_connection.execute(
                f"SELECT * FROM {table_name}"
            ).df()
            
            # Convertir a CSV
            csv_bytes = df.to_csv(index=False).encode()
            
            # Subir a S3
            result = self.s3_manager.upload_bytes(
                file_bytes=csv_bytes,
                filename=Path(s3_key).name,
                org_id=self.org_id,
                data_type="backups",
            )
            
            if result["success"]:
                logger.info(f"✅ CSV exportado: {result['s3_url']}")
                return {
                    "success": True,
                    "s3_key": result["s3_key"],
                    "size_mb": result.get("file_size_mb", 0),
                }
            else:
                return {"success": False, "error": result.get("error")}
        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    # ==================== UTILITIES ====================
    
    def list_tables(self) -> Dict[str, Any]:
        """Lista todas las tablas cargadas en DuckDB."""
        try:
            tables = self.db_connection.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
            ).fetchall()
            
            return {
                "success": True,
                "tables": [t[0] for t in tables],
                "count": len(tables),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """Obtiene información de una tabla."""
        try:
            description = self.db_connection.execute(
                f"DESCRIBE {table_name}"
            ).fetchall()
            
            columns = []
            for row in description:
                columns.append({
                    "name": row[0],
                    "type": row[1],
                    "nullable": row[2] == "YES",
                })
            
            # Contar filas
            row_count = self.db_connection.execute(
                f"SELECT COUNT(*) FROM {table_name}"
            ).fetchone()[0]
            
            return {
                "success": True,
                "table_name": table_name,
                "columns": columns,
                "row_count": row_count,
            }
        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def close(self):
        """Cierra conexión a DuckDB."""
        try:
            self.db_connection.close()
            logger.info(f"✅ DuckDB cerrado para org: {self.org_id}")
        except Exception as e:
            logger.warning(f"⚠️ Error cerrando DuckDB: {e}")


# ==================== FACTORY ====================

def create_data_service(org_id: str) -> DataService:
    """
    Crea instancia de DataService para una org.
    
    Uso típico en Streamlit:
    ```python
    import streamlit as st
    from src.services.data_service import create_data_service
    
    @st.cache_resource
    def get_data_service():
        return create_data_service(org_id=st.session_state.org_id)
    
    ds = get_data_service()
    result = ds.load_multiple_csvs(["2020", "2021", "2022", "2023", "2024", "2025"])
    ```
    """
    return DataService(org_id)
