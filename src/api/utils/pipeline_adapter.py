"""
Adaptador para usar DataPipeline con archivos desde S3 (bytes).

Permite procesar CSVs que vienen como bytes (descargados de S3)
usando la pipeline original del dashboard.
"""

from __future__ import annotations
import io
from typing import List, BinaryIO
import pandas as pd

from src.data.pipeline import DataPipeline
from src.utils.logger import Logger


class BytesFile:
    """
    Simula un archivo uploadeado (UploadedFile de Streamlit)
    pero recibe bytes directamente.
    
    Usa el mismo interfaz que Streamlit para que DataPipeline 
    funcione sin cambios.
    """
    
    def __init__(self, file_bytes: bytes, filename: str):
        self.bytes_content = file_bytes
        self.name = filename
        self._stream = io.BytesIO(file_bytes)
    
    def getvalue(self) -> bytes:
        """Retorna el contenido completo (como UploadedFile)"""
        return self.bytes_content
    
    def read(self) -> bytes:
        """Lee del stream"""
        return self._stream.read()
    
    def seek(self, pos: int) -> int:
        """Busca posición en el stream"""
        return self._stream.seek(pos)


def process_csv_bytes_with_pipeline(
    csv_bytes: bytes,
    filename: str,
    logger: Logger | None = None
) -> dict:
    """
    Procesa un CSV desde bytes usando la DataPipeline original.
    
    Args:
        csv_bytes: Contenido del archivo en bytes
        filename: Nombre del archivo (para referencia)
        logger: Logger opcional
    
    Returns:
        {
            "success": bool,
            "movements": pd.DataFrame,      # Datos crudos reconciliados
            "demand_monthly": pd.DataFrame, # Demanda mensual
            "stock_monthly": pd.DataFrame,  # Stock mensual
            "error": str (si falla)
        }
    """
    try:
        if logger:
            logger.info(f"📥 Procesando {filename} ({len(csv_bytes)} bytes)")
        
        # Crear un objeto "fake file" que simule Streamlit's UploadedFile
        fake_file = BytesFile(csv_bytes, filename)
        
        # Usar la pipeline original con el archivo simulado
        pipeline = DataPipeline(logger=logger)
        result = pipeline.run([fake_file])  # Espera lista de archivos
        
        if logger:
            logger.info(f"✅ Pipeline completado: {len(result.movements)} movimientos")
        
        return {
            "success": True,
            "movements": result.movements,
            "demand_monthly": result.demand_monthly,
            "stock_monthly": result.stock_monthly,
        }
        
    except Exception as e:
        error_msg = str(e)
        if logger:
            logger.error(f"❌ Error procesando {filename}: {error_msg}")
        return {
            "success": False,
            "error": error_msg
        }


def extract_product_demand(
    demand_monthly: pd.DataFrame,
    product_name: str | None = None
) -> pd.DataFrame:
    """
    Extrae la demanda histórica de un producto específico.
    
    Si product_name es None, retorna toda la demanda amalgamada.
    
    Args:
        demand_monthly: DataFrame de demanda mensual (del pipeline)
        product_name: Nombre del producto a filtrar (opcional)
    
    Returns:
        DataFrame con columnas: Mes, Demanda_Unid (el nombre exacto de la columna)
    """
    if demand_monthly.empty:
        return pd.DataFrame()
    
    # Si no se especifica producto, usar toda la demanda
    if product_name is None:
        # Buscar columna de demanda
        demand_cols = [col for col in demand_monthly.columns 
                       if 'demanda' in col.lower() or 'quantity' in col.lower()]
        if demand_cols:
            return demand_monthly[['Mes', demand_cols[0]]].rename(
                columns={demand_cols[0]: 'Demanda_Unid'}
            )
        return demand_monthly
    
    # Filtrar por producto si existe
    if 'Producto_Nombre' in demand_monthly.columns:
        product_data = demand_monthly[
            demand_monthly['Producto_Nombre'].str.contains(product_name, case=False, na=False)
        ]
        if not product_data.empty:
            demand_cols = [col for col in product_data.columns 
                           if 'demanda' in col.lower() or 'quantity' in col.lower()]
            if demand_cols:
                return product_data[['Mes', demand_cols[0]]].rename(
                    columns={demand_cols[0]: 'Demanda_Unid'}
                ).sort_values('Mes')
    
    return pd.DataFrame()
