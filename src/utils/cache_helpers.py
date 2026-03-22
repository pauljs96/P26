"""
Utilidades para serialización de DataFrames a JSON y viceversa.
Necesary para guardar/cargar datos del cache en Supabase.
"""

import json
import pandas as pd
import gzip
import base64
from typing import Optional, Union
from datetime import datetime


def dataframe_to_json(df: pd.DataFrame) -> str:
    """
    Convierte DataFrame a JSON string (preserva tipos de datos usando orient='split')
    
    Args:
        df: DataFrame a serializar
    
    Returns:
        JSON string con estructura preservada
    """
    if df is None or df.empty:
        return json.dumps({})
    
    try:
        # orient='split' preserva índices, columnas y tipos de datos
        # date_format='iso' convierte datetimes a ISO strings
        return df.to_json(orient='split', date_format='iso')
    
    except Exception as e:
        raise ValueError(f"Error serializando DataFrame: {str(e)}")


def compress_json(json_str: str) -> str:
    """
    Comprime JSON string usando gzip y lo codifica en base64
    Típicamente reduce tamaño 10:1
    
    Args:
        json_str: JSON string a comprimir
    
    Returns:
        JSON comprimido en base64
    """
    if not json_str:
        return ""
    
    try:
        # Comprimir con gzip
        compressed = gzip.compress(json_str.encode('utf-8'))
        # Codificar en base64 para transportar como texto
        encoded = base64.b64encode(compressed).decode('utf-8')
        print(f"[COMPRESS] {len(json_str)} bytes → {len(encoded)} bytes (compression ratio: {len(json_str) / len(encoded):.1f}:1)")
        return encoded
    
    except Exception as e:
        raise ValueError(f"Error comprimiendo JSON: {str(e)}")


def decompress_json(compressed_str: str) -> str:
    """
    Descomprime JSON que fue comprimido con compress_json()
    
    Args:
        compressed_str: JSON comprimido en base64
    
    Returns:
        JSON string original
    """
    if not compressed_str:
        return "{}"
    
    try:
        # Decodificar base64
        compressed = base64.b64decode(compressed_str.encode('utf-8'))
        # Descomprimir con gzip
        decompressed = gzip.decompress(compressed).decode('utf-8')
        return decompressed
    
    except Exception as e:
        raise ValueError(f"Error descomprimiendo JSON: {str(e)}")


def json_to_dataframe(json_str: str) -> pd.DataFrame:
    """
    Convierte JSON string de vuelta a DataFrame (preserva tipos de datos)
    
    Args:
        json_str: JSON string generado por dataframe_to_json()
    
    Returns:
        DataFrame reconstruido with proper dtypes
    """
    if not json_str or json_str == '{}':
        return pd.DataFrame()
    
    try:
        # read_json con orient='split' restaura preserva índices y tipos automáticamente
        from io import StringIO
        df = pd.read_json(StringIO(json_str), orient='split')
        
        # Asegurar que columnas con 'Fecha' sean datetime (por si acaso)
        for col in df.columns:
            col_lower = col.lower()
            if any(date_col in col_lower for date_col in ['fecha', 'date', 'time', 'timestamp', 'mes']):
                if not pd.api.types.is_datetime64_any_dtype(df[col]):
                    try:
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                    except:
                        pass
        
        return df
    
    except Exception as e:
        raise ValueError(f"Error deserializando JSON a DataFrame: {str(e)}")


def validate_dataframestamp(df: pd.DataFrame, required_columns: list = None) -> bool:
    """
    Valida que un DataFrame tenga estructura correcta
    
    Args:
        df: DataFrame a validar
        required_columns: Lista de columnas requeridas (opcional)
    
    Returns:
        True si válido, False si no
    """
    if df is None or not isinstance(df, pd.DataFrame):
        return False
    
    if df.empty:
        return False
    
    if required_columns:
        return all(col in df.columns for col in required_columns)
    
    return True


# Funciones de serialización para PipelineResult
def serialize_pipeline_result(movements, demand_monthly, stock_monthly) -> tuple:
    """
    Serializa salida de DataPipeline.run() para guardar en cache
    Comprime DataFrames grandes para reducir tamaño (típicamente 10:1)

    Returns:
        (movements_json, demand_json, stock_json) - comprimidos en base64
    """
    try:
        movements_json = dataframe_to_json(movements) if movements is not None else None
        demand_json = dataframe_to_json(demand_monthly)
        stock_json = dataframe_to_json(stock_monthly)
        
        # Comprimir para reducir tamaño en Supabase
        movements_json = compress_json(movements_json) if movements_json else None
        demand_json = compress_json(demand_json)
        stock_json = compress_json(stock_json)
        
        return movements_json, demand_json, stock_json
    except Exception as e:
        raise ValueError(f"Error serializando PipelineResult: {str(e)}")


def deserialize_pipeline_result(movements_json, demand_json, stock_json) -> tuple:
    """
    Deserializa datos del cache a DataFrames (primero descomprime, luego deserializa)

    Returns:
        (movements_df, demand_df, stock_df)
    """
    try:
        # Descomprimir primero
        movements_json = decompress_json(movements_json) if movements_json else None
        demand_json = decompress_json(demand_json)
        stock_json = decompress_json(stock_json)
        
        # Luego convertir a DataFrames
        movements_df = json_to_dataframe(movements_json) if movements_json else None
        demand_df = json_to_dataframe(demand_json)
        stock_df = json_to_dataframe(stock_json)
        return movements_df, demand_df, stock_df
    except Exception as e:
        raise ValueError(f"Error deserializando PipelineResult: {str(e)}")
