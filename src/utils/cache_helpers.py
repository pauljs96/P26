"""
Utilidades para serialización de DataFrames a JSON y viceversa.
Necesary para guardar/cargar datos del cache en Supabase.
"""

import json
import pandas as pd
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
        df = pd.read_json(json_str, orient='split')
        
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
    
    Returns:
        (movements_json, demand_json, stock_json)
    """
    try:
        movements_json = dataframe_to_json(movements) if movements is not None else None
        demand_json = dataframe_to_json(demand_monthly)
        stock_json = dataframe_to_json(stock_monthly)
        return movements_json, demand_json, stock_json
    except Exception as e:
        raise ValueError(f"Error serializando PipelineResult: {str(e)}")


def deserialize_pipeline_result(movements_json, demand_json, stock_json) -> tuple:
    """
    Deserializa datos del cache a DataFrames
    
    Returns:
        (movements_df, demand_df, stock_df)
    """
    try:
        movements_df = json_to_dataframe(movements_json) if movements_json else None
        demand_df = json_to_dataframe(demand_json)
        stock_df = json_to_dataframe(stock_json)
        return movements_df, demand_df, stock_df
    except Exception as e:
        raise ValueError(f"Error deserializando PipelineResult: {str(e)}")
