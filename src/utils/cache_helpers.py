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
    Convierte DataFrame a JSON string (preserva tipos de datos)
    
    Args:
        df: DataFrame a serializar
    
    Returns:
        JSON string con estructura: {"columns": [...], "index": [...], "data": [...]}
    """
    if df is None or df.empty:
        return json.dumps({"columns": [], "index": [], "data": []})
    
    try:
        # Convertir índice a lista (puede ser DatetimeIndex, etc)
        index_data = df.index.tolist() if hasattr(df.index, 'tolist') else list(df.index)
        
        # Convertir a ISO format si es DatetimeIndex
        if isinstance(df.index, pd.DatetimeIndex):
            index_data = [d.isoformat() if isinstance(d, datetime) else str(d) for d in index_data]
        
        # Serializar valores
        data = {
            "columns": df.columns.tolist(),
            "index": index_data,
            "data": df.fillna('null').astype(str).values.tolist()
        }
        
        return json.dumps(data, default=str)
    
    except Exception as e:
        raise ValueError(f"Error serializando DataFrame: {str(e)}")


def json_to_dataframe(json_str: str) -> pd.DataFrame:
    """
    Convierte JSON string de vuelta a DataFrame
    
    Args:
        json_str: JSON string generado por dataframe_to_json()
    
    Returns:
        DataFrame reconstruido
    """
    if not json_str or json_str == '{}':
        return pd.DataFrame()
    
    try:
        data = json.loads(json_str)
        
        if not data.get("columns"):
            return pd.DataFrame()
        
        # Reconstruir DataFrame
        df = pd.DataFrame(
            data=data["data"],
            columns=data["columns"],
            index=data.get("index")
        )
        
        # Intentar convertir index a datetime si parece una fecha
        try:
            df.index = pd.to_datetime(df.index)
        except:
            pass  # Si no, dejar como está
        
        # Procesar cada columna
        for col in df.columns:
            # Reemplazar 'null' con NaN
            df[col] = df[col].replace('null', None)
            
            # Detectar columnas de fecha por nombre
            col_lower = col.lower()
            if any(date_col in col_lower for date_col in ['fecha', 'date', 'fecha_', 'date_', 'time', 'timestamp']):
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    continue  # Pasar a siguiente columna
                except:
                    pass
            
            # Intentar conversión numérica para otras columnas
            try:
                df[col] = pd.to_numeric(df[col], errors='ignore')
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
