"""
Cache Management Integrations
Funciones para integrar caching en el flujo de upload y dashboard
"""

import pandas as pd
from typing import Optional, Tuple
from src.db.supabase import SupabaseDB
from src.utils.cache_helpers import (
    serialize_pipeline_result,
    deserialize_pipeline_result,
    json_to_dataframe
)


def check_and_load_org_cache(
    db: SupabaseDB, 
    org_id: str
) -> Tuple[bool, Optional[dict]]:
    """
    Verifica si una organización tiene data cacheada.
    
    Returns:
        (has_cache: bool, data_dict: dict or None)
        data_dict contiene: {demand_monthly, stock_monthly, movements, csv_files_count, updated_at}
    """
    try:
        # Verificar flag
        is_loaded = db.is_data_loaded(org_id)
        
        if not is_loaded:
            return False, None
        
        # Cargar datos
        cache_data = db.load_org_data(org_id)
        
        if not cache_data.get("success"):
            return False, None
        
        # Deserializar DataFrames
        try:
            movements_json = cache_data.get("movements")
            demand_json = cache_data.get("demand_monthly")
            stock_json = cache_data.get("stock_monthly")
            
            movements, demand_monthly, stock_monthly = deserialize_pipeline_result(
                movements_json, demand_json, stock_json
            )
            
            return True, {
                "movements": movements,
                "demand_monthly": demand_monthly,
                "stock_monthly": stock_monthly,
                "csv_files_count": cache_data.get("csv_files_count", 0),
                "updated_at": cache_data.get("updated_at"),
                "demand_json": demand_json,
                "stock_json": stock_json,
                "movements_json": movements_json
            }
        
        except Exception as e:
            print(f"❌ Error deserializing cache: {str(e)}")
            return False, None
    
    except Exception as e:
        print(f"❌ Error checking cache: {str(e)}")
        return False, None


def save_org_cache(
    db: SupabaseDB,
    org_id: str,
    movements: pd.DataFrame,
    demand_monthly: pd.DataFrame,
    stock_monthly: pd.DataFrame,
    processed_by: str,
    csv_files_count: int = 0
) -> bool:
    """
    Serializa y guarda resultados de DataPipeline en org_cache.
    
    Args:
        db: SupabaseDB instance
        org_id: Organization ID
        movements: DataFrame de movimientos
        demand_monthly: DataFrame de demanda mensual
        stock_monthly: DataFrame de stock mensual
        processed_by: User ID que procesó
        csv_files_count: Número de CSVs procesados
    
    Returns:
        True si fue exitoso, False otherwise
    """
    try:
        # Serializar
        movements_json, demand_json, stock_json = serialize_pipeline_result(
            movements, demand_monthly, stock_monthly
        )
        
        # Guardar en BD
        result = db.save_org_data(
            org_id=org_id,
            demand_monthly_json=demand_json,
            stock_monthly_json=stock_json,
            movements_json=movements_json,
            processed_by=processed_by,
            csv_files_count=csv_files_count
        )
        
        return result.get("success", False)
    
    except Exception as e:
        print(f"❌ Error saving cache: {str(e)}")
        return False
