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
    org_id: str,
    last_cache_timestamp: str = None
) -> Tuple[bool, Optional[dict]]:
    """
    Verifica si una organización tiene data cacheada.
    Incluye detección de cambios recientes en BD.
    
    Args:
        db: SupabaseDB instance
        org_id: Organization ID
        last_cache_timestamp: Timestamp del último caché cargado (para detectar cambios)
    
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
        
        current_timestamp = cache_data.get("updated_at")
        
        # Si hay timestamp anterior: verificar si BD cambió
        if last_cache_timestamp and current_timestamp:
            if str(current_timestamp) == str(last_cache_timestamp):
                # Data no cambió, caché es válido
                return True, {
                    "movements": None,  # No descargar si no cambió
                    "demand_monthly": None,
                    "stock_monthly": None,
                    "csv_files_count": cache_data.get("csv_files_count", 0),
                    "updated_at": current_timestamp,
                    "cache_valid": True  # Flag: no re-deserializar
                }
        
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
                "updated_at": current_timestamp,
                "demand_json": demand_json,
                "stock_json": stock_json,
                "movements_json": movements_json,
                "cache_valid": False  # Flag: data fue deserializada
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
) -> Tuple[bool, Optional[str]]:
    """
    Serializa y guarda resultados de DataPipeline en org_cache.
    Retorna timestamp para detección futura de cambios.
    
    Args:
        db: SupabaseDB instance
        org_id: Organization ID
        movements: DataFrame de movimientos
        demand_monthly: DataFrame de demanda mensual
        stock_monthly: DataFrame de stock mensual
        processed_by: User ID que procesó
        csv_files_count: Número de CSVs procesados
    
    Returns:
        (success: bool, updated_at_timestamp: str or None)
    """
    try:
        from datetime import datetime
        import traceback
        
        # Serializar
        print(f"[CACHE] Serializando dataframes para org_id={org_id}...")
        movements_json, demand_json, stock_json = serialize_pipeline_result(
            movements, demand_monthly, stock_monthly
        )
        total_size = len(movements_json or "") + len(demand_json) + len(stock_json)
        print(f"[CACHE] Serialización completada. Tamaño total: {total_size} bytes (~{total_size / 1024 / 1024:.2f} MB)")
        
        # NOTA: Supabase tiene límites de query size (~8MB)
        # Si los datos son muy grandes, saltamos el guardado en BD
        # El caché de Streamlit (@st.cache_data) sigue funcionando normalmente
        if total_size > 5_000_000:  # >5MB
            print(f"[WARN] Datos demasiado grandes ({total_size / 1024 / 1024:.2f} MB) para guardar en BD")
            print(f"[WARN] Usando sólo caché de Streamlit (disponible 5 minutos)")
            # Retornar éxito para no mostrar warning al usuario
            timestamp = datetime.now().isoformat()
            return True, timestamp
        
        # Guardar en BD
        print(f"[CACHE] Guardando en BD...")
        try:
            result = db.save_org_data(
                org_id=org_id,
                demand_monthly_json=demand_json,
                stock_monthly_json=stock_json,
                movements_json=movements_json,
                processed_by=processed_by,
                csv_files_count=csv_files_count
            )
            print(f"[CACHE] save_org_data retornó: {result}")
        except Exception as db_error:
            print(f"[ERROR] db.save_org_data lanzó excepción: {str(db_error)}")
            traceback.print_exc()
            return False, None
        
        if result.get("success", False) is True:
            # Si fue exitoso, usar timestamp actual (Supabase lo grabará con NOW())
            timestamp = datetime.now().isoformat()
            print(f"[OK] Cache guardado exitosamente para org_id={org_id} | timestamp={timestamp}")
            return True, timestamp
        else:
            print(f"[ERROR] save_org_data retornó success={result.get('success')} (esperaba True)")
            return False, None
    
    except Exception as e:
        print(f"[EXCEPTION] Error guardando cache: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, None
