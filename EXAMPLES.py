"""
EJEMPLOS DE CÓDIGO - Cómo usar los nuevos servicios
===================================================

Este archivo muestra ejemplos reales de cómo usar:
- DataService (DuckDB + Polars)
- S3Manager (org-aware)
- SupabaseDB (RBAC)
"""

# ============================================================
# EJEMPLO 1: Cargar datos desde S3 con DuckDB
# ============================================================

from src.services.data_service import create_data_service

def ejemplo_cargar_datos():
    """Carga 6 años de datos desde S3 para una org."""
    
    # 1. Crear servicio para la org
    org_id = "123e4567-e89b-12d3-a456-426614174000"  # UUID real
    ds = create_data_service(org_id)
    
    # 2. Cargar múltiples años (automático desde S3)
    result = ds.load_multiple_csvs(
        year_list=["2020", "2021", "2022", "2023", "2024", "2025"],
        prefix="raw"
    )
    
    # 3. Verificar resultado
    if result["success"]:
        print(f"✅ Años cargados: {list(result['loaded_tables'].keys())}")
        print(f"📊 Total filas: {result['total_rows']}")
        print(f"💾 Tamaño: {result['total_size_mb']:.2f} MB")
    else:
        print(f"❌ Error: {result['errors']}")
    
    return ds


# ============================================================
# EJEMPLO 2: Query rápida sin cargar todo en memoria
# ============================================================

def ejemplo_queries(ds):
    """Ejecuta queries rápidas con DuckDB."""
    
    # Query 1: Demanda por mes y año
    demand_query = """
        SELECT 
            Año,
            Mes,
            COUNT(*) as registros,
            SUM(Demanda) as total_demanda,
            AVG(Demanda) as demanda_promedio,
            MAX(Demanda) as demanda_max
        FROM data_2024
        GROUP BY Año, Mes
        ORDER BY Año DESC, Mes DESC
    """
    
    result = ds.query(demand_query)
    if result["success"]:
        df_demand = result["data"]
        print(f"📈 Demanda por mes: {len(df_demand)} registros")
        print(df_demand.head())
    
    # Query 2: Comparar años
    comparison_query = """
        SELECT 
            Año,
            ROUND(AVG(Demanda), 2) as demanda_anual_promedio,
            ROUND(AVG(Precio), 2) as precio_promedio,
            SUM(Cantidad) as cantidad_total
        FROM (
            SELECT * FROM data_2020 UNION ALL
            SELECT * FROM data_2021 UNION ALL
            SELECT * FROM data_2022 UNION ALL
            SELECT * FROM data_2023 UNION ALL
            SELECT * FROM data_2024 UNION ALL
            SELECT * FROM data_2025
        )
        GROUP BY Año
        ORDER BY Año
    """
    
    result = ds.query(comparison_query)
    if result["success"]:
        df_comp = result["data"]
        print(f"📊 Comparativa 6 años: {len(df_comp)} años")
        print(df_comp)


# ============================================================
# EJEMPLO 3: Agregación sin cargar full dataset
# ============================================================

def ejemplo_agregaciones(ds):
    """Agrupaciones y estadísticas rápidas."""
    
    # Agregar datos por mes
    result = ds.aggregate_data(
        table_name="data_2024",
        group_by=["Mes"],
        agg_columns={
            "Demanda": "sum",
            "Precio": "avg",
            "Cantidad": "count",
        }
    )
    
    if result["success"]:
        df_agg = result["data"]
        print(f"📊 Agregados por mes: {len(df_agg)} filas")
        print(df_agg)
    
    # Obtener estadísticas descriptivas
    stats_result = ds.get_stats("data_2024")
    if stats_result["success"]:
        print(f"📈 Estadísticas: {stats_result['stats']}")


# ============================================================
# EJEMPLO 4: Exportar a Parquet para procesamiento posterior
# ============================================================

def ejemplo_exportar(ds):
    """Exporta tabla procesada a parquet en S3."""
    
    # Ejecutar query para procesar
    processed_query = """
        SELECT 
            *,
            Demanda * Precio as revenue,
            ROW_NUMBER() OVER (PARTITION BY Mes ORDER BY Demanda DESC) as rank_mes
        FROM data_2024
    """
    
    result = ds.query(processed_query)
    if result["success"]:
        # Crear tabla temporal con resultados
        ds.db_connection.register("processed_2024", result["data"])
    
    # Exportar a Parquet
    export_result = ds.export_to_parquet("processed_2024")
    if export_result["success"]:
        print(f"✅ Parquet guardado en S3: {export_result['s3_key']}")
        print(f"💾 Tamaño: {export_result['size_mb']:.2f} MB")


# ============================================================
# EJEMPLO 5: Gestionar S3 directamente (org-aware)
# ============================================================

from src.storage.s3_manager_v2 import get_s3_manager

def ejemplo_s3_manager():
    """Ejemplos de uso de S3Manager com validación org."""
    
    s3 = get_s3_manager()
    org_id = "123e4567-e89b-12d3-a456-426614174000"
    
    # 1. Listar archivos de la org
    files_result = s3.list_org_files(org_id, data_type="raw")
    if files_result["success"]:
        print(f"📁 Archivos en org ({org_id}):")
        for file in files_result["files"]:
            print(f"  - {file['filename']}: {file['size_mb']:.2f} MB")
        print(f"📊 Total: {files_result['total_size_mb']:.2f} MB")
    
    # 2. Descargar archivo específico
    download_result = s3.download_file(
        s3_key=f"{org_id}/raw/data_2024.csv",
        org_id=org_id,
        save_path=None  # Mantener en memoria como bytes
    )
    if download_result["success"]:
        file_bytes = download_result["file_bytes"]
        print(f"✅ Archivo descargado: {len(file_bytes)/1024:.2f} KB")
    
    # 3. Generar URL presignada para compartir
    url_result = s3.get_presigned_url(
        s3_key=f"{org_id}/processed/forecast_2025.parquet",
        org_id=org_id,
        expires_days=7
    )
    if url_result["success"]:
        print(f"🔗 URL presignada (válida 7 días):")
        print(f"   {url_result['presigned_url']}")


# ============================================================
# EJEMPLO 6: Gestionar usuarios y RBAC
# ============================================================

from src.db.supabase_v2 import get_supabase_db

def ejemplo_rbac():
    """Gestión de usuarios y roles."""
    
    db = get_supabase_db()
    
    # 1. Crear nuevo usuario
    user_result = db.register_user(
        email="john.doe@company.com",
        password="SecurePass@123",
        full_name="John Doe",
        is_master_admin=False
    )
    
    if user_result["success"]:
        user_id = user_result["user_id"]
        print(f"✅ Usuario creado: {user_id}")
        
        # 2. Asignar a organización como viewer
        org_result = db.assign_user_to_org(
            user_id=user_id,
            org_id="ORG_UUID_123",
            role="viewer"  # o "org_admin"
        )
    
    # 3. Obtener orgs del usuario
    orgs_result = db.get_user_organizations("USER_UUID_456")
    if orgs_result["success"]:
        print(f"👤 Usuario pertenece a {orgs_result['count']} orgs:")
        for org in orgs_result["organizations"]:
            print(f"  - {org['org_name']} ({org['role']})")
    
    # 4. Obtener miembros de una org
    members_result = db.get_org_members("ORG_UUID_123")
    if members_result["success"]:
        print(f"👥 Miembros de org: {members_result['count']}")
        for member in members_result["members"]:
            print(f"  - {member['email']} ({member['role']})")


# ============================================================
# EJEMPLO 7: Validación RBAC antes de acciones
# ============================================================

def ejemplo_rbac_validacion():
    """Validar permisos antes de ejecutar operaciones."""
    
    db = get_supabase_db()
    
    user_id = "USER_UUID"
    org_id = "ORG_UUID"
    
    # Obtener rol del usuario en la org
    role_result = db.get_user_role_in_org(user_id, org_id)
    
    if role_result["success"]:
        role = role_result["role"]
        
        # Validar permisos
        if role == "master_admin":
            print("✅ Acceso: Master - sin restricciones")
            # Permitir: crear orgs, eliminar usuarios, etc.
        
        elif role == "org_admin":
            print("✅ Acceso: Admin - puede gestionar su org")
            # Permitir: subir CSV, agregar usuarios a su org
        
        elif role == "viewer":
            print("✅ Acceso: Viewer - solo lectura")
            # Permitir: ver datos
        
        else:
            print("❌ Rol desconocido")
    else:
        print(f"⚠️ Usuario no asignado a org: {role_result['error']}")


# ============================================================
# EJEMPLO 8: Auditoría - registrar acciones
# ============================================================

def ejemplo_auditoria(org_id: str, user_id: str):
    """Registrar uploads y análisis para auditoría."""
    
    db = get_supabase_db()
    
    # Ejemplo: Usuario sube un CSV
    upload_result = db.log_upload(
        org_id=org_id,
        uploaded_by=user_id,
        file_name="data_2025.csv",
        s3_path=f"{org_id}/raw/data_2025.csv",
        year="2025",
        file_size_mb=45.5,
        rows_processed=5000000
    )
    
    if upload_result["success"]:
        print(f"✅ Upload registrado: {upload_result['upload_id']}")
    
    # Ejemplo: Guardar resultado de análisis
    analysis_result = db.save_analysis_result(
        org_id=org_id,
        performed_by=user_id,
        analysis_type="demand_forecast",
        results_summary={
            "model": "ETS",
            "rmse": 0.0234,
            "accuracy": 0.9876,
            "forecast_periods": 12,
        },
        s3_results_path=f"{org_id}/processed/forecast_2025.parquet"
    )
    
    if analysis_result["success"]:
        print(f"✅ Análisis guardado: {analysis_result['result_id']}")


# ============================================================
# EJEMPLO 9: Uso en Streamlit (con caching)
# ============================================================

"""
# En dashboard.py:

import streamlit as st
from src.services.data_service import create_data_service
from src.db.supabase_v2 import get_supabase_db

# Cache recursos caros
@st.cache_resource
def get_data_service():
    return create_data_service(st.session_state.org_id)

@st.cache_resource
def get_db():
    return get_supabase_db()

# Usar en app
st.title("Dashboard Multi-Tenant")

# Obtener org del usuario logueado
org_id = st.session_state.org_id

# Cargar datos
ds = get_data_service()
load_result = ds.load_multiple_csvs(["2020", "2021", "2022", "2023", "2024", "2025"])

if load_result["success"]:
    st.success(f"✅ {load_result['total_rows']} filas cargadas")
    
    # Query y mostrar
    demand = ds.query("SELECT * FROM data_2024 LIMIT 10")
    st.dataframe(demand["data"])
else:
    st.error(f"❌ Error: {load_result['error']}")
"""


# ============================================================
# EJEMPLO 10: Error handling completo
# ============================================================

def ejemplo_error_handling():
    """Ejemplo de manejo robusto de errores."""
    
    try:
        # Cargar datos
        ds = create_data_service("org-uuid")
        result = ds.load_multiple_csvs(["2024", "2025"])
        
        if not result["success"]:
            if result["errors"]:
                print(f"⚠️ Algunos archivos fallaron:")
                for error in result["errors"]:
                    print(f"  - {error}")
            else:
                print(f"❌ Error total: {result}")
            return
        
        # Query con validación
        query_result = ds.query("SELECT * FROM data_2024")
        
        if not query_result["success"]:
            print(f"❌ Query falló: {query_result['error']}")
            # Fallback: retornar vacío o datos por defecto
            return None
        
        df = query_result["data"]
        print(f"✅ Query exitosa: {len(df)} filas")
        
    except ValueError as e:
        print(f"⚠️ Validación: {e}")
    except FileNotFoundError as e:
        print(f"❌ Archivo no encontrado: {e}")
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        # Log en servicios reales


# ============================================================
# MAIN - Ejecutar ejemplos
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("EJEMPLOS DE CÓDIGO - Multi-Tenant Sistema Tesis")
    print("=" * 60)
    
    # Nota: Estos son ejemplos de REFERENCIA.
    # Para ejecutarlos, necesitas:
    # 1. Setup variables de entorno (.env)
    # 2. Datos reales en S3
    # 3. Supabase schema creado
    
    print("\n💡 Ejemplos disponibles:")
    print("   1. ejemplo_cargar_datos() - Load from S3")
    print("   2. ejemplo_queries() - Fast queries with DuckDB")
    print("   3. ejemplo_agregaciones() - Aggregations")
    print("   4. ejemplo_exportar() - Export to parquet")
    print("   5. ejemplo_s3_manager() - S3 operations")
    print("   6. ejemplo_rbac() - User management")
    print("   7. ejemplo_rbac_validacion() - Permission checks")
    print("   8. ejemplo_auditoria() - Logging")
    print("   9. [Código embed] - Streamlit usage")
    print("   10. ejemplo_error_handling() - Error handling")
    
    print("\n📚 Para usar:")
    print("   # Importar y llamar la función deseada")
    print("   from examples import ejemplo_cargar_datos")
    print("   ds = ejemplo_cargar_datos()")
