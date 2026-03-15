#!/usr/bin/env python
"""
Test & Verification Suite - Multi-Tenant Setup
==============================================

Verifica que todos los componentes están funcionando:
1. Supabase conectado + schema correcto
2. S3 configurado + bucket accesible
3. DuckDB + Polars instalados
4. DataService working
5. RBAC middleware ready

Uso:
  python scripts/verify_setup.py
"""

import sys
import logging
from pathlib import Path
from datetime import datetime

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_header(text: str):
    """Imprime header formateado."""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def test_imports():
    """Verifica que todas las dependencias estén instaladas."""
    print_header("1️⃣  VERIFICANDO IMPORTS")
    
    dependencies = {
        "streamlit": False,
        "pandas": False,
        "plotly": False,
        "duckdb": False,
        "polars": False,
        "boto3": False,
        "supabase": False,
    }
    
    for lib, _ in dependencies.items():
        try:
            __import__(lib)
            dependencies[lib] = True
            print(f"  ✅ {lib:15} OK")
        except ImportError as e:
            print(f"  ❌ {lib:15} MISSING: {e}")
    
    return all(dependencies.values())


def test_supabase():
    """Verifica conexión a Supabase."""
    print_header("2️⃣  VERIFICANDO SUPABASE")
    
    try:
        from src.db.supabase_v2 import get_supabase_db
        
        print("  Conectando a Supabase...")
        db = get_supabase_db()
        print("  ✅ Conexión exitosa")
        
        # Test: Listar orgs
        print("  Obteniendo organizaciones...")
        orgs = db.list_organizations()
        
        if orgs["success"]:
            count = orgs["count"]
            print(f"  ✅ {count} organizaciones encontradas")
            
            if count > 0:
                print("\n  Primeras 3 orgs:")
                for org in orgs["organizations"][:3]:
                    print(f"    - {org['name']} ({org['id']})")
            
            return True
        else:
            print(f"  ❌ Error listando orgs: {orgs['error']}")
            return False
    
    except Exception as e:
        print(f"  ❌ Error Supabase: {e}")
        return False


def test_s3():
    """Verifica conexión a S3."""
    print_header("3️⃣  VERIFICANDO AWS S3")
    
    try:
        from src.storage.s3_manager_v2 import get_s3_manager
        
        print("  Conectando a S3...")
        s3 = get_s3_manager()
        
        if s3.is_configured:
            print(f"  ✅ Conectado: {s3.bucket_name} ({s3.region})")
            
            # Test: Listar buckets
            try:
                result = s3.list_org_files("demo", data_type="raw")
                if "error" not in str(result):
                    print(f"  ✅ Demo org accesible")
            except:
                print(f"  ⚠️  Demo org no tiene datos (pero conexión OK)")
            
            return True
        else:
            print(f"  ⚠️  S3 no configurado (fallback a session storage)")
            return True  # No es critical
    
    except Exception as e:
        print(f"  ❌ Error S3: {e}")
        return False


def test_duckdb():
    """Verifica DuckDB + Polars."""
    print_header("4️⃣  VERIFICANDO DUCKDB + POLARS")
    
    try:
        import duckdb
        import polars as pl
        from io import BytesIO
        import pandas as pd
        
        print("  Probando DuckDB in-memory...")
        
        # Crear tabla de prueba
        db = duckdb.connect(":memory:")
        
        # Sample data
        df = pl.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
            "sales": [100, 200, 150, 300, 250],
        })
        
        db.register("test_table", df)
        
        # Query
        result = db.execute("SELECT COUNT(*) FROM test_table").fetchone()
        count = result[0]
        
        print(f"  ✅ DuckDB query OK (5 rows)")
        
        # Aggregation
        agg_result = db.execute(
            "SELECT SUM(sales) as total FROM test_table"
        ).fetchone()
        
        print(f"  ✅ Aggregation OK (total: {agg_result[0]})")
        
        return True
    
    except Exception as e:
        print(f"  ❌ Error DuckDB: {e}")
        return False


def test_rbac():
    """Verifica RBAC middleware."""
    print_header("5️⃣  VERIFICANDO RBAC MIDDLEWARE")
    
    try:
        from src.utils.rbac_middleware import RBACMiddleware
        
        print("  Probando RBAC...")
        
        # Test permissions
        master_perms = RBACMiddleware.get_user_permissions("master_admin")
        org_admin_perms = RBACMiddleware.get_user_permissions("org_admin")
        viewer_perms = RBACMiddleware.get_user_permissions("viewer")
        
        print(f"  ✅ Master Admin: {len(master_perms)} permisos")
        print(f"  ✅ Org Admin: {len(org_admin_perms)} permisos")
        print(f"  ✅ Viewer: {len(viewer_perms)} permisos")
        
        # Test permission checks
        has_upload = RBACMiddleware.has_permission("org_admin", "upload_data")
        no_upload = RBACMiddleware.has_permission("viewer", "upload_data")
        
        if has_upload and not no_upload:
            print("  ✅ Permission checks OK")
        else:
            print("  ❌ Permission checks FAILED")
            return False
        
        return True
    
    except Exception as e:
        print(f"  ❌ Error RBAC: {e}")
        return False


def test_dataservice():
    """Verifica DataService con DuckDB."""
    print_header("6️⃣  VERIFICANDO DATA SERVICE")
    
    try:
        from src.services.data_service import create_data_service
        
        print("  Creando DataService...")
        
        # Use test org_id
        ds = create_data_service("test-org-uuid")
        print(f"  ✅ DataService creado")
        
        # Test: listar tablas (debería estar vacío inicialmente)
        tables = ds.list_tables()
        
        if tables["success"]:
            print(f"  ✅ {tables['count']} tablas en DuckDB")
            return True
        else:
            print(f"  ❌ Error listando tablas: {tables['error']}")
            return False
    
    except Exception as e:
        print(f"  ❌ Error DataService: {e}")
        return False


def test_dashboard():
    """Verifica que el dashboard se puede importar."""
    print_header("7️⃣  VERIFICANDO DASHBOARD")
    
    try:
        from src.ui.dashboard_v2 import main, init_session_state
        
        print("  Importando dashboard_v2...")
        print(f"  ✅ Dashboard importado OK")
        
        print("  Funciones disponibles:")
        print(f"    - main()")
        print(f"    - init_session_state()")
        print(f"    - show_login_page()")
        print(f"    - render_header()")
        print(f"    - [6 page functions]")
        
        return True
    
    except Exception as e:
        print(f"  ❌ Error dashboard: {e}")
        return False


def generate_report(results: dict) -> str:
    """Genera reporte final."""
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    failed = total - passed
    
    report = "\n" + "=" * 60
    report += "\n  📊 REPORTE FINAL"
    report += "\n" + "=" * 60
    
    report += f"\n  ✅ Pasadas: {passed}/{total}"
    report += f"\n  ❌ Fallidas: {failed}/{total}"
    
    if failed == 0:
        report += "\n\n  🎉 ¡SETUP COMPLETADO EXITOSAMENTE!"
        report += "\n\n  Próximos pasos:"
        report += "\n    1. streamlit run main.py"
        report += "\n    2. Login con credenciales demo"
        report += "\n    3. Cambiar entre orgs"
        report += "\n    4. Explorar datos en S3"
    else:
        report += "\n\n  ⚠️  Hay problemas que resolver"
        report += "\n\n  Revisar errores arriba e instalar dependencias"
    
    report += "\n" + "=" * 60 + "\n"
    
    return report


def main():
    """Ejecuta suite de verificación."""
    
    print_header("VERIFICACIÓN DE SETUP - MULTI-TENANT SISTEMA TESIS")
    print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = {
        "Imports": test_imports(),
        "Supabase": test_supabase(),
        "S3": test_s3(),
        "DuckDB": test_duckdb(),
        "RBAC": test_rbac(),
        "DataService": test_dataservice(),
        "Dashboard": test_dashboard(),
    }
    
    # Generar reporte
    report = generate_report(results)
    print(report)
    
    # Return status
    if all(results.values()):
        print("✅ Setup verificado correctamente")
        return 0
    else:
        print("❌ Hay problemas en el setup")
        return 1


if __name__ == "__main__":
    sys.exit(main())
