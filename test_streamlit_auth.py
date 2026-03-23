"""Script para diagnosticar problemas con Streamlit Cloud authentication"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()

print("=" * 60)
print("DIAGNOSTICO DE AUTENTICACION")
print("=" * 60)

# 1. Verificar variables de entorno
print("\n1. VARIABLES DE ENTORNO:")
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")

print(f"   SUPABASE_URL: {'[OK] Configurada' if supabase_url else '[ERROR] NO CONFIGURADA'}")
if supabase_url:
    print(f"      -> {supabase_url[:50]}...")

print(f"   SUPABASE_KEY: {'[OK] Configurada' if supabase_key else '[ERROR] NO CONFIGURADA'}")
if supabase_key:
    print(f"      -> {supabase_key[:30]}...")

# 2. Intentar importar módulos
print("\n2. IMPORTS:")
try:
    from src.db import get_db
    print("   [OK] src.db.get_db importable")
except Exception as e:
    print(f"   [ERROR] Error en import: {str(e)}")
    sys.exit(1)

try:
    from src.ui.dashboard import Dashboard
    print("   [OK] src.ui.dashboard.Dashboard importable")
except Exception as e:
    print(f"   [ERROR] Error en import: {str(e)}")
    sys.exit(1)

# 3. Intentar inicializar Supabase DB
print("\n3. INICIALIZAR SUPABASE:")
try:
    db = get_db()
    print("   [OK] SupabaseDB inicializado correctamente")
except ValueError as e:
    print(f"   [WARN] ValueError: {str(e)}")
    print("      -> SOLUCION: Configura SUPABASE_URL y SUPABASE_KEY en .env")
except Exception as e:
    print(f"   [ERROR] Error inesperado: {type(e).__name__}: {str(e)}")
    import traceback
    traceback.print_exc()

# 4. Verificar conectividad Supabase
print("\n4. CONECTIVIDAD SUPABASE:")
try:
    db = get_db()
    print("   [INFO] Intentando conectar a Supabase...")
    
    # Intento simple: listar organizaciones (operación de lectura)
    orgs = db.list_all_organizations()
    print(f"   [OK] Supabase accesible - {len(orgs)} organizaciones encontradas")
except Exception as e:
    print(f"   [ERROR] Error de conectividad: {type(e).__name__}")
    print(f"      -> {str(e)[:100]}")

# 5. Resumen
print("\n" + "=" * 60)
print("RESUMEN:")
if not supabase_url or not supabase_key:
    print("[ERROR] PROBLEMA: Falta configuracion de Supabase")
    print("   SOLUCION:")
    print("   1. Abre Streamlit Cloud Dashboard")
    print("   2. Selecciona tu app")
    print("   3. Settings -> Secrets")
    print("   4. Agrega:")
    print("   ")
    print("   SUPABASE_URL=xxxxx")
    print("   SUPABASE_KEY=xxxxx")
else:
    print("[OK] Configuracion basica correcta")
    print("   El error podria estar en:")
    print("   - Tablas no existen en Supabase")
    print("   - Credenciales invalidas")
    print("   - Esquema de BD incorrecto")

print("=" * 60)
