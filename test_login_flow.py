"""Test del flujo de login paso a paso"""

import os
from dotenv import load_dotenv

load_dotenv()

from src.db import get_db

print("=" * 60)
print("TESTING LOGIN FLOW")
print("=" * 60)

db = get_db()

# 1. Verificar que las tablas existan
print("\n1. VERIFICAR TABLAS EN SUPABASE:")
print("   Consultando usuarios...")

try:
    users = db.get_all_users()
    print(f"   [OK] Tabla 'users' accesible - {len(users)} usuarios encontrados")
except Exception as e:
    print(f"   [ERROR] No se puede acceder a tabla 'users':")
    print(f"      {type(e).__name__}: {str(e)}")

# 2. Intentar login con usuario inexistente
print("\n2. INTENTO DE LOGIN (usuario inexistente):")
try:
    result = db.login_user("test@example.com", "wrong_password")
    print(f"   Resultado: {result}")
    if result.get("success"):
        print("   [ERROR] Login deberia fallar!")
    else:
        print(f"   [OK] Login fallo como se esperaba: {result.get('error')}")
except Exception as e:
    print(f"   [ERROR] Exception en login_user():")
    print(f"      {type(e).__name__}: {str(e)[:200]}")
    import traceback
    traceback.print_exc()

# 3. Listar usuarios existentes
print("\n3. USUARIOS EXISTENTES EN SISTEMA:")
try:
    users = db.get_all_users()
    print(f"   Total: {len(users)}")
    for user in users[:5]:  # Mostrar primeros 5
        email = user.get("email", "N/A")
        org = user.get("organization_name", "Sin org")
        role = user.get("role_name", "N/A")
        print(f"      - {email} | Org: {org} | Rol: {role}")
    if len(users) > 5:
        print(f"      ... y {len(users) - 5} usuarios mas")
except Exception as e:
    print(f"   [ERROR] Error listando usuarios:")
    print(f"      {type(e).__name__}: {str(e)}")

# 4. Si hay usuarios, intentar login con el primero
print("\n4. INTENTO DE LOGIN CON USUARIO EXISTENTE:")
try:
    users = db.get_all_users()
    if users:
        test_user = users[0]
        test_email = test_user.get("email")
        print(f"   Email para test: {test_email}")
        print("   Nota: No se puede testear la password correcta sin tener acceso directo")
        print("   (La password hasheada no se puede extraer de Supabase)")
        
        # Intentar con password vacia (probablemente fallara pero veremos el error)
        result = db.login_user(test_email, "test_password_123")
        print(f"   Login result: {result}")
    else:
        print("   [SKIP] No hay usuarios en la BD para testear")
except Exception as e:
    print(f"   [ERROR] Exception en login test:")
    print(f"      {type(e).__name__}: {str(e)[:200]}")

print("\n" + "=" * 60)
print("DIAGNOSTICO COMPLETO")
print("=" * 60)
