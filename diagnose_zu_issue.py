"""
Script para diagnosticar por qué zu@gmail.com puede estar dando error

Verifica:
1. Si la organización de zu@gmail.com tiene datos cacheados
2. Si hay datos en org_cache para Empresa4
3. Estado general de la BD
"""

import os
from dotenv import load_dotenv

# Cargar .env ANTES de importar get_db
load_dotenv()

from src.db import get_db
import json

print("=" * 60)
print("DIAGNOSTICO: ESTADO DE DATOS EN SUPABASE")
print("=" * 60)

db = get_db()

# 1. Listar todas las organizaciones y su estado
print("\n1. ORGANIZACIONES Y ESTADO DE DATOS:")
print("-" * 60)
orgs = db.list_all_organizations()
for org in orgs:
    org_id = org.get('id')
    org_name = org.get('name')
    data_loaded = org.get('data_loaded', False)
    is_active = org.get('is_active', True)
    
    status = "[datos cargados]" if data_loaded else "[sin datos]"
    active = "[activa]" if is_active else "[inactiva]"
    
    print(f"\n  {org_name} ({active})")
    print(f"    ID: {org_id}")
    print(f"    Estado: {status}")
    
    # Verificar si tiene cache
        try:
            cache = db.load_org_data(org_id)
            if cache.get('success'):
                print(f"    Cache: OK (actualizado: {cache.get('updated_at')})")
                print(f"    CSV procesados: {cache.get('csv_files_count')}")
            else:
                print(f"    Cache: NO (vacio)")
        except:
            print(f"    Cache: NO accesible")

# 2. Listar usuarios y su estado
print("\n\n2. USUARIOS Y SUS ORGANIZACIONES:")
print("-" * 60)
usuarios = db.get_all_users()
for user in usuarios:
    email = user.get('email')
    org_id = user.get('organization_id')
    org_name = user.get('organization_name', 'Sin Org')
    role = user.get('role_name', 'viewer')
    
    print(f"\n  {email}")
    print(f"    Organización: {org_name}")
    print(f"    Rol: {role}")
    
    # Verificar cache de su org
    if org_id:
        try:
            cache = db.load_org_data(org_id)
            if cache.get('success'):
                print(f"    Estado: [OK] Tiene datos cacheados")
            else:
                print(f"    Estado: [WARN] Sin datos cacheados")
        except:
            print(f"    Estado: [ERROR] Error al acceder al cache")

# 3. Verificar zu@gmail.com específicamente
print("\n\n3. ESTADO DE zu@gmail.com (DETALLADO):")
print("-" * 60)
zu_user = [u for u in usuarios if u.get('email') == 'zu@gmail.com']
if zu_user:
    user = zu_user[0]
    print(f"  Email: {user.get('email')}")
    print(f"  User ID: {user.get('id')}")
    print(f"  Organización: {user.get('organization_name')}")
    print(f"  Org ID: {user.get('organization_id')}")
    print(f"  Rol: {user.get('role_name')}")
    print(f"  Admin: {user.get('is_admin')}")
    
    org_id = user.get('organization_id')
    if org_id:
        org = db.get_organization(org_id)
        print(f"\n  Detalles de Organización:")
        print(f"    Nombre: {org.get('name')}")
        print(f"    Datos cargados: {org.get('data_loaded')}")
        print(f"    Activa: {org.get('is_active')}")
        print(f"    Creada: {org.get('created_at')}")
        
        # Intentar cargar datos del cache
        print(f"\n  Verificando cache de datos:")
        cache = db.load_org_data(org_id)
        if cache.get('success'):
            print(f"    [OK] CACHE ENCONTRADO")
            print(f"    Actualizado: {cache.get('updated_at')}")
            print(f"    Procesado por: {cache.get('processed_by')}")
            print(f"    Archivos: {cache.get('csv_files_count')}")
            
            # Mostrar primeras líneas de datos
            import pandas as pd
            from src.utils.cache_helpers import json_to_dataframe
            
            demand_json = cache.get('demand_monthly')
            if demand_json:
                demand_df = json_to_dataframe(demand_json)
                print(f"\n    Datos en cache:")
                print(f"      Demanda: {len(demand_df)} registros")
                print(f"      Productos unicos: {demand_df['Codigo'].nunique() if 'Codigo' in demand_df.columns else 'N/A'}")
        else:
            print(f"    [ERROR] SIN CACHE: {cache.get('error')}")
else:
    print("  zu@gmail.com NO ENCONTRADO EN BASE DE DATOS")

print("\n" + "=" * 60)
print("RECOMENDACION:")
print("=" * 60)
print("""
Si zu@gmail.com no tiene datos:
  1. Asegúrate de cargar los archivos CSV después del login
  2. Espera a que aparezca "✅ Procesados X archivos"
  3. Recarga la página (F5)
  4. Los datos quedarán cacheados en Supabase

Si el error persiste:
  1. Revisa que Supabase esté accesible
  2. Verifica las credenciales en .env
  3. Contacta al administrador
""")
print("=" * 60)
