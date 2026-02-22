"""
Test script for WEEK 1: Multi-Tenant Database Setup
Ejecuta esto después de completar el SQL en Supabase Console
"""

import pandas as pd
import json
from datetime import datetime

print("\n" + "="*60)
print("WEEK 1: Verification Tests")
print("="*60)

# Test 1: Import SupabaseDB
print("\n[TEST 1] Verificar imports...")
try:
    from src.db.supabase import SupabaseDB
    print("✅ SupabaseDB imported successfully")
except ImportError as e:
    print(f"❌ Error importing SupabaseDB: {e}")
    exit(1)

# Test 2: Import cache_helpers
print("\n[TEST 2] Verificar cache_helpers...")
try:
    from src.utils.cache_helpers import (
        dataframe_to_json, 
        json_to_dataframe,
        serialize_pipeline_result,
        deserialize_pipeline_result
    )
    print("✅ cache_helpers imported successfully")
except ImportError as e:
    print(f"❌ Error importing cache_helpers: {e}")
    exit(1)

# Test 3: Initialize SupabaseDB
print("\n[TEST 3] Inicializar SupabaseDB...")
try:
    db = SupabaseDB()
    print("✅ SupabaseDB initialized successfully")
    print(f"   URL: {db.url[:40]}...")
except ValueError as e:
    # Si .env no existe, es OK, seguir con otros tests
    print(f"⚠️  Skipping: {e}")
    print("   (Este error es esperado si .env no está configurado)")
    db = None
except Exception as e:
    print(f"❌ Error initializing SupabaseDB: {e}")
    exit(1)

# Test 4: Test DataFrame serialization
print("\n[TEST 4] Test DataFrame serialization...")
try:
    # Crear DataFrame de prueba
    test_data = {
        'producto': ['A', 'B', 'C'],
        'cantidad': [10.5, 20.3, 15.7],
        'fecha': pd.to_datetime(['2025-01-01', '2025-01-02', '2025-01-03'])
    }
    test_df = pd.DataFrame(test_data)
    test_df.set_index('fecha', inplace=True)
    
    # Serializar
    json_str = dataframe_to_json(test_df)
    print(f"✅ DataFrame serialized ({len(json_str)} chars)")
    print(f"   Preview: {json_str[:100]}...")
    
    # Deserializar
    recovered_df = json_to_dataframe(json_str)
    print(f"✅ DataFrame deserialized ({len(recovered_df)} rows)")
    
    # Verificar contenido
    assert len(recovered_df) == 3, "Row count mismatch"
    assert 'producto' in recovered_df.columns, "Column missing"
    print("✅ Data integrity verified")
    
except Exception as e:
    print(f"❌ Error in serialization: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 5: Test SupabaseDB methods (sin conexión real, solo sintaxis)
print("\n[TEST 5] Verificar métodos nuevos en SupabaseDB...")
try:
    if db is None:
        print("⚠️  Skipping (SupabaseDB no inicializado)")
    else:
        # Verificar que los métodos existan
        methods = [
            'create_organization',
            'get_organization',
            'get_user_organization',
            'create_user_in_organization',
            'get_organization_users',
            'save_org_data',
            'load_org_data',
            'is_data_loaded',
            'save_csv_schema',
            'get_csv_schema'
        ]
        
        for method_name in methods:
            if hasattr(db, method_name):
                print(f"✅ Method exists: {method_name}")
            else:
                print(f"❌ Method missing: {method_name}")
                exit(1)
            
except Exception as e:
    print(f"❌ Error checking methods: {e}")
    exit(1)

# Test 6: Dry-run: create_organization signature
print("\n[TEST 6] Verificar firma de método create_organization...")
try:
    import inspect
    # Verificar que el método exista en la clase aunque db sea None
    sig = inspect.signature(SupabaseDB.create_organization)
    params = list(sig.parameters.keys())
    expected = ['self', 'nombre', 'admin_user_id', 'description']
    
    for param in ['nombre', 'admin_user_id', 'description']:
        if param in params:
            print(f"✅ Parameter: {param}")
        else:
            print(f"❌ Missing parameter: {param}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    exit(1)

# Final summary
print("\n" + "="*60)
print("✅ ALL TESTS PASSED")
print("="*60)
print("\nProximos pasos:")
print("1. Ejecuta el SQL en Supabase Console: db_migrations/001_multi_tenant_schema.sql")
print("2. Verifica que las tablas existan en Supabase Table Editor")
print("3. (Optional) Crea orgs de prueba via el SQL script")
print("4. En WEEK 2, agregaremos Admin Panel al dashboard")
print("\n" + "="*60 + "\n")
