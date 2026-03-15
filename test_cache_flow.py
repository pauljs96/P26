#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de prueba para el flujo completo de cache
Valida: serialize -> save -> load -> deserialize
"""

import sys
import os
import io

# Set UTF-8 encoding for stdout
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Test imports
try:
    from src.db.supabase import SupabaseDB
    from src.services.cache_service import save_org_cache, check_and_load_org_cache
    from src.utils.cache_helpers import (
        serialize_pipeline_result,
        deserialize_pipeline_result,
        dataframe_to_json,
        json_to_dataframe
    )
    print("[OK] Imports exitosos")
except Exception as e:
    print(f"[ERROR] Error en imports: {e}")
    sys.exit(1)

# Test 1: Crear datos de prueba
print("\n" + "="*60)
print("TEST 1: Crear DataFrames de prueba")
print("="*60)
try:
    # DataFrames pequenos para testing
    movements = pd.DataFrame({
        'product': ['ProductA', 'ProductB', 'ProductC'],
        'quantity': [100, 200, 150],
        'date': pd.date_range('2024-01-01', periods=3)
    })
    
    demand_monthly = pd.DataFrame({
        'month': pd.date_range('2024-01-01', periods=12, freq='MS'),
        'product': ['ProductA'] * 12,
        'demand': range(100, 112)
    })
    
    stock_monthly = pd.DataFrame({
        'month': pd.date_range('2024-01-01', periods=12, freq='MS'),
        'product': ['ProductA'] * 12,
        'stock': range(500, 512)
    })
    
    print(f"[OK] movements: {movements.shape}")
    print(f"[OK] demand_monthly: {demand_monthly.shape}")
    print(f"[OK] stock_monthly: {stock_monthly.shape}")
except Exception as e:
    print(f"[ERROR] Error creando DataFrames: {e}")
    sys.exit(1)

# Test 2: Serializar
print("\n" + "="*60)
print("TEST 2: Serializar DataFrames")
print("="*60)
try:
    movements_json, demand_json, stock_json = serialize_pipeline_result(
        movements, demand_monthly, stock_monthly
    )
    print(f"[OK] movements_json: {len(movements_json)} chars")
    print(f"[OK] demand_json: {len(demand_json)} chars")
    print(f"[OK] stock_json: {len(stock_json)} chars")
except Exception as e:
    print(f"[ERROR] Error serializando: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Inicializar DB
print("\n" + "="*60)
print("TEST 3: Inicializar SupabaseDB")
print("="*60)
try:
    db = SupabaseDB()
    print(f"[OK] SupabaseDB inicializado")
    print(f"   URL: {db.url[:50]}...")
except Exception as e:
    print(f"[ERROR] Error inicializando SupabaseDB: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Guardar en cache
print("\n" + "="*60)
print("TEST 4: Guardar en cache (save_org_cache)")
print("="*60)
try:
    import uuid
    test_org_id = str(uuid.uuid4())  # Usar UUID valido
    # Crear un user_id valido - puede ser el user_id del usuario autenticado
    # Para testing, usar un UUID fijo
    test_user = str(uuid.uuid4())  # Generar UUID para el usuario
    
    print(f"[INFO] Usando org_id: {test_org_id}")
    print(f"[INFO] Usando user: {test_user}")
    
    success, timestamp = save_org_cache(
        db=db,
        org_id=test_org_id,
        movements=movements,
        demand_monthly=demand_monthly,
        stock_monthly=stock_monthly,
        processed_by=test_user,
        csv_files_count=6
    )
    
    print(f"[OK] save_org_cache retorno: success={success}, timestamp={timestamp}")
    
    if not success:
        print("[ERROR] save_org_cache fallo!")
        sys.exit(1)
except Exception as e:
    print(f"[ERROR] Error guardando cache: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Cargar del cache
print("\n" + "="*60)
print("TEST 5: Cargar del cache (check_and_load_org_cache)")
print("="*60)
try:
    has_cache, cached_data = check_and_load_org_cache(
        db=db,
        org_id=test_org_id,
        last_cache_timestamp=None
    )
    
    print(f"[OK] has_cache={has_cache}")
    
    if has_cache and cached_data:
        print(f"[OK] Datos cacheados loaded:")
        print(f"   - movements shape: {cached_data['movements'].shape if cached_data['movements'] is not None else 'None'}")
        print(f"   - demand_monthly shape: {cached_data['demand_monthly'].shape}")
        print(f"   - stock_monthly shape: {cached_data['stock_monthly'].shape}")
        print(f"   - updated_at: {cached_data['updated_at']}")
    else:
        print("[ERROR] No cache found!")
        sys.exit(1)
except Exception as e:
    print(f"[ERROR] Error cargando cache: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Verificar integridad
print("\n" + "="*60)
print("TEST 6: Verificar integridad de datos")
print("="*60)
try:
    # Comparar shapes
    original_movements_shape = movements.shape
    loaded_movements_shape = cached_data['movements'].shape if cached_data['movements'] is not None else (0, 0)
    
    if original_movements_shape == loaded_movements_shape:
        print(f"[OK] movements shape coincide: {original_movements_shape}")
    else:
        print(f"[ERROR] movements shape NO coincide: {original_movements_shape} vs {loaded_movements_shape}")
    
    if demand_monthly.shape == cached_data['demand_monthly'].shape:
        print(f"[OK] demand_monthly shape coincide: {demand_monthly.shape}")
    else:
        print(f"[ERROR] demand_monthly shape NO coincide")
        
    if stock_monthly.shape == cached_data['stock_monthly'].shape:
        print(f"[OK] stock_monthly shape coincide: {stock_monthly.shape}")
    else:
        print(f"[ERROR] stock_monthly shape NO coincide")
    
except Exception as e:
    print(f"[ERROR] Error verificando: {e}")
    sys.exit(1)

print("\n" + "="*60)
print("[OK] TODOS LOS TESTS PASARON")
print("="*60)
