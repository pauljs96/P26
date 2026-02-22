#!/usr/bin/env python
"""Test E2E - Verifica integracion correcta del API con logica original"""

import sys
from pathlib import Path
import pandas as pd
import io
import requests
import json
from time import sleep

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.storage.s3_manager import S3Manager, get_storage_manager
from src.db.supabase import SupabaseDB
from src.api.utils.pipeline_adapter import process_csv_bytes_with_pipeline
from src.utils.logger import Logger

def test_pipeline_adapter():
    """Test: Pipeline adapter puede procesar bytes"""
    print("\n" + "="*60)
    print("TEST 1: Pipeline Adapter con bytes")
    print("="*60)
    
    logger = Logger(enabled=True)
    
    # Crear CSV de prueba
    df_test = pd.DataFrame({
        'Mes': pd.date_range('2024-01-01', periods=24, freq='MS'),
        'Producto': ['ProductoA'] * 24,
        'Cantidad': [100 + i*5 for i in range(24)],
        'Documento': [f'DOC-{i:04d}' for i in range(24)]
    })
    
    csv_bytes = df_test.to_csv(index=False).encode('utf-8')
    print(f"OK: CSV de prueba creado: {len(csv_bytes)} bytes")
    
    result = process_csv_bytes_with_pipeline(csv_bytes, "test.csv", logger=logger)
    
    if result.get("success"):
        print(f"OK: Pipeline procesó correctamente")
        print(f"   - Movimientos: {len(result.get('movements', []))} registros")
        print(f"   - Demanda: {len(result.get('demand_monthly', []))} periodos")
        return True
    else:
        print(f"ERROR: {result.get('error')}")
        return False

def test_api_upload_endpoint():
    """Test: Endpoint /uploads/process funciona"""
    print("\n" + "="*60)
    print("TEST 2: API Upload Processing")
    print("="*60)
    
    # Crear CSV de prueba y subirlo a S3
    storage = get_storage_manager()
    
    df_test = pd.DataFrame({
        'Mes': pd.date_range('2024-01-01', periods=24, freq='MS'),
        'Producto': ['Test Product'] * 24,
        'Cantidad': list(range(100, 124)),
        'Documento': [f'DOC-{i:04d}' for i in range(24)]
    })
    
    csv_bytes = df_test.to_csv(index=False).encode('utf-8')
    
    # Upload a S3
    upload_result = storage.upload_file_bytes(
        csv_bytes,
        "test_api_upload.csv",
        user_id="test_user",
        project_id="test_project"
    )
    
    if not upload_result.get("success"):
        print(f"ERROR: No se pudo subir a S3: {upload_result.get('error')}")
        return False
    
    s3_key = upload_result.get("s3_key")
    print(f"OK: Archivo subido a S3: {s3_key}")
    
    # Crear registro en Supabase
    db = SupabaseDB()
    upload_data = db.save_upload(
        user_id="test_user",
        project_id="test_project",
        filename="test_api_upload.csv",
        s3_path=s3_key,
        file_size=len(csv_bytes)
    )
    
    if not upload_data.get("success"):
        print(f"ERROR: No se pudo guardar en Supabase: {upload_data.get('error')}")
        return False
    
    upload_id = upload_data.get("upload_id")
    print(f"OK: Registro en Supabase: {upload_id}")
    
    # Llamar endpoint
    try:
        response = requests.post(
            "http://localhost:8000/uploads/process",
            json={
                "upload_id": upload_id,
                "s3_path": s3_key,
                "filename": "test_api_upload.csv"
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"OK: API procesó correctamente")
            print(f"   - Status: {result.get('status')}")
            print(f"   - Message: {result.get('message')}")
            return True
        else:
            print(f"ERROR: API {response.status_code}: {response.text}")
            return False
            
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return False

def test_imports():
    """Test: Verificar importaciones"""
    print("\n" + "="*60)
    print("TEST 0: Import Validation")
    print("="*60)
    
    try:
        from src.api.utils.pipeline_adapter import (
            process_csv_bytes_with_pipeline,
            extract_product_demand,
            BytesFile
        )
        print("OK: pipeline_adapter")
        
        from src.services.ml_service import compare_models, forecast_next_month
        print("OK: services.ml_service (original)")
        
        from src.api.routers.uploads import router as uploads_router
        print("OK: routers.uploads")
        
        from src.api.routers.forecasts import router as forecasts_router
        print("OK: routers.forecasts")
        
        return True
    except ImportError as e:
        print(f"ERROR: {str(e)}")
        return False

if __name__ == "__main__":
    print("\n" + "="*60)
    print("[TEST E2E] API Integration con logica original")
    print("="*60)
    
    results = {
        "imports": test_imports(),
        "pipeline_adapter": test_pipeline_adapter(),
        "api_upload": test_api_upload_endpoint(),
    }
    
    print("\n" + "="*60)
    print("[RESULTADOS]")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status}: {test_name}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*60)
    if all_passed:
        print("[OK] TODOS LOS TESTS PASARON")
        print("El API integra correctamente la logica original")
    else:
        print(f"[FAILED] {sum(1 for v in results.values() if not v)} test(s) fallaron")
    print("="*60 + "\n")
    
    sys.exit(0 if all_passed else 1)
