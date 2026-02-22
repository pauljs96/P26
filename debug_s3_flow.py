#!/usr/bin/env python
"""Debug script para verificar flujo de S3 upload/download"""

import sys
from pathlib import Path
import pandas as pd
import io

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from src.storage.s3_manager import S3Manager, get_storage_manager

def test_s3_flow():
    """Test upload y download de archivos S3"""
    
    print("\n" + "="*60)
    print("🧪 TEST: S3 Upload/Download Flow")
    print("="*60)
    
    # Inicializar storage
    storage = get_storage_manager()
    print(f"\n1️⃣ Storage Manager inicializado")
    print(f"   ├─ is_configured: {storage.is_configured}")
    print(f"   ├─ bucket_name: {storage.bucket_name}")
    print(f"   ├─ region: {storage.region}")
    print(f"   └─ s3_client exists: {storage.s3_client is not None}")
    
    # Crear CSV de prueba
    print(f"\n2️⃣ Creando CSV de prueba...")
    df_test = pd.DataFrame({
        'Producto': ['A', 'B', 'C', 'A', 'B', 'C'],
        'Fecha': pd.date_range('2024-01-01', periods=6),
        'Cantidad': [100, 150, 200, 120, 140, 180]
    })
    
    csv_bytes = df_test.to_csv(index=False).encode('utf-8')
    print(f"   └─ CSV creado: {len(csv_bytes)} bytes, {len(df_test)} filas")
    
    # Upload bytes a S3
    print(f"\n3️⃣ Subiendo CSV a S3...")
    result = storage.upload_file_bytes(
        csv_bytes,
        "test_file.csv",
        user_id="test_user",
        project_id="test_project"
    )
    
    print(f"   ├─ success: {result.get('success')}")
    print(f"   ├─ s3_key: {result.get('s3_key')}")
    print(f"   ├─ s3_url: {result.get('s3_url')}")
    print(f"   ├─ presigned_url: {'✅ Present' if result.get('presigned_url') else '❌ None'}")
    
    if not result.get('success'):
        print(f"   └─ ERROR: {result.get('error')}")
        return
    
    s3_key = result.get('s3_key')
    
    # Download desde S3
    print(f"\n4️⃣ Descargando CSV desde S3...")
    downloaded_bytes = storage.download_file(s3_key)
    
    if downloaded_bytes:
        print(f"   ├─ Descargado: {len(downloaded_bytes)} bytes")
        
        # Parsear CSV
        try:
            df_downloaded = pd.read_csv(io.BytesIO(downloaded_bytes))
            print(f"   ├─ CSV parseado: {len(df_downloaded)} filas, {len(df_downloaded.columns)} columnas")
            print(f"   ├─ Columnas: {list(df_downloaded.columns)}")
            print(f"   └─ ✅ Flujo S3 está funcionando correctamente")
        except Exception as e:
            print(f"   └─ ❌ Error al parsear CSV: {str(e)}")
    else:
        print(f"   └─ ❌ download_file retornó None")
        print(f"       Verifique:")
        print(f"       - S3 está configurado (is_configured={storage.is_configured})")
        print(f"       - Archivo existe en S3: {s3_key}")
        print(f"       - Credenciales AWS son correctas")
    
    # Test con URL completa (simulando lo que pasaba antes)
    print(f"\n5️⃣ Test con URL completa (debug)...")
    if result.get('s3_url'):
        s3_url = result.get('s3_url')
        print(f"   ├─ URL: {s3_url}")
        downloaded_from_url = storage.download_file(s3_url)
        if downloaded_from_url:
            print(f"   └─ ✅ download_file maneja URLs completas correctamente")
        else:
            print(f"   └─ ✅ download_file maneja URLs completas (extrapoladas a claves)")
    
    print(f"\n" + "="*60)
    print(f"✅ Test completado")
    print(f"="*60 + "\n")

if __name__ == "__main__":
    test_s3_flow()
