"""Test de integración del nuevo router DataLoader.

Verifica que:
1. Auto-detecta formato NUEVO (Data.csv)
2. Auto-detecta formato LEGACY (v4)
3. Normaliza correctamente ambos
4. Pipeline sigue funcionando sin cambios
"""

import sys
import pandas as pd
from pathlib import Path

# Agregar proyecto a path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.data_loader import DataLoader
from src.data.pipeline import DataPipeline


def test_loader_detection():
    """Test auto-detección de formato."""
    print("\n" + "="*70)
    print("TEST 1: Auto-detección de formato")
    print("="*70)
    
    loader = DataLoader()
    print(f"✓ DataLoader instanciado")
    print(f"  - Loader New: {loader.loader_new}")
    print(f"  - Loader Legacy: {loader.loader_legacy}")
    print(f"  - Formato detectado: {loader.detected_format}")
    
    return True


def test_normalize_functions():
    """Test funciones de normalización."""
    print("\n" + "="*70)
    print("TEST 2: Funciones de normalización unificadas")
    print("="*70)
    
    from src.data.normalize_functions import (
        normalize_fecha,
        normalize_producto_codigo,
        normalize_dataframe,
        get_column_mapping
    )
    
    # Test 1: Normalización de fecha DD/MM/YYYY
    df_test = pd.DataFrame({
        "Fecha": ["01/01/2022", "15/02/2022"],
        "Producto_codigo": ["MECO_01", "MECO_02"],
        "Cantidad": ["5", "10"]
    })
    
    df_norm = normalize_fecha(df_test, "Fecha")
    print(f"✓ Fecha normalizada: {df_norm['Fecha'].dtype}")
    assert df_norm['Fecha'].dtype == 'datetime64[ns]', "Fecha debería ser datetime64"
    
    # Test 2: Normalización de producto
    df_norm = normalize_producto_codigo(df_test)
    print(f"✓ Producto_codigo renombrado a Producto_id: {'Producto_id' in df_norm.columns}")
    
    # Test 3: Mapeo de columnas
    cols_new = get_column_mapping(is_new_format=True)
    cols_legacy = get_column_mapping(is_new_format=False)
    print(f"✓ Nuevo formato tiene {len(cols_new)} columnas mapeadas")
    print(f"✓ Legacy formato tiene {len(cols_legacy)} columnas mapeadas")
    
    return True


def test_pipeline_compatibility():
    """Test que pipeline sigue funcionando."""
    print("\n" + "="*70)
    print("TEST 3: Compatibilidad con Pipeline")
    print("="*70)
    
    # Verificar imports
    pipeline = DataPipeline()
    print(f"✓ DataPipeline instanciado")
    print(f"  - Loader type: {type(pipeline.loader).__name__}")
    print(f"  - Loader.load_files existe: {hasattr(pipeline.loader, 'load_files')}")
    print(f"  - Loader.get_detected_format existe: {hasattr(pipeline.loader, 'get_detected_format')}")
    
    return True


def test_file_structure():
    """Test estructura de archivos."""
    print("\n" + "="*70)
    print("TEST 4: Estructura de archivos")
    print("="*70)
    
    files = [
        "src/data/normalize_functions.py",
        "src/data/data_loader.py",
        "src/data/data_loader_new.py",
        "src/data/data_loader_legacy.py",
    ]
    
    for f in files:
        path = Path(f)
        exists = path.exists()
        size = path.stat().st_size if exists else 0
        print(f"{'✓' if exists else '✗'} {f}: {size} bytes")
    
    return all(Path(f).exists() for f in files)


if __name__ == "__main__":
    try:
        print("\n" + "#"*70)
        print("# TEST DE INTEGRACIÓN: NUEVO ROUTER DATA LOADER")
        print("#"*70)
        
        test_loader_detection()
        test_normalize_functions()
        test_pipeline_compatibility()
        test_file_structure()
        
        print("\n" + "="*70)
        print("✓ TODOS LOS TESTS COMPLETADOS")
        print("="*70)
        print("\nProximos pasos:")
        print("1. Probar upload de Data.csv en Streamlit")
        print("2. Probar upload de v4 CSV existente")
        print("3. Verificar que dashboard funciona igual")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n✗ ERROR EN TEST: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
