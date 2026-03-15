"""Diagnóstico: Simular carga de 6 archivos separados como lo hace el dashboard."""

import pandas as pd
import io
from src.data.data_loader import DataLoader
from src.data.data_cleaner import DataCleaner

print("="*70)
print("DIAGNOSTICO: Cargar 6 archivos separados (como en el dashboard)")
print("="*70)

archivos = ['D_2020.csv', 'D_2021.csv', 'D_2022.csv', 'D_2023.csv', 'D_2024.csv', 'D_2025.csv']

# Simular objeto de Streamlit
class FakeUploadedFile:
    def __init__(self, name, content):
        self.name = name
        self.content = content
    
    def getvalue(self):
        return self.content

# Cargar los 6 archivos como si fueran subidos a Streamlit
uploaded_files = []
for archivo in archivos:
    try:
        with open(archivo, 'rb') as f:
            content = f.read()
        uploaded_files.append(FakeUploadedFile(archivo, content))
        print(f"✓ {archivo}: {len(content) / 1024:.1f} KB")
    except FileNotFoundError:
        print(f"✗ {archivo} no encontrado")

if len(uploaded_files) < 2:
    print("No hay archivos suficientes para probar")
    exit(1)

print(f"\nTotal: {len(uploaded_files)} archivos para procesar")

# Usar el loader real
print("\n" + "="*70)
print("Paso 1: DataLoader.load_files() con múltiples archivos")
print("="*70)

loader = DataLoader()
try:
    df_raw = loader.load_files(uploaded_files)
    print(f"✓ Cargado: {len(df_raw):,} filas")
    print(f"  Columnas: {list(df_raw.columns)[:5]}...(+ {len(df_raw.columns)-5} más)")
except Exception as e:
    print(f"✗ Error en load_files: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Pasar por DataCleaner
print("\n" + "="*70)
print("Paso 2: DataCleaner.clean() ")
print("="*70)

cleaner = DataCleaner()
try:
    df_clean = cleaner.clean(df_raw)
    print(f"✓ Limpieza exitosa: {len(df_clean):,} filas")
    print(f"  Columnas: {list(df_clean.columns)}")
except Exception as e:
    print(f"✗ Error en clean: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

if df_clean.empty:
    print("\n✗ PROBLEMA ENCONTRADO: DataFrame vacío después de limpieza")
    print("  Investigando...")
    
    # Ver qué tiene df_raw
    print(f"\n  df_raw tiene:")
    print(f"    - Filas: {len(df_raw)}")
    print(f"    - Columnas: {list(df_raw.columns)}")
    print(f"    - Tipos: {df_raw.dtypes.to_dict()}")
    
    # Intentar ver qué columnas buscaba
    from src.utils import config
    print(f"\n  Columnas requeridas:")
    print(f"    Codigo: {config.COL_CODIGO}")
    print(f"    Fecha: {config.COL_FECHA}")
    print(f"    Documento: {config.COL_DOCUMENTO}")
    
    exit(1)

print("\n✅ TODO OK - Los 6 archivos se cargan correctamente")
print("  El problema está en otra parte del pipeline")

