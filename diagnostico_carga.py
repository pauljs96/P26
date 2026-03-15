"""Diagnóstico del error de carga - Simula exactamente el flujo del dashboard."""

import pandas as pd
import io
from src.utils.config import CSV_SEPARATORS, CSV_ENCODINGS
from src.data.data_loader import DataLoader
from src.data.data_cleaner import DataCleaner

print("="*70)
print("DIAGNOSTICO: Cargando D_2020.csv como lo hace el dashboard")
print("="*70)

# Simular lo que hace Streamlit con el archivo
with open('D_2020.csv', 'rb') as f:
    file_content = f.read()

print(f"\n1. Tamaño archivo: {len(file_content)} bytes")
print(f"   Primeros 200 caracteres (hex):")
print(f"   {file_content[:200]}")

# Crear objeto simulado de Streamlit
class FakeUploadedFile:
    def __init__(self, name, content):
        self.name = name
        self.content = content
    
    def getvalue(self):
        return self.content

fake_file = FakeUploadedFile('D_2020.csv', file_content)

# Usar el loader real del sistema
print("\n2. Intentando cargar con DataLoader...")
loader = DataLoader()
try:
    df_raw = loader.load_files([fake_file])
    print(f"   ✓ Cargado exitosamente")
    print(f"   Filas: {len(df_raw)}")
    print(f"   Columnas ({len(df_raw.columns)}):")
    for i, col in enumerate(df_raw.columns, 1):
        print(f"      {i}. '{col}'")
except Exception as e:
    print(f"   ✗ Error en carga: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Limpiar con el cleaner
print("\n3. Limpiando con DataCleaner...")
cleaner = DataCleaner()
df_clean = cleaner.clean(df_raw)

if df_clean.empty:
    print(f"   ✗ ERROR: DataFrame vacío después de limpieza")
    print(f"   Esto significa que DataCleaner no encontró las columnas requeridas")
    
    # Investigar por qué
    from src.utils import config
    print(f"\n4. Investigación - Columnas requeridas vs presentes:")
    
    required_cols = [
        ('Codigo', config.COL_CODIGO),
        ('Fecha', config.COL_FECHA),
        ('Documento', config.COL_DOCUMENTO),
        ('Numero', config.COL_NUMERO),
        ('Bodega', config.COL_BODEGA),
        ('Entrada_unid', config.COL_ENTRADA_UNID),
        ('Salida_unid', config.COL_SALIDA_UNID),
        ('Saldo_unid', config.COL_SALDO_UNID),
    ]
    
    for req_name, candidates in required_cols:
        found = False
        for candidate in candidates:
            if candidate in df_raw.columns or candidate.lower() in [c.lower() for c in df_raw.columns]:
                print(f"   ✓ {req_name}: '{candidate}' encontrado")
                found = True
                break
        
        if not found:
            print(f"   ✗ {req_name}: NO ENCONTRADO")
            print(f"      Buscaba uno de: {candidates}")
            print(f"      Columnas disponibles: {list(df_raw.columns)}")
    
else:
    print(f"   ✓ Limpieza exitosa")
    print(f"   Filas: {len(df_clean)}")
    print(f"   Columnas: {list(df_clean.columns)}")
    print(f"\n✅ SISTEMA LISTO - Los datos se cargarían correctamente")

