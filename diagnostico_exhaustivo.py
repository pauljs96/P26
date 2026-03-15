"""
Verificación exhaustiva - Qué EXACTAMENTE va mal al procesar el archivo en S3

Este script simula lo que hace Streamlit Cloud en Streamlit Cloud.
"""

import pandas as pd
import io
from src.data.data_loader import DataLoader
from src.data.data_cleaner import DataCleaner

print("="*70)
print("DIAGNÓSTICO EXHAUSTIVO - Simular Streamlit Cloud")
print("="*70)

archivo = 'Datos_Completo_2020_2025.csv'

# 1. Leer archivo RAW
print(f"\n1. LECTURA RAW (primeros 500 bytes):")
with open(archivo, 'rb') as f:
    raw_bytes = f.read()

print(f"   Tamaño: {len(raw_bytes):,} bytes")
print(f"   Primeros 500 bytes (text):")
try:
    print(f"   {raw_bytes[:500].decode('utf-8')}")
except:
    print(f"   (No se puede decodificar como UTF-8)")

# 2. Verificar que tiene ';' como separador
print(f"\n2. VERIFICACIÓN DEL SEPARADOR:")
linea1 = raw_bytes.split(b'\n')[0].decode('utf-8')
print(f"   Primera línea: {linea1[:100]}")
print(f"   ¿Tiene ';'? {';' in linea1}")
print(f"   ¿Tiene ','? {',' in linea1}")
print(f"   ¿Tiene '|'? {'|' in linea1}")
print(f"   Conteo de ';': {linea1.count(';')}")

# 3. Simular lo que hace Streamlit uploader
print(f"\n3. SIMULACIÓN STREAMLIT UPLOADER:")

class FakeStreamlitFile:
    def __init__(self, filepath):
        with open(filepath, 'rb') as f:
            self.content = f.read()
        self.name = filepath
    
    def getvalue(self):
        return self.content

fake_file = FakeStreamlitFile(archivo)

# 4. Intentar cargarlo con DataLoader
print(f"\n4. CARGA CON DataLoader:")
loader = DataLoader()
try:
    df_raw = loader.load_files([fake_file])
    print(f"   ✓ Cargado: {len(df_raw)} filas, {len(df_raw.columns)} columnas")
    print(f"   Columnas: {list(df_raw.columns)[:8]}")
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 5. Ver qué recibe DataCleaner
print(f"\n5. QUÉ RECIBE DataCleaner:")
print(f"   Filas: {len(df_raw)}")
print(f"   Columnas: {len(df_raw.columns)}")
print(f"   Tipos: {df_raw.dtypes.to_dict()}")
print(f"   ¿Alguna columna vacía?")
for col in df_raw.columns[:5]:
    nulls = df_raw[col].isna().sum()
    print(f"     - {col}: {nulls} nulls de {len(df_raw)}")

# 6. Intentar limpiar
print(f"\n6. LIMPIEZA CON DataCleaner:")
cleaner = DataCleaner()
try:
    df_clean = cleaner.clean(df_raw)
    if df_clean.empty:
        print(f"   ✗ DataFrame vacío después de limpieza")
    else:
        print(f"   ✓ Limpieza exitosa: {len(df_clean)} filas")
except Exception as e:
    print(f"   ✗ Exception: {e}")
    import traceback
    traceback.print_exc()

# 7. Investigación: ¿Qué busca DataCleaner?
print(f"\n7. INVESTIGACIÓN - ¿Qué busca DataCleaner?")
from src.utils import config
from src.data.data_cleaner import _pick_col

print(f"   Buscando columnas en df_raw con {len(df_raw.columns)} columnas:")
print(f"   Codigo candidates: {config.COL_CODIGO}")
c = _pick_col(df_raw, config.COL_CODIGO)
print(f"     → Encontrado: {c}")

print(f"   Fecha candidates: {config.COL_FECHA}")
c = _pick_col(df_raw, config.COL_FECHA)
print(f"     → Encontrado: {c}")

print(f"   Documento candidates: {config.COL_DOCUMENTO}")
c = _pick_col(df_raw, config.COL_DOCUMENTO)
print(f"     → Encontrado: {c}")

print(f"   Entrada_unid candidates: {config.COL_ENTRADA_UNID}")
c = _pick_col(df_raw, config.COL_ENTRADA_UNID)
print(f"     → Encontrado: {c}")

print(f"\n" + "="*70)
print("✅ ANÁLISIS COMPLETO - Si llegaste aquí, el archivo está ok")
print("="*70)

