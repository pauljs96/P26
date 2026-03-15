"""
Debugging: Simular exactamente lo que hace Streamlit Cloud + S3

Flujo real en prod:
1. User sube archivos en Dashboard (Streamlit Cloud)
2. Se guardan en S3
3. Se leen de S3
4. Se procesan con pipeline
5. Se guardan en Supabase

¿Dónde falla?
"""

import pandas as pd
import io
from src.data.data_loader import DataLoader
from src.data.data_cleaner import DataCleaner
from src.utils import config

print("="*70)
print("DEBUG: Simulando lectura local de D_2020.csv")
print("="*70)

# Ver qué tiene el archivo crudo
print("\n1. Lectura DIRECTA del archivo (sin procesar):")
df_directo = pd.read_csv('D_2020.csv', sep=';', decimal=',', dtype=str)
print(f"   Tipo lectura: directo con pandas")
print(f"   Filas: {len(df_directo)}")
print(f"   Columnas ({len(df_directo.columns)}):")
for col in df_directo.columns:
    print(f"     - '{col}'")

# Ver primeros valores
print(f"\n   Primeros 2 valores (fila 0):")
print(f"   {df_directo.iloc[0, :5].to_dict()}")

# Ahora simular lo que hace DataLoader
print("\n" + "="*70)
print("2. Lectura con DataLoader (como lo hace Streamlit):")
print("="*70)

class FakeStreamlitFile:
    """Simula un archivo cargado en Streamlit"""
    def __init__(self, filepath):
        with open(filepath, 'rb') as f:
            self.content = f.read()
        self.name = filepath
    
    def getvalue(self):
        return self.content
    
    def seek(self, pos):
        pass  # No needed for this test

fake_file = FakeStreamlitFile('D_2020.csv')
loader = DataLoader()

try:
    df_loaded = loader.load_files([fake_file])
    print(f"   ✓ Cargado con DataLoader")
    print(f"   Filas: {len(df_loaded)}")
    print(f"   Columnas ({len(df_loaded.columns)}):")
    for col in df_loaded.columns:
        print(f"     - '{col}'")
except Exception as e:
    print(f"   ✗ Error en DataLoader: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Ahora pasar por DataCleaner
print("\n" + "="*70)
print("3. Procesamiento con DataCleaner:")
print("="*70)

cleaner = DataCleaner()

# Debug: Ver qué busca en config
print(f"\n   Columnas que busca DataCleaner:")
print(f"     Codigo: {config.COL_CODIGO}")
print(f"     Descripcion: {config.COL_DESCRIPCION}")
print(f"     Fecha: {config.COL_FECHA}")
print(f"     Documento: {config.COL_DOCUMENTO}")
print(f"     Numero: {config.COL_NUMERO}")
print(f"     Bodega: {config.COL_BODEGA}")
print(f"     Entrada_unid: {config.COL_ENTRADA_UNID}")
print(f"     Salida_unid: {config.COL_SALIDA_UNID}")
print(f"     Saldo_unid: {config.COL_SALDO_UNID}")

# Intentar limpiar
df_clean = cleaner.clean(df_loaded)

if df_clean.empty:
    print(f"\n   ✗ ENCONTRADO: DataFrame vacío después de limpieza")
    print(f"   Esto significa que falta una columna requerida")
    
    # Buscar qué columna falta
    print(f"\n   Investigando...")
    from src.data.data_cleaner import _pick_col
    
    missing = []
    for req_name, candidates in [
        ('Codigo', config.COL_CODIGO),
        ('Fecha', config.COL_FECHA),
        ('Documento', config.COL_DOCUMENTO),
        ('Numero', config.COL_NUMERO),
        ('Bodega', config.COL_BODEGA),
        ('Entrada_unid', config.COL_ENTRADA_UNID),
        ('Salida_unid', config.COL_SALIDA_UNID),
        ('Saldo_unid', config.COL_SALDO_UNID),
    ]:
        found = _pick_col(df_loaded, candidates)
        if found is None:
            missing.append((req_name, candidates))
            print(f"   ✗ {req_name}: NO ENCONTRADO")
            print(f"      Buscaba: {candidates}")
        else:
            print(f"   ✓ {req_name}: encontrado como '{found}'")
    
    if missing:
        print(f"\n   COLUMNAS FALTANTES:")
        for name, candidates in missing:
            print(f"   - {name}")
    
    exit(1)
else:
    print(f"\n   ✓ Limpieza exitosa")
    print(f"   Filas: {len(df_clean)}")
    print(f"   Columnas ({len(df_clean.columns)}):")
    for col in df_clean.columns[:5]:
        print(f"     - '{col}'")
    print(f"   ... ({len(df_clean.columns)-5} más)")

print("\n" + "="*70)
print("✅ TODO OK - Los archivos se cargan correctamente")
print("="*70)
print("\nSi ves esto, el problema está DESPUÉS del DataCleaner")
print("(en GuideReconciler, DemandBuilder, StockBuilder, o al guardar)")

