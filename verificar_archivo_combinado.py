"""Verificar que el archivo combinado sea procesado correctamente por el pipeline."""

import pandas as pd
import io
from src.data.data_loader import DataLoader
from src.data.data_cleaner import DataCleaner

print("="*70)
print("VERIFICACION: Datos_Completo_2020_2025.csv")
print("="*70)

# Simular lo que hace Streamlit con el archivo
with open('Datos_Completo_2020_2025.csv', 'rb') as f:
    file_content = f.read()

print(f"\n✓ Tamaño archivo: {len(file_content) / 1024 / 1024:.2f} MB")

# Crear objeto simulado de Streamlit
class FakeUploadedFile:
    def __init__(self, name, content):
        self.name = name
        self.content = content
    
    def getvalue(self):
        return self.content

fake_file = FakeUploadedFile('Datos_Completo_2020_2025.csv', file_content)

# Usar el loader real del sistema
print("\n1. Cargando con DataLoader...")
loader = DataLoader()
try:
    df_raw = loader.load_files([fake_file])
    print(f"   ✓ Cargado exitosamente")
    print(f"   {len(df_raw):,} filas")
    print(f"   {len(df_raw.columns)} columnas")
except Exception as e:
    print(f"   ✗ Error: {e}")
    exit(1)

# Limpiar con el cleaner
print("\n2. Ejecutando DataCleaner...")
cleaner = DataCleaner()
df_clean = cleaner.clean(df_raw)

if df_clean.empty:
    print(f"   ✗ ERROR: DataFrame vacío después de limpieza")
    exit(1)
else:
    print(f"   ✓ Limpieza exitosa")
    print(f"   {len(df_clean):,} filas")
    print(f"   {len(df_clean.columns)} columnas")

# Ejecutar reconciliación
print("\n3. Ejecutando reconciliador de guías...")
from src.data.guide_reconciliation import GuideReconciler
reconciler = GuideReconciler()
rec = reconciler.reconcile(df_clean)
print(f"   ✓ {len(rec):,} filas reconciliadas")

# Construir demanda
print("\n4. Construyendo demanda mensual...")
from src.data.demand_builder import DemandBuilder
demand_builder = DemandBuilder()
demand = demand_builder.build_monthly(rec)
print(f"   ✓ {len(demand):,} registros de demanda")

# Construir stock
print("\n5. Construyendo stock mensual...")
from src.data.stock_builder import StockBuilder
stock_builder = StockBuilder()
stock = stock_builder.build_monthly(rec)
print(f"   ✓ {len(stock):,} registros de stock")

print("\n" + "="*70)
print("✅ ARCHIVO LISTO PARA CARGAR AL DASHBOARD")
print("="*70)
print(f"Instrucciones:")
print(f"1. Abre el dashboard de PREDICAST")
print(f"2. Como Admin, ve a 'Subir Datos' en la barra lateral")
print(f"3. Selecciona: Datos_Completo_2020_2025.csv")
print(f"4. El archivo se procesará automáticamente")

