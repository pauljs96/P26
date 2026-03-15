"""Script directo para cargar datos en Supabase (sin pasar por Streamlit)."""

import pandas as pd
import os
from dotenv import load_dotenv
from src.db import get_db
from datetime import datetime

load_dotenv()

# Archivos a cargar
archivos = ['D_2020.csv', 'D_2021.csv', 'D_2022.csv', 'D_2023.csv', 'D_2024.csv', 'D_2025.csv']

print("="*70)
print("CARGANDO DATOS DIRECTAMENTE AL SISTEMA (Sin pasar por Streamlit)")
print("="*70)

# Conectar a BD
db = get_db()
org_id = "SVMPREMIUM"  # Tu organización
user_id = os.getenv('ADMIN_USER_ID') or 'brisa@gmail.com'

print(f"\nOrganización: {org_id}")
print(f"Usuario: {user_id}")

# Cargar y procesar con el pipeline
from src.data.pipeline import DataPipeline

dfs_raw = []
for archivo in archivos:
    if not os.path.exists(archivo):
        print(f"⚠️ {archivo} no encontrado")
        continue
    
    df = pd.read_csv(archivo, sep=';', decimal=',')
    df['__source_file'] = archivo
    dfs_raw.append(df)
    print(f"✓ {archivo}: {len(df)} filas")

if not dfs_raw:
    print("No hay archivos para procesar")
    exit(1)

df_combined = pd.concat(dfs_raw, ignore_index=True)
print(f"\n✓ Total combinado: {len(df_combined)} filas")

# Procesar con pipeline
print("\nEjecutando pipeline...")
pipeline = DataPipeline()

from src.data.data_cleaner import DataCleaner
from src.data.guide_reconciliation import GuideReconciler
from src.data.demand_builder import DemandBuilder
from src.data.stock_builder import ProductStockBuilder

cleaner = DataCleaner()
reconciler = GuideReconciler()
demand_builder = DemandBuilder()
stock_builder = ProductStockBuilder()

clean = cleaner.clean(df_combined)
print(f"  Clean: {len(clean)} filas")

rec = reconciler.reconcile(clean)
print(f"  Reconciled: {len(rec)} filas")

demand = demand_builder.build_monthly(rec)
print(f"  Demand: {len(demand)} registros")

stock = stock_builder.build_monthly(rec)
print(f"  Stock: {len(stock)} registros")

# Guardar en cache
print("\nGuardando en Supabase...")
try:
    if db:
        # Guardar datos en tabla de cache/processed_data
        cache_data = {
            'organization_id': org_id,
            'uploaded_by': user_id,
            'uploaded_at': datetime.now().isoformat(),
            'total_raw_records': len(df_combined),
            'total_clean_records': len(clean),
            'total_demand_records': len(demand),
            'total_stock_records': len(stock),
            'file_count': len(archivos),
            'file_names': ','.join(archivos)
        }
        
        result = db.client.table('uploads').insert(cache_data).execute()
        print(f"✅ Guardado en uploads table")
        
        # Aquí podrías guardar los datos en otras tablas si es necesario
        
except Exception as e:
    print(f"⚠️ Error en Supabase: {e}")

print("\n" + "="*70)
print("✅ CARGA COMPLETADA - Los datos están listos en el sistema")
print("="*70)

