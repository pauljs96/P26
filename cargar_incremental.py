"""
Cargador incremental - Permite cargar datos año por año sin pasar por S3.
Ideal para escalabilidad futura (2027, 2028, etc.)
"""

import pandas as pd
import os
from datetime import datetime
from dotenv import load_dotenv
from src.data.pipeline import DataPipeline
from src.db import get_db

load_dotenv()

def cargar_datos_por_ano(ano: int, org_id: str = "SVMPREMIUM", user_id: str = None):
    """
    Carga datos de un año específico al sistema sin pasar por S3.
    
    Uso:
        cargar_datos_por_ano(2020)  # Carga D_2020.csv
        cargar_datos_por_ano(2026)  # Cuando llegue 2027, solo ejecuta esto
    """
    
    if user_id is None:
        user_id = os.getenv('ADMIN_USER_ID', 'brisa@gmail.com')
    
    # Archivo del año
    archivo = f'D_{ano}.csv'
    
    if not os.path.exists(archivo):
        print(f"✗ Archivo no encontrado: {archivo}")
        return False
    
    print("="*70)
    print(f"CARGANDO: {archivo} para {org_id}")
    print("="*70)
    
    # Cargar CSV
    print(f"\n1. Leyendo {archivo}...")
    df = pd.read_csv(archivo, sep=';', decimal=',')
    print(f"   ✓ {len(df):,} filas cargadas")
    
    # Procesar con pipeline
    print(f"\n2. Ejecutando pipeline...")
    
    from src.data.data_cleaner import DataCleaner
    from src.data.guide_reconciliation import GuideReconciler
    from src.data.demand_builder import DemandBuilder
    from src.data.stock_builder import StockBuilder
    from src.data.series_completion import complete_monthly_demand
    
    # Añadir columna de fuente
    df['__source_file'] = archivo
    
    cleaner = DataCleaner()
    reconciler = GuideReconciler()
    demand_builder = DemandBuilder()
    stock_builder = StockBuilder()
    
    clean = cleaner.clean(df)
    print(f"   ✓ Limpieza: {len(clean):,} filas")
    
    rec = reconciler.reconcile(clean)
    print(f"   ✓ Reconciliación: {len(rec):,} filas")
    
    demand = demand_builder.build_monthly(rec)
    print(f"   ✓ Demanda: {len(demand):,} registros")
    
    stock = stock_builder.build_monthly(rec)
    print(f"   ✓ Stock: {len(stock):,} registros")
    
    # Guardar en BD
    print(f"\n3. Guardando en Supabase...")
    try:
        db = get_db()
        if db:
            # Guardar metadata de carga
            upload_data = {
                'organization_id': org_id,
                'uploaded_by': user_id,
                'uploaded_at': datetime.now().isoformat(),
                'year': ano,
                'total_raw_records': len(df),
                'total_clean_records': len(clean),
                'total_demand_records': len(demand),
                'total_stock_records': len(stock),
                'filename': archivo,
                'status': 'processed'
            }
            
            result = db.client.table('uploads').insert(upload_data).execute()
            print(f"   ✓ Metadata guardada")
            
            # Aquí puedes guardar los datos procesados en tablas específicas si es necesario
            # Por ahora solo guardamos la metadata de que fue procesado
            
    except Exception as e:
        print(f"   ⚠️ Aviso en Supabase: {e}")
        print(f"   (Datos están listos para análisis, pero metadata podría no estar guardada)")
    
    print("\n" + "="*70)
    print(f"✅ {archivo} CARGADO EXITOSAMENTE")
    print("="*70)
    return True


def cargar_multiples_anos(anos: list, org_id: str = "SVMPREMIUM"):
    """
    Carga múltiples años en secuencia.
    
    Uso:
        cargar_multiples_anos([2020, 2021, 2022, 2023, 2024, 2025])
        # O para agregar nuevos:
        cargar_multiples_anos([2026, 2027])
    """
    print("\n" + "#"*70)
    print("# CARGA INCREMENTAL DE DATOS")
    print("#"*70)
    
    exitosos = 0
    fallos = 0
    
    for ano in anos:
        try:
            if cargar_datos_por_ano(ano, org_id):
                exitosos += 1
            else:
                fallos += 1
        except Exception as e:
            print(f"✗ Error procesando {ano}: {e}")
            fallos += 1
        
        print()
    
    print("="*70)
    print(f"RESUMEN: {exitosos} cargados, {fallos} fallos")
    print("="*70)


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        # Uso: python cargar_incremental.py 2020 2021 2022
        anos = [int(x) for x in sys.argv[1:]]
        cargar_multiples_anos(anos)
    else:
        # Por defecto, cargar los 6 años
        print("Uso:")
        print("  python cargar_incremental.py 2020 2021 2022 2023 2024 2025")
        print("  python cargar_incremental.py 2026     # En 2027, solo agregar nuevos")
        print("  python cargar_incremental.py 2020     # Cargar un solo año")
        print()
        
        respuesta = input("¿Cargar datos de 2020-2025? (s/n): ").strip().lower()
        if respuesta == 's':
            cargar_multiples_anos([2020, 2021, 2022, 2023, 2024, 2025])

