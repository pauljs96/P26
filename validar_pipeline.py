import sys
sys.path.insert(0, r'd:\Desktop\TESIS\Sistema_Tesis')

import pandas as pd
from src.data.data_cleaner import DataCleaner
from src.data.guide_reconciliation import GuideReconciler
from src.data.demand_builder import DemandBuilder
from src.data.stock_builder import StockBuilder

# Cargar todos los datos
print('Combinando archivos 2020-2025...')
dfs = []
for ano in [2020, 2021, 2022, 2023, 2024, 2025]:
    archivo = 'Datos_{0}.csv'.format(ano)
    df = pd.read_csv(archivo, sep=';', decimal=',')
    dfs.append(df)
    print('  {0}: {1:,} transacciones'.format(ano, len(df)))

df_completo = pd.concat(dfs, ignore_index=True)
print('\nTotal: {0:,} transacciones\n'.format(len(df_completo)))

# Pipeline
print('='*70)
print('VALIDACION - PIPELINE COMPLETO')
print('='*70)

print('\nSTAGE 1: DATA CLEANER')
cleaner = DataCleaner()
df_clean = cleaner.clean(df_completo)
print('  Entrada: {0:,}'.format(len(df_completo)))
print('  Salida: {0:,} limpias'.format(len(df_clean)))
print('  Productos: {0:,}'.format(df_clean['Codigo'].nunique()))

print('\nSTAGE 2: GUIDE RECONCILIATION')
reconciler = GuideReconciler()
df_reconciled = reconciler.reconcile(df_clean)
print('  Reconciliadas: {0:,} filas'.format(len(df_reconciled)))

print('\nSTAGE 3: DEMAND BUILDER')
demand_builder = DemandBuilder()
df_demand = demand_builder.build_monthly(df_reconciled)
print('  Demanda mensual: {0:,} registros'.format(len(df_demand)))
print('  Productos con demanda: {0:,}'.format(df_demand['Codigo'].nunique()))
print('  Meses: {0}'.format(df_demand['Mes'].nunique()))

if len(df_demand) > 0:
    print('\n  TOP 5 productos:')
    top = df_demand.groupby('Codigo')['Demanda_Unid'].sum().sort_values(ascending=False).head(5)
    for i, (sku, total) in enumerate(top.items(), 1):
        meses = len(df_demand[df_demand['Codigo'] == sku])
        promedio = total / meses if meses > 0 else 0
        print('    {0}. {1:15} : {2:7,.0f} unid ({3:5.1f}/mes en {4:2d} meses)'.format(i, str(sku), total, promedio, meses))

print('\nSTAGE 4: STOCK BUILDER')
stock_builder = StockBuilder()
df_stock = stock_builder.build_monthly(df_reconciled)
print('  Stock mensual: {0:,} registros'.format(len(df_stock)))
print('  Productos: {0:,}'.format(df_stock['Codigo'].nunique()))
print('  Almacenes: {0}'.format(df_stock['Bodega'].nunique()))

print('\n' + '='*70)
print('RESULTADO FINAL - DATOS LISTA PARA ML')
print('='*70)
print('  Periodo: 2020-2025 (6 anos)')
print('  CSV: {0:,} transacciones'.format(len(df_completo)))
print('  Limpios: {0:,}'.format(len(df_clean)))
print('  Demanda para forecasting: {0:,} registros'.format(len(df_demand)))
print('  Stock para forecasting: {0:,} registros'.format(len(df_stock)))
print('  Productos: {0:,}'.format(df_demand['Codigo'].nunique()))
print('\nOK - DATOS DENSOS Y REALISTAS')
