import sys
sys.path.insert(0, r'd:\Desktop\TESIS\Sistema_Tesis')

import pandas as pd
from src.data.data_cleaner import DataCleaner
from src.data.guide_reconciliation import GuideReconciler
from src.data.demand_builder import DemandBuilder
from src.data.stock_builder import StockBuilder

print('Pipeline - Validacion con datos balanceados\n')

dfs = []
for ano in [2020, 2021, 2022, 2023, 2024, 2025]:
    archivo = 'Datos_Balanceado_{0}.csv'.format(ano)
    df = pd.read_csv(archivo, sep=';', decimal=',')
    dfs.append(df)

df_completo = pd.concat(dfs, ignore_index=True)
print('Total: {0:,} transacciones\n'.format(len(df_completo)))

print('='*70)
print('PIPELINE - STAGE 1: DATA CLEANER')
print('='*70)
cleaner = DataCleaner()
df_clean = cleaner.clean(df_completo)
print('  Entrada: {0:,}'.format(len(df_completo)))
print('  Salida: {0:,}'.format(len(df_clean)))
print('  Productos: {0:,}'.format(df_clean['Codigo'].nunique()))

print('\nSTAGE 2: GUIDE RECONCILIATION')
reconciler = GuideReconciler()
df_reconciled = reconciler.reconcile(df_clean)
print('  Reconciliadas: {0:,}'.format(len(df_reconciled)))

print('\nSTAGE 3: DEMAND BUILDER')
demand_builder = DemandBuilder()
df_demand = demand_builder.build_monthly(df_reconciled)
print('  Demanda: {0:,} registros'.format(len(df_demand)))
print('  Productos: {0:,}'.format(df_demand['Codigo'].nunique()))
print('  Meses: {0}'.format(df_demand['Mes'].nunique()))

print('\nSTAGE 4: STOCK BUILDER')
stock_builder = StockBuilder()
df_stock = stock_builder.build_monthly(df_reconciled)
print('  Stock: {0:,} registros'.format(len(df_stock)))
print('  Productos: {0:,}'.format(df_stock['Codigo'].nunique()))
print('  Almacenes: {0}'.format(df_stock['Bodega'].nunique()))

print('\n' + '='*70)
print('FINAL - DATOS LISTA PARA ML')
print('='*70)
print('  Periodo: 2020-2025 (6 anos)')
print('  CSV: {0:,}'.format(len(df_completo)))
print('  Demanda para forecasting: {0:,}'.format(len(df_demand)))
print('  Stock para forecasting: {0:,}'.format(len(df_stock)))
print('  Productos activos: {0:,}'.format(df_demand['Codigo'].nunique()))
print('\nOK - PIPELINE VALIDADO')
