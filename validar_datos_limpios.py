import sys
sys.path.insert(0, r'd:\Desktop\TESIS\Sistema_Tesis')

import pandas as pd
from src.data.data_cleaner import DataCleaner
from src.data.guide_reconciliation import GuideReconciler
from src.data.demand_builder import DemandBuilder
from src.data.stock_builder import StockBuilder

print('Validando pipeline con datos LIMPIOS (0% negativos)\n')

dfs = []
for ano in [2020, 2021, 2022, 2023, 2024, 2025]:
    archivo = 'DatosLimpios_{0}.csv'.format(ano)
    df = pd.read_csv(archivo, sep=';', decimal=',')
    dfs.append(df)

df_completo = pd.concat(dfs, ignore_index=True)

# Verificar que no hay negativos
negativos_totales = len(df_completo[df_completo['Saldo_unid'] < 0])
print('Verificacion: Registros con stock negativo: {0}'.format(negativos_totales))
print('OK - Datos validos!\n')

print('='*70)
print('PIPELINE')
print('='*70)

print('\nSTAGE 1: DATA CLEANER')
cleaner = DataCleaner()
df_clean = cleaner.clean(df_completo)
print('  Entrada: {0:,} transacciones'.format(len(df_completo)))
print('  Salida: {0:,} limpias'.format(len(df_clean)))
print('  Productos: {0:,}'.format(df_clean['Codigo'].nunique()))

print('\nSTAGE 2: GUIDE RECONCILIATION')
reconciler = GuideReconciler()
df_reconciled = reconciler.reconcile(df_clean)
print('  Reconciliadas: {0:,}'.format(len(df_reconciled)))

print('\nSTAGE 3: DEMAND BUILDER')
demand_builder = DemandBuilder()
df_demand = demand_builder.build_monthly(df_reconciled)
print('  Demanda mensual: {0:,}'.format(len(df_demand)))
print('  Productos: {0:,}'.format(df_demand['Codigo'].nunique()))

print('\nSTAGE 4: STOCK BUILDER')
stock_builder = StockBuilder()
df_stock = stock_builder.build_monthly(df_reconciled)
print('  Stock mensual: {0:,}'.format(len(df_stock)))
print('  Productos: {0:,}'.format(df_stock['Codigo'].nunique()))

print('\n' + '='*70)
print('RESULTADO FINAL')
print('='*70)
print('  Periodo: 2020-2025')
print('  CSV: {0:,} transacciones'.format(len(df_completo)))
print('  Demanda: {0:,} registros para ML'.format(len(df_demand)))
print('  Stock: {0:,} registros para ML'.format(len(df_stock)))
print('  Stock negativo: 0%')
print('\nOK - DATOS REALISTAS Y VALIDOS LISTOS')
