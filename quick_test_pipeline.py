#!/usr/bin/env python3
"""Quick test for v4 pipeline"""

import pandas as pd
from src.data.pipeline import DataPipeline
from src.utils.logger import Logger

# Cargar CSV v4
csv_path = 'Inventario_ML_Completo_v4.csv'
print(f"Cargando {csv_path}...")
df = pd.read_csv(csv_path, sep=',', encoding='utf-8', dtype=str)
print(f'OK: {len(df)} filas cargadas')

# Mock file para simular upload
class MockFile:
    def __init__(self, df_raw):
        self.df = df_raw
        self.name = 'test.csv'
    
    def getvalue(self):
        return self.df.to_csv(index=False).encode('utf-8')

mock = MockFile(df)

# Ejecutar pipeline
print("\nEjecutando pipeline...")
pipeline = DataPipeline(logger_obj=Logger(enabled=False))
result = pipeline.run([mock])

# Resultados
print('\n' + '='*60)
print('RESULTADOS DEL PIPELINE v4')
print('='*60)
print(f'Movimientos: {len(result.movements)} filas')
print(f'Demanda: {len(result.demand_monthly)} registros')
print(f'Stock: {len(result.stock_monthly)} registros')

if result.movements.shape[0] > 0:
    print(f'Productos: {result.movements["Producto_id"].nunique()}')
    
    # Estadísticas de movimientos
    venta_count = (result.movements['Tipo_movimiento'] == 'Venta').sum()
    prod_count = (result.movements['Tipo_movimiento'] == 'Producción').sum()
    print(f'Ventas: {venta_count} | Producciones: {prod_count}')

if result.demand_monthly.shape[0] > 0:
    print(f'\nDemanda total: {result.demand_monthly["Cantidad_total"].sum():.0f} unidades')
    print(f'Promedio/mes: {result.demand_monthly["Cantidad_total"].mean():.2f} unidades')
    
if result.stock_monthly.shape[0] > 0:
    print(f'\nStock prom: {result.stock_monthly["Stock_final"].mean():.2f}')
    print(f'Stock max: {result.stock_monthly["Stock_final"].max():.0f}')
    print(f'Stock min: {result.stock_monthly["Stock_final"].min():.0f}')

print('\nOK: Pipeline completado exitosamente!')

