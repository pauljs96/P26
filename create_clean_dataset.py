"""
Script para crear dataset limpio de 20 productos importantes
Selecciona: Cajas de pase, Tableros eléctricos, sin apóstrofes en el código
"""

import pandas as pd
import sys

print("Leyendo archivo completo (puede tomar un tiempo)...")
df = pd.read_csv('Inventario_ML_Completo_v4.csv', encoding='utf-8')

print(f"Total de registros: {len(df):,}")
print(f"Columnas: {list(df.columns)}")

# 1. Filtrar productos sin apóstrofes o caracteres especiales en Producto_id
df_clean = df.copy()

# Excluir productos que comienzan con "(" (automáticos) o tienen apóstrofes
df_clean = df_clean[~df_clean['Producto_id'].astype(str).str.contains(r"['\(\)]", regex=True, na=False)]

print(f"\nDespués de filtrar caracteres especiales: {len(df_clean):,} registros")
print(f"Productos únicos únicos: {df_clean['Producto_id'].nunique()}")

# 2. Obtener lista de productos únicos con sus nombres
productos_unicos = df_clean[['Producto_id', 'Producto_nombre']].drop_duplicates().reset_index(drop=True)
print(f"\nProductos únicos disponibles: {len(productos_unicos)}")

# 3. Identificar productos relevantes (cajas, tableros)
relevantes = productos_unicos[
    (productos_unicos['Producto_nombre'].str.contains('CAJA|TABLERO', case=False, na=False))
].copy()

print(f"\nProductos relevantes (CAJA o TABLERO): {len(relevantes)}")

if len(relevantes) < 20:
    print(f"Encontrados {len(relevantes)} productos relevantes, complementando con otros...")
    otros = productos_unicos[
        ~productos_unicos['Producto_id'].isin(relevantes['Producto_id'])
    ].head(20 - len(relevantes))
    seleccionados = pd.concat([relevantes, otros], ignore_index=True).head(20)
else:
    seleccionados = relevantes.head(20)

print(f"\nProductos seleccionados ({len(seleccionados)}):")
for idx, row in seleccionados.iterrows():
    print(f"  {idx+1}. [{row['Producto_id']:>5}] {row['Producto_nombre']}")

# 4. Crear nuevo dataset con solo estos productos
df_filtered = df_clean[df_clean['Producto_id'].isin(seleccionados['Producto_id'])].copy()

print(f"\nRegistros en dataset filtrado: {len(df_filtered):,}")
print(f"Rango de fechas: {df_filtered['Fecha'].min()} a {df_filtered['Fecha'].max()}")

# 5. Guardar en CSV nuevo
output_file = 'Inventario_v4_20PRODUCTOS.csv'
df_filtered.to_csv(output_file, index=False, encoding='utf-8')
print(f"\n✅ Archivo guardado: {output_file}")

# 6. Mostrar resumen
print(f"\nResumen del archivo:")
print(f"  Registros: {len(df_filtered):,}")
print(f"  Productos únicos: {df_filtered['Producto_id'].nunique()}")
print(f"  Años: {sorted(df_filtered['Año'].unique())}")
print(f"  Tamaño: {df_filtered.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
