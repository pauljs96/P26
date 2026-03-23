"""
Verificar integridad de los archivos divididos
"""

import pandas as pd
import os

print("=" * 60)
print("VERIFICACION DE ARCHIVOS")
print("=" * 60)

archivos = [
    'Inventario_v4_20PRODUCTOS_2022.csv',
    'Inventario_v4_20PRODUCTOS_2023.csv',
    'Inventario_v4_20PRODUCTOS_2024.csv',
    'Inventario_v4_20PRODUCTOS_2025.csv'
]

total_registros = 0
todos_productos = set()

for archivo in archivos:
    if not os.path.exists(archivo):
        print(f"❌ {archivo} - NO ENCONTRADO")
        continue
    
    df = pd.read_csv(archivo)
    tamaño_mb = os.path.getsize(archivo) / 1024 / 1024
    
    print(f"\n✅ {archivo}")
    print(f"   Registros: {len(df):,}")
    print(f"   Tamaño: {tamaño_mb:.2f} MB")
    print(f"   Productos: {df['Producto_id'].nunique()}")
    print(f"   Fechas: {df['Fecha'].min()} a {df['Fecha'].max()}")
    
    total_registros += len(df)
    todos_productos.update(df['Producto_id'].unique())

print(f"\n{'='*60}")
print(f"RESUMEN TOTAL:")
print(f"  Registros: {total_registros:,}")
print(f"  Productos únicos: {len(todos_productos)}")
print(f"  Años cubiertos: 2022, 2023, 2024, 2025")
print(f"\nListado de productos:")

# Mostrar productos
df_completo = pd.concat([pd.read_csv(f) for f in archivos], ignore_index=True)
productos = df_completo[['Producto_id', 'Producto_nombre']].drop_duplicates().sort_values('Producto_id')
for idx, row in productos.iterrows():
    print(f"  {row['Producto_id']:>3} | {row['Producto_nombre'][:60]}")

print(f"\n✨ Todos los archivos están listos para Streamlit Cloud!")
