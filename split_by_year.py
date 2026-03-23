"""
Script para dividir el dataset en archivos anuales
Crea: Inventario_v4_20PRODUCTOS_2022.csv, 2023, 2024, 2025
"""

import pandas as pd

# Leer archivo
df = pd.read_csv('Inventario_v4_20PRODUCTOS.csv', encoding='utf-8')

print(f"Archivo original: {len(df):,} registros\n")

# Dividir por año
años = sorted(df['Año'].unique())
print(f"Años disponibles: {años}\n")

for año in años:
    df_año = df[df['Año'] == int(año)].copy()
    filename = f'Inventario_v4_20PRODUCTOS_{año}.csv'
    df_año.to_csv(filename, index=False, encoding='utf-8')
    
    tamaño_mb = df_año.memory_usage(deep=True).sum() / 1024 / 1024
    print(f"✅ {filename}")
    print(f"   Registros: {len(df_año):,}")
    print(f"   Tamaño: {tamaño_mb:.2f} MB\n")

print("=" * 60)
print("📋 RESUMEN FINAL:")
print("=" * 60)
print("\nArchivos listos para cargar en Streamlit Cloud:")
for año in años:
    print(f"  • Inventario_v4_20PRODUCTOS_{año}.csv")
    
print("\nTodos los archivos son < 200MB (límite de Streamlit Cloud)")
print("\n✨ PRÓXIMOS PASOS:")
print("  1. Descarga estos 4 archivos")
print("  2. En Streamlit Cloud: Settings → Secrets")
print("  3. Configura tus credenciales Supabase")
print("  4. Ingresa con zu@gmail.com")
print("  5. Carga los 4 archivos CSV")
