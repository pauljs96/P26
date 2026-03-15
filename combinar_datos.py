"""Combinar todos los archivos D_2020 a D_2025 en uno solo para facilitar la carga."""

import pandas as pd
import os

print("="*70)
print("COMBINANDO ARCHIVOS D_2020 - D_2025 EN UNO SOLO")  
print("="*70)

archivos = ['D_2020.csv', 'D_2021.csv', 'D_2022.csv', 'D_2023.csv', 'D_2024.csv', 'D_2025.csv']
dfs = []

total_filas = 0
for archivo in archivos:
    if os.path.exists(archivo):
        df = pd.read_csv(archivo, sep=';', decimal=',')
        dfs.append(df)
        print(f"✓ {archivo}: {len(df)} filas")
        total_filas += len(df)
    else:
        print(f"⚠️ {archivo} no encontrado - saltando")

if not dfs:
    print("No hay archivos para combinar")
    exit(1)

# Combinar
df_combined = pd.concat(dfs, ignore_index=True)

# Guardar
output_file = 'Datos_Completo_2020_2025.csv'
df_combined.to_csv(output_file, sep=';', decimal=',', index=False, encoding='utf-8')

print(f"\n{'='*70}")
print(f"✅ ARCHIVO COMBINADO CREADO")
print(f"{'='*70}")
print(f"Nombre: {output_file}")
print(f"Total filas: {len(df_combined):,}")
print(f"Columnas: {len(df_combined.columns)}")
print(f"\n✓ Ahora SUBE SOLO ESTE ARCHIVO al dashboard")
print(f"✓ Versión original (6 archivos) separados ya NO es necesaria")

