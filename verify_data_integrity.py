#!/usr/bin/env python3
"""Verificar integridad: 4 archivos anuales vs archivo completo"""

import pandas as pd
import os

print("="*70)
print("VERIFICACION: ¿Los 4 anuales = Archivo completo?")
print("="*70)

# Leer archivo completo
print("\n1. Leyendo archivo COMPLETO...")
df_completo = pd.read_csv('Inventario_ML_Completo_v4.csv')
print(f"   Filas: {len(df_completo):,}")
print(f"   Columnas: {len(df_completo.columns)}")
completo_size = os.path.getsize('Inventario_ML_Completo_v4.csv') / 1024 / 1024
print(f"   Tamaño en disco: {completo_size:.1f}MB")

# Leer archivos anuales
print("\n2. Leyendo archivos ANUALES...")
filas_total_anuales = 0
for year in [2022, 2023, 2024, 2025]:
    filename = f'Inventario_v4_{year}.csv'
    df = pd.read_csv(filename)
    tamaño = os.path.getsize(filename) / 1024 / 1024
    filas_total_anuales += len(df)
    print(f"   {filename}: {len(df):,} filas ({tamaño:.1f}MB)")

total_anual_mb = sum(os.path.getsize(f'Inventario_v4_{year}.csv') / 1024 / 1024 
                     for year in [2022, 2023, 2024, 2025])

print(f"\n   TOTAL anuales: {filas_total_anuales:,} filas ({total_anual_mb:.1f}MB)")

# Comparación
print("\n" + "="*70)
print("COMPARACION:")
print("="*70)
print(f"Archivo completo:  {len(df_completo):,} filas × {completo_size:.1f}MB")
print(f"4 archivos anuales: {filas_total_anuales:,} filas × {total_anual_mb:.1f}MB")
print(f"\nDiferencia de FILAS: {len(df_completo) - filas_total_anuales:,} (¿perdidas?)")
print(f"Diferencia de TAMAÑO: {completo_size - total_anual_mb:.1f}MB ({100*(completo_size - total_anual_mb)/completo_size:.1f}%)")
print(f"Factor de compresión: {completo_size / total_anual_mb:.2f}x")

# Verificar columnas
print("\n3. Comparando COLUMNAS...")
df_2022 = pd.read_csv('Inventario_v4_2022.csv', nrows=1)
cols_completo = set(df_completo.columns)
cols_anual = set(df_2022.columns)

if cols_completo == cols_anual:
    print(f"   ✓ Columnas IDENTICAS: {len(cols_completo)} columnas")
else:
    print(f"   ✗ DIFERENTES COLUMNAS!")
    print(f"   Completo tiene: {cols_completo - cols_anual}")
    print(f"   Anuales tienen: {cols_anual - cols_completo}")

# Verificar tipos de dato
print("\n4. Comparando TIPOS DE DATO...")
print("   Completo:")
for col in list(df_completo.columns[:5]):
    print(f"     {col}: {df_completo[col].dtype}")
print("\n   Anual (2022):")
for col in list(df_2022.columns[:5]):
    print(f"     {col}: {df_2022[col].dtype}")

# Conclusión
print("\n" + "="*70)
if len(df_completo) == filas_total_anuales:
    print("✓ FILAS: IDENTICAS")
    print(f"   {len(df_completo):,} = {filas_total_anuales:,}")
else:
    print("✗ FILAS: DIFERENTES (pérdida de datos)")
    print(f"   Completo: {len(df_completo):,}")
    print(f"   Anuales: {filas_total_anuales:,}")

print("\nTAMAÑO:")
print(f"   El archivo original es {completo_size/total_anual_mb:.2f}x más grande")
print(f"   Diferencia: {completo_size - total_anual_mb:.1f}MB")
print(f"   Causa probable: Formato CSV original menos comprimido")
print("="*70)
