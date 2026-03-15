"""
Dividir el archivo grande en 3 partes más pequeñas.

Esto evita problemas con timeouts o memory limits en Streamlit Cloud.

En lugar de subir Datos_Completo_2020_2025.csv (36 MB),
subiras 3 archivos de ~12 MB cada uno.
"""

import pandas as pd
import os

print("="*70)
print("DIVIDIENDO ARCHIVO GRANDE EN 3 PARTES")
print("="*70)

archivo_entrada = 'Datos_Completo_2020_2025.csv'

if not os.path.exists(archivo_entrada):
    print(f"✗ {archivo_entrada} no encontrado")
    exit(1)

# Leer archivo
print(f"\n1. Leyendo {archivo_entrada}...")
df = pd.read_csv(archivo_entrada, sep=';', decimal=',')
print(f"   Total: {len(df):,} filas")

# Dividir en 3 partes
total = len(df)
parte1_end = total // 3
parte2_end = 2 * total // 3

print(f"\n2. Dividiendo en 3 partes:")
print(f"   Parte 1: filas 0 - {parte1_end:,}")
print(f"   Parte 2: filas {parte1_end + 1:,} - {parte2_end:,}")
print(f"   Parte 3: filas {parte2_end + 1:,} - {total:,}")

df1 = df.iloc[:parte1_end]
df2 = df.iloc[parte1_end:parte2_end]
df3 = df.iloc[parte2_end:]

# Guardar partes
print(f"\n3. Guardando archivos...")

archivo1 = 'Datos_Parte1_2020_2021.csv'
df1.to_csv(archivo1, sep=';', decimal=',', index=False, encoding='utf-8')
print(f"   ✓ {archivo1}: {len(df1):,} filas ({os.path.getsize(archivo1) / 1024 / 1024:.1f} MB)")

archivo2 = 'Datos_Parte2_2022_2023.csv'
df2.to_csv(archivo2, sep=';', decimal=',', index=False, encoding='utf-8')
print(f"   ✓ {archivo2}: {len(df2):,} filas ({os.path.getsize(archivo2) / 1024 / 1024:.1f} MB)")

archivo3 = 'Datos_Parte3_2024_2025.csv'
df3.to_csv(archivo3, sep=';', decimal=',', index=False, encoding='utf-8')
print(f"   ✓ {archivo3}: {len(df3):,} filas ({os.path.getsize(archivo3) / 1024 / 1024:.1f} MB)")

print(f"\n" + "="*70)
print(f"✅ ARCHIVOS CREADOS - AHORA PUEDES SUBIR POR PARTES")
print(f"="*70)

print(f"\nPARA CARGAR AL DASHBOARD:")
print(f"1. Sube Datos_Parte1_2020_2021.csv")
print(f"2. Espera a que termine")
print(f"3. Sube Datos_Parte2_2022_2023.csv")
print(f"4. Espera a que termine")
print(f"5. Sube Datos_Parte3_2024_2025.csv")
print(f"6. Listo - tendrás todos los datos 2020-2025")

print(f"\nVentajas:")
print(f"- Cada parte es ~12 MB (más rápido)")
print(f"- Menos probabilidad de timeout")
print(f"- Menos problemas de memory")
print(f"- Si falla una parte, reintentas solo esa")

