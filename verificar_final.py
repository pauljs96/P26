"""Verificación final del archivo combinado antes de subir."""

import pandas as pd
import os

print("="*70)
print("VERIFICACION FINAL: Datos_Completo_2020_2025.csv")
print("="*70)

archivo = 'Datos_Completo_2020_2025.csv'

if not os.path.exists(archivo):
    print(f"✗ Archivo no encontrado: {archivo}")
    exit(1)

# 1. Verificación básica
print(f"\n1. INFO DEL ARCHIVO:")
tamaño = os.path.getsize(archivo)
print(f"   Tamaño: {tamaño / 1024 / 1024:.2f} MB")
print(f"   Ruta: {os.path.abspath(archivo)}")

# 2. Leer con pandas
print(f"\n2. LECTURA CON PANDAS:")
df = pd.read_csv(archivo, sep=';', decimal=',')
print(f"   ✓ Filas: {len(df):,}")
print(f"   ✓ Columnas: {len(df.columns)}")

# 3. Columnas
print(f"\n3. COLUMNAS ({len(df.columns)}):")
for i, col in enumerate(df.columns, 1):
    print(f"   {i:2d}. '{col}'")

# 4. Valores de ejemplo
print(f"\n4. VALORES DE EJEMPLO (primeras 3 filas):")
print(f"   Codigo | Documento | Bodega | Entrada | Salida | Saldo")
for idx in range(min(3, len(df))):
    row = df.iloc[idx]
    print(f"   {row['Codigo']:>6} | {row['Documento'][:20]:20} | {row['Bodega']:>15} | {row['Entrada_unid']:>7} | {row['Salida_unid']:>6} | {row['Saldo_unid']:>6}")

# 5. Verificación de datos
print(f"\n5. INTEGRIDAD DE DATOS:")
print(f"   Filas sin nulls en columnas clave:")
print(f"   - Codigo: {df['Codigo'].notna().sum():,} (falta {df['Codigo'].isna().sum()})")
print(f"   - Fecha: {df['Fecha'].notna().sum():,} (falta {df['Fecha'].isna().sum()})")
print(f"   - Documento: {df['Documento'].notna().sum():,} (falta {df['Documento'].isna().sum()})")
print(f"   - Bodega: {df['Bodega'].notna().sum():,} (falta {df['Bodega'].isna().sum()})")

# 6. Distribución por año
print(f"\n6. CANTIDAD DE REGISTROS POR AÑO:")
if 'Fecha' in df.columns:
    df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce')
    df['Año'] = df['Fecha'].dt.year
    for año in sorted(df['Año'].dropna().unique()):
        count = len(df[df['Año'] == año])
        pct = 100 * count / len(df)
        print(f"   {int(año)}: {count:,} registros ({pct:.1f}%)")

# 7.  Tipos de dato
print(f"\n7. TIPOS DE DATO:")
numeric_cols = df.select_dtypes(include=['number']).columns
string_cols = df.select_dtypes(include=['object']).columns
print(f"   Numéricos: {list(numeric_cols)}")
print(f"   Text: {list(string_cols)}")

print(f"\n" + "="*70)
print(f"✅ ARCHIVO OK - Listo para cargar al dashboard")
print(f"="*70)
print(f"\nPARA SUBIR AL DASHBOARD:")
print(f"1. Ve a Streamlit Cloud")
print(f"2. Abre PREDICAST Dashboard")
print(f"3. Como Admin → Subir Datos")
print(f"4. Carga el archivo de esta carpeta: {archivo}")
print(f"5. Espera a que procese (~30 segundos)")

