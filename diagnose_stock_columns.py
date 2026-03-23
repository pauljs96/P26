"""
Script para diagnosticar qué columnas tiene res_stock después del pipeline

Esto ayuda a identificar por qué falla la detección de Stock
"""

import pandas as pd
import sys
from pathlib import Path

print("=" * 80)
print("DIAGNOSTICO: COLUMNAS DE RES_STOCK")
print("=" * 80)

# Simular lo que pasa en el pipeline
csv_file = Path('Inventario_v4_20PRODUCTOS.csv')

if not csv_file.exists():
    print(f"❌ Archivo no encontrado: {csv_file}")
    sys.exit(1)

# Cargar archivo como lo hace el pipeline
print(f"\n1. Cargando {csv_file}...")
df = pd.read_csv(csv_file, encoding='utf-8')
print(f"   Registros: {len(df):,}")
print(f"   Columnas: {len(df.columns)}")

# Ver columnas exactas
print(f"\n2. Columnas exactas en el archivo:")
for i, col in enumerate(df.columns, 1):
    dtype = str(df[col].dtype)
    non_null = df[col].notna().sum()
    print(f"   {i:2}. '{col:30}' | {dtype:15} | {non_null:,} valores")

# Simular lo que hace detect_dataset_version
print(f"\n3. Detectando versión del dataset...")
v4_required = {'Fecha', 'Producto_id', 'Tipo_movimiento', 'Cantidad', 'Stock_anterior', 'Stock_posterior'}
legacy_required = {'Codigo', 'Fecha', 'Documento', 'Entrada_unid', 'Salida_unid', 'Saldo_unid'}

v4_cols = v4_required & set(df.columns)
legacy_cols = legacy_required & set(df.columns)

print(f"   v4 columns encontradas: {len(v4_cols)}/{len(v4_required)}")
print(f"   legacy columns encontradas: {len(legacy_cols)}/{len(legacy_required)}")

is_v4 = len(v4_cols) >= 5  # Al menos 5 de 6 columns v4

if is_v4:
    print(f"   → Dataset V4 detectado")
else:
    print(f"   → Dataset Legacy detectado")

# Simular normalize_stock_to_legacy
print(f"\n4. Normalizando stock...")

# Simular la extracción del stock_monthly del pipeline
# El pipeline crea stock_monthly agrupando por Producto_id, Año, Mes
if is_v4:
    print(f"   Creando stock_monthly desde v4...")
    stock_monthly = df[['Producto_id', 'Año', 'Mes', 'Stock_posterior']].copy()
    stock_monthly = stock_monthly.groupby(['Producto_id', 'Año', 'Mes']).agg({
        'Stock_posterior': 'last'
    }).reset_index()
    stock_monthly.columns = ['Producto_id', 'Año', 'Mes', 'Stock_posterior']
else:
    print(f"   Usando stock_monthly legacy...")
    stock_monthly = df[['Codigo', 'Saldo_unid']].copy()

print(f"   Registros en stock_monthly: {len(stock_monthly):,}")

# Aplicar normalize_stock_to_legacy
print(f"\n5. Aplicar normalize_stock_to_legacy...")
d = stock_monthly.copy()

# Ver columnas ANTES de normalización
print(f"   ANTES - Columnas: {list(d.columns)}")

# Detectar si ya fue normalizado
if 'Mes' in d.columns and pd.api.types.is_datetime64_any_dtype(d['Mes']):
    print(f"   → Ya está normalizado (Mes es datetime)")
else:
    print(f"   → NO está normalizado, aplicando transformación...")
    
    # Construir rename map
    rename_map = {}
    if 'Producto_id' in d.columns and 'Codigo' not in d.columns:
        rename_map['Producto_id'] = 'Codigo'
        print(f"     - Renombrando Producto_id → Codigo")
    
    if 'Saldo_unid' not in d.columns:
        for stock_col in ['Stock_posterior', 'Stock_Unid']:
            if stock_col in d.columns:
                rename_map[stock_col] = 'Saldo_unid'
                print(f"     - Renombrando {stock_col} → Saldo_unid")
                break
    
    if rename_map:
        d = d.rename(columns=rename_map)
        print(f"     - Rename map aplicado: {rename_map}")
    
    # Convertir Año+Mes a datetime
    if 'Año' in d.columns and 'Mes' in d.columns:
        print(f"     - Convirtiendo Año+Mes a datetime...")
        d['Mes'] = pd.to_datetime(
            d['Año'].astype(str) + '-' + d['Mes'].astype(str).str.zfill(2) + '-01',
            errors='coerce'
        )

print(f"   DESPUES - Columnas: {list(d.columns)}")

# Ver si tiene Saldo_unid ahora
if 'Saldo_unid' in d.columns:
    print(f"   ✅ Saldo_unid presente")
    print(f"      Rango: {d['Saldo_unid'].min():.0f} a {d['Saldo_unid'].max():.0f}")
else:
    print(f"   ❌ Saldo_unid NO presente")
    print(f"      Columnas disponibles: {list(d.columns)}")

print(f"\n" + "=" * 80)
print("CONCLUSIÓN")
print("=" * 80)

if 'Saldo_unid' not in d.columns:
    print(f"""
❌ PROBLEMA ENCONTRADO:

La función normalize_stock_to_legacy NO logró crear la columna 'Saldo_unid'.

Esto probablemente es porque:
1. El archivo no tiene 'Stock_posterior' o 'Stock_Unid'
2. O el Rename no se aplicó correctamente

Columnas que SÍ tiene: {list(d.columns)}

SOLUCIÓN:
Verifica que el CSV que se cargó tiene una columna para stock con alguno de
estos nombres:
  • Stock_posterior (v4)
  • Stock_Unid
  • Saldo_unid (legacy)
""")
else:
    print(f"""
✅ CORRECTO:

La función normalize_stock_to_legacy creó correctamente la columna 'Saldo_unid'.

El Dashboard debería poder acceder a ella sin errores KeyError.
""")
