"""
ANALISIS DETALLADO DE LA DATA CARGADA

Este script analiza el archivo Inventario_v4_20PRODUCTOS.csv
para entender su estructura y cómo el sistema debe procesarlo.
"""

import pandas as pd
import numpy as np

print("=" * 80)
print("ANÁLISIS DE ESTRUCTURA DE DATOS")
print("=" * 80)

# Cargar archivo
df = pd.read_csv('Inventario_v4_20PRODUCTOS.csv', encoding='utf-8')

# 1. DIMENSIONES Y COLUMNAS
print("\n1. DIMENSIONES Y COLUMNAS")
print("-" * 80)
print(f"Registros: {len(df):,}")
print(f"Columnas: {len(df.columns)}")
print(f"\nListado de columnas:")
for i, col in enumerate(df.columns, 1):
    dtype = str(df[col].dtype)
    non_null = df[col].notna().sum()
    print(f"  {i:2}. {col:25} | Tipo: {dtype:15} | Non-null: {non_null:,}")

# 2. ANALISIS POR COLUMNA CRITICA
print("\n\n2. ANÁLISIS DE COLUMNAS CRÍTICAS")
print("-" * 80)

# Producto_id / Codigo
print("\nPRODUCTO_ID (debería ser Codigo):")
print(f"  Valores únicos: {df['Producto_id'].nunique()}")
print(f"  Tipo de datos: {df['Producto_id'].dtype}")
print(f"  Ejemplos: {df['Producto_id'].unique()[:5].tolist()}")

# Tipo_movimiento
print("\nTIPO_MOVIMIENTO (debería ser Documento):")
print(f"  Valores únicos: {df['Tipo_movimiento'].nunique()}")
print(f"  Valores: {sorted(df['Tipo_movimiento'].unique())}")
print(f"  Conteo:")
for tipo in sorted(df['Tipo_movimiento'].unique()):
    count = (df['Tipo_movimiento'] == tipo).sum()
    print(f"    - {tipo:15} : {count:,} registros ({count/len(df)*100:.1f}%)")

# Cantidad
print("\nCANTIDAD (valores de movimiento):")
print(f"  Mínimo: {df['Cantidad'].min():,}")
print(f"  Máximo: {df['Cantidad'].max():,}")
print(f"  Promedio: {df['Cantidad'].mean():.2f}")
print(f"  Desv. Est.: {df['Cantidad'].std():.2f}")
print(f"  % valores positivos: {(df['Cantidad'] > 0).sum() / len(df) * 100:.1f}%")
print(f"  % valores negativos: {(df['Cantidad'] < 0).sum() / len(df) * 100:.1f}%")
print(f"  % ceros: {(df['Cantidad'] == 0).sum() / len(df) * 100:.1f}%")

# Stock
print("\nSTOCK_POSTERIOR (Stock actual):")
print(f"  Mínimo: {df['Stock_posterior'].min():,}")
print(f"  Máximo: {df['Stock_posterior'].max():,}")
print(f"  Promedio: {df['Stock_posterior'].mean():,.2f}")
print(f"  Valores únicos: {df['Stock_posterior'].nunique()}")

# Stock_anterior
print("\nSTOCK_ANTERIOR (para validación):")
print(f"  Mínimo: {df['Stock_anterior'].min():,}")
print(f"  Máximo: {df['Stock_anterior'].max():,}")
print(f"  Promedio: {df['Stock_anterior'].mean():,.2f}")

# Fecha
print("\nFECHA (rango temporal):")
print(f"  Mínima: {df['Fecha'].min()}")
print(f"  Máxima: {df['Fecha'].max()}")
print(f"  Formato: {df['Fecha'].dtype}")

# Año, Mes
print("\nAÑO y MES (desagregación):")
print(f"  Años: {sorted(df['Año'].unique())}")
print(f"  Meses: {sorted(df['Mes'].unique())}")
meses_disponibles = df.groupby(['Año', 'Mes']).size()
print(f"  Combinaciones Año-Mes: {len(meses_disponibles)}")

# Empresa_cliente / Departamento
print("\nCLIENTE (para análisis de canal):")
print(f"  Empresas únicas: {df['Empresa_cliente'].nunique()}")
print(f"  Departamentos únicos: {df['Departamento_cliente'].nunique()}")
print(f"  Canales únicos: {df['Canal_venta'].nunique()}")
print(f"  Canales: {df['Canal_venta'].unique()[:10].tolist()}")

# 3. VALIDACIONES
print("\n\n3. VALIDACIONES DE CONSISTENCIA")
print("-" * 80)

# Validar Stock_anterior + Cantidad = Stock_posterior
df['Stock_calculado'] = df['Stock_anterior'] + df['Cantidad']
diferencia = (df['Stock_calculado'] != df['Stock_posterior']).sum()
print(f"\nValidación: Stock_anterior + Cantidad = Stock_posterior")
print(f"  Registros que cumplen: {len(df) - diferencia:,} ({(1-diferencia/len(df))*100:.1f}%)")
print(f"  Registros que NO cumplen: {diferencia:,}")

if diferencia > 0:
    ejemplos = df[df['Stock_calculado'] != df['Stock_posterior']].head(3)
    print(f"\n  Ejemplos de inconsistencias:")
    for idx, row in ejemplos.iterrows():
        print(f"    {row['Producto_id']:5} | Stock_anterior: {row['Stock_anterior']:6} | Cantidad: {row['Cantidad']:6} | Calculado: {row['Stock_calculado']:6} | Real: {row['Stock_posterior']:6}")

# 4. MAPEO RECOMENDADO
print("\n\n4. MAPEO RECOMENDADO PARA EL SISTEMA")
print("-" * 80)
print("""
Format actual (tu data):
  Producto_id         → Codigo (usar como string)
  Tipo_movimiento     → Documento
  Cantidad            → Salida_unid (para ventas, positivo)
                     → Entrada_unid (para producción, positivo)
  Stock_posterior     → Saldo_unid (renombrar)
  Fecha               → Mantener igual
  Año, Mes, Dia       → Mantener igual

Transforma Tipo_movimiento a Documento:
  'Producción'        → 'Producción' (o 'Entrada')
  'Venta'             → 'Venta' (o 'Salida')
  Otros               → Dejar como está o mapear

Cálculo de Salida_unid y Entrada_unid:
  Si Tipo_movimiento == 'Venta':
    Salida_unid = abs(Cantidad)  (siempre positivo)
  Si Tipo_movimiento == 'Producción':
    Entrada_unid = Cantidad (puede ser positivo o negativo, pero usuallumente positivo)
    Salida_unid = 0
  Else:
    Salida_unid = 0
    Entrada_unid = 0
""")

# 5. MUESTRA DE DATOS DESPUÉS DE MAPEO
print("\n\n5. MUESTRA DE DATOS (primeras 5 filas)")
print("-" * 80)
print(df[['Fecha', 'Producto_id', 'Producto_nombre', 'Tipo_movimiento', 'Cantidad', 'Stock_anterior', 'Stock_posterior']].head())

print("\n\n" + "=" * 80)
print("CONCLUSIÓN")
print("=" * 80)
print(f"""
Tu archivo tiene {len(df):,} registros de {df['Producto_id'].nunique()} productos,
abarcando desde {df['Fecha'].min()} hasta {df['Fecha'].max()}.

El sistema espera ciertos nombres de columnas, pero tu data tiene nombres diferentes.
Los últimos commits han añadido DETECCIÓN FLEXIBLE de columnas, así que debería
funcionar automáticamente detectando:
  - Stock_posterior → como columna de stock
  - Cantidad → como cantidad de movimiento
  - Tipo_movimiento → como documento type

La clave es asegurar que:
  1. La normalización convierta Producto_id → Codigo
  2. El pipeline genere Saldo_unid a partir de Stock_posterior
  3. Las columnas demanda y stock sean consistentes en nombre
""")
