"""
TRANSFORMADOR DE DATOS: De tu formato a formato del sistema

Convierte tu archivo CSV al formato que el dashboard espera.
"""

import pandas as pd
import numpy as np

print("=" * 80)
print("TRANSFORMADOR DE DATOS CSV")
print("=" * 80)

# Cargar archivo original
df = pd.read_csv('Inventario_v4_20PRODUCTOS.csv', encoding='utf-8')
print(f"\nArchivo original: {len(df):,} registros")

# PASO 1: Renombrar columnas principales
print("\n[PASO 1] Renombrando columnas...")
df_transformed = df.copy()

rename_map = {
    'Producto_id': 'Codigo',
    'Tipo_movimiento': 'Documento',
    'Stock_posterior': 'Saldo_unid',
}

df_transformed = df_transformed.rename(columns=rename_map)
print(f"  Columnas renombradas: {list(rename_map.items())}")

# PASO 2: Convertir Codigo a string
print("\n[PASO 2] Convirtiendo Codigo a string...")
df_transformed['Codigo'] = df_transformed['Codigo'].astype(str).str.strip()
print(f"  Codigo ahora es string. Ejemplos: {df_transformed['Codigo'].unique()[:5].tolist()}")

# PASO 3: Crear Salida_unid según Documento
print("\n[PASO 3] Creando Salida_unid y Entrada_unid...")
df_transformed['Salida_unid'] = 0
df_transformed['Entrada_unid'] = 0

# Para Venta: Salida_unid = Cantidad positiva
mask_venta = df_transformed['Documento'] == 'Venta'
df_transformed.loc[mask_venta, 'Salida_unid'] = df_transformed.loc[mask_venta, 'Cantidad'].abs()

# Para Producción: Entrada_unid = Cantidad positiva
mask_produccion = df_transformed['Documento'] == 'Producción'
df_transformed.loc[mask_produccion, 'Entrada_unid'] = df_transformed.loc[mask_produccion, 'Cantidad'].abs()

print(f"  Salida_unid creada para {mask_venta.sum():,} registros de Venta")
print(f"  Entrada_unid creada para {mask_produccion.sum():,} registros de Producción")

# PASO 4: Convertir Fecha a datetime
print("\n[PASO 4] Convirtiendo Fecha a datetime...")
df_transformed['Fecha'] = pd.to_datetime(df_transformed['Fecha'])
print(f"  Rango: {df_transformed['Fecha'].min()} a {df_transformed['Fecha'].max()}")

# PASO 5: Crear columna Mes en formato datetime (1er día del mes)
print("\n[PASO 5] Creando columna Mes (datetime)...")
df_transformed['Mes_datetime'] = pd.to_datetime(
    df_transformed['Año'].astype(str) + '-' + df_transformed['Mes'].astype(str).str.zfill(2) + '-01'
)
print(f"  Mes_datetime creado. Ejemplos: {df_transformed['Mes_datetime'].unique()[:5]}")

# PASO 6: Organizar columnas en orden lógico
print("\n[PASO 6] Reorganizando columnas...")
columnas_orden = [
    'Fecha', 'Año', 'Mes', 'Mes_datetime', 'Dia',  # Temporal
    'Codigo', 'Producto_nombre',  # Producto
    'Documento', 'Cantidad', 'Salida_unid', 'Entrada_unid',  # Movimiento
    'Stock_anterior', 'Saldo_unid',  # Stock
    'Empresa_cliente', 'Departamento_cliente', 'Canal_venta', 'Punto_venta',  # Cliente
    'Precio_unitario', 'Descuento_pct', 'Valor_total', 'Campana', 'Costo_unitario'  # Financiero
]

df_transformed = df_transformed[[col for col in columnas_orden if col in df_transformed.columns]]

# PASO 7: Guardar archivo transformado
output_file = 'Inventario_v4_20PRODUCTOS_TRANSFORMADO.csv'
df_transformed.to_csv(output_file, index=False, encoding='utf-8')
print(f"\n✅ Archivo transformado guardado: {output_file}")

# PASO 8: Validación
print("\n" + "=" * 80)
print("VALIDACIÓN")
print("=" * 80)

print(f"\nArchivo original:")
print(f"  Registros: {len(df):,}")
print(f"  Columnas: {list(df.columns)[:5]}... ({len(df.columns)} totales)")

print(f"\nArchivo transformado:")
print(f"  Registros: {len(df_transformed):,}")
print(f"  Columnas: {list(df_transformed.columns)[:8]}... ({len(df_transformed.columns)} totales)")

print(f"\nColumnas clave después de transformación:")
print(f"  ✅ Codigo: {df_transformed['Codigo'].nunique()} productos únicos")
print(f"  ✅ Documento: {df_transformed['Documento'].unique().tolist()}")
print(f"  ✅ Saldo_unid: Rango {df_transformed['Saldo_unid'].min():,} a {df_transformed['Saldo_unid'].max():,}")
print(f"  ✅ Fecha: {df_transformed['Fecha'].min()} a {df_transformed['Fecha'].max()}")
print(f"  ✅ Mes_datetime: {len(df_transformed['Mes_datetime'].unique())} períodos")

# PASO 9: Mostrar muestra
print(f"\nMuestra de datos transformados (5 primeras filas):")
print(df_transformed[['Fecha', 'Codigo', 'Documento', 'Cantidad', 'Salida_unid', 'Entrada_unid', 'Stock_anterior', 'Saldo_unid']].head())

print("\n" + "=" * 80)
print("PRÓXIMOS PASOS")
print("=" * 80)
print(f"""
1. El archivo {output_file} está listo para usar

2. Para cargar en Streamlit Cloud:
   - Puedes usar este archivo transformado directamente
   - O seguir usando el original (el sistema debería detectar automáticamente)

3. Estructura esperada después de transformación:
   - Codigo: ID del producto (string)
   - Documento: Tipo de movimiento ('Venta' o 'Producción')
   - Saldo_unid: Stock después del movimiento
   - Salida_unid: Cantidad vendida (solo para Venta)
   - Entrada_unid: Cantidad producida (solo para Producción)
   - Fecha: Datetime del movimiento
   - Mes_datetime: Primer día del mes (para agregación)

4. El dashboard puede ahora:
   ✅ Agrupar por Mes_datetime para obtener demanda mensual
   ✅ Usar Saldo_unid directamente como stock
   ✅ Identificar tipo de movimiento por Documento
   ✅ Separar ventas de producción
""")
