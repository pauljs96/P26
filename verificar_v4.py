import pandas as pd

df = pd.read_csv('Inventario_ML_Completo_v4.csv')

print("=" * 80)
print("[VERIFICACIÓN] Dataset v4")
print("=" * 80)

# 1. Columnas
print("\n1. COLUMNAS DEL DATASET:")
print("-" * 80)
print(df.columns.tolist())

# 2. Verificar Estacionalidad_factor, etc estén eliminadas
removed_cols = ['Estacionalidad_factor', 'Dia_semana_factor', 'Evento_impacto']
print("\n2. COLUMNAS ELIMINADAS:")
for col in removed_cols:
    if col in df.columns:
        print(f"  ❌ {col} - AÚN PRESENTE")
    else:
        print(f"  ✓ {col} - Eliminada")

# 3. Verificar Costo_unitario constante por producto
print("\n3. COSTO_UNITARIO POR PRODUCTO (verificando que sea constante):")
print("-" * 80)

productos_verificar = df['Producto_id'].unique()[:10]

for producto_id in productos_verificar:
    df_prod = df[df['Producto_id'] == producto_id]
    df_prod_costs = df_prod[df_prod['Tipo_movimiento'] == 'Producción']['Costo_unitario'].dropna().unique()
    
    if len(df_prod_costs) == 1:
        costo = df_prod_costs[0]
        print(f"✓ {producto_id:20} → Costo constante: ${costo:.2f}")
    else:
        print(f"❌ {producto_id:20} → VARIABLE: {sorted(df_prod_costs)[:3]}")

# 4. Estructura de Venta vs Producción
print("\n4. ESTRUCTURA - VENTA:")
print("-" * 80)
venta_sample = df[df['Tipo_movimiento'] == 'Venta'].iloc[0]
for col in ['Empresa_cliente', 'Departamento_cliente', 'Canal_venta', 'Punto_venta', 
            'Precio_unitario', 'Descuento_pct', 'Costo_unitario']:
    valor = venta_sample[col]
    if pd.isna(valor):
        print(f"  {col:25} None (NULL)")
    else:
        print(f"  {col:25} {valor}")

print("\n5. ESTRUCTURA - PRODUCCIÓN:")
print("-" * 80)
prod_sample = df[df['Tipo_movimiento'] == 'Producción'].iloc[0]
for col in ['Empresa_cliente', 'Departamento_cliente', 'Canal_venta', 'Punto_venta',
            'Precio_unitario', 'Descuento_pct', 'Costo_unitario']:
    valor = prod_sample[col]
    if pd.isna(valor):
        print(f"  {col:25} None (NULL)")
    else:
        print(f"  {col:25} {valor}")

# 6. Verificar transacciones múltiples por día
print("\n6. TRANSACCIONES POR DÍA (muestra):")
print("-" * 80)
df['Fecha'] = pd.to_datetime(df['Fecha'])
transac_por_dia = df.groupby('Fecha').size()
print(f"  Promedio transacciones/día: {transac_por_dia.mean():.1f}")
print(f"  Min: {transac_por_dia.min()}, Max: {transac_por_dia.max()}")

# 7. Punto de venta
print("\n7. DISTRIBUCIÓN PUNTO DE VENTA:")
print("-" * 80)
tienda_con_pv = len(df[(df['Canal_venta'] == 'Tienda Física') & (df['Punto_venta'].notna())])
tienda_sin_pv = len(df[(df['Canal_venta'] == 'Tienda Física') & (df['Punto_venta'].isna())])
online = len(df[df['Canal_venta'] == 'Online'])

print(f"  Tienda Física CON punto_venta: {tienda_con_pv:,}")
print(f"  Tienda Física SIN punto_venta: {tienda_sin_pv:,}")
print(f"  Online (sin punto_venta): {online:,}")

print("\n" + "=" * 80)
print("✅ VERIFICACIÓN COMPLETADA")
print("=" * 80)
