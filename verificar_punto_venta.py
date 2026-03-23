import pandas as pd

df = pd.read_csv('Inventario_ML_v3.csv')

print("=" * 80)
print("VENTA en TIENDA FÍSICA:")
print("=" * 80)
venta_tienda = df[(df['Tipo_movimiento'] == 'Venta') & (df['Canal_venta'] == 'Tienda Física')].iloc[0]
for col in ['Fecha', 'Producto_id', 'Empresa_cliente', 'Canal_venta', 'Punto_venta', 'Cantidad', 'Valor_total']:
    valor = venta_tienda[col]
    if pd.isna(valor):
        valor = "None (NULL)"
    print(f"{col:25} {valor}")

print("\n" + "=" * 80)
print("VENTA ONLINE:")
print("=" * 80)
venta_online = df[(df['Tipo_movimiento'] == 'Venta') & (df['Canal_venta'] == 'Online')].iloc[0]
for col in ['Fecha', 'Producto_id', 'Empresa_cliente', 'Canal_venta', 'Punto_venta', 'Cantidad', 'Valor_total']:
    valor = venta_online[col]
    if pd.isna(valor):
        valor = "None (NULL)"
    print(f"{col:25} {valor}")

print("\n" + "=" * 80)
print("PRODUCCIÓN:")
print("=" * 80)
produccion = df[df['Tipo_movimiento'] == 'Producción'].iloc[0]
for col in ['Fecha', 'Producto_id', 'Empresa_cliente', 'Canal_venta', 'Punto_venta', 'Cantidad', 'Costo_unitario']:
    valor = produccion[col]
    if pd.isna(valor):
        valor = "None (NULL)"
    print(f"{col:25} {valor}")

print("\n" + "=" * 80)
print("DISTRIBUCIÓN: Ventas por Punto de Venta")
print("=" * 80)
venta_tienda_count = len(df[(df['Tipo_movimiento'] == 'Venta') & (df['Canal_venta'] == 'Tienda Física')])
punto_1 = len(df[(df['Tipo_movimiento'] == 'Venta') & (df['Punto_venta'] == 'Punto de venta 1')])
punto_2 = len(df[(df['Tipo_movimiento'] == 'Venta') & (df['Punto_venta'] == 'Punto de venta 2')])

print(f"Total ventas en Tienda Física: {venta_tienda_count:,}")
print(f"  - Punto de venta 1: {punto_1:,} ({100*punto_1/venta_tienda_count:.1f}%)")
print(f"  - Punto de venta 2: {punto_2:,} ({100*punto_2/venta_tienda_count:.1f}%)")

print(f"\nVentas Online (sin punto de venta): {len(df[(df['Tipo_movimiento'] == 'Venta') & (df['Canal_venta'] == 'Online')]):,}")

print("\n" + "=" * 80)
print("VERIFICACIÓN: Punto_venta no debe existir en Online ni Producción")
print("=" * 80)
online_con_pv = len(df[(df['Canal_venta'] == 'Online') & (df['Punto_venta'].notna())])
prod_con_pv = len(df[(df['Tipo_movimiento'] == 'Producción') & (df['Punto_venta'].notna())])

print(f"Online con Punto_venta: {online_con_pv} (debería ser 0) ✓" if online_con_pv == 0 else f"Online con Punto_venta: {online_con_pv} (ERROR)")
print(f"Producción con Punto_venta: {prod_con_pv} (debería ser 0) ✓" if prod_con_pv == 0 else f"Producción con Punto_venta: {prod_con_pv} (ERROR)")
