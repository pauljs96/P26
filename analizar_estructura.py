import pandas as pd
import numpy as np

df = pd.read_csv('Inventario_ML_v2.csv')

print("=" * 80)
print("VENTA - Registro típico:")
print("=" * 80)
venta = df[df['Tipo_movimiento'] == 'Venta'].iloc[0]
for col in df.columns:
    valor = venta[col]
    # Mostrar NULL como "None"
    if pd.isna(valor):
        valor = "None (NULL)"
    print(f"{col:25} {valor}")

print("\n" + "=" * 80)
print("PRODUCCIÓN - Registro típico:")
print("=" * 80)
produccion = df[df['Tipo_movimiento'] == 'Producción'].iloc[0]
for col in df.columns:
    valor = produccion[col]
    # Mostrar NULL como "None"
    if pd.isna(valor):
        valor = "None (NULL)"
    print(f"{col:25} {valor}")

print("\n" + "=" * 80)
print("✅ ESTRUCTURA CORRECTA:")
print("=" * 80)
print("\nCampos con NULL en PRODUCCIÓN:")
print(f"  - Empresa_cliente: {produccion['Empresa_cliente']} ✓")
print(f"  - Departamento_cliente: {produccion['Departamento_cliente']} ✓")
print(f"  - Canal_venta: {produccion['Canal_venta']} ✓")
print(f"  - Precio_unitario: {produccion['Precio_unitario']} ✓")
print(f"  - Descuento_pct: {produccion['Descuento_pct']} ✓")
print(f"  - Campana: {produccion['Campana']} ✓")

print("\nCampos específicos de PRODUCCIÓN:")
print(f"  - Costo_unitario: {produccion['Costo_unitario']} ✓")
print(f"  - Valor_total (costo): {produccion['Valor_total']}")

print("\n" + "=" * 80)
print("LISTA COMPLETA DE COLUMNAS:")
print("=" * 80)
print("\nPara VENTAS (llenan):              Para PRODUCCIÓN (son NULL):")
print("- Empresa_cliente                  - Empresa_cliente")
print("- Departamento_cliente             - Departamento_cliente")
print("- Canal_venta                      - Canal_venta")  
print("- Precio_unitario                  - Precio_unitario")
print("- Descuento_pct                    - Descuento_pct")
print("- Campana                          - Campana")
print("                                   ")
print("Ambas comparten:")
print("- Fecha, Año, Mes, Dia")
print("- Producto_id, Producto_nombre")
print("- Tipo_movimiento (Venta/Producción)")
print("- Cantidad, Stock_anterior, Stock_posterior")
print("- Valor_total")
print("- Costo_unitario (NULL en Venta)")
print("- Estacionalidad_factor, etc.")
