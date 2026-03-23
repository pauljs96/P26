import pandas as pd
import numpy as np

df = pd.read_csv('Inventario_ML_v3.csv')

# Filtrar solo producción
df_prod = df[df['Tipo_movimiento'] == 'Producción'].copy()

print("=" * 80)
print("[ANÁLISIS] Variación de Costo_unitario en Producción")
print("=" * 80)

# Analizar por producto
print("\n1. ANÁLISIS POR PRODUCTO")
print("-" * 80)

for producto_id in sorted(df_prod['Producto_id'].unique()):
    df_prod_p = df_prod[df_prod['Producto_id'] == producto_id]
    
    costo_unitario_values = df_prod_p['Costo_unitario'].dropna().unique()
    
    print(f"\n{producto_id}:")
    print(f"  Total registros de producción: {len(df_prod_p):,}")
    print(f"  Costo_unitario único: {len(costo_unitario_values)}")
    
    if len(costo_unitario_values) > 1:
        print(f"  ❌ VARIACIÓN DETECTADA")
        print(f"  Rango: ${costo_unitario_values.min():.2f} - ${costo_unitario_values.max():.2f}")
        print(f"  Valores diferentes: {sorted(costo_unitario_values)[:10]}")  # Mostrar primeros 10
    else:
        print(f"  ✓ Valor constante: ${costo_unitario_values[0]:.2f}")

# Analizar por fecha
print("\n" + "=" * 80)
print("2. ANÁLISIS POR MISMO DÍA Y PRODUCTO")
print("-" * 80)

df_prod['Fecha'] = pd.to_datetime(df_prod['Fecha'])
grouped = df_prod.groupby(['Fecha', 'Producto_id'])

dias_con_variacion = []

for (fecha, producto_id), group in grouped:
    if len(group) > 0:
        costos = group['Costo_unitario'].dropna().unique()
        if len(costos) > 1:
            dias_con_variacion.append({
                'Fecha': fecha,
                'Producto': producto_id,
                'Registros': len(group),
                'Costos_diferentes': len(costos),
                'Valores': sorted(costos)
            })

if dias_con_variacion:
    print(f"Se encontraron {len(dias_con_variacion)} días con VARIACIÓN en Costo_unitario")
    print("\nEjemplos:")
    for i, item in enumerate(dias_con_variacion[:5]):
        print(f"\n{i+1}. {item['Fecha'].date()} - {item['Producto']}")
        print(f"   Registros en el día: {item['Registros']}")
        print(f"   Costos diferentes: {item['Costos_diferentes']}")
        print(f"   Valores: {[f'${v:.2f}' for v in item['Valores']]}")
else:
    print("✓ No se detectó variación de Costo_unitario en el mismo día para mismo producto")

# Analizar por mes
print("\n" + "=" * 80)
print("3. ANÁLISIS POR MES Y PRODUCTO")
print("-" * 80)

df_prod['AñoMes'] = df_prod['Fecha'].dt.to_period('M')
grouped_mes = df_prod.groupby(['AñoMes', 'Producto_id'])

meses_con_variacion = 0

for (periodo, producto_id), group in grouped_mes:
    costos = group['Costo_unitario'].dropna().unique()
    if len(costos) > 1:
        meses_con_variacion += 1
        if meses_con_variacion <= 3:  # Mostrar primeros 3
            print(f"\n{periodo} - {producto_id}:")
            print(f"  Registros: {len(group)}")
            print(f"  Costos diferentes: {len(costos)}")
            print(f"  Rango: ${costos.min():.2f} - ${costos.max():.2f}")

if meses_con_variacion > 0:
    print(f"\n❌ TOTAL: {meses_con_variacion} combinaciones mes-producto con variación de Costo_unitario")
else:
    print("\n✓ No hay variación de Costo_unitario dentro de un mismo mes para mismo producto")

# Análisis detallado de un ejemplo
print("\n" + "=" * 80)
print("4. EJEMPLO DETALLADO - Variaciones en un producto específico")
print("-" * 80)

# Tomar un producto con variación
producto_test = df_prod[df_prod['Producto_id'] == 'CPE-00063'].head(5)

print(f"\nÚltimos 5 registros de producción para CPE-00063:")
for idx, row in producto_test.iterrows():
    print(f"\n  Fecha: {row['Fecha']}")
    print(f"    Cantidad: {row['Cantidad']} unidades")
    print(f"    Costo_unitario: ${row['Costo_unitario']:.2f}")
    print(f"    Valor_total (costo): ${row['Valor_total']:.2f}")
    print(f"    Cálculo: {row['Cantidad']} × ${row['Costo_unitario']:.2f} = ${row['Cantidad'] * row['Costo_unitario']:.2f}")

# Análisis lógico
print("\n" + "=" * 80)
print("5. CONCLUSIÓN ESPERADA")
print("-" * 80)
print("""
¿POR QUÉ PODRÍA VARIAR?

Opciones encontradas en el código:

1. Si se genera NUEVO precio_base cada día/registro
   → Costo_unitario sería aleatorio
   
2. Si se genera CONSTANTE precio_base por producto-empresa
   → Costo_unitario debería ser IGUAL para mismo producto
   
3. Si hay factor aleatorio en costo_produccion
   → El Valor_total variaría, pero Costo_unitario sería igual
   
Necesitamos revisar el código generador para identificar la causa.
""")
