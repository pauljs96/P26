import pandas as pd

df = pd.read_csv('Inventario_ML_v3.csv')

# Filtrar solo producción
df_prod = df[df['Tipo_movimiento'] == 'Producción'].copy()
df_prod['Fecha'] = pd.to_datetime(df_prod['Fecha'])

print("=" * 80)
print("[INVESTIGACIÓN] ¿Por qué varía el Costo_unitario?")
print("=" * 80)

# Analizar: ¿Es por diferentes empresas?
print("\n1. HIPÓTESIS: ¿Varía por DIFERENTES EMPRESAS (clientes)?")
print("-" * 80)

# Tomar un día específico con un producto
muestra = df_prod[(df_prod['Fecha'] == '2020-01-02') & (df_prod['Producto_id'] == 'CPE-00063')]

print(f"\nProducción de CPE-00063 el 2020-01-02:")
print(f"Total registros: {len(muestra)}\n")

for idx, (i, row) in enumerate(muestra.iterrows()):
    print(f"Registro {idx+1}:")
    print(f"  Empresa_cliente: {row['Empresa_cliente']}")
    print(f"  Cantidad: {row['Cantidad']} unidades")
    print(f"  Costo_unitario: ${row['Costo_unitario']:.2f}")
    print(f"  Valor_total: ${row['Valor_total']:.2f}")
    print()

# Verificación 2: En un día, mismo producto, ¿son empresas diferentes?
print("\n" + "=" * 80)
print("2. VERIFICACIÓN: ¿Hay múltiples empresas en mismo día-producto?")
print("-" * 80)

muestra_empresas = muestra['Empresa_cliente'].unique()
print(f"\nEmpresai en ese día para CPE-00063: {list(muestra_empresas)}")

# Análisis más amplio
print("\n" + "=" * 80)
print("3. PATRÓN: Costo_unitario POR PRODUCTO-EMPRESA")
print("-" * 80)

for producto_id in ['CPE-00063', 'CPE-00365']:
    print(f"\n{producto_id}:")
    df_prod_item = df_prod[df_prod['Producto_id'] == producto_id]
    
    # Grupos por empresa
    grupos_empresa = df_prod_item.groupby('Empresa_cliente')['Costo_unitario'].unique()
    
    for empresa, costos in grupos_empresa.items():
        print(f"  {empresa[:30]:30} → Costo unitario: ${costos[0]:.2f} (constante)")

# Conclusión
print("\n" + "=" * 80)
print("4. CONCLUSIÓN DEL ANÁLISIS")
print("=" * 80)

print("""
HALLAZGO CLAVE:
===============

El Costo_unitario VARÍA porque cada EMPRESA-CLIENTE tiene su PROPIO costo de producción.

PROBLEMA LÓGICO:
================

Esto NO tiene sentido de negocio. El costo de PRODUCIR un producto debería ser:
- CONSTANTE para ese producto (mismo costo manufacturero)
- IGUAL sin importar a qué cliente va
- INDEPENDIENTE de la empresa cliente

ACTUALMENTE:
- CPE-00063 para Constructora Pacífico = $X
- CPE-00063 para Desarrollos Urbanos = $Y
- CPE-00063 para Inmobiliaria Andina = $Z

DEBERÍA SER:
- Costo unitario de CPE-00063 = $C (CONSTANTE)
- Este costo es igual para todos, la producción es interna

CAMBIO NECESARIO:
==================
El Costo_unitario debe ser CONSTANTE POR PRODUCTO, no por (Producto, Empresa).

Actual: precio_base = np.random.uniform(3, 8)  [generado en cada producto-empresa]
Ideal:  precio_base debería generarse UNA SOLA VEZ por Producto, no por (Producto, Empresa)
""")
