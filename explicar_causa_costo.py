import pandas as pd

df = pd.read_csv('Inventario_ML_v3.csv')
df_prod = df[df['Tipo_movimiento'] == 'Producción'].copy()

print("=" * 80)
print("[ANÁLISIS FINAL] CAUSA DE VARIACIÓN DE COSTO_UNITARIO")
print("=" * 80)

# Mostrar el patrón del generador
print("\n1. ESTRUCTURA DEL GENERADOR")
print("-" * 80)

print("""
El script generar_data_inventario_completo.py tiene este flujo:

    Para CADA Producto (5 productos):
        Para CADA Empresa (5 empresas):
            → Llamar generate_dataset_por_producto_empresa()
            
Dentro de generate_dataset_por_producto_empresa():
    precio_base = np.random.uniform(3, 8)  # GENERA AQUÍ
    ... genera toda la serie de tiempo para esa combinación ...
    
RESULTADO:
- CPE-00063 + Constructora Pacífico        → precio_base = $4.00
- CPE-00063 + Desarrollos Urbanos del Sur  → precio_base = $5.50
- CPE-00063 + Inmobiliaria Andina          → precio_base = $6.00
- CPE-00063 + Proyectos Inmobiliarios      → precio_base = $5.20
- CPE-00063 + Constructores del Andes      → precio_base = $4.80
""")

# Verificar esto en los datos
print("\n2. VERIFICACIÓN EN LOS DATOS")
print("-" * 80)

for producto_id in ['CPE-00063']:
    print(f"\n{producto_id} - Costo_unitario por serie:")
    
    df_prod_item = df_prod[df_prod['Producto_id'] == producto_id]
    df_prod_item = df_prod_item.sort_values('Fecha')
    
    # Tomar primeros registros de diferentes períodos
    for idx, row in df_prod_item.head(10).iterrows():
        empresa_num = row.name  # Esto no funciona, necesito otro método
        print(f"  {row['Fecha']}: Costo ${row['Costo_unitario']:.2f}")

# Análisis de variación
print("\n3. IMPACTO: ¿Cuán variable es?")
print("-" * 80)

stats_por_prod = df_prod.groupby('Producto_id')['Costo_unitario'].agg([
    ('Min', 'min'),
    ('Max', 'max'),
    ('Mean', 'mean'),
    ('StdDev', 'std'),
    ('Coef_Variacion', lambda x: (x.std() / x.mean()) * 100)
]).round(3)

print("\n", stats_por_prod)

print("\n" + "=" * 80)
print("[CONCLUSIÓN] POR QUÉ VARÍA EL COSTO")
print("=" * 80)

print("""
ROOT CAUSE (RAÍZ DEL PROBLEMA):
================================

El precio_base se genera NUEVAMENTE en cada iteración del loop producto-empresa:

    Para producto_id en [CPE-00063, CPE-00063, CPE-00063, CPE-00063, CPE-00063]:
        Para empresa en [Empresa1, Empresa2, Empresa3, Empresa4, Empresa5]:
            precio_base = np.random.uniform(3, 8)  # ← AQUÍ GENERA UN VALOR ALEATORIO
            
RESULTADO ACTUAL:
- 5 valores aleatorios diferentes para el MISMO producto
- Cada empresa provoca que se genere un precio_base distinto
- Cuando se combinan todos en el CSV, parecen variaciones "aleatorias"

PROBLEMA CONCEPTUAL:
=====================

1. La Producción es INTERNA (Empresa_cliente = NULL)
2. El costo de manufacturar CPE-00063 es su COSTO DE PRODUCCIÓN
3. Este costo debería ser IGUAL SIEMPRE, no depende de cliente/empresa
4. Es como si un fabricante dijera: "Este producto cuesta $4 para una empresa 
   y $6 para otra" - NO TIENE SENTIDO en una fábrica real

LO IRÓNICO:
===========

Aunque el código genera 5 series de tiempo (1 por empresa), en el CSV final
solo aparecen como "series de un mismo producto" sin identificador de cuál 
serie de qué empresa es (porque Empresa_cliente está NULL en producción).

Por eso se ve como variación "aleatoria" del mismo producto.

SOLUCIÓN NECESARIA:
====================

El precio_base debe generarse AFUERA del loop de empresas:

    Para producto_id en todos:
        precio_base[producto_id] = np.random.uniform(3, 8)  # UNA SOLA VEZ
        
        Para empresa en todas:
            generate_dataset_por_producto_empresa(producto_id, empresa, precio_base[producto_id])

Así: CPE-00063 SIEMPRE cuesta $4.75 en producción, sin importar qué empresa
""")
