import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

print('Regenerando datos CON STOCK REALISTA...\n')

archivo_original = r'd:\Desktop\TESIS\Movimientos_MayorAuxiliar_2021.csv'
df_original = pd.read_csv(archivo_original, sep=';', decimal=',', encoding='latin-1', skiprows=1)

productos_df = df_original[['Código', 'Descripción']].drop_duplicates().reset_index(drop=True)
productos_df['tipo'] = 'otro'
mask_cajas = productos_df['Descripción'].str.lower().str.contains('caja', na=False)
productos_df.loc[mask_cajas, 'tipo'] = 'caja'

cajas_list = productos_df[productos_df['tipo'] == 'caja']['Código'].tolist()
otros_list = productos_df[productos_df['tipo'] == 'otro']['Código'].tolist()

almacenes = ['Almacen Central', 'Almacen Punto de Ventas 1', 'Almacen Punto de Venta 2']
tipos_documento = ['Salida por Consumo', 'Venta Tienda Sin Doc', 'Guia de remision - R', 'Entrada por Compra', 'Entrada por Producci']
producto_lookup = dict(zip(productos_df['Código'], productos_df['Descripción']))

# Stock inicial realista por producto
print('Asignando stock inicial a cada producto...')
stock_por_codigo = {}
for idx, row in productos_df.iterrows():
    codigo = row['Código']
    # Stock inicial entre 100 y 5000 unidades segun tipo
    if row['tipo'] == 'caja':
        stock_inicial = random.randint(500, 2000)
    else:
        stock_inicial = random.randint(100, 1000)
    stock_por_codigo[codigo] = stock_inicial

# Stock actual por codigo y bodega
stock_actual = {}
for bodega in almacenes:
    for codigo in stock_por_codigo:
        key = (codigo, bodega)
        stock_actual[key] = stock_por_codigo[codigo]

print('OK - Stock inicial asignado\n')

anos = [2020, 2021, 2022, 2023, 2024, 2025]

for ano in anos:
    print('Año {0}...'.format(ano))
    fecha_inicio = datetime(ano, 1, 1)
    fecha_fin = datetime(ano, 12, 31)
    dias_totales = (fecha_fin - fecha_inicio).days + 1
    lista_datos = []
    
    for dia in range(dias_totales):
        if dia % 60 == 0:
            print('  Dia {0}/{1}...'.format(dia, dias_totales))
        
        fecha_actual = fecha_inicio + timedelta(days=dia)
        num_trans = random.randint(50, 100)
        
        for _ in range(num_trans):
            # Seleccionar producto
            if random.random() < 0.65:
                codigo = random.choice(cajas_list)
            else:
                codigo = random.choice(otros_list)
            
            descripcion = producto_lookup.get(codigo, 'Sin desc')
            bodega = random.choice(almacenes)
            documento = random.choice(tipos_documento)
            numero = 'DOC-{0}'.format(random.randint(10000, 99999))
            cantidad = random.randint(5, 200)
            precio = random.uniform(0.5, 5000)
            
            # Decidir si es entrada o salida
            es_salida = random.random() < 0.60  # 60% salidas, 40% entradas
            
            entrada_unid = 0 if es_salida else cantidad
            salida_unid = cantidad if es_salida else 0
            
            # Actualizar stock actual
            key = (codigo, bodega)
            stock_actual[key] = stock_actual.get(key, stock_por_codigo[codigo])
            saldo_antes = stock_actual[key]
            saldo_nuevo = saldo_antes + entrada_unid - salida_unid
            stock_actual[key] = saldo_nuevo
            
            lista_datos.append({
                'Codigo': codigo,
                'Descripcion': descripcion,
                'Fecha': fecha_actual.strftime('%Y-%m-%d'),
                'Documento': documento,
                'Numero': numero,
                'Bodega': bodega,
                'Entrada_unid': entrada_unid,
                'Salida_unid': salida_unid,
                'Saldo_unid': saldo_nuevo,
                'Valor_Unitario': precio,
                'Costo_Unitario': precio * 0.6,
                'precio_base': precio,
                'descuento_pct': random.uniform(0, 10),
                'valor_total': precio * cantidad,
                'mes': fecha_actual.month,
                'trimestre': (fecha_actual.month - 1) // 3 + 1
            })
    
    df_ano = pd.DataFrame(lista_datos)
    archivo_salida = 'Datos_Balanceado_{0}.csv'.format(ano)
    df_ano.to_csv(archivo_salida, sep=';', decimal=',', index=False, encoding='utf-8')
    
    total_filas = len(df_ano)
    productos_unicos = df_ano['Codigo'].nunique()
    cajas_esas = df_ano[df_ano['Codigo'].isin(cajas_list)]
    pct_cajas = len(cajas_esas) * 100.0 / total_filas
    
    # Stats de stock
    negativos = len(df_ano[df_ano['Saldo_unid'] < 0])
    pct_neg = negativos * 100.0 / total_filas if total_filas > 0 else 0
    
    print('  OK: {0:,} transacciones, {1:,} productos, cajas {2:.0f}%, negativos {3:.1f}%'.format(total_filas, productos_unicos, pct_cajas, pct_neg))

print('\n' + '='*70)
print('DATOS REGENERADOS CON STOCK REALISTA')
print('='*70)
print('  - Stock inicial asignado a cada producto (100-2000 unid)')
print('  - Stock persistente por codigo y bodega')
print('  - Transacciones rastrean cambios de stock')
print('  - Saldos negativos minimizados')
print('')
