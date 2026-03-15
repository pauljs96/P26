import pandas as pd
import random
from datetime import datetime, timedelta

print('Generando datos SIN STOCK NEGATIVO...\n')

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

# Stock inicial por producto y bodega
print('Asignando stock inicial...')
stock_actual = {}
for bodega in almacenes:
    for idx, row in productos_df.iterrows():
        codigo = row['Código']
        if row['tipo'] == 'caja':
            stock = random.randint(500, 2000)
        else:
            stock = random.randint(100, 1000)
        key = (codigo, bodega)
        stock_actual[key] = stock

print('OK\n')

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
            
            key = (codigo, bodega)
            stock_antes = stock_actual.get(key, 0)
            
            # Intentar salida (60% de probabilidad)
            intenta_salida = random.random() < 0.60
            
            if intenta_salida and stock_antes >= cantidad:
                # Hay stock: hacer la salida
                entrada_unid = 0
                salida_unid = cantidad
                stock_nuevo = stock_antes - cantidad
            elif intenta_salida and stock_antes < cantidad:
                # NO hay stock: convertir a entrada (reabastecimiento)
                entrada_unid = cantidad
                salida_unid = 0
                stock_nuevo = stock_antes + cantidad
            else:
                # Hacer entrada
                entrada_unid = cantidad
                salida_unid = 0
                stock_nuevo = stock_antes + cantidad
            
            # Actualizar stock
            stock_actual[key] = stock_nuevo
            
            # Validar que nunca sea negativo
            if stock_nuevo < 0:
                print('ERROR: Stock negativo detectado!')
                raise ValueError('Stock negativo en {0} bodega {1}'.format(codigo, bodega))
            
            lista_datos.append({
                'Codigo': codigo,
                'Descripcion': descripcion,
                'Fecha': fecha_actual.strftime('%Y-%m-%d'),
                'Documento': documento,
                'Numero': numero,
                'Bodega': bodega,
                'Entrada_unid': entrada_unid,
                'Salida_unid': salida_unid,
                'Saldo_unid': stock_nuevo,
                'Valor_Unitario': precio,
                'Costo_Unitario': precio * 0.6,
                'precio_base': precio,
                'descuento_pct': random.uniform(0, 10),
                'valor_total': precio * cantidad,
                'mes': fecha_actual.month,
                'trimestre': (fecha_actual.month - 1) // 3 + 1
            })
    
    df_ano = pd.DataFrame(lista_datos)
    archivo_salida = 'DatosLimpios_{0}.csv'.format(ano)
    df_ano.to_csv(archivo_salida, sep=';', decimal=',', index=False, encoding='utf-8')
    
    total_filas = len(df_ano)
    productos_unicos = df_ano['Codigo'].nunique()
    negativos = len(df_ano[df_ano['Saldo_unid'] < 0])
    
    print('  OK: {0:,} transacciones, {1:,} productos, negativos {2}'.format(total_filas, productos_unicos, negativos))

print('\n' + '='*70)
print('LISTO - DATOS SIN JUNCA STOCK NEGATIVO')
print('='*70)
print('  - Stock inicial: 100-2000 unid por producto')
print('  - Salidas: Solo si hay stock disponible')
print('  - Sin stock: Automaticamente convertido a entrada')
print('  - Resultado: 0% stock negativo')
print('')
