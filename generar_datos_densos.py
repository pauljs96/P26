import pandas as pd
import random
from datetime import datetime, timedelta
import os

# Asegurar que estamos en el directorio correcto
os.chdir(r'd:\Desktop\TESIS\Sistema_Tesis')

archivo_original = r'd:\Desktop\TESIS\Movimientos_MayorAuxiliar_2021.csv'
df_original = pd.read_csv(archivo_original, sep=';', decimal=',', encoding='latin-1', skiprows=1)

print('Preparando catalogo...')
productos_df = df_original[['Código', 'Descripción']].drop_duplicates().reset_index(drop=True)
productos_df['tipo'] = 'otro'
mask_cajas = productos_df['Descripción'].str.lower().str.contains('caja', na=False)
productos_df.loc[mask_cajas, 'tipo'] = 'caja'

cajas_list = productos_df[productos_df['tipo'] == 'caja']['Código'].tolist()
otros_list = productos_df[productos_df['tipo'] == 'otro']['Código'].tolist()

print('  CAJAS: {0}'.format(len(cajas_list)))
print('  OTROS: {0}'.format(len(otros_list)))
print('')

almacenes = ['Almacen Central', 'Almacen Punto de Ventas 1', 'Almacen Punto de Venta 2']
tipos_documento = ['Salida por Consumo', 'Venta Tienda Sin Doc', 'Guia de remision - R', 'Entrada por Compra', 'Entrada por Producci']
producto_lookup = dict(zip(productos_df['Código'], productos_df['Descripción']))

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
            if random.random() < 0.65:
                codigo = random.choice(cajas_list)
            else:
                codigo = random.choice(otros_list)
            
            descripcion = producto_lookup.get(codigo, 'Sin desc')
            es_salida = random.random() < 0.75
            bodega = random.choice(almacenes)
            documento = random.choice(tipos_documento)
            numero = 'DOC-{0}'.format(random.randint(10000, 99999))
            cantidad = random.randint(5, 200)
            precio = random.uniform(0.5, 5000)
            
            entrada_unid = 0 if es_salida else cantidad
            salida_unid = cantidad if es_salida else 0
            
            lista_datos.append({
                'Codigo': codigo,
                'Descripcion': descripcion,
                'Fecha': fecha_actual.strftime('%Y-%m-%d'),
                'Documento': documento,
                'Numero': numero,
                'Bodega': bodega,
                'Entrada_unid': entrada_unid,
                'Salida_unid': salida_unid,
                'Saldo_unid': entrada_unid - salida_unid,
                'Valor_Unitario': precio,
                'Costo_Unitario': precio * 0.6,
                'precio_base': precio,
                'descuento_pct': random.uniform(0, 10),
                'valor_total': precio * cantidad,
                'mes': fecha_actual.month,
                'trimestre': (fecha_actual.month - 1) // 3 + 1
            })
    
    df_ano = pd.DataFrame(lista_datos)
    archivo_salida = 'Datos_{0}.csv'.format(ano)
    df_ano.to_csv(archivo_salida, sep=';', decimal=',', index=False, encoding='utf-8')
    
    total_filas = len(df_ano)
    productos_unicos = df_ano['Codigo'].nunique()
    cajas_esas = df_ano[df_ano['Codigo'].isin(cajas_list)]
    pct_cajas = len(cajas_esas) * 100.0 / total_filas
    
    print('  OK: {0:,} transacciones, {1:,} productos, cajas {2:.0f}%'.format(total_filas, productos_unicos, int(pct_cajas)))

print('')
print('='*70)
print('LISTO - 6 archivos generados')
print('='*70)
print('  Datos_2020.csv')
print('  Datos_2021.csv')
print('  Datos_2022.csv')
print('  Datos_2023.csv')
print('  Datos_2024.csv')
print('  Datos_2025.csv')
print('')
print('Cada archivo contiene 50-100 transacciones por dia')
print('65% de las transacciones son de productos CAJA')
print('Esto genera historial denso y realista para ML/forecasting')
