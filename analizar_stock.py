import pandas as pd
import numpy as np

print('Analizando stock negativo en los datos...\n')

# Cargar todos los datos
dfs = []
for ano in [2020, 2021, 2022, 2023, 2024, 2025]:
    archivo = 'Datos_{0}.csv'.format(ano)
    df = pd.read_csv(archivo, sep=';', decimal=',')
    dfs.append(df)

df_completo = pd.concat(dfs, ignore_index=True)

print('='*70)
print('ANALISIS DE STOCK')
print('='*70)

print('\nESTADISTICAS GENERALES:')
print('  Total de registros: {0:,}'.format(len(df_completo)))
print('  Saldo_unid minimo: {0:,.0f}'.format(df_completo['Saldo_unid'].min()))
print('  Saldo_unid maximo: {0:,.0f}'.format(df_completo['Saldo_unid'].max()))
print('  Saldo_unid promedio: {0:,.1f}'.format(df_completo['Saldo_unid'].mean()))

# Registros con stock negativo
negativos = df_completo[df_completo['Saldo_unid'] < 0]
print('\nREGISTROS CON STOCK NEGATIVO:')
print('  Total: {0:,} ({1:.2f}%)'.format(len(negativos), len(negativos)*100/len(df_completo)))

if len(negativos) > 0:
    print('  Rango: [{0:,.0f} a {1:,.0f}]'.format(negativos['Saldo_unid'].min(), negativos['Saldo_unid'].max()))
    print('  Promedio negativo: {0:,.1f}'.format(negativos['Saldo_unid'].mean()))
    print('  Mediana negativa: {0:,.1f}'.format(negativos['Saldo_unid'].median()))
    
    # Distribucion de negatividad
    muy_negativo = len(negativos[negativos['Saldo_unid'] < -100])
    moderado_negativo = len(negativos[(negativos['Saldo_unid'] >= -100) & (negativos['Saldo_unid'] < -10)])
    poco_negativo = len(negativos[negativos['Saldo_unid'] >= -10])
    
    print('\n  Distribucion por severidad:')
    print('    - Mayor a -100 unid (muy negativo): {0:,}'.format(muy_negativo))
    print('    - Entre -100 y -10 (moderado): {0:,}'.format(moderado_negativo))
    print('    - Entre -10 y 0 (poco negativo): {0:,}'.format(poco_negativo))

print('\nPRODUCTOS CON ALGUN STOCK NEGATIVO:')
productos_negativo = df_completo[df_completo['Saldo_unid'] < 0]['Codigo'].unique()
print('  Total productos: {0:,}'.format(len(productos_negativo)))
print('  % del catalogo (2,885 total): {0:.1f}%'.format(len(productos_negativo)*100/2885))

# Top 10 productos con menor stock
print('\nTOP 10 PEORES SALDOS (mas negativos):')
peores = df_completo.nsmallest(10, 'Saldo_unid')[['Codigo', 'Descripcion', 'Fecha', 'Saldo_unid']]
for i, row in peores.iterrows():
    print('  {0:15} : {1:7,.0f} unid ({2})'.format(str(row['Codigo']), row['Saldo_unid'], row['Fecha']))

# Analisis por bodega
print('\nSTOCK POR BODEGA:')
for bodega in df_completo['Bodega'].unique():
    df_bodega = df_completo[df_completo['Bodega'] == bodega]
    neg_bodega = len(df_bodega[df_bodega['Saldo_unid'] < 0])
    pct = neg_bodega * 100 / len(df_bodega)
    print('  {0:30} : {1:6.2f}% registros negativos ({2:,}/{3:,})'.format(bodega, pct, neg_bodega, len(df_bodega)))

print('\n' + '='*70)
print('CONCLUSION')
print('='*70)

if len(negativos) / len(df_completo) > 0.10:
    print('  ALTO: Mas del 10% de registros tienen stock negativo')
    print('  RECOMENDACION: Regenerar datos con stock inicial mas alto')
elif len(negativos) / len(df_completo) > 0.02:
    print('  MODERADO: Entre 2-10% negativos')
    print('  RECOMENDACION: Es aceptable (simula backorders reales)')
else:
    print('  BAJO: Muy pocos registros negativos (<2%)')
    print('  RECOMENDACION: Datos realistas')

print('\n')
