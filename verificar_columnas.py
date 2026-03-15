import pandas as pd

# Ver qué columnas tenemos
df = pd.read_csv('DatosLimpios_2020.csv', sep=';', decimal=',', nrows=1)
print('COLUMNAS EN DatosLimpios_2020.csv:')
print('='*70)
for i, col in enumerate(df.columns, 1):
    print('{0:2d}. {1}'.format(i, col))

print('\n' + '='*70)
print('VERIFICACION:')
print('='*70)
print('Total columnas: {0}'.format(len(df.columns)))
print('Archivo: DatosLimpios_2020.csv')
print('Sep: ;')
print('Decimales: ,')

# Ver primeros datos
print('\n' + '='*70)
print('PRIMERAS FILAS:')
print('='*70)
print(df.head(2).to_string())
