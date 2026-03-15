import pandas as pd

print('Analizando nuevos datos...\n')

dfs = []
for ano in [2020, 2021, 2022, 2023, 2024, 2025]:
    archivo = 'Datos_Real_{0}.csv'.format(ano)
    df = pd.read_csv(archivo, sep=';', decimal=',')
    dfs.append(df)
    
    # Stats rapidas
    negativos = len(df[df['Saldo_unid'] < 0])
    pct_neg = negativos * 100.0 / len(df)
    min_saldo = df['Saldo_unid'].min()
    max_saldo = df['Saldo_unid'].max()
    
    entradas = df['Entrada_unid'].sum()
    salidas = df['Salida_unid'].sum()
    
    print('{0}: Neg {1:5.1f}% | Min {2:7.0f}, Max {3:7.0f} | Entradas {4:7,.0f}, Salidas {5:7,.0f}'.format(
        ano, pct_neg, min_saldo, max_saldo, entradas, salidas))

print('\n' + '='*70)
print('PROBLEMA Y SOLUCION')
print('='*70)
print('  Problema: Las salidas (75%) superan entradas (25%)')
print('  Resultado: Stock se agota con el tiempo')
print('')
print('  Solucion: Cambiar a 60% salidas / 40% entradas')
print('  Esto mantendra stock positivo la mayoria del tiempo')
