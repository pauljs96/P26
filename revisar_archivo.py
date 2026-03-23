import pandas as pd

# Leer el archivo con diferentes codificaciones
try:
    df = pd.read_csv('Movimientos_MayorAuxiliar_2025.csv', sep=';', encoding='utf-8')
except:
    try:
        df = pd.read_csv('Movimientos_MayorAuxiliar_2025.csv', sep=';', encoding='latin-1')
    except:
        df = pd.read_csv('Movimientos_MayorAuxiliar_2025.csv', sep=';', encoding='iso-8859-1')

print("Columnas encontradas:")
print(df.columns.tolist())
print("\nPrimeras líneas:")
print(df.head(10))
