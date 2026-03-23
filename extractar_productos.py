import pandas as pd
import re

# Leer el archivo - skipea la primera fila y usa la segunda como header
df = pd.read_csv('Movimientos_MayorAuxiliar_2025.csv', sep=';', encoding='latin-1', skiprows=1)

# Extraer productos únicos
print("Extrayendo productos del archivo...")

# Columna de código es probablemente la primera
codigo_col = df.columns[0]  # Primera columna
desc_col = df.columns[1]     # Segunda columna

df_clean = df.dropna(subset=[codigo_col])

# Tomar duplicados
productos_unicos = df_clean[[codigo_col, desc_col]].drop_duplicates()

print(f"Total de productos únicos encontrados: {len(productos_unicos)}")

# Tomar ~200 productos
productos_seleccionados = productos_unicos.head(200).reset_index(drop=True)
productos_seleccionados.columns = ['codigo', 'descripcion']

print(f"Productos seleccionados: {len(productos_seleccionados)}")

# Mostrar algunos ejemplos
print("\nEjemplos de productos:")
for idx in [0, 10, 50, 100, 150, 199]:
    if idx < len(productos_seleccionados):
        code = str(productos_seleccionados.iloc[idx]['codigo']).strip().strip("'")
        desc = str(productos_seleccionados.iloc[idx]['descripcion'])[:60]
        print(f"  {code}: {desc}")

# Guardar para usar en el generador
productos_seleccionados.to_csv('productos_seleccionados.csv', index=False)
print(f"\n✓ Productos guardados en productos_seleccionados.csv")
