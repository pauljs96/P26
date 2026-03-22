#!/usr/bin/env python3
"""Split v4 dataset by year for Streamlit Cloud"""

import pandas as pd
import os

csv_path = 'Inventario_ML_Completo_v4.csv'
print(f'Leyendo {csv_path}...')
df = pd.read_csv(csv_path, dtype=str)
print(f'OK: {len(df)} filas')

# Convertir Fecha a datetime para filtrar por año
df['Fecha'] = pd.to_datetime(df['Fecha'])

# Dividir por año
for year in [2022, 2023, 2024, 2025]:
    year_df = df[df['Fecha'].dt.year == year].copy()
    out_file = f'Inventario_v4_{year}.csv'
    
    # Convertir Fecha de vuelta a string para guardar
    year_df['Fecha'] = year_df['Fecha'].dt.strftime('%Y-%m-%d')
    
    year_df.to_csv(out_file, index=False)
    
    size_mb = os.path.getsize(out_file) / 1024 / 1024
    print(f'{out_file}: {len(year_df):,} filas ({size_mb:.1f}MB)')

print('\nOK! Archivos listos para subir al dashboard')
