#!/usr/bin/env python3
"""
Fix remaining res.* references in dashboard.py
"""

# Read file
with open('src/ui/dashboard.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Reemplazos
replacements = [
    ('res.demand_monthly', 'res_demand'),
    ('res.movements', 'res_movements'),
    ('res.stock_monthly', 'res_stock'),
]

# Lineas a excluir (donde 'res' es variable local de pipeline.run())
exclude_lines = [
    'res = pipeline.run(',  # Define res
    'if res.movements.empty:',  # Check res
    'movements=res.movements,',  # Use res
    'demand_monthly=res.demand_monthly,',  # Use res
    'stock_monthly=res.stock_monthly,',  # Use res
    'res_movements = res.movements',  # Assign res
    'res_demand = res.demand_monthly',  # Assign res
    'res_stock = res.stock_monthly',  # Assign res
]

# Separar por líneas
lines = content.split('\n')

# Procesar cada línea
new_lines = []
for line in lines:
    # Verificar si es una línea a excluir
    if any(exclude in line for exclude in exclude_lines):
        new_lines.append(line)
    else:
        # Hacer reemplazos
        for old, new in replacements:
            if old in line:
                line = line.replace(old, new)
        new_lines.append(line)

# Escribir
new_content = '\n'.join(new_lines)
with open('src/ui/dashboard.py', 'w', encoding='utf-8') as f:
    f.write(new_content)

print("✅ Fixed all res.* references")
