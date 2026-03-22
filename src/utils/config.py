"""Configuración y constantes del proyecto.

Centraliza:
- Lectura de CSV (separadores/encodings)
- Nombres de documentos relevantes
- Tolerancias para reconciliación de guías
- Candidatos de nombres de columnas (por si cambian en el ERP)
"""

# ============================================
# DATASET v4 - Inventario ML Completo
# ============================================
# Formato: CSV limpio, diario, con histórico 2022-2025
# 200+ productos, ~1.9M transacciones
# Separador: coma, Encoding: UTF-8

CSV_SEPARATORS = [","]
CSV_ENCODINGS = ["utf-8"]

# Tipos de movimiento (v4)
MOVEMENT_SALE = "Venta"
MOVEMENT_PRODUCTION = "Producción"
DEMAND_DIRECT_DOCS = [MOVEMENT_SALE]  # Solo ventas son demanda

# -----------------------------
# ============================================
# COLUMNAS ESPERADAS - DATASET v4
# ============================================

# Columnas requeridas (debe tener todas)
REQUIRED_COLUMNS = [
    "Fecha",
    "Producto_id",
    "Producto_nombre",
    "Tipo_movimiento",
    "Cantidad",
    "Stock_anterior",
    "Stock_posterior",
]

# Columnas opcionales (si existen, se usan para análisis)
OPTIONAL_COLUMNS = [
    "Año",
    "Mes",
    "Dia",
    "Empresa_cliente",
    "Departamento_cliente",
    "Canal_venta",
    "Punto_venta",
    "Precio_unitario",
    "Descuento_pct",
    "Valor_total",
    "Campana",
    "Costo_unitario",
]

# Tolerancia para validación de stock
STOCK_TOLERANCE = 0.01  # 1% tolerancia
