"""Configuración y constantes del proyecto.

Centraliza:
- Lectura de CSV (separadores/encodings)
- Nombres de documentos relevantes
- Tolerancias para reconciliación de guías
- Candidatos de nombres de columnas (por si cambian en el ERP)
"""

# Separadores y encodings típicos en exportaciones ERP
CSV_SEPARATORS = [";", ",", "\t", "|"]
CSV_ENCODINGS = ["utf-8", "latin-1"]

# -----------------------------
# Documentos (texto exacto esperado, con strip() aplicado)
# -----------------------------
DOC_VENTA_TIENDA = "Venta Tienda Sin Doc"
DOC_SALIDA_CONSUMO = "Salida por Consumo"

# Documentos que representan demanda directa (consumo/venta real)
DEMAND_DIRECT_DOCS = [DOC_VENTA_TIENDA, DOC_SALIDA_CONSUMO]

# Documento guía (puede ser transferencia interna o venta externa)
GUIDE_DOC = "Guía de remisión - R"

# -----------------------------
# Reconciliación de guías
# -----------------------------
# Tolerancia absoluta (unidades) y relativa (porcentaje) para considerar S≈E
GUIDE_TOL_ABS = 0.5
GUIDE_TOL_REL = 0.02

# -----------------------------
# Candidatos de nombres de columnas (por si el ERP cambia mayúsculas/acentos)
# -----------------------------
COL_CODIGO = ["Código", "Codigo", "CODIGO", "CÓDIGO"]
COL_DESCRIPCION = ["Descripción", "Descripcion", "DESCRIPCIÓN", "DESCRIPCION"]
COL_FECHA = ["Fecha", "FECHA", "Date", "DATE"]
COL_DOCUMENTO = ["Documento", "DOCUMENTO"]
COL_NUMERO = ["Número", "Numero", "NÚMERO", "NUMERO"]
COL_BODEGA = ["Bodega", "BODEGA"]

# Unidades (físicas)
COL_ENTRADA_UNID = ["Entrada"]
COL_SALIDA_UNID = ["Salida"]
COL_SALDO_UNID = ["Saldo"]

# Valores monetarios / costos
COL_VALOR_UNIT = ["Valor Unitario", "ValorUnitario", "VALOR UNITARIO"]
COL_COSTO_UNIT = ["Costo Unitario", "CostoUnitario", "COSTO UNITARIO"]

# Si existen duplicadas por monto (Entrada/Salida/Saldo repetidas), el loader renombra con __2
COL_ENTRADA_MONTO = ["Entrada__2", "Entrada (monto)", "Entrada_monto"]
COL_SALIDA_MONTO = ["Salida__2", "Salida (monto)", "Salida_monto"]
COL_SALDO_MONTO = ["Saldo__2", "Saldo (monto)", "Saldo_monto"]
