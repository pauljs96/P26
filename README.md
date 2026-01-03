# Sistema de Planificación (MVP) - Tesis

Este proyecto construye un pipeline reproducible para:
- Cargar múltiples CSV transaccionales de inventario (kardex) del ERP.
- Limpiar y tipar columnas.
- Reconciliar "Guía de remisión - R" para separar transferencias internas vs. ventas externas.
- Construir:
  - Demanda mensual real a nivel empresa (producto-mes)
  - Stock mensual por bodega (producto-bodega-mes)
- Visualizar resultados en un dashboard Streamlit.

## Ejecutar
```bash
pip install -r requirements.txt
streamlit run main.py
```

## Notas sobre el CSV del ERP
- El CSV puede traer una primera fila tipo "Periodo del;...".
- Los encabezados reales suelen venir en la segunda fila.
- Las columnas "Entrada/Salida/Saldo" pueden repetirse (unidades y montos). El loader desduplica nombres.
