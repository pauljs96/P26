#!/usr/bin/env python3
"""
Test de pipeline v4 con Inventario_ML_Completo_v4.csv

Este script carga el dataset v4 y ejecuta el pipeline completo,
mostrando estadísticas de entrada y salida.
"""

import sys
import pandas as pd
import logging
from pathlib import Path

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.pipeline import DataPipeline
from src.utils.logger import Logger


def main():
    # Rutas
    csv_path = Path(__file__).parent / "Inventario_ML_Completo_v4.csv"
    
    if not csv_path.exists():
        logger.error(f"❌ Archivo no encontrado: {csv_path}")
        sys.exit(1)
    
    logger.info(f"📁 Usando dataset: {csv_path}")
    
    # Leer CSV v4
    logger.info("Cargando CSV v4...")
    try:
        df = pd.read_csv(csv_path, sep=",", encoding="utf-8", dtype=str)
        logger.info(f"✓ {len(df)} filas, {len(df.columns)} columnas cargadas")
        logger.info(f"  Columnas: {list(df.columns)}")
    except Exception as e:
        logger.error(f"❌ Error al cargar CSV: {e}")
        sys.exit(1)
    
    # Crear datos de entrada simulando upload
    # Para el pipeline, creamos una lista con un objeto que tiene getvalue()
    class MockFile:
        def __init__(self, df_raw):
            self.df = df_raw
            self.name = csv_path.name
        
        def getvalue(self):
            # Convertir DataFrame a bytes CSV
            return self.df.to_csv(index=False).encode("utf-8")
    
    mock_file = MockFile(df)
    
    # Ejecutar pipeline
    logger.info("=== Iniciando Pipeline v4 ===")
    pipeline = DataPipeline(logger_obj=Logger(enabled=True))
    result = pipeline.run([mock_file])
    
    # Verificar errores
    if result.error_message:
        logger.error(f"❌ Pipeline falló: {result.error_message}")
        sys.exit(1)
    
    # Mostrar resultados
    logger.info("\n=== RESULTADOS DEL PIPELINE ===")
    
    movements = result.movements
    demand = result.demand_monthly
    stock = result.stock_monthly
    
    logger.info(f"\n1. MOVIMIENTOS (limpiados y validados)")
    logger.info(f"   Total: {len(movements)} filas")
    if not movements.empty:
        logger.info(f"   Período: {movements['Fecha'].min()} a {movements['Fecha'].max()}")
        logger.info(f"   Productos: {movements['Producto_id'].nunique()}")
        logger.info(f"   Tipos: {movements['Tipo_movimiento'].unique().tolist()}")
        logger.info(f"   Cantidad total vendida: {movements[movements['Tipo_movimiento']=='Venta']['Cantidad'].sum():.0f} unidades")
        logger.info(f"   Cantidad total producida: {movements[movements['Tipo_movimiento']=='Producción']['Cantidad'].sum():.0f} unidades")
    
    logger.info(f"\n2. DEMANDA MENSUAL")
    logger.info(f"   Total: {len(demand)} registros (producto-mes)")
    if not demand.empty:
        logger.info(f"   Productos cubiertos: {demand['Producto_id'].nunique()}")
        logger.info(f"   Meses: {demand['Año'].nunique() * 12} períodos (aprox)")
        logger.info(f"   Demanda total: {demand['Cantidad_total'].sum():.0f} unidades")
        logger.info(f"   Demanda promedio/mes: {demand['Cantidad_total'].mean():.2f} unidades")
        
        # Top 5 productos por demanda
        top5 = demand.groupby('Producto_id')['Cantidad_total'].sum().nlargest(5)
        logger.info(f"   Top 5 productos por demanda:")
        for prod_id, cant in top5.items():
            logger.info(f"     - {prod_id}: {cant:.0f} unidades")
    
    logger.info(f"\n3. STOCK MENSUAL")
    logger.info(f"   Total: {len(stock)} registros (producto-mes)")
    if not stock.empty:
        logger.info(f"   Productos cubiertos: {stock['Producto_id'].nunique()}")
        logger.info(f"   Stock promedio final/mes: {stock['Stock_final'].mean():.2f} unidades")
        logger.info(f"   Stock máximo: {stock['Stock_final'].max():.0f} unidades")
        logger.info(f"   Stock mínimo: {stock['Stock_final'].min():.0f} unidades")
    
    logger.info("\n✅ Pipeline v4 completado exitosamente!")


if __name__ == "__main__":
    main()
