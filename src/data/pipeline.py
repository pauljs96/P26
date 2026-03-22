"""Pipeline de datos del proyecto - Dataset v4.

Orquesta:
- Carga de CSV v4
- Limpieza y validación
- Construcción de demanda mensual (Tipo_movimiento='Venta')
- Construcción de stock mensual (último Stock_posterior del mes)
"""

from __future__ import annotations
import pandas as pd
from dataclasses import dataclass
import logging

from src.data.data_loader import DataLoader
from src.data.data_cleaner import DataCleaner
from src.data.guide_reconciliation import GuideReconciler
from src.data.demand_builder import DemandBuilder
from src.data.ProductStockBuilder import ProductStockBuilder
from src.utils.logger import Logger

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    movements: pd.DataFrame
    demand_monthly: pd.DataFrame
    stock_monthly: pd.DataFrame
    error_message: str = None


class DataPipeline:
    def __init__(self, logger_obj: Logger | None = None):
        self.logger = logger_obj or Logger(enabled=False)
        self.loader = DataLoader()
        self.cleaner = DataCleaner()
        self.reconciler = GuideReconciler()  # Pass-through para v4
        self.demand_builder = DemandBuilder()
        self.stock_builder = ProductStockBuilder()
        

    def run(self, uploaded_files) -> PipelineResult:
        """Ejecuta el pipeline completo.
        
        Entrada: Archivos CSV subidos (o lista de archivos)
        Salida: PipelineResult con movements, demand_monthly, stock_monthly
        """
        try:
            self.logger.info("=== Pipeline v4 Iniciado ===")
            
            self.logger.info("1. Cargando CSV...")
            raw = self.loader.load_files(uploaded_files)
            self.logger.info(f"   ✓ {len(raw)} filas cargadas, {len(raw.columns)} columnas")

            self.logger.info("2. Limpiando y validando tipos...")
            try:
                clean = self.cleaner.clean(raw)
            except ValueError as e:
                error_msg = f"Error en limpieza: {str(e)}"
                self.logger.error(error_msg)
                return PipelineResult(
                    pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
                    error_message=error_msg
                )
            
            if clean.empty:
                error_msg = "DataFrame vacío tras limpieza"
                self.logger.error(error_msg)
                return PipelineResult(
                    pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
                    error_message=error_msg
                )
            
            self.logger.info(f"   ✓ {len(clean)} filas limpias")

            # GuideReconciler es pass-through para v4 (sin transformación)
            self.logger.info("3. Validación de datos (pass-through para v4)...")
            rec = self.reconciler.reconcile(clean)
            self.logger.info(f"   ✓ {len(rec)} filas validadas")

            self.logger.info("4. Construyendo demanda mensual...")
            demand = self.demand_builder.build_monthly(rec)
            self.logger.info(f"   ✓ {len(demand)} registros de demanda")

            self.logger.info("5. Construyendo stock mensual...")
            stock = self.stock_builder.build_monthly(rec)
            self.logger.info(f"   ✓ {len(stock)} registros de stock")

            # Información de rango de fechas
            if not rec.empty:
                min_date = rec["Fecha"].min()
                max_date = rec["Fecha"].max()
                self.logger.info(f"   Período: {min_date.strftime('%Y-%m-%d')} a {max_date.strftime('%Y-%m-%d')}")
                self.logger.info(f"   Productos: {rec['Producto_id'].nunique()}")
                self.logger.info(f"   Movimientos: {len(rec)}")

            self.logger.info("=== Pipeline completado exitosamente ===")
            
            return PipelineResult(rec, demand, stock)
        
        except Exception as e:
            error_msg = f"Error en pipeline: {str(e)}"
            self.logger.error(error_msg)
            import traceback
            self.logger.error(traceback.format_exc())
            return PipelineResult(
                pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
                error_message=error_msg
            )
