"""Pipeline de datos del proyecto - Dataset v4 y Data.csv.

Orquesta:
- Carga de CSV (auto-detección formato: v4 o Data.csv)
- Limpieza y validación
- Construcción de demanda mensual (Tipo_movimiento='Venta')
- Construcción de stock mensual (último Stock_posterior del mes)
- Análisis opcionales si existen columnas de Data.csv:
  - Campaign analysis (si Campana)
  - Channel performance (si Canal_venta)
  - Client segmentation (si Empresa_cliente)
  - Profit analysis (si Precio_unitario, Costo_unitario)
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
from src.data.campaign_builder import CampaignBuilder
from src.data.channel_builder import ChannelBuilder
from src.data.client_builder import ClientBuilder
from src.data.profit_builder import ProfitBuilder
from src.utils.logger import Logger

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    movements: pd.DataFrame
    demand_monthly: pd.DataFrame
    stock_monthly: pd.DataFrame
    demand_campaign: pd.DataFrame = None
    demand_channel: pd.DataFrame = None
    demand_client: pd.DataFrame = None
    profit_monthly: pd.DataFrame = None
    channel_performance: pd.DataFrame = None
    client_segmentation: pd.DataFrame = None
    error_message: str = None


class DataPipeline:
    def __init__(self, logger_obj: Logger | None = None):
        self.logger = logger_obj or Logger(enabled=False)
        self.loader = DataLoader()
        self.cleaner = DataCleaner()
        self.reconciler = GuideReconciler()  # Pass-through para v4
        self.demand_builder = DemandBuilder()
        self.stock_builder = ProductStockBuilder()
        
        # Builders adicionales para Data.csv (inicializados pero opcionales)
        self.campaign_builder = CampaignBuilder()
        self.channel_builder = ChannelBuilder()
        self.client_builder = ClientBuilder()
        self.profit_builder = ProfitBuilder()

    def run(self, uploaded_files) -> PipelineResult:
        """Ejecuta el pipeline completo.
        
        Entrada: Archivos CSV subidos (o lista de archivos)
        Salida: PipelineResult con movements, demand_monthly, stock_monthly, y análisis opcionales
        """
        try:
            self.logger.info("=== Pipeline Iniciado ===")
            
            self.logger.info("1. Cargando CSV...")
            raw = self.loader.load_files(uploaded_files)
            self.logger.info(f"   ✓ {len(raw)} filas cargadas, {len(raw.columns)} columnas")
            
            if self.loader.get_detected_format():
                self.logger.info(f"   Formato detectado: {self.loader.get_detected_format()}")

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

            # GuideReconciler es pass-through para v4
            self.logger.info("3. Validación de datos...")
            rec = self.reconciler.reconcile(clean)
            self.logger.info(f"   ✓ {len(rec)} filas validadas")

            self.logger.info("4. Construyendo demanda mensual...")
            demand = self.demand_builder.build_monthly(rec)
            self.logger.info(f"   ✓ {len(demand)} registros de demanda")

            self.logger.info("5. Construyendo stock mensual...")
            stock = self.stock_builder.build_monthly(rec)
            self.logger.info(f"   ✓ {len(stock)} registros de stock")

            # Análisis opcionales si existen columnas Data.csv
            demand_campaign = None
            demand_channel = None
            demand_client = None
            profit_monthly = None
            channel_perf = None
            client_seg = None

            if "Campana" in rec.columns:
                self.logger.info("6a. Analizando demanda por campaña...")
                demand_campaign = self.demand_builder.build_by_campaign(rec)
                self.logger.info(f"   ✓ {len(demand_campaign)} registros de campañas")

            if "Canal_venta" in rec.columns:
                self.logger.info("6b. Analizando demanda por canal...")
                demand_channel = self.demand_builder.build_by_channel(rec)
                self.logger.info(f"   ✓ {len(demand_channel)} registros de canales")
                
                self.logger.info("6c. Analizando performance de canales...")
                channel_perf = self.channel_builder.build_channel_performance(rec)
                self.logger.info(f"   ✓ {len(channel_perf)} canales analizados")

            if "Empresa_cliente" in rec.columns:
                self.logger.info("6d. Analizando demanda por cliente...")
                demand_client = self.demand_builder.build_by_client(rec)
                self.logger.info(f"   ✓ {len(demand_client)} registros de clientes")
                
                self.logger.info("6e. Analizando segmentación de clientes...")
                client_seg = self.client_builder.build_client_segmentation(rec)
                self.logger.info(f"   ✓ {len(client_seg)} clientes segmentados")

            if "Costo_unitario" in rec.columns and "Precio_unitario" in rec.columns:
                self.logger.info("6f. Analizando ganancias y márgenes...")
                profit_monthly = self.profit_builder.build_product_profit_monthly(rec)
                self.logger.info(f"   ✓ {len(profit_monthly)} registros de ganancias")

            # Información de rango de fechas
            if not rec.empty:
                min_date = rec["Fecha"].min()
                max_date = rec["Fecha"].max()
                self.logger.info(f"   Período: {min_date.strftime('%Y-%m-%d')} a {max_date.strftime('%Y-%m-%d')}")
                self.logger.info(f"   Productos: {rec['Producto_id'].nunique()}")
                self.logger.info(f"   Movimientos: {len(rec)}")

            self.logger.info("=== Pipeline completado exitosamente ===")
            
            return PipelineResult(
                rec, demand, stock,
                demand_campaign=demand_campaign,
                demand_channel=demand_channel,
                demand_client=demand_client,
                profit_monthly=profit_monthly,
                channel_performance=channel_perf,
                client_segmentation=client_seg
            )
        
        except Exception as e:
            error_msg = f"Error en pipeline: {str(e)}"
            self.logger.error(error_msg)
            import traceback
            self.logger.error(traceback.format_exc())
            return PipelineResult(
                pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
                error_message=error_msg
            )
