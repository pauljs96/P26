"""Pipeline de datos del proyecto.

Orquesta:
- carga de archivos
- limpieza
- reconciliación de guías
- construcción de demanda mensual
- construcción de stock mensual por bodega
"""

from __future__ import annotations
import pandas as pd
from dataclasses import dataclass

from src.data.data_loader import DataLoader
from src.data.data_cleaner import DataCleaner
from src.data.guide_reconciliation import GuideReconciler
from src.data.demand_builder import DemandBuilder
from src.data.ProductStockBuilder import ProductStockBuilder
from src.utils.logger import Logger
from src.data.series_completion import complete_monthly_demand


@dataclass
class PipelineResult:
    movements: pd.DataFrame
    demand_monthly: pd.DataFrame
    stock_monthly: pd.DataFrame


class DataPipeline:
    def __init__(self, logger: Logger | None = None):
        self.logger = logger or Logger(enabled=False)
        self.loader = DataLoader()
        self.cleaner = DataCleaner()
        self.reconciler = GuideReconciler()
        self.demand_builder = DemandBuilder()
        self.stock_builder = ProductStockBuilder()
        

    def run(self, uploaded_files) -> PipelineResult:
        self.logger.info("Cargando CSV...")
        raw = self.loader.load_files(uploaded_files)

        self.logger.info("Limpiando y tipando data...")
        clean = self.cleaner.clean(raw)
        if clean.empty:
            return PipelineResult(pd.DataFrame(), pd.DataFrame(), pd.DataFrame())

        self.logger.info("Reconciliando guías de remisión...")
        rec = self.reconciler.reconcile(clean)

        self.logger.info("Construyendo demanda mensual...")
        demand = self.demand_builder.build_monthly(rec)

        self.logger.info("Completando meses faltantes con 0 (serie uniforme)...")
        demand = complete_monthly_demand(demand, start="2021-01-01", end="2025-05-01")


        self.logger.info("Construyendo stock mensual EMPRESA (último saldo del mes)...")
        stock = self.stock_builder.build_monthly(rec)



        return PipelineResult(rec, demand, stock)
