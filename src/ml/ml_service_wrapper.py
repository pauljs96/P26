"""Wrapper de compatibilidad para transición a nuevo ML.

Permite usar MLOrchestrator como drop-in replacement de ETSForecaster/RFForecaster,
manteniendo compatibilidad con código existente mientras se mejora performance.
"""

from __future__ import annotations
import pandas as pd


class MLServiceWrapper:
    """
    Wrapper que encapsula el Orchestrator y proporciona interfaz compatible
    con ETSForecaster y RFForecaster.
    
    Uso:
        wrapper = MLServiceWrapper(method="ensemble")
        yhat = wrapper.forecast_demand(history_df, product_id)
        info = wrapper.last_forecast_info
    """

    def __init__(self, method: str = "auto"):
        """
        Args:
            method: "auto" (default), "contextual", "univariante", "ensemble"
        """
        from src.ml.ml_orchestrator import MLOrchestrator
        
        self.orchestrator = MLOrchestrator(
            use_rf_contextual=True,
            use_rf_univariante=True,
            use_ets=True,
        )
        self.method = method
        self.last_forecast_info = {}

    def forecast_demand(
        self,
        history: pd.DataFrame,
        y_col: str = "Demanda_Unid",
        product_id: str | None = None,
    ) -> float:
        """
        Pronostica demanda para t+1.
        
        Args:
            history: DataFrame con histórico (puede incluir columnas de contexto)
            y_col: Columna de demanda (default "Demanda_Unid")
            product_id: Opcional, para logging
        
        Returns:
            Predicción como float (>= 0)
        """
        yhat, info = self.orchestrator.forecast_1step(
            history, 
            y_col=y_col, 
            method=self.method
        )
        
        self.last_forecast_info = {
            "product_id": product_id,
            "method": self.method,
            **info
        }
        
        return float(max(0.0, yhat))

    def forecast_all_products(
        self,
        pipeline_result,
        method: str | None = None,
    ) -> dict[str, float]:
        """
        Pronostica demanda para todos los productos en el pipeline.
        
        Args:
            pipeline_result: Resultado del pipeline con demand_monthly
            method: Opcionalmente override del método (default: usa self.method)
        
        Returns:
            Dict {product_id: predicted_demand}
        """
        method = method or self.method
        forecasts = {}

        if not hasattr(pipeline_result, 'demand_monthly') or pipeline_result.demand_monthly is None:
            return forecasts

        demand_df = pipeline_result.demand_monthly
        
        for product_id in demand_df["Producto_id"].unique():
            hist = demand_df[demand_df["Producto_id"] == product_id].copy()
            
            if hist.empty:
                continue

            try:
                yhat = self.forecast_demand(hist, product_id=product_id)
                forecasts[product_id] = yhat
            except Exception as e:
                print(f"⚠️ Forecast failed for {product_id}: {str(e)[:50]}")
                forecasts[product_id] = 0.0

        return forecasts

    def get_performance_info(self) -> dict:
        """Retorna información de desempeño de último forecast."""
        return self.last_forecast_info

    def get_ensemble_breakdown(self) -> dict:
        """Retorna desglose de ensemble si fue usado."""
        info = self.last_forecast_info
        if info.get("method") == "ensemble":
            return {
                "n_models": info.get("n_models", 0),
                "methods_used": info.get("methods_used", []),
                "predictions": info.get("predictions", {}),
                "weights": info.get("weights", {}),
                "confidence": info.get("confidence", 0.0),
            }
        return {}
