"""Orchestrador de ML - Elige mejor modelo según datos disponibles.

Integra:
- Modelos univariantes (legacy)
- Modelos contextuales (RF con features de negocio)
- Conjunto de modelos con ponderación automática
"""

from __future__ import annotations
import pandas as pd
import numpy as np


class MLOrchestrator:
    """
    Selecciona y ejecuta el mejor modelo según disponibilidad de datos.
    
    Lógica:
    1. Si hay features contextuales (Canal, Campaña, Cliente, etc) → RF Contextual
    2. Si datos v4 (univariante) → RF Univariante + ETS
    3. Ensemble ponderado si ambos disponibles
    """

    def __init__(
        self,
        use_rf_contextual: bool = True,
        use_rf_univariante: bool = True,
        use_ets: bool = True,
        ensemble_weights: dict | None = None,
    ):
        """
        Args:
            use_rf_contextual: Usar RF mejorado con features de negocio
            use_rf_univariante: Usar RF base (univariante)
            use_ets: Usar Exponential Smoothing
            ensemble_weights: Pesos para ensemble {rf_contextual, rf_univariante, ets}
        """
        self.use_rf_contextual = use_rf_contextual
        self.use_rf_univariante = use_rf_univariante
        self.use_ets = use_ets
        
        # Pesos por defecto: RF contextual > RF univariante > ETS
        self.ensemble_weights = ensemble_weights or {
            "rf_contextual": 0.5,
            "rf_univariante": 0.3,
            "ets": 0.2,
        }
        
        self._normalize_weights()

    def _normalize_weights(self):
        """Normaliza los pesos para que sumen 1."""
        total = sum(self.ensemble_weights.values())
        if total > 0:
            for key in self.ensemble_weights:
                self.ensemble_weights[key] /= total

    def _detect_data_type(self, history: pd.DataFrame) -> str:
        """Detecta tipo de datos: 'contextual' o 'univariante'."""
        contextual_cols = ["Canal_venta", "Campana", "Empresa_cliente", "Producto_id"]
        has_contextual = any(col in history.columns for col in contextual_cols)
        return "contextual" if has_contextual else "univariante"

    def forecast_1step(
        self,
        history: pd.DataFrame,
        y_col: str = "Demanda_Unid",
        method: str = "auto",  # "auto", "contextual", "univariante", "ensemble"
    ) -> tuple[float, dict]:
        """
        Predice t+1 con lógica de selección automática.
        
        Args:
            history: DataFrame histórico
            y_col: Columna de target
            method: 'auto' = selecciona mejor según datos
                   'contextual' = fuerza RF contextual
                   'univariante' = fuerza RF univariante
                   'ensemble' = combina todos
        
        Returns:
            (predicción, diccionario de información del modelo)
        """
        if history is None or history.empty:
            return 0.0, {"method": "empty", "confidence": 0.0}

        data_type = self._detect_data_type(history)

        # === AUTO: Selecciona según disponibilidad ===
        if method == "auto":
            if data_type == "contextual" and self.use_rf_contextual:
                return self._forecast_rf_contextual(history, y_col)
            elif self.use_rf_univariante:
                return self._forecast_rf_univariante(history, y_col)
            elif self.use_ets:
                return self._forecast_ets(history, y_col)
            else:
                return 0.0, {"method": "no_models_available", "confidence": 0.0}

        # === CONTEXTUAL: Fuerza RF contextual ===
        elif method == "contextual":
            if self.use_rf_contextual:
                return self._forecast_rf_contextual(history, y_col)
            else:
                return self._forecast_rf_univariante(history, y_col)

        # === UNIVARIANTE: Fuerza RF univariante ===
        elif method == "univariante":
            if self.use_rf_univariante:
                return self._forecast_rf_univariante(history, y_col)
            elif self.use_ets:
                return self._forecast_ets(history, y_col)
            else:
                return self._forecast_rf_contextual(history, y_col)

        # === ENSEMBLE: Combina todos los disponibles ===
        elif method == "ensemble":
            return self._forecast_ensemble(history, y_col)

        else:
            return 0.0, {"method": "unknown", "confidence": 0.0}

    def _forecast_rf_contextual(
        self,
        history: pd.DataFrame,
        y_col: str,
    ) -> tuple[float, dict]:
        """RF con features contextuales."""
        try:
            from src.ml.rf_model_contextual import RFForecasterContextual

            model = RFForecasterContextual(n_estimators=500, min_samples_leaf=2)
            yhat = model.forecast_1step(history, y_col=y_col)
            
            return float(max(0.0, yhat)), {
                "method": model.model_used or "rf_contextual",
                "confidence": 0.9 if model.model_used == "rf_contextual" else 0.6,
                "model_info": model.get_model_info(),
            }
        except Exception as e:
            print(f"⚠️ RF Contextual: {str(e)[:50]}")
            return 0.0, {"method": "rf_contextual_error", "confidence": 0.0}

    def _forecast_rf_univariante(
        self,
        history: pd.DataFrame,
        y_col: str,
    ) -> tuple[float, dict]:
        """RF sin features contextuales (modelo base)."""
        try:
            from src.ml.rf_model import RFForecaster

            model = RFForecaster(n_estimators=400, min_samples_leaf=1)
            yhat = model.forecast_1step(history, y_col=y_col)
            
            return float(max(0.0, yhat)), {
                "method": "rf_univariante",
                "confidence": 0.75,
                "n_estimators": 400,
            }
        except Exception as e:
            print(f"⚠️ RF Univariante: {str(e)[:50]}")
            return 0.0, {"method": "rf_univariante_error", "confidence": 0.0}

    def _forecast_ets(
        self,
        history: pd.DataFrame,
        y_col: str,
    ) -> tuple[float, dict]:
        """ETS agregado (Holt-Winters)."""
        try:
            from src.ml.ets_model_contextual import ETSForecasterContextual

            model = ETSForecasterContextual()
            yhat = model.forecast_1step(history, y_col=y_col)
            
            return float(max(0.0, yhat)), {
                "method": model.model_used or "ets",
                "confidence": 0.7 if model.model_used in ["ets_agregado", "ets_estratificado_Canal_venta"] else 0.4,
                "model_info": model.get_model_info(),
            }
        except Exception as e:
            print(f"⚠️ ETS: {str(e)[:50]}")
            return 0.0, {"method": "ets_error", "confidence": 0.0}

    def _forecast_ensemble(
        self,
        history: pd.DataFrame,
        y_col: str,
    ) -> tuple[float, dict]:
        """Ensemble ponderado de RF contextiual + RF univariante + ETS."""
        predictions = []
        confidences = []
        methods_used = []

        # RF Contextual
        if self.use_rf_contextual:
            yhat, info = self._forecast_rf_contextual(history, y_col)
            if yhat > 0 or np.isfinite(yhat):
                predictions.append(yhat)
                confidences.append(info.get("confidence", 0.5))
                methods_used.append(info.get("method", "rf_contextual"))

        # RF Univariante
        if self.use_rf_univariante:
            yhat, info = self._forecast_rf_univariante(history, y_col)
            if yhat > 0 or np.isfinite(yhat):
                predictions.append(yhat)
                confidences.append(info.get("confidence", 0.5))
                methods_used.append(info.get("method", "rf_univariante"))

        # ETS
        if self.use_ets:
            yhat, info = self._forecast_ets(history, y_col)
            if yhat > 0 or np.isfinite(yhat):
                predictions.append(yhat)
                confidences.append(info.get("confidence", 0.5))
                methods_used.append(info.get("method", "ets"))

        if not predictions:
            return 0.0, {"method": "ensemble_all_failed", "confidence": 0.0}

        # Combinar por promedio ponderado (pesas más las predicciones con mayor confianza)
        confidences = np.array(confidences)
        weights = confidences / confidences.sum()
        yhat_ensemble = float(np.average(predictions, weights=weights))

        avg_confidence = float(np.mean(confidences))

        return float(max(0.0, yhat_ensemble)), {
            "method": "ensemble",
            "confidence": min(0.95, avg_confidence),
            "n_models": len(predictions),
            "methods_used": methods_used,
            "weights": dict(zip(methods_used, weights.tolist())),
            "predictions": dict(zip(methods_used, [float(p) for p in predictions])),
        }
