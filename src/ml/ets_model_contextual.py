"""ETS contextual con capacidad de estratificación por negocio.

Extiende ETSForecaster para:
- Usar ETS agregado (general) o estratificado por Canal/Campaña
- Auto-detectar dimensión con más datos
- Fallback a univariante si hay poco datos
"""

from __future__ import annotations
import pandas as pd
import numpy as np

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
except Exception:
    ExponentialSmoothing = None

from src.ml.baselines import naive_last, seasonal_naive_12


class ETSForecasterContextual:
    """
    Exponential Smoothing mejorado con capacidad de estratificación.
    
    - ETS agregado (default)
    - ETS por canal (si hay datos suficientes y disponible columna)
    - ETS por campaña (si hay datos suficientes y disponible columna)
    - Fallback automático a univariante
    """

    def __init__(
        self,
        seasonal_periods: int = 12,
        trend: str | None = "add",
        seasonal: str | None = "add",
        damped_trend: bool = False,
        min_obs: int = 24,
        stratify_by: str | None = None,  # "channel", "campaign", "client", None
        min_obs_stratum: int = 12,  # Min obs por estrato
    ):
        self.seasonal_periods = seasonal_periods
        self.trend = trend
        self.seasonal = seasonal
        self.damped_trend = damped_trend
        self.min_obs = min_obs
        self.stratify_by = stratify_by  # Dimensión para estratificar
        self.min_obs_stratum = min_obs_stratum
        self.model_used = None

    def forecast_1step(
        self, 
        history: pd.DataFrame, 
        y_col: str = "Demanda_Unid"
    ) -> float:
        """
        Predice t+1 usando ETS (estratificado o agregado).
        
        Args:
            history: DataFrame con histórico
            y_col: Columna de target (demanda)
        
        Returns:
            Predicción como float positivo
        """
        if history is None or history.empty:
            self.model_used = "empty"
            return 0.0

        # Fallback si statsmodels no disponible
        if ExponentialSmoothing is None:
            self.model_used = "fallback_no_statsmodels"
            return naive_last(history, y_col=y_col)

        h = history.copy().sort_values("Mes")
        y = pd.to_numeric(h[y_col], errors="coerce").fillna(0.0).astype(float).to_numpy()

        if len(y) < self.min_obs:
            self.model_used = "fallback_small_data"
            return naive_last(history, y_col=y_col)

        if np.allclose(y, 0.0):
            self.model_used = "all_zeros"
            return 0.0

        # === INTENTAR ESTRATIFICACIÓN (si está disponible dimensión) ===
        stratum_col = None
        if self.stratify_by == "channel" and "Canal_venta" in h.columns:
            stratum_col = "Canal_venta"
        elif self.stratify_by == "campaign" and "Campana" in h.columns:
            stratum_col = "Campana"
        elif self.stratify_by == "client" and "Empresa_cliente" in h.columns:
            stratum_col = "Empresa_cliente"

        if stratum_col is not None:
            try:
                return self._forecast_stratified(h, y_col, stratum_col)
            except Exception as e:
                print(f"⚠️  ETS estratificado falló: {str(e)[:50]}. Usando agregado...")

        # === ETS AGREGADO (default) ===
        try:
            model = ExponentialSmoothing(
                y,
                error="add",
                trend=self.trend,
                seasonal=self.seasonal,
                seasonal_periods=self.seasonal_periods,
                damped_trend=self.damped_trend,
            )
            fitted = model.fit(optimized=True, disp=False)
            yhat = float(fitted.forecast(steps=1).iloc[0])

            if not np.isfinite(yhat):
                raise ValueError("Predicción no-finita")

            self.model_used = "ets_agregado"
            return float(max(0.0, yhat))

        except Exception as e:
            # Fallback a baseline
            print(f"⚠️ ETS agregado falló: {str(e)[:50]}")
            self.model_used = "fallback_baseline"
            try:
                return float(seasonal_naive_12(history, y_col=y_col))
            except Exception:
                return float(naive_last(history, y_col=y_col))

    def _forecast_stratified(
        self,
        history: pd.DataFrame,
        y_col: str,
        stratum_col: str,
    ) -> float:
        """
        ETS estratificado por dimensión (canal, campaña, etc).
        
        Agrupa por estrato, predice dentro de cada uno, combina.
        """
        grouped = history.groupby(stratum_col)
        forecasts = []

        last_mes = pd.to_datetime(history["Mes"].max())

        for stratum_val, group in grouped:
            group_sorted = group.sort_values("Mes")
            y_stratum = pd.to_numeric(
                group_sorted[y_col], errors="coerce"
            ).fillna(0.0).astype(float).to_numpy()

            # Skip si muy pocos datos en este estrato
            if len(y_stratum) < self.min_obs_stratum:
                continue

            # Skip si todo ceros
            if np.allclose(y_stratum, 0.0):
                continue

            try:
                model = ExponentialSmoothing(
                    y_stratum,
                    error="add",
                    trend=self.trend,
                    seasonal=self.seasonal,
                    seasonal_periods=min(self.seasonal_periods, len(y_stratum) // 2),
                    damped_trend=self.damped_trend,
                )
                fitted = model.fit(optimized=True, disp=False)
                yhat = float(fitted.forecast(steps=1).iloc[0])

                if np.isfinite(yhat) and yhat >= 0:
                    forecasts.append(yhat)

            except Exception:
                # Skip stratum problemático
                continue

        if not forecasts:
            raise ValueError("No estratos con predicciones válidas")

        # Combinar predicciones (media ponderada por tamaño grupo)
        combined = float(np.mean(forecasts))
        self.model_used = f"ets_estratificado_{stratum_col}"
        return float(max(0.0, combined))

    def get_model_info(self) -> dict:
        """Retorna información del modelo usado en última predicción."""
        return {
            "model_type": self.model_used or "unknown",
            "seasonal_periods": self.seasonal_periods,
            "trend": self.trend,
            "seasonal": self.seasonal,
        }
