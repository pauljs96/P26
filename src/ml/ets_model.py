from __future__ import annotations
import pandas as pd
import numpy as np

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
except Exception:  # pragma: no cover
    ExponentialSmoothing = None

from src.ml.baselines import naive_last


class ETSForecaster:
    """
    Holt–Winters / ETS para demanda mensual (univariante).

    - Entrena con historia hasta t
    - Predice t+1 (por defecto)
    - Si hay poca data o falla el ajuste, usa fallback (naive_last)
    """

    def __init__(
        self,
        seasonal_periods: int = 12,
        trend: str | None = "add",
        seasonal: str | None = "add",
        damped_trend: bool = False,
        min_obs: int = 24,
    ):
        self.seasonal_periods = seasonal_periods
        self.trend = trend
        self.seasonal = seasonal
        self.damped_trend = damped_trend
        self.min_obs = min_obs

    def forecast_1step(self, history: pd.DataFrame, y_col: str = "Demanda_Unid") -> float:
        """
        Forecast t+1 usando Holt–Winters (ETS). Devuelve un float.
        """
        if history is None or history.empty:
            return 0.0

        # Fallback si statsmodels no está disponible
        if ExponentialSmoothing is None:
            return naive_last(history, y_col=y_col)

        h = history.copy().sort_values("Mes")
        y = pd.to_numeric(h[y_col], errors="coerce").fillna(0.0).astype(float).to_numpy()

        # Reglas mínimas para intentar estacionalidad
        # - Para seasonal_periods=12, una regla práctica es tener >= 2 ciclos (~24 obs)
        if len(y) < self.min_obs:
            return naive_last(history, y_col=y_col)

        # Si la serie es casi todo ceros, ETS puede volverse inestable: fallback simple
        if np.allclose(y, 0.0):
            return 0.0

        try:
            model = ExponentialSmoothing(
                y,
                trend=self.trend,
                damped_trend=self.damped_trend,
                seasonal=self.seasonal,
                seasonal_periods=self.seasonal_periods,
                initialization_method="estimated",
            )
            fit = model.fit(optimized=True)
            yhat = float(fit.forecast(1)[0])

            # ETS puede dar negativos; en demanda no tiene sentido -> truncar
            if not np.isfinite(yhat):
                return naive_last(history, y_col=y_col)
            return float(max(0.0, yhat))

        except Exception:
            # robustez: si falla el fit, volvemos al baseline
            return naive_last(history, y_col=y_col)
