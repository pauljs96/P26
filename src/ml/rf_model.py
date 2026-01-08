from __future__ import annotations
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor

from src.ml.baselines import seasonal_naive_12, naive_last
from src.ml.rf_features import make_supervised_features, build_next_month_row


class RFForecaster:
    """
    Random Forest para forecasting univariante con features temporales.
    - Entrenamiento sobre dataset supervisado (lags/rolling/calendario)
    - Predicción t+1
    - Fallback si hay poca data: seasonal_naive_12 -> naive_last
    """

    def __init__(
        self,
        n_estimators: int = 400,
        random_state: int = 42,
        min_obs: int = 24,
        max_depth: int | None = None,
        min_samples_leaf: int = 1,
    ):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.min_obs = min_obs
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf

    def _make_model(self) -> RandomForestRegressor:
        return RandomForestRegressor(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
        )

    def forecast_1step(self, history: pd.DataFrame, y_col: str = "Demanda_Unid") -> float:
        if history is None or history.empty:
            return 0.0

        h = history.copy().sort_values("Mes")
        h[y_col] = pd.to_numeric(h[y_col], errors="coerce").fillna(0.0).astype(float)

        if len(h) < self.min_obs:
            # Fallback: primero seasonal, si no alcanza, naive
            try:
                return float(seasonal_naive_12(h, y_col=y_col))
            except Exception:
                return float(naive_last(h, y_col=y_col))

        last_mes = pd.to_datetime(h["Mes"].max()).to_period("M").to_timestamp()
        next_mes = last_mes + pd.offsets.MonthBegin(1)

        feats = make_supervised_features(h, y_col=y_col, date_col="Mes")
        # Quitamos filas donde falten lags clave (lag_12 típicamente)
        # En intermitencia puede haber NaN por no tener suficiente historia

        # Nos quedamos SOLO con columnas numéricas (robusto ante errores)
        candidate_cols = [c for c in feats.columns if c not in ["Mes", y_col, "y"]]

        numeric_cols = []
        for c in candidate_cols:
            if pd.api.types.is_numeric_dtype(feats[c]):
                numeric_cols.append(c)

        train = feats.dropna(subset=["y"]).copy()
        train[numeric_cols] = train[numeric_cols].fillna(0.0)

        X = train[numeric_cols].to_numpy(dtype=float)
        y = train["y"].to_numpy(dtype=float)


        # Entrenar
        model = self._make_model()
        model.fit(X, y)

        # Features para next month
        x_next = build_next_month_row(h, next_mes, y_col=y_col, date_col="Mes")
        x_next[numeric_cols] = x_next[numeric_cols].fillna(0.0)
        yhat = float(model.predict(x_next[numeric_cols].to_numpy(dtype=float))[0])


        if not np.isfinite(yhat):
            return float(seasonal_naive_12(h, y_col=y_col))

        return float(max(0.0, yhat))
