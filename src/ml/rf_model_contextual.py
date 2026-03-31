"""Random Forest contextual con features de negocio.

Extiende RFForecaster para usar features de Canal, Campaña, Cliente, Producto.
- Automáticamente detecta columnas disponibles
- Fallback graceful a modelo univariante si faltan datos contextuales
- Mejor performance con datos completos de negocio
"""

from __future__ import annotations
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor

from src.ml.baselines import seasonal_naive_12, naive_last
from src.ml.ml_features_enhanced import (
    make_supervised_features_contextual,
    build_next_month_row_contextual,
)


class RFForecasterContextual:
    """
    Random Forest mejorado con features contextuales de negocio.
    
    - Usa características de Canal, Campaña, Cliente, Producto si disponibles
    - Fallback automático a modelo univariante si faltan datos contextuales
    - Compatible con datos v4 (univariante) y Data.csv (contextual)
    """

    def __init__(
        self,
        n_estimators: int = 500,
        random_state: int = 42,
        min_obs: int = 24,
        max_depth: int | None = None,
        min_samples_leaf: int = 2,
        include_client: bool = True,
        include_campaign: bool = True,
        include_channel: bool = True,
        include_product: bool = True,
    ):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.min_obs = min_obs
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        
        # Banderas para features contextuales
        self.include_client = include_client
        self.include_campaign = include_campaign
        self.include_channel = include_channel
        self.include_product = include_product
        
        self.model_used = None  # Para logging qué modelo se usó

    def _make_model(self) -> RandomForestRegressor:
        return RandomForestRegressor(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
        )

    def _detect_available_features(self, hist: pd.DataFrame) -> dict:
        """Detecta qué columnas de contexto están disponibles."""
        available = {
            "has_client": "Empresa_cliente" in hist.columns,
            "has_campaign": "Campana" in hist.columns,
            "has_channel": "Canal_venta" in hist.columns,
            "has_product": "Producto_id" in hist.columns,
            "has_valor_total": "Valor_total" in hist.columns,
        }
        return available

    def forecast_1step(
        self, 
        history: pd.DataFrame, 
        y_col: str = "Demanda_Unid",
        use_contextual: bool = True
    ) -> float:
        """
        Predice t+1 usando features contextuales si están disponibles.
        
        Args:
            history: DataFrame con histórico
            y_col: Columna de target (demanda)
            use_contextual: Si False, fuerza modelo univariante
        
        Returns:
            Predicción como float positivo
        """
        if history is None or history.empty:
            self.model_used = "empty"
            return 0.0

        h = history.copy().sort_values("Mes")
        h[y_col] = pd.to_numeric(h[y_col], errors="coerce").fillna(0.0).astype(float)

        if len(h) < self.min_obs:
            self.model_used = "fallback_small_data"
            try:
                return float(seasonal_naive_12(h, y_col=y_col))
            except Exception:
                return float(naive_last(h, y_col=y_col))

        # Detectar features disponibles
        available = self._detect_available_features(h)
        use_context = use_contextual and any([
            available["has_client"],
            available["has_campaign"],
            available["has_channel"],
            available["has_product"],
        ])

        # === PATH 1: CONTEXTUAL (si hay features de negocio) ===
        if use_context:
            try:
                feats = make_supervised_features_contextual(
                    h,
                    y_col=y_col,
                    date_col="Mes",
                    include_client=self.include_client and available["has_client"],
                    include_campaign=self.include_campaign and available["has_campaign"],
                    include_channel=self.include_channel and available["has_channel"],
                    include_product=self.include_product and available["has_product"],
                )
                
                # Features para entrenamiento
                candidate_cols = [
                    c for c in feats.columns 
                    if c not in ["Mes", y_col, "y"]
                ]
                numeric_cols = [
                    c for c in candidate_cols
                    if pd.api.types.is_numeric_dtype(feats[c])
                ]

                # Preparar datos de entrenamiento
                train = feats.dropna(subset=["y"]).copy()
                train[numeric_cols] = train[numeric_cols].fillna(0.0)

                if len(train) < self.min_obs:
                    raise ValueError(f"Tras limpieza, {len(train)} < {self.min_obs} observaciones")

                X = train[numeric_cols].to_numpy(dtype=float)
                y = train["y"].to_numpy(dtype=float)

                # Entrenar modelo contextual
                model = self._make_model()
                model.fit(X, y)

                # Predecir t+1
                last_mes = pd.to_datetime(h["Mes"].max()).to_period("M").to_timestamp()
                next_mes = last_mes + pd.offsets.MonthBegin(1)
                
                x_next = build_next_month_row_contextual(
                    h,
                    next_mes,
                    y_col=y_col,
                    date_col="Mes",
                    include_client=self.include_client and available["has_client"],
                    include_campaign=self.include_campaign and available["has_campaign"],
                    include_channel=self.include_channel and available["has_channel"],
                    include_product=self.include_product and available["has_product"],
                )
                
                x_next[numeric_cols] = x_next[numeric_cols].fillna(0.0)
                yhat = float(model.predict(x_next[numeric_cols].to_numpy(dtype=float))[0])

                if not np.isfinite(yhat):
                    raise ValueError("Predicción no-finita")

                self.model_used = "rf_contextual"
                return float(max(0.0, yhat))

            except Exception as e:
                # Fallback a univariante si hay error en contextual
                print(f"⚠️ RF Contextual falló: {str(e)[:50]}. Usando univariante...")

        # === PATH 2: UNIVARIANTE (fallback o sin features contextuales) ===
        try:
            from src.ml.rf_features import make_supervised_features, build_next_month_row
            
            feats = make_supervised_features(h, y_col=y_col, date_col="Mes")
            candidate_cols = [c for c in feats.columns if c not in ["Mes", y_col, "y"]]
            numeric_cols = [c for c in candidate_cols if pd.api.types.is_numeric_dtype(feats[c])]

            train = feats.dropna(subset=["y"]).copy()
            train[numeric_cols] = train[numeric_cols].fillna(0.0)

            X = train[numeric_cols].to_numpy(dtype=float)
            y = train["y"].to_numpy(dtype=float)

            model = self._make_model()
            model.fit(X, y)

            last_mes = pd.to_datetime(h["Mes"].max()).to_period("M").to_timestamp()
            next_mes = last_mes + pd.offsets.MonthBegin(1)

            x_next = build_next_month_row(h, next_mes, y_col=y_col, date_col="Mes")
            x_next[numeric_cols] = x_next[numeric_cols].fillna(0.0)
            yhat = float(model.predict(x_next[numeric_cols].to_numpy(dtype=float))[0])

            if not np.isfinite(yhat):
                raise ValueError("Predicción no-finita")

            self.model_used = "rf_univariante"
            return float(max(0.0, yhat))

        except Exception as e:
            self.model_used = "fallback_baseline"
            try:
                return float(seasonal_naive_12(h, y_col=y_col))
            except Exception:
                return float(naive_last(h, y_col=y_col))

    def get_model_info(self) -> dict:
        """Retorna información del modelo usado en última predicción."""
        return {
            "model_type": self.model_used or "unknown",
            "n_estimators": self.n_estimators,
            "min_obs": self.min_obs,
        }
