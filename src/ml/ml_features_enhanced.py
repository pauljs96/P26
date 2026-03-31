"""Feature engineering avanzado para ML contextual.

Extiende rf_features.py con contexto de negocio:
- Features por Canal, Campaña, Producto, Cliente
- Histórico y recencia de cliente
- Estacionalidad contextual
- Características de campaña (activa, tipo)

Compatible backward: si faltan columnas, usa features base.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def add_client_features(
    hist: pd.DataFrame,
    date_col: str = "Mes",
    client_col: str = "Empresa_cliente",
) -> pd.DataFrame:
    """Agrega features históricos de cliente."""
    df = hist.copy()
    
    if client_col not in df.columns:
        return df
    
    try:
        # Recencia: cuántos meses sin comprar (por cliente si aplica)
        df["client_last_purchase"] = df.groupby(client_col)[date_col].transform(
            lambda x: (pd.to_datetime(df[date_col].max()) - pd.to_datetime(x.max())).days / 30.0
        )
        
        # Frecuencia: # de compras por cliente en últimos 12 meses
        last_12m = pd.to_datetime(df[date_col].max()) - pd.DateOffset(months=12)
        df["client_freq_12m"] = df[df[date_col] >= last_12m].groupby(client_col).cumcount() + 1
        df["client_freq_12m"] = df["client_freq_12m"].fillna(0)
        
        # Valor promedio por cliente
        df["client_avg_value"] = df.groupby(client_col)["Valor_total"].transform("mean")
        df["client_avg_value"] = df["client_avg_value"].fillna(0.0)
        
        # Segmentación simple: cliente alto/medio/bajo valor
        qts = df["client_avg_value"].quantile([0.33, 0.67])
        df["client_tier"] = pd.cut(
            df["client_avg_value"],
            bins=[0, qts[0.33], qts[0.67], float('inf')],
            labels=[1, 2, 3],
            include_lowest=True
        ).astype(float)
        df["client_tier"] = df["client_tier"].fillna(1.0)
        
    except Exception as e:
        print(f"Warning: No se pudo calcular features de cliente: {e}")
    
    return df


def add_campaign_features(
    hist: pd.DataFrame,
    campaign_col: str = "Campana",
    date_col: str = "Mes",
) -> pd.DataFrame:
    """Agrega features relacionadas a campañas."""
    df = hist.copy()
    
    if campaign_col not in df.columns:
        return df
    
    try:
        # Campaign activa (dummy): indicador si hay ventas en el mes
        df["campaign_active"] = df.groupby(campaign_col)[date_col].transform("count").fillna(0).astype(bool).astype(float)
        
        # Momentum de campaña: suma últimos 3 meses
        df_sorted = df.sort_values(date_col).reset_index(drop=True)
        df["campaign_momentum_3m"] = df_sorted.groupby(campaign_col)["Valor_total"].transform(
            lambda x: x.rolling(window=3, min_periods=1).sum()
        ).fillna(0.0)
        
        # Número de campañas activas en el mes
        df["campaigns_active_count"] = df.groupby(date_col)[campaign_col].transform("nunique").fillna(0).astype(float)
        
    except Exception as e:
        print(f"Warning: No se pudo calcular features de campaña: {e}")
    
    return df


def add_channel_features(
    hist: pd.DataFrame,
    channel_col: str = "Canal_venta",
    date_col: str = "Mes",
    y_col: str = "Demanda_Unid",
) -> pd.DataFrame:
    """Agrega features por canal de venta."""
    df = hist.copy()
    
    if channel_col not in df.columns:
        return df
    
    try:
        # Market share del canal (% de demanda)
        df["channel_share"] = (
            df.groupby([date_col, channel_col])[y_col].transform("sum") /
            df.groupby(date_col)[y_col].transform("sum")
        ).fillna(0.0)
        
        # Volatilidad del canal (últimos 6 meses)
        df_sorted = df.sort_values(date_col).reset_index(drop=True)
        df["channel_volatility_6m"] = df_sorted.groupby(channel_col)[y_col].transform(
            lambda x: x.rolling(window=6, min_periods=1).std()
        ).fillna(0.0)
        
        # Tendencia: uptrend/downtrend últimos 3 meses
        df["channel_trend_3m"] = df_sorted.groupby(channel_col)[y_col].transform(
            lambda x: np.where(
                x.rolling(window=3, min_periods=1).mean().diff() > 0, 1.0, -1.0
            )
        ).fillna(0.0)
        
    except Exception as e:
        print(f"Warning: No se pudo calcular features de canal: {e}")
    
    return df


def add_product_features(
    hist: pd.DataFrame,
    product_col: str = "Producto_id",
    date_col: str = "Mes",
    y_col: str = "Demanda_Unid",
) -> pd.DataFrame:
    """Agrega features por producto."""
    df = hist.copy()
    
    if product_col not in df.columns:
        return df
    
    try:
        # Rotación de producto (# meses con demanda > 0)
        df["product_rotation"] = df.groupby(product_col).apply(
            lambda x: (x[y_col] > 0).sum()
        ).reindex(df[product_col]).values.astype(float)
        
        # Demanda promedio por producto
        df["product_avg_demand"] = df.groupby(product_col)[y_col].transform("mean").fillna(0.0)
        
        # Concentración: si este producto es >50% de la demanda del mes
        df["product_concentration"] = (
            df.groupby([date_col, product_col])[y_col].transform("sum") /
            df.groupby(date_col)[y_col].transform("sum")
        ).fillna(0.0)
        
    except Exception as e:
        print(f"Warning: No se pudo calcular features de producto: {e}")
    
    return df


def make_supervised_features_contextual(
    hist: pd.DataFrame,
    y_col: str = "Demanda_Unid",
    date_col: str = "Mes",
    lags: tuple[int, ...] = (1, 2, 3, 6, 12),
    roll_windows: tuple[int, ...] = (3, 6, 12),
    include_client: bool = True,
    include_campaign: bool = True,
    include_channel: bool = True,
    include_product: bool = True,
) -> pd.DataFrame:
    """
    Features mejoradas + contextuales.
    
    Primero agrega features base (de rf_features), luego contextuales.
    Si faltan columnas de contexto, las salta gracefully.
    """
    from src.ml.rf_features import make_supervised_features
    
    # Features base (temporal + calendario)
    df = make_supervised_features(
        hist, 
        y_col=y_col, 
        date_col=date_col,
        lags=lags,
        roll_windows=roll_windows
    )
    
    # Features contextuales (opcionales)
    if include_client:
        df = add_client_features(df, date_col=date_col, client_col="Empresa_cliente")
    
    if include_campaign:
        df = add_campaign_features(df, campaign_col="Campana", date_col=date_col)
    
    if include_channel:
        df = add_channel_features(df, channel_col="Canal_venta", date_col=date_col, y_col=y_col)
    
    if include_product:
        df = add_product_features(df, product_col="Producto_id", date_col=date_col, y_col=y_col)
    
    return df


def build_next_month_row_contextual(
    hist: pd.DataFrame,
    next_mes: pd.Timestamp,
    y_col: str = "Demanda_Unid",
    date_col: str = "Mes",
    lags: tuple[int, ...] = (1, 2, 3, 6, 12),
    roll_windows: tuple[int, ...] = (3, 6, 12),
    include_client: bool = True,
    include_campaign: bool = True,
    include_channel: bool = True,
    include_product: bool = True,
) -> pd.DataFrame:
    """Construye fila de features contextual para t+1."""
    from src.ml.rf_features import build_next_month_row
    
    # Fila base
    row = build_next_month_row(
        hist,
        next_mes,
        y_col=y_col,
        date_col=date_col,
        lags=lags,
        roll_windows=roll_windows
    )
    
    # Agregar features contextuales forward-filling desde última fila disponible
    feats_full = make_supervised_features_contextual(
        hist,
        y_col=y_col,
        date_col=date_col,
        lags=lags,
        roll_windows=roll_windows,
        include_client=include_client,
        include_campaign=include_campaign,
        include_channel=include_channel,
        include_product=include_product,
    )
    
    if len(feats_full) > 0:
        last_row = feats_full.iloc[-1:]
        contextual_cols = [
            c for c in last_row.columns 
            if c not in ["Mes", date_col, y_col, "y"] 
            and c not in row.columns  # Solo agregar si no está
        ]
        
        for col in contextual_cols:
            if col in last_row.columns:
                row[col] = last_row[col].values[0]
    
    return row
