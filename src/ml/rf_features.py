from __future__ import annotations
import numpy as np
import pandas as pd


def make_supervised_features(
    hist: pd.DataFrame,
    y_col: str = "Demanda_Unid",
    date_col: str = "Mes",
    lags: tuple[int, ...] = (1, 2, 3, 6, 12),
    roll_windows: tuple[int, ...] = (3, 6, 12),
) -> pd.DataFrame:
    """
    Crea features temporales para una serie mensual univariante con ceros/intermitencia.
    IMPORTANTE: Todo rolling y lags se calculan con shift(1) para evitar leakage.
    Entrada: hist con columnas [Mes, Demanda_Unid]
    Salida: DataFrame con columnas: Mes, y, features...
    """
    df = hist.copy()
    df[date_col] = pd.to_datetime(df[date_col]).dt.to_period("M").dt.to_timestamp()
    df = df.sort_values(date_col).reset_index(drop=True)
    df[y_col] = pd.to_numeric(df[y_col], errors="coerce").fillna(0.0).astype(float)

    # Target
    df["y"] = df[y_col]

    # Lags
    for L in lags:
        df[f"lag_{L}"] = df["y"].shift(L)

    # Rolling stats (shift(1) para no usar y(t) en features de t)
    y_shift = df["y"].shift(1)

    for w in roll_windows:
        df[f"roll_mean_{w}"] = y_shift.rolling(window=w, min_periods=1).mean()
        df[f"roll_std_{w}"] = y_shift.rolling(window=w, min_periods=1).std().fillna(0.0)
        df[f"roll_nonzero_{w}"] = (y_shift > 0).rolling(window=w, min_periods=1).sum()

    # Intermitencia: conteo de ceros en ventanas
    for w in (6, 12):
        df[f"zero_count_{w}"] = (y_shift == 0).rolling(window=w, min_periods=1).sum()

    # Calendario: mes del año (one-hot simple o cíclico)
    df["month"] = df[date_col].dt.month.astype(int)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)

    return df


def build_next_month_row(
    hist: pd.DataFrame,
    next_mes: pd.Timestamp,
    y_col: str = "Demanda_Unid",
    date_col: str = "Mes",
    lags: tuple[int, ...] = (1, 2, 3, 6, 12),
    roll_windows: tuple[int, ...] = (3, 6, 12),
) -> pd.DataFrame:
    """
    Construye una sola fila de features para predecir el próximo mes (t+1),
    usando únicamente el historial disponible.
    """
    h = hist.copy()
    h[date_col] = pd.to_datetime(h[date_col]).dt.to_period("M").dt.to_timestamp()
    h = h.sort_values(date_col).reset_index(drop=True)
    h[y_col] = pd.to_numeric(h[y_col], errors="coerce").fillna(0.0).astype(float)

    # Usamos el mismo generador y tomamos la última fila como base,
    # pero "simulamos" el mes siguiente.
    feats = make_supervised_features(h, y_col=y_col, date_col=date_col, lags=lags, roll_windows=roll_windows)

    last = feats.iloc[-1:].copy()
    last[date_col] = pd.to_datetime(next_mes).to_period("M").to_timestamp()

    # Recalcular calendario para next_mes
    last["month"] = last[date_col].dt.month.astype(int)
    last["month_sin"] = np.sin(2 * np.pi * last["month"] / 12.0)
    last["month_cos"] = np.cos(2 * np.pi * last["month"] / 12.0)

    # OJO: las columnas de lag/rolling ya están basadas en el historial (shift),
    # y para predecir next_mes, esas features son válidas (porque dependen de historia).
    return last
