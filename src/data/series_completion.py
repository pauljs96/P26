from __future__ import annotations
import pandas as pd


def complete_monthly_demand(
    demand_monthly: pd.DataFrame,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """
    Completa meses faltantes con 0 por producto.

    Espera columnas: Codigo, Mes, Demanda_Unid
    - start/end (opcionales): límites globales en formato 'YYYY-MM-01' o 'YYYY-MM'
      Si no se pasan, usa min/max de Mes en el dataframe.

    Retorna: DataFrame con TODOS los meses para TODOS los productos y Demanda_Unid=0 cuando falte.
    """
    if demand_monthly is None or demand_monthly.empty:
        return pd.DataFrame(columns=["Codigo", "Mes", "Demanda_Unid"])

    df = demand_monthly.copy()

    # Normalizar tipos
    df["Codigo"] = df["Codigo"].astype(str).str.strip()
    df["Mes"] = pd.to_datetime(df["Mes"]).dt.to_period("M").dt.to_timestamp()
    df["Demanda_Unid"] = pd.to_numeric(df["Demanda_Unid"], errors="coerce").fillna(0.0)

    # Rango global de meses
    min_mes = df["Mes"].min()
    max_mes = df["Mes"].max()

    if start:
        min_mes = pd.to_datetime(start).to_period("M").to_timestamp()
    if end:
        max_mes = pd.to_datetime(end).to_period("M").to_timestamp()

    all_months = pd.date_range(min_mes, max_mes, freq="MS")
    codigos = df["Codigo"].unique()

    idx = pd.MultiIndex.from_product([codigos, all_months], names=["Codigo", "Mes"])
    out = (
        df.set_index(["Codigo", "Mes"])
          .reindex(idx, fill_value=0.0)
          .reset_index()
    )

    # Asegurar orden y tipo
    out["Demanda_Unid"] = out["Demanda_Unid"].astype(float)
    return out.sort_values(["Codigo", "Mes"]).reset_index(drop=True)
