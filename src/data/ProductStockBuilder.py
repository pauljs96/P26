from __future__ import annotations
import pandas as pd


class ProductStockBuilder:
    """
    Stock mensual por PRODUCTO a nivel empresa.

    Dato: Saldo_unid es saldo consolidado del producto (no por bodega).
    Regla:
      - Para cada (Codigo, Mes): tomar el último Saldo_unid del mes según Fecha
    """

    def build_monthly(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame(columns=["Codigo", "Mes", "Stock_Unid"])

        d = df.copy()
        
        # Parsear fechas con múltiples formatos soportados (igual que guide_reconciliation)
        d["Fecha"] = pd.to_datetime(d["Fecha"], format='%Y-%m-%d', errors='coerce')
        
        # Si hay NaNs, intentar con formato europeo DD/MM/YYYY
        nan_mask = d["Fecha"].isna()
        if nan_mask.any():
            d.loc[nan_mask, "Fecha"] = pd.to_datetime(
                d.loc[nan_mask, "Fecha"], 
                format='%d/%m/%Y', 
                errors='coerce'
            )
        
        d = d.dropna(subset=["Fecha"])

        d["Codigo"] = d["Codigo"].astype(str).str.strip()
        d["Mes"] = d["Fecha"].dt.to_period("M").dt.to_timestamp()

        d["Saldo_unid"] = pd.to_numeric(d["Saldo_unid"], errors="coerce").fillna(0.0)

        # Orden por fecha y por orden de aparición (índice) para desempatar
        d["_row"] = range(len(d))
        d = d.sort_values(["Codigo", "Fecha", "_row"])

        # Tomar último saldo del mes para cada producto
        monthly = (
            d.groupby(["Codigo", "Mes"], as_index=False)
             .tail(1)[["Codigo", "Mes", "Saldo_unid"]]
             .rename(columns={"Saldo_unid": "Stock_Unid"})
             .sort_values(["Codigo", "Mes"])
             .reset_index(drop=True)
        )

        return monthly
