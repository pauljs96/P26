"""Construcción de demanda mensual real (empresa).

Demanda mensual por producto se construye como la suma de 3 componentes:

1) Venta Tienda Sin Doc  -> sum(Salida_unid)
2) Salida por Consumo    -> sum(Salida_unid)
3) Guía de remisión - R  -> solo la "salida externa neta" por guía-producto:
      externa = max(0, sum(Salida_unid) - sum(Entrada_unid))
   calculada por grupo (Documento, Numero, Codigo). Esto evita el doble conteo
   cuando la guía tiene múltiples filas.

Nota:
- La demanda se agrega a nivel empresa (NO por bodega) para evitar duplicación
  por transferencias internas entre almacenes.
"""

from __future__ import annotations
import pandas as pd
from src.utils import config


class DemandBuilder:
    def build_monthly(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame(columns=["Codigo", "Mes", "Demanda_Unid"])

        d = df.copy()

        # Normalizar textos para comparación robusta
        for col in ["Documento", "Codigo", "Numero"]:
            if col in d.columns:
                d[col] = d[col].astype(str).str.strip()

        d["Mes"] = d["Fecha"].dt.to_period("M").dt.to_timestamp()

        # ------------------------------
        # 1) Demanda directa (ventas + consumo)
        # ------------------------------
        direct = d[d["Documento"].isin(config.DEMAND_DIRECT_DOCS)].copy()
        direct_monthly = (
            direct.groupby(["Codigo", "Mes"], as_index=False)["Salida_unid"]
                  .sum()
                  .rename(columns={"Salida_unid": "Demanda_Unid"})
        )

        # ------------------------------
        # 2) Demanda por guías (YA reconciliadas)
        #    Usamos el output de guide_reconciliation.py:
        #    - Tipo_Guia == "VENTA_EXTERNA"
        #    - Guia_Salida_Externa_Unid (prorrateada y reconciliada por MES)
        # ------------------------------
        if "Tipo_Guia" in d.columns and "Guia_Salida_Externa_Unid" in d.columns:
            guia_ext = d[d["Tipo_Guia"] == "VENTA_EXTERNA"].copy()

            guia_monthly = (
                guia_ext.groupby(["Codigo", "Mes"], as_index=False)["Guia_Salida_Externa_Unid"]
                        .sum()
                        .rename(columns={"Guia_Salida_Externa_Unid": "Demanda_Unid"})
            )
        else:
            guia_monthly = pd.DataFrame(columns=["Codigo", "Mes", "Demanda_Unid"])

        # Unir demanda directa + demanda guías externas
        monthly = pd.concat([direct_monthly, guia_monthly], ignore_index=True)
        monthly = (
            monthly.groupby(["Codigo", "Mes"], as_index=False)["Demanda_Unid"]
                .sum()
        )


        return monthly.sort_values(["Codigo", "Mes"]).reset_index(drop=True)
