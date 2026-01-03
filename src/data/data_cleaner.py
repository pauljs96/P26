"""Limpieza y normalización de columnas.

Convierte el DataFrame 'crudo' del ERP a un esquema estable para el proyecto:

- Codigo, Descripcion, Fecha, Documento, Numero, Bodega
- Entrada_unid, Salida_unid, Saldo_unid
- Valor_Unitario, Costo_Unitario
- Entrada_monto, Salida_monto, Saldo_monto (si existen)
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Optional, List

from src.utils import config


def _pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    # case-insensitive
    mapping = {str(col).strip().lower(): col for col in df.columns}
    for c in candidates:
        key = str(c).strip().lower()
        if key in mapping:
            return mapping[key]
    return None


def _to_num(series: pd.Series) -> pd.Series:
    # Limpia formatos comunes: separador de miles, comas decimales
    s = series.astype(str).str.replace(" ", "", regex=False)
    # si vienen con coma decimal, convertir a punto
    s = s.str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")


class DataCleaner:
    def clean(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        if df_raw is None or df_raw.empty:
            return pd.DataFrame()

        df = df_raw.copy()

        c_codigo = _pick_col(df, config.COL_CODIGO)
        c_desc = _pick_col(df, config.COL_DESCRIPCION)
        c_fecha = _pick_col(df, config.COL_FECHA)
        c_doc = _pick_col(df, config.COL_DOCUMENTO)
        c_num = _pick_col(df, config.COL_NUMERO)
        c_bodega = _pick_col(df, config.COL_BODEGA)

        c_ent_u = _pick_col(df, config.COL_ENTRADA_UNID)
        c_sal_u = _pick_col(df, config.COL_SALIDA_UNID)
        c_saldo_u = _pick_col(df, config.COL_SALDO_UNID)

        c_val_unit = _pick_col(df, config.COL_VALOR_UNIT)
        c_cost_unit = _pick_col(df, config.COL_COSTO_UNIT)

        c_ent_m = _pick_col(df, config.COL_ENTRADA_MONTO)
        c_sal_m = _pick_col(df, config.COL_SALIDA_MONTO)
        c_saldo_m = _pick_col(df, config.COL_SALDO_MONTO)

        required = [c_codigo, c_fecha, c_doc, c_num, c_bodega, c_ent_u, c_sal_u, c_saldo_u]
        if any(c is None for c in required):
            # Devolver vacío para que UI muestre columnas encontradas
            return pd.DataFrame()

        out = pd.DataFrame()
        out["Codigo"] = df[c_codigo].astype(str).str.strip()
        out["Descripcion"] = df[c_desc].astype(str).str.strip() if c_desc else ""
        out["Documento"] = df[c_doc].astype(str).str.strip()
        out["Numero"] = df[c_num].astype(str).str.strip()
        out["Bodega"] = df[c_bodega].astype(str).str.strip()

        out["Fecha"] = pd.to_datetime(df[c_fecha], errors="coerce", dayfirst=True)

        out["Entrada_unid"] = _to_num(df[c_ent_u]).fillna(0.0)
        out["Salida_unid"] = _to_num(df[c_sal_u]).fillna(0.0)
        out["Saldo_unid"] = _to_num(df[c_saldo_u])

        out["Valor_Unitario"] = _to_num(df[c_val_unit]) if c_val_unit else np.nan
        out["Costo_Unitario"] = _to_num(df[c_cost_unit]) if c_cost_unit else np.nan

        out["Entrada_monto"] = _to_num(df[c_ent_m]).fillna(0.0) if c_ent_m else 0.0
        out["Salida_monto"] = _to_num(df[c_sal_m]).fillna(0.0) if c_sal_m else 0.0
        out["Saldo_monto"] = _to_num(df[c_saldo_m]) if c_saldo_m else np.nan

        # Limpieza final
        out = out.dropna(subset=["Fecha"])
        out = out[out["Codigo"] != ""]

        # No negativos en movimientos
        out["Entrada_unid"] = out["Entrada_unid"].clip(lower=0)
        out["Salida_unid"] = out["Salida_unid"].clip(lower=0)

        return out.reset_index(drop=True)
