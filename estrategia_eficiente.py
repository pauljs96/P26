"""
ESTRATEGIA: Parametrizar archivos existentes para soportar v4 + legacy
NO crear nuevos archivos, NO duplicar código
"""

print("""
╔════════════════════════════════════════════════════════════════════════════╗
║  ESTRATEGIA EFICIENTE: Modificar existentes, no crear nuevos              ║
╚════════════════════════════════════════════════════════════════════════════╝

═══════════════════════════════════════════════════════════════════════════
PRINCIPIO: Un solo archivo, dos modos
═══════════════════════════════════════════════════════════════════════════

En lugar de:
  ✗ data_loader.py (legacy)
  ✗ data_loader_v4.py (duplicado)

Usar:
  ✓ data_loader.py (detecta formato automáticamente)


═══════════════════════════════════════════════════════════════════════════
CAMBIOS MÍNIMOS POR ARCHIVO
═══════════════════════════════════════════════════════════════════════════


1. src/data/data_loader.py
═════════════════════════════════════════════════════════════════════════════

CAMBIO: Agregar detección automática de formato

def load_files(self, uploaded_files: List) -> tuple[pd.DataFrame, str]:
    '''Retorna (df, format_type)'''
    dfs = []
    format_detected = None
    
    for f in uploaded_files:
        df = self._load_single_file(f)
        detected_fmt = self._detect_format(df)  # ← NUEVO
        
        if format_detected is None:
            format_detected = detected_fmt
        elif format_detected != detected_fmt:
            raise ValueError("Archivos con formatos mixtos no permitidos")
        
        df["__source_file"] = getattr(f, "name", "uploaded.csv")
        dfs.append(df)
    
    return pd.concat(dfs, ignore_index=True), format_detected


def _detect_format(self, df: pd.DataFrame) -> str:
    '''Detecta si es 'legacy' o 'v4' '''
    cols_lower = [c.lower() for c in df.columns]
    
    # Si tiene estas columnas → es v4
    v4_markers = ['tipo_movimiento', 'producto_id', 'stock_anterior', 'punto_venta']
    if all(any(marker in c for c in cols_lower) for marker in v4_markers):
        return 'v4'
    
    # Si tiene estas → es legacy ERP
    legacy_markers = ['documento', 'entrada', 'salida', 'bodega']
    if any(any(marker in c for c in cols_lower) for marker in legacy_markers):
        return 'legacy'
    
    raise ValueError("Formato desconocido - ni v4 ni legacy")


═══════════════════════════════════════════════════════════════════════════


2. src/data/data_cleaner.py
═════════════════════════════════════════════════════════════════════════════

CAMBIO: Parametrizar limpieza según formato

class DataCleaner:
    def clean(self, df_raw: pd.DataFrame, format_type: str = 'legacy') -> pd.DataFrame:
        '''
        format_type: 'legacy' | 'v4'
        '''
        if format_type == 'v4':
            return self._clean_v4(df_raw)
        else:
            return self._clean_legacy(df_raw)
    
    
    def _clean_legacy(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """Limpieza actual (rename existing clean() content)"""
        # TODO: mover código actual aquí
        ...
    
    
    def _clean_v4(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """Limpieza simplificada para v4"""
        df = df_raw.copy()
        
        # Validar columnas requeridas
        required = [
            'Fecha', 'Producto_id', 'Tipo_movimiento', 'Cantidad',
            'Stock_anterior', 'Stock_posterior'
        ]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Columnas faltantes: {missing}")
        
        # Convertir tipos
        df['Fecha'] = pd.to_datetime(df['Fecha'])
        df['Cantidad'] = pd.to_numeric(df['Cantidad'], errors='coerce').fillna(0).astype(int)
        df['Stock_anterior'] = pd.to_numeric(df['Stock_anterior'], errors='coerce').fillna(0).astype(int)
        df['Stock_posterior'] = pd.to_numeric(df['Stock_posterior'], errors='coerce').fillna(0).astype(int)
        
        # ← NUEVO: Validar coherencia
        incoherent = df[
            (df['Stock_anterior'] + df['Cantidad'] != df['Stock_posterior']) &
            (df['Tipo_movimiento'] == 'Venta') |  # Para venta es -Cantidad
            (df['Tipo_movimiento'] == 'Producción')  # Para prod es +Cantidad
        ]
        
        if len(incoherent) > 0:
            logger.warning(f"⚠️ {len(incoherent)} filas con stock incoherente")
        
        # Limpiar
        df = df.dropna(subset=['Fecha', 'Producto_id'])
        df = df[df['Producto_id'] != ""]
        
        return df.reset_index(drop=True)


═══════════════════════════════════════════════════════════════════════════


3. src/data/guide_reconciliation.py
═════════════════════════════════════════════════════════════════════════════

CAMBIO: Hacer opcional

class GuideReconciler:
    def reconcile(self, df: pd.DataFrame, format_type: str = 'legacy') -> pd.DataFrame:
        '''
        format_type: 'legacy' | 'v4'
        '''
        if format_type == 'v4':
            # v4 ya está coherente, solo retornar
            return df
        else:
            # legacy: aplicar lógica actual
            return self._reconcile_legacy(df)
    
    
    def _reconcile_legacy(self, df: pd.DataFrame) -> pd.DataFrame:
        """Renombrar método actual + mover código"""
        # TODO: mover código actual aquí
        ...


═══════════════════════════════════════════════════════════════════════════


4. src/data/demand_builder.py
═════════════════════════════════════════════════════════════════════════════

CAMBIO: Agregar capacidades v4 sin cambiar legacy

class DemandBuilder:
    def build_monthly(self, df: pd.DataFrame, format_type: str = 'legacy') -> pd.DataFrame:
        '''Agrupa demanda mensual'''
        if format_type == 'v4':
            return self._build_monthly_v4(df)
        else:
            return self._build_monthly_legacy(df)
    
    
    def _build_monthly_legacy(self, df: pd.DataFrame) -> pd.DataFrame:
        """Renombrar método actual + mover código"""
        # TODO: mover código actual aquí
        ...
    
    
    def _build_monthly_v4(self, df: pd.DataFrame) -> pd.DataFrame:
        """Nueva agregación para v4 - más simple"""
        # Filtrar solo ventas
        sales = df[df['Tipo_movimiento'] == 'Venta'].copy()
        
        # Agregar por (Producto, Año, Mes)
        result = sales.groupby([
            'Producto_id',
            sales['Fecha'].dt.year.rename('Año'),
            sales['Fecha'].dt.month.rename('Mes')
        ]).agg({
            'Cantidad': 'sum',
            'Valor_total': 'sum',
            'Descuento_pct': 'mean'  ← NUEVO en v4
        }).reset_index()
        
        result.columns = ['Producto_id', 'Año', 'Mes', 'Demanda_unid', 'Valor_total', 'Descuento_promedio']
        
        return result
    
    
    # ← NUEVO: Opcional - análisis por cliente
    def build_demand_by_client(self, df: pd.DataFrame) -> pd.DataFrame:
        '''v4: demanda por cliente (solo si tiene columna)'''
        if 'Empresa_cliente' not in df.columns:
            return pd.DataFrame()
        
        sales = df[df['Tipo_movimiento'] == 'Venta'].copy()
        
        result = sales.groupby([
            'Producto_id',
            'Empresa_cliente',
            sales['Fecha'].dt.year.rename('Año'),
            sales['Fecha'].dt.month.rename('Mes')
        ]).agg({
            'Cantidad': 'sum',
            'Valor_total': 'sum'
        }).reset_index()
        
        return result


═══════════════════════════════════════════════════════════════════════════


5. src/data/ProductStockBuilder.py
═════════════════════════════════════════════════════════════════════════════

CAMBIO: Simplificar para v4, mantener legacy complejo

class ProductStockBuilder:
    def build(self, df: pd.DataFrame, format_type: str = 'legacy') -> pd.DataFrame:
        if format_type == 'v4':
            return self._build_v4(df)
        else:
            return self._build_legacy(df)
    
    
    def _build_legacy(self, df: pd.DataFrame) -> pd.DataFrame:
        """Renombrar método actual + mover código"""
        # TODO: mover código actual aquí
        ...
    
    
    def _build_v4(self, df: pd.DataFrame) -> pd.DataFrame:
        """v4: tomar última transacción del mes = stock final"""
        # Agrupar por producto y mes
        df['AñoMes'] = df['Fecha'].dt.to_period('M')
        
        # Última transacción del mes
        last_tx = df.sort_values('Fecha').groupby(['Producto_id', 'AñoMes']).tail(1)
        
        result = last_tx[[
            'Producto_id', 'Año', 'Mes', 'Stock_posterior'
        ]].copy()
        
        result.columns = ['Producto_id', 'Año', 'Mes', 'Stock_final']
        
        # ← BONUS: también calcular promedio del mes
        monthly_avg = df.groupby(['Producto_id', 'AñoMes'])['Stock_posterior'].mean()
        result['Stock_promedio'] = monthly_avg.values
        
        return result.reset_index(drop=True)


═══════════════════════════════════════════════════════════════════════════


6. src/data/pipeline.py
═════════════════════════════════════════════════════════════════════════════

CAMBIO: Parametrizar toda la orquestación

class DataPipeline:
    def run(self, uploaded_files) -> PipelineResult:
        try:
            # ← NUEVO: Detectar formato automáticamente
            self.logger.info("Cargando CSV...")
            raw, format_type = self.loader.load_files(uploaded_files)  # Retorna (df, type)
            self.logger.info(f"✓ {len(raw)} filas cargadas (formato: {format_type})")
            
            self.logger.info("Limpiando y tipando data...")
            clean = self.cleaner.clean(raw, format_type=format_type)  # ← Pasar formato
            self.logger.info(f"✓ {len(clean)} filas limpias")
            
            # ← NUEVO: GuideReconciler es opcional
            if format_type == 'legacy':
                self.logger.info("Reconciliando guías de remisión...")
                rec = self.reconciler.reconcile(clean, format_type=format_type)
                self.logger.info(f"✓ {len(rec)} filas reconciliadas")
            else:
                rec = clean  # v4 ya está coherente
                self.logger.info("✓ v4: Sin reconciliación necesaria")
            
            self.logger.info("Construyendo demanda mensual...")
            demand = self.demand_builder.build_monthly(rec, format_type=format_type)
            self.logger.info(f"✓ {len(demand)} registros de demanda")
            
            self.logger.info("Construyendo stock mensual...")
            stock = self.stock_builder.build(rec, format_type=format_type)
            self.logger.info(f"✓ {len(stock)} registros de stock")
            
            # ← NUEVO: análisis adicionales SOLO para v4
            if format_type == 'v4':
                self.logger.info("Construyendo demanda por cliente...")
                demand_client = self.demand_builder.build_demand_by_client(rec)
                if not demand_client.empty:
                    self.logger.info(f"✓ {len(demand_client)} registros demanda×cliente")
            
            return PipelineResult(
                movements=rec,
                demand_monthly=demand,
                stock_monthly=stock,
                format_type=format_type  # ← Pasar formato al resultado
            )
        
        except Exception as e:
            ...


═══════════════════════════════════════════════════════════════════════════


7. src/utils/config.py
═════════════════════════════════════════════════════════════════════════════

CAMBIO: Agregar config para v4 (adicional, no reemplazar legacy)

# Legacy config (MANTENER - sin cambios)
DOC_VENTA_TIENDA = "Venta Tienda Sin Doc"
...

# ← NUEVO: Config para v4
V4_REQUIRED_COLUMNS = [
    'Fecha', 'Producto_id', 'Tipo_movimiento', 'Cantidad',
    'Stock_anterior', 'Stock_posterior', 'Producto_nombre'
]

V4_OPTIONAL_COLUMNS = [
    'Empresa_cliente', 'Departamento_cliente', 'Canal_venta',
    'Punto_venta', 'Precio_unitario', 'Descuento_pct',
    'Valor_total', 'Campana', 'Costo_unitario'
]

V4_FORMAT_MARKERS = {
    'tipo_movimiento': 'v4',
    'stock_anterior': 'v4',
    'documento': 'legacy'
}


═══════════════════════════════════════════════════════════════════════════


8. main.py / dashboard.py
═════════════════════════════════════════════════════════════════════════════

CAMBIO: Mínimo - el sistema detecta automáticamente

# NO NECESITA CAMBIO EXPLÍCITO
# El usuario simplemente sube el CSV (legacy o v4)
# El sistema detecta y procesa correctamente


═══════════════════════════════════════════════════════════════════════════
RESUMEN: ARCHIVOS A MODIFICAR (NO crear nuevos)
═══════════════════════════════════════════════════════════════════════════

✏️ data_loader.py
   + Agregar _detect_format()
   + Retornar (df, format_type)
   
✏️ data_cleaner.py
   + Parametrizar clean(df, format_type)
   + Agregar _clean_v4()
   + Renombrar clean() → _clean_legacy()
   
✏️ guide_reconciliation.py
   + Parametrizar reconcile(df, format_type)
   + Si v4: retornar df sin cambios
   
✏️ demand_builder.py
   + Parametrizar build_monthly()
   + Agregar _build_monthly_v4()
   + Agregar build_demand_by_client() (bonus)
   
✏️ ProductStockBuilder.py
   + Parametrizar build()
   + Agregar _build_v4() (simple)
   
✏️ pipeline.py
   + Pasar format_type a todos los métodos
   + Saltear reconciliation si v4
   
✏️ config.py
   + Agregar V4_REQUIRED_COLUMNS
   + Agregar V4_FORMAT_MARKERS
   
⚠️ main.py / dashboard.py
   - Sin cambios (auto-detecta)


═══════════════════════════════════════════════════════════════════════════
VENTAJAS DE ESTA ESTRATEGIA
═══════════════════════════════════════════════════════════════════════════

✓ Sin archivos obsoletos
✓ Un solo código por componente (legacy + v4 combinado)
✓ Auto-detección automática - usuario no elige
✓ Compatible backward (legacy sigue funcionando)
✓ Menos mantenimiento
✓ Más limpio y profesional
✓ Fácil agregar nuevos formatos en futuro


═══════════════════════════════════════════════════════════════════════════
IMPACTO EN NÚMEROS
═══════════════════════════════════════════════════════════════════════════

Opción 1 (crear nuevos):    +1,500 líneas en 5 archivos nuevos
Opción 2 (parametrizar):    +600 líneas en 7 archivos existentes (-400 renombrados)

Ganancia: -60% líneas nuevas, 0 archivos obsoletos


═══════════════════════════════════════════════════════════════════════════
""")
