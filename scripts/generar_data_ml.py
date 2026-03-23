"""
Generador de Data Sintética para ML - Predicción de Demanda con Factores Exógenos

Objetivo: Crear dataset que demuestre valor de ML para predecir demanda considerando:
- Factores temporales (estacionalidad, tendencia)
- Factores de precios (elasticidad)
- Factores de promociones (campañas)
- Factores de canal (online vs tienda)
- Factores demográficos (región)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')


class DemandaMLDataGenerator:
    """Generador de data sintética realista para ML de demanda"""
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        self.seed = seed
        
        # Parámetros de demanda base por producto
        self.productos = {
            'CPE-00365': {'nombre': 'CAJA DE PASE 800X800X300MM', 'demanda_base': 150, 'elasticidad': -0.8},
            'CPE-00063': {'nombre': 'CAJA DE PASE 200X200X50MM', 'demanda_base': 250, 'elasticidad': -1.2},
            'CPE-01068': {'nombre': 'CAJA DE PASE 100X100X50MM', 'demanda_base': 300, 'elasticidad': -0.9},
            'BANDEJA-01': {'nombre': 'BANDEJA RANURADA 200X100MM', 'demanda_base': 120, 'elasticidad': -1.1},
            'GABINETE-01': {'nombre': 'GABINETE ADOSABLE 450X350X150MM', 'demanda_base': 80, 'elasticidad': -0.7},
        }
        
        self.empresas = [
            'Sistemas Constructivos SA',
            'Inmobiliaria Andina SRL',
            'Constructora Pacífico Ltda',
            'Desarrollos Urbanos del Sur',
            'Proyectos Inmobiliarios Nacional'
        ]
        
        self.departamentos = ['Lima', 'Arequipa', 'Cusco', 'Tacna', 'Junín']
        self.canales = ['Online', 'Tienda Física']
        self.campanas = ['Ninguna', 'Promo Constructor', 'Descuento Volumen', 'Black Friday']
        
        # Fechas festivas/eventos en Perú que afectan demanda
        self.eventos_peru = {
            '2024-01-01': 'Año Nuevo',
            '2024-04-09': 'Semana Santa',
            '2024-05-01': 'Día del Trabajo',
            '2024-06-29': 'San Pedro y San Pablo',
            '2024-07-28': 'Fiestas Patrias',
            '2024-08-30': 'Santa Rosa',
            '2024-10-08': 'Batalla de Angamos',
            '2024-11-01': 'Día de Difuntos',
            '2024-12-09': 'Batalla de Ayacucho',
            '2024-12-25': 'Navidad',
        }
    
    def get_base_price(self, producto_id: str) -> float:
        """Precio base del producto (1-10 USD)"""
        return np.random.uniform(2.5, 8.0)
    
    def get_seasonality_factor(self, mes: int, producto_id: str) -> float:
        """Factor de estacionalidad según mes y producto
        
        Construcción es más activa: Feb-Mayo, Sept-Nov (luego de vacaciones)
        Baja en: Enero (vacaciones), Agosto, Diciembre
        """
        estacionalidad = {
            1: 0.7,   # Enero (vacaciones)
            2: 1.1,   # Febrero
            3: 1.3,   # Marzo (fuerte)
            4: 1.25,  # Abril (fuerte)
            5: 1.15,  # Mayo
            6: 0.95,  # Junio
            7: 0.9,   # Julio (mitad vacaciones)
            8: 0.75,  # Agosto (bajo)
            9: 1.2,   # Septiembre (retorno)
            10: 1.3,  # Octubre (fuerte)
            11: 1.25, # Noviembre (fuerte)
            12: 0.8,  # Diciembre (vacaciones)
        }
        return estacionalidad.get(mes, 1.0)
    
    def get_event_impact(self, fecha: pd.Timestamp) -> float:
        """Impacto de eventos/festivos en demanda"""
        fecha_str = fecha.strftime('%Y-%m-%d')
        
        # Semana antes/después de eventos impacta
        if fecha_str in self.eventos_peru:
            return 0.85  # Reducción durante festivo
        
        # Semana previa a festivo (compras anticipadas)
        for evento_fecha in self.eventos_peru.keys():
            evento_dt = pd.to_datetime(evento_fecha)
            if (evento_dt - timedelta(days=7) <= fecha <= evento_dt):
                return 1.1  # Aumento por compras anticipadas
        
        return 1.0
    
    def get_campaign_discount(self, mes: int, campaña: str) -> tuple:
        """(descuento%, impacto en demanda)
        
        Descuento afecta elasticidad del precio
        """
        campaign_effects = {
            'Ninguna': (0, 1.0),
            'Promo Constructor': (8, 1.3),  # 8% desc, +30% demanda
            'Descuento Volumen': (5, 1.15),  # 5% desc, +15% demanda
            'Black Friday': (15, 1.5),  # 15% desc, +50% demanda (solo Nov)
        }
        
        # Black Friday solo es viável en Noviembre
        if mes != 11 and campaña == 'Black Friday':
            return (0, 1.0)
        
        return campaign_effects.get(campaña, (0, 1.0))
    
    def generate_demand_for_product(
        self,
        producto_id: str,
        empresa_id: int,
        fecha_inicio: str,
        fecha_fin: str,
        ruido: float = 0.15
    ) -> pd.DataFrame:
        """Genera serie temporal realista de demanda para un producto-empresa"""
        
        fecha_inicio = pd.to_datetime(fecha_inicio)
        fecha_fin = pd.to_datetime(fecha_fin)
        
        datos = []
        fecha_actual = fecha_inicio
        
        producto_info = self.productos[producto_id]
        empresa = self.empresas[empresa_id % len(self.empresas)]
        departamento = np.random.choice(self.departamentos)
        canal = np.random.choice(self.canales)
        
        # Tendencia leve (crecimiento/decrecimiento del sector)
        trend_direction = np.random.choice([-0.02, -0.01, 0.0, 0.01, 0.02])
        
        # Patrones específicos del producto
        intermitencia = np.random.uniform(0.1, 0.4)  # % de meses sin demanda
        
        dias_procesados = 0
        while fecha_actual <= fecha_fin:
            mes = fecha_actual.month
            ano = fecha_actual.year
            
            # 1. Demanda base
            demanda_base = producto_info['demanda_base']
            
            # 2. Tendencia
            dias_desde_inicio = (fecha_actual - fecha_inicio).days
            tendencia = 1 + (trend_direction * dias_desde_inicio / 365)
            
            # 3. Estacionalidad
            factor_estacional = self.get_seasonality_factor(mes, producto_id)
            
            # 4. Impacto de eventos
            factor_evento = self.get_event_impact(fecha_actual)
            
            # 5. Campaña promocional
            campaña = np.random.choice(self.campanas, p=[0.5, 0.25, 0.15, 0.1])
            desc_pct, impacto_campaña = self.get_campaign_discount(mes, campaña)
            
            # 6. Precio (varía con campaña y contexto)
            precio_base = self.get_base_price(producto_id)
            precio = precio_base * (1 - desc_pct / 100)
            
            # 7. Elasticidad de precio (si baja precio -> sube demanda)
            precio_relativo = precio / precio_base  # 0.85-1.0 (si hay descuento)
            impacto_precio = 1 + (precio_relativo - 1) * producto_info['elasticidad']
            
            # 8. Intermitencia: algunos meses sin demanda
            if np.random.random() < intermitencia:
                demanda_unidades = 0
            else:
                # Combinar todos los factores
                demanda_unidades = demanda_base * tendencia * factor_estacional * factor_evento * impacto_campaña * impacto_precio
                
                # 9. Ruido aleatorio
                ruido_aleatorio = np.random.normal(1.0, ruido)
                demanda_unidades = demanda_unidades * ruido_aleatorio
                demanda_unidades = max(0, int(demanda_unidades))
            
            # Calcular valor total
            valor_total = demanda_unidades * precio
            
            datos.append({
                'Fecha': fecha_actual,
                'Año': ano,
                'Mes': mes,
                'Producto_id': producto_id,
                'Producto_nombre': producto_info['nombre'],
                'Empresa': empresa,
                'Departamento': departamento,
                'Canal_venta': canal,
                'Demanda_unid': int(demanda_unidades),
                'Precio_unitario': round(precio, 2),
                'Descuento_pct': desc_pct,
                'Valor_total': round(valor_total, 2),
                'Campana': campaña,
                'Estacionalidad_factor': round(factor_estacional, 3),
                'Evento_impacto': round(factor_evento, 3),
                'Tendencia_factor': round(tendencia, 3),
                'Precio_elasticidad_factor': round(impacto_precio, 3),
            })
            
            fecha_actual += pd.DateOffset(months=1)
            dias_procesados += 1
        
        return pd.DataFrame(datos)
    
    def generate_full_dataset(
        self,
        fecha_inicio: str = '2021-01-01',
        fecha_fin: str = '2025-12-31',
        n_empresas: int = 5
    ) -> pd.DataFrame:
        """Genera dataset completo: todos los productos × todas las empresas"""
        
        print(f"\n{'='*70}")
        print(f"[GENERANDO] Dataset ML de Demanda")
        print(f"{'='*70}")
        print(f"Período: {fecha_inicio} a {fecha_fin}")
        print(f"Productos: {len(self.productos)}")
        print(f"Empresas: {n_empresas}")
        
        todas_las_series = []
        
        for i, (prod_id, prod_info) in enumerate(self.productos.items()):
            for j in range(n_empresas):
                print(f"\n[{i+1}/{len(self.productos)}, Empresa {j+1}/{n_empresas}] {prod_id}")
                
                df_prod = self.generate_demand_for_product(
                    producto_id=prod_id,
                    empresa_id=j,
                    fecha_inicio=fecha_inicio,
                    fecha_fin=fecha_fin
                )
                
                todas_las_series.append(df_prod)
                print(f"  -> {len(df_prod)} meses generados")
        
        # Combinar todo
        df_completo = pd.concat(todas_las_series, ignore_index=True)
        df_completo = df_completo.sort_values(['Fecha', 'Producto_id', 'Empresa']).reset_index(drop=True)
        
        print(f"\n{'='*70}")
        print(f"[DATASET COMPLETADO]")
        print(f"Total de filas: {len(df_completo):,}")
        print(f"Rango: {df_completo['Fecha'].min().date()} a {df_completo['Fecha'].max().date()}")
        print(f"Demanda total: {df_completo['Demanda_unid'].sum():,} unidades")
        print(f"Valor total: ${df_completo['Valor_total'].sum():,.2f}")
        
        return df_completo
    
    def save_dataset(self, df: pd.DataFrame, output_path: str):
        """Guarda dataset en CSV"""
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"\n[GUARDADO] {output_path}")
        print(f"Tamaño: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")


if __name__ == "__main__":
    generator = DemandaMLDataGenerator(seed=42)
    
    # Generar dataset: 5 productos × 5 empresas, 5 años
    df_ml = generator.generate_full_dataset(
        fecha_inicio='2020-01-01',
        fecha_fin='2025-12-31',
        n_empresas=5
    )
    
    # Guardar
    output_file = "Demanda_ML_Training_Data.csv"
    generator.save_dataset(df_ml, output_file)
    
    # Resumen estadístico
    print(f"\n{'='*70}")
    print(f"[ESTADÍSTICAS DEL DATASET]")
    print(f"{'='*70}")
    print(f"\nDemanda por Producto:")
    print(df_ml.groupby('Producto_id')['Demanda_unid'].agg(['count', 'mean', 'std', 'min', 'max']))
    
    print(f"\nDemanda por Canal:")
    print(df_ml.groupby('Canal_venta')['Demanda_unid'].agg(['count', 'sum', 'mean']))
    
    print(f"\nImpacto de Campaña en Demanda:")
    print(df_ml.groupby('Campana')['Demanda_unid'].agg(['count', 'sum', 'mean', 'max']))
    
    print(f"\nCorrelaciones clave:")
    print(df_ml[['Demanda_unid', 'Precio_unitario', 'Descuento_pct', 'Estacionalidad_factor', 'Precio_elasticidad_factor']].corr())
