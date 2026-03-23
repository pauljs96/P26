"""
Generador de Dataset Completo de Inventario para ML
Incluye: Ingresos (Producción), Egresos (Ventas), Stock en tiempo real
Con coherencia lógica: Producción responde a demanda y niveles de stock
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class InventarioMLDataGenerator:
    def __init__(self, seed=42):
        np.random.seed(seed)
        
        # Productos con demanda base diferenciada
        self.productos = {
            'CPE-00365': {'nombre': 'CAJAS PLASTICO', 'demanda_base': 120, 'lead_time': 7},
            'CPE-00063': {'nombre': 'CAJAS CORRUGADO', 'demanda_base': 250, 'lead_time': 5},
            'CPE-01068': {'nombre': 'BANDEJAS PLASTICO', 'demanda_base': 260, 'lead_time': 10},
            'BANDEJA-01': {'nombre': 'BANDEJA RANURADA', 'demanda_base': 110, 'lead_time': 8},
            'GABINETE-01': {'nombre': 'GABINETE METALICO', 'demanda_base': 65, 'lead_time': 14},
        }
        
        # Empresas/Clientes
        self.empresas = [
            'Constructora Pacífico Ltda',
            'Desarrollos Urbanos del Sur',
            'Inmobiliaria Andina SRL',
            'Proyectos Inmobiliarios Nacional',
            'Constructores del Andes'
        ]
        
        # Departamentos
        self.departamentos = ['Lima', 'Arequipa', 'Cusco', 'Tacna', 'Junín']
        
        # Canales
        self.canales = ['Online', 'Tienda Física']
        
        # Puntos de venta
        self.puntos_venta = ['Punto de venta 1', 'Punto de venta 2']
        self.campanas = ['Ninguna', 'Promo Constructor', 'Descuento Volumen', 'Black Friday']
        
        # Holidays peruanos que afectan demanda
        self.holidays_2020_2025 = [
            ('2020-01-01', 1.0), ('2020-12-31', 0.5),
            ('2021-01-01', 1.0), ('2021-12-31', 0.5),
            ('2022-01-01', 1.0), ('2022-03-28', 0.3), ('2022-12-31', 0.5),
            ('2023-01-01', 1.0), ('2023-03-30', 0.3), ('2023-12-31', 0.5),
            ('2024-01-01', 1.0), ('2024-03-28', 0.3), ('2024-12-31', 0.5),
            ('2025-01-01', 1.0), ('2025-03-27', 0.3), ('2025-12-31', 0.5),
        ]
        
    def get_seasonality_factor(self, fecha):
        """Factor de estacionalidad por mes (industria construcción)"""
        month = fecha.month
        # Construcción picos: Feb-May, Sept-Nov
        seasonality = {
            1: 0.65, 2: 1.20, 3: 1.35, 4: 1.40, 5: 1.25,
            6: 0.75, 7: 0.70, 8: 0.68, 9: 1.10, 10: 1.30,
            11: 1.25, 12: 0.65
        }
        return seasonality.get(month, 0.8)
    
    def get_day_of_week_factor(self, fecha):
        """Factor de venta por día de semana"""
        dow = fecha.weekday()  # 0=Monday, 6=Sunday
        factors = {
            0: 1.10,   # Lunes: inicio semana
            1: 1.15,   # Martes: fuerte
            2: 1.10,   # Miércoles
            3: 0.95,   # Jueves
            4: 0.90,   # Viernes: fin semana
            5: 0.50,   # Sábado: bajo
            6: 0.30    # Domingo: muy bajo/cerrado
        }
        return factors.get(dow, 1.0)
    
    def get_event_impact(self, fecha):
        """Impacto de eventos/holidays peruanos"""
        fecha_str = fecha.strftime('%Y-%m-%d')
        for holiday_date, impact in self.holidays_2020_2025:
            holiday = datetime.strptime(holiday_date, '%Y-%m-%d')
            days_diff = abs((fecha - holiday).days)
            if days_diff <= 7:  # Efecto 7 días antes y después
                return impact
        return 1.0
    
    def get_campaign_discount(self, fecha, campaign_name):
        """Descuento y factor de demanda por campaña"""
        if campaign_name == 'Ninguna':
            return 0, 1.0
        elif campaign_name == 'Promo Constructor':
            return np.random.uniform(8, 12), np.random.uniform(1.25, 1.50)
        elif campaign_name == 'Descuento Volumen':
            return np.random.uniform(5, 10), np.random.uniform(1.15, 1.35)
        elif campaign_name == 'Black Friday':
            return np.random.uniform(10, 20), np.random.uniform(1.20, 1.50)
        return 0, 1.0
    
    def get_reorder_point(self, demanda_base):
        """Punto de reorden: ~2 semanas de demanda promedio"""
        dias_demanda_promedio = demanda_base / 30  # Demanda diaria promedio
        return dias_demanda_promedio * 14  # 2 semanas
    
    def get_max_stock(self, demanda_base):
        """Stock máximo: ~2 meses de demanda"""
        dias_demanda_promedio = demanda_base / 30
        return dias_demanda_promedio * 60  # 2 meses
    
    def generar_demanda_diaria(self, producto_id, fecha, stock_actual, precio_unitario):
        """Genera una venta diaria coherente"""
        producto = self.productos[producto_id]
        demanda_base = producto['demanda_base']
        
        # Factores multiplicadores
        estacionalidad = self.get_seasonality_factor(fecha)
        dia_semana = self.get_day_of_week_factor(fecha)
        evento = self.get_event_impact(fecha)
        
        # Demanda diaria esperada (base/30 días)
        demanda_diaria = (demanda_base / 30) * estacionalidad * dia_semana * evento
        
        # Añadir ruido
        demanda_diaria = demanda_diaria * np.random.uniform(0.7, 1.3)
        
        # Aplicar probabilidad de venta (no todos los días hay venta)
        if np.random.random() > 0.25:  # 75% de probabilidad de venta
            # Elasticidad precio
            precio_elasticidad = np.random.uniform(-0.8, -1.2)
            precio_factor = (precio_unitario / 5.0) ** precio_elasticidad  # Precio base ~5
            demanda_diaria = demanda_diaria * precio_factor
        else:
            demanda_diaria = 0
        
        # No puede vender más de lo que existe en stock
        venta_real = min(int(max(0, demanda_diaria)), int(stock_actual))
        
        return venta_real, demanda_diaria
    
    def generar_produccion(self, producto_id, stock_actual, demanda_promedio_7dias):
        """Genera producción basada en nivel de stock y demanda esperada"""
        reorder_point = self.get_reorder_point(self.productos[producto_id]['demanda_base'])
        max_stock = self.get_max_stock(self.productos[producto_id]['demanda_base'])
        
        # Política de producción: Si stock < reorder point, producir
        if stock_actual < reorder_point:
            # Producir hasta el máximo, considerando demanda esperada
            cantidad_produccion = int(max_stock - stock_actual)
            # Limitar a producción realista
            cantidad_produccion = min(cantidad_produccion, int(demanda_promedio_7dias * 30))
            return max(0, cantidad_produccion)
        elif stock_actual > max_stock:
            return 0  # No producir si hay sobrestock
        else:
            # Producción preventiva si hay campaña esperada
            if np.random.random() > 0.7:
                return int(demanda_promedio_7dias * 7)
            return 0
    
    def generate_dataset_por_producto_empresa(self, producto_id, empresa, departamento):
        """Genera serie de tiempo completa (diaria) para un producto-empresa"""
        datos = []
        
        fecha_inicio = datetime(2020, 1, 1)
        fecha_fin = datetime(2025, 12, 31)
        
        producto = self.productos[producto_id]
        precio_base = np.random.uniform(3, 8)
        
        # Stock inicial
        stock_actual = int(self.get_max_stock(producto['demanda_base']) / 2)
        
        # Tracking para demanda promedio de últimos 7 días
        demandas_7dias = []
        
        canal = np.random.choice(self.canales)
        
        fecha_actual = fecha_inicio
        while fecha_actual <= fecha_fin:
            # Determinar si hay campaña este mes
            campaña = np.random.choice(self.campanas, p=[0.65, 0.15, 0.12, 0.08])
            descuento, factor_campana = self.get_campaign_discount(fecha_actual, campaña)
            
            # Precio con descuento
            precio_con_descuento = max(1, precio_base * (1 - descuento/100))
            
            # Generar venta diaria
            venta, demanda_esperada = self.generar_demanda_diaria(
                producto_id, fecha_actual, stock_actual, precio_con_descuento
            )
            
            # Tracking demanda histórica (últimos 7 días)
            demandas_7dias.append(demanda_esperada)
            if len(demandas_7dias) > 7:
                demandas_7dias.pop(0)
            demanda_promedio_7dias = np.mean(demandas_7dias) if demandas_7dias else 0
            
            # Stock después de venta
            stock_post_venta = stock_actual - venta
            
            # Registrar venta si la hay
            if venta > 0:
                # Asignar punto de venta si es Tienda Física
                punto_venta = np.random.choice(self.puntos_venta) if canal == 'Tienda Física' else None
                
                datos.append({
                    'Fecha': fecha_actual,
                    'Año': fecha_actual.year,
                    'Mes': fecha_actual.month,
                    'Dia': fecha_actual.day,
                    'Producto_id': producto_id,
                    'Producto_nombre': producto['nombre'],
                    'Empresa_cliente': empresa,
                    'Departamento_cliente': departamento,
                    'Canal_venta': canal,
                    'Punto_venta': punto_venta,
                    'Tipo_movimiento': 'Venta',
                    'Cantidad': venta,
                    'Stock_anterior': stock_actual,
                    'Stock_posterior': max(0, stock_post_venta),
                    'Precio_unitario': round(precio_con_descuento, 2),
                    'Descuento_pct': round(descuento, 2),
                    'Valor_total': round(venta * precio_con_descuento, 2),
                    'Campana': campaña,
                    'Costo_unitario': None,
                    'Estacionalidad_factor': round(self.get_seasonality_factor(fecha_actual), 3),
                    'Dia_semana_factor': round(self.get_day_of_week_factor(fecha_actual), 3),
                    'Evento_impacto': round(self.get_event_impact(fecha_actual), 3),
                })
            
            # Actualizar stock post-venta
            stock_actual = max(0, stock_post_venta)
            
            # Generar producción
            cantidad_produccion = self.generar_produccion(
                producto_id, stock_actual, demanda_promedio_7dias
            )
            
            # Registrar producción si la hay
            if cantidad_produccion > 0:
                costo_produccion = cantidad_produccion * precio_base * np.random.uniform(0.5, 0.7)
                datos.append({
                    'Fecha': fecha_actual,
                    'Año': fecha_actual.year,
                    'Mes': fecha_actual.month,
                    'Dia': fecha_actual.day,
                    'Producto_id': producto_id,
                    'Producto_nombre': producto['nombre'],
                    'Empresa_cliente': None,
                    'Departamento_cliente': None,
                    'Canal_venta': None,
                    'Punto_venta': None,
                    'Tipo_movimiento': 'Producción',
                    'Cantidad': cantidad_produccion,
                    'Stock_anterior': stock_actual,
                    'Stock_posterior': stock_actual + cantidad_produccion,
                    'Precio_unitario': None,
                    'Descuento_pct': None,
                    'Valor_total': round(costo_produccion, 2),
                    'Campana': None,
                    'Costo_unitario': round(precio_base, 2),
                    'Estacionalidad_factor': round(self.get_seasonality_factor(fecha_actual), 3),
                    'Dia_semana_factor': round(self.get_day_of_week_factor(fecha_actual), 3),
                    'Evento_impacto': round(self.get_event_impact(fecha_actual), 3),
                })
                stock_actual += cantidad_produccion
            
            fecha_actual += timedelta(days=1)
        
        return datos
    
    def generate_full_dataset(self):
        """Genera dataset completo de inventario"""
        all_data = []
        
        total_items = len(self.productos) * len(self.empresas)
        current = 0
        
        for producto_id in self.productos.keys():
            for empresa in self.empresas:
                current += 1
                # Asignar departamento aleatorio pero consistente por empresa
                departamento = self.departamentos[
                    hash(empresa) % len(self.departamentos)
                ]
                
                print(f"[{current}/{total_items}] {producto_id} - {empresa}")
                
                datos_series = self.generate_dataset_por_producto_empresa(
                    producto_id, empresa, departamento
                )
                all_data.extend(datos_series)
        
        # Convertir a DataFrame
        df = pd.DataFrame(all_data)
        
        # Ordenar por fecha y producto
        df = df.sort_values(['Fecha', 'Producto_id']).reset_index(drop=True)
        
        return df
    
    def save_dataset(self, df, filepath='Inventario_ML_Completo.csv'):
        """Guarda el dataset en CSV"""
        df.to_csv(filepath, index=False, encoding='utf-8')
        print(f"\n[GUARDADO] {filepath}")
        print(f"Tamaño: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Estadísticas de calidad
        print("\n" + "="*70)
        print("[ESTADÍSTICAS DEL DATASET]")
        print("="*70 + "\n")
        
        print(f"Período: {df['Fecha'].min().date()} a {df['Fecha'].max().date()}")
        print(f"Total de transacciones: {len(df):,}")
        print(f"Transacciones de VENTA: {len(df[df['Tipo_movimiento']=='Venta']):,}")
        print(f"Transacciones de PRODUCCIÓN: {len(df[df['Tipo_movimiento']=='Producción']):,}")
        
        print(f"\nUnidades vendidas totales: {df[df['Tipo_movimiento']=='Venta']['Cantidad'].sum():,.0f}")
        print(f"Unidades producidas totales: {df[df['Tipo_movimiento']=='Producción']['Cantidad'].sum():,.0f}")
        
        print(f"\nValor total de ventas: ${df[df['Tipo_movimiento']=='Venta']['Valor_total'].sum():,.2f}")
        print(f"Costo total de producción: ${df[df['Tipo_movimiento']=='Producción']['Valor_total'].sum():,.2f}")
        
        print("\n--- Venta por Producto ---")
        venta_por_prod = df[df['Tipo_movimiento']=='Venta'].groupby('Producto_id').agg({
            'Cantidad': ['count', 'sum', 'mean'],
            'Valor_total': 'sum'
        }).round(2)
        print(venta_por_prod)
        
        print("\n--- Producción por Producto ---")
        prod_por_prod = df[df['Tipo_movimiento']=='Producción'].groupby('Producto_id').agg({
            'Cantidad': ['count', 'sum', 'mean'],
            'Valor_total': 'sum'  # Costo total
        }).round(2)
        print(prod_por_prod)
        
        print("\n--- Ventas por Punto de Venta ---")
        venta_por_pv = df[df['Tipo_movimiento']=='Venta'].groupby('Punto_venta').agg({
            'Cantidad': ['count', 'sum', 'mean'],
            'Valor_total': 'sum'
        }).round(2)
        print(venta_por_pv)
        
        print("\n--- Stock Promedio por Producto ---")
        stock_promedio = df.groupby('Producto_id')['Stock_posterior'].mean().round(0)
        print(stock_promedio)
        
        print("\n--- Transacciones por Mes (primeros 12) ---")
        df['Año-Mes'] = df['Fecha'].dt.to_period('M')
        transac_mes = df.groupby('Año-Mes').size()
        print(transac_mes.head(12))
        
        print("\n" + "="*70)


if __name__ == '__main__':
    print("\n" + "="*70)
    print("[GENERANDO] Dataset Completo de Inventario ML")
    print("="*70)
    print("Período: 2020-01-01 a 2025-12-31")
    print("Movimientos: Ventas Diarias + Producción bajo demanda")
    print("Variables: Stock en tiempo real, Demanda, Campaña, Estacionalidad")
    print("="*70 + "\n")
    
    generator = InventarioMLDataGenerator()
    df = generator.generate_full_dataset()
    
    print("\n" + "="*70)
    print("[DATASET COMPLETADO]")
    print("="*70)
    
    generator.save_dataset(df, 'Inventario_ML_v3.csv')
    
    print("\n✅ Listo para ML!")
