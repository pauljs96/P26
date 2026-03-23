"""
Generador de Dataset Completo de Inventario para ML - v4
Incluye: 200+ Productos, Costo_unitario CONSTANTE por producto
Transacciones diarias múltiples con coherencia lógica
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class InventarioMLDataGenerator:
    def __init__(self, seed=42):
        np.random.seed(seed)
        
        # CARGAR PRODUCTOS DESDE ARCHIVO
        print("[CARGANDO] Productos desde archivo...")
        df_productos = pd.read_csv('productos_seleccionados.csv')
        
        # Crear diccionario de productos con costo FIJO
        self.productos = {}
        self.costo_unitario_por_producto = {}
        
        for idx, row in df_productos.iterrows():
            codigo = str(row['codigo']).strip().strip("'")
            descripcion = str(row['descripcion']).strip()
            
            # GENERAR COSTO UNA SOLA VEZ por producto
            costo_unitario = np.random.uniform(2, 150)  # Rango realista
            
            self.productos[codigo] = {
                'nombre': descripcion[:80],  # Limitar longitud
                'demanda_base': np.random.randint(30, 300),  # Demanda varía por producto
                'lead_time': np.random.randint(5, 20)
            }
            
            self.costo_unitario_por_producto[codigo] = round(costo_unitario, 2)
        
        print(f"✓ {len(self.productos)} productos cargados")
        
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
        
        # Campañas
        self.campanas = ['Ninguna', 'Promo Constructor', 'Descuento Volumen', 'Black Friday']
        
    def get_seasonality_factor(self, fecha):
        """Factor de estacionalidad por mes"""
        month = fecha.month
        seasonality = {
            1: 0.65, 2: 1.20, 3: 1.35, 4: 1.40, 5: 1.25,
            6: 0.75, 7: 0.70, 8: 0.68, 9: 1.10, 10: 1.30,
            11: 1.25, 12: 0.65
        }
        return seasonality.get(month, 0.8)
    
    def get_day_of_week_factor(self, fecha):
        """Factor de venta por día de semana"""
        dow = fecha.weekday()
        factors = {
            0: 1.10, 1: 1.15, 2: 1.10, 3: 0.95, 4: 0.90, 5: 0.50, 6: 0.30
        }
        return factors.get(dow, 1.0)
    
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
        """Punto de reorden"""
        dias_demanda_promedio = demanda_base / 30
        return dias_demanda_promedio * 14
    
    def get_max_stock(self, demanda_base):
        """Stock máximo"""
        dias_demanda_promedio = demanda_base / 30
        return dias_demanda_promedio * 60
    
    def generar_demanda_diaria(self, producto_id, fecha, stock_actual, precio_unitario):
        """Genera una venta diaria coherente"""
        producto = self.productos[producto_id]
        demanda_base = producto['demanda_base']
        
        estacionalidad = self.get_seasonality_factor(fecha)
        dia_semana = self.get_day_of_week_factor(fecha)
        
        demanda_diaria = (demanda_base / 30) * estacionalidad * dia_semana
        demanda_diaria = demanda_diaria * np.random.uniform(0.7, 1.3)
        
        # Mayor probabilidad de venta
        if np.random.random() > 0.20:  # 80% de probabilidad
            precio_elasticidad = np.random.uniform(-0.8, -1.2)
            precio_factor = (precio_unitario / 50.0) ** precio_elasticidad
            demanda_diaria = demanda_diaria * precio_factor
        else:
            demanda_diaria = 0
        
        venta_real = min(int(max(0, demanda_diaria)), int(stock_actual))
        return venta_real, demanda_diaria
    
    def generar_produccion(self, producto_id, stock_actual, demanda_promedio_7dias):
        """Genera producción basada en nivel de stock"""
        reorder_point = self.get_reorder_point(self.productos[producto_id]['demanda_base'])
        max_stock = self.get_max_stock(self.productos[producto_id]['demanda_base'])
        
        if stock_actual < reorder_point:
            cantidad_produccion = int(max_stock - stock_actual)
            cantidad_produccion = min(cantidad_produccion, int(demanda_promedio_7dias * 30))
            return max(0, cantidad_produccion)
        elif stock_actual > max_stock:
            return 0
        else:
            if np.random.random() > 0.7:
                return int(demanda_promedio_7dias * 7)
            return 0
    
    def generate_dataset_por_producto_empresa(self, producto_id, empresa, departamento, costo_unitario_fijo):
        """Genera serie de tiempo para producto-empresa"""
        datos = []
        
        fecha_inicio = datetime(2022, 1, 1)  # Desde 2022 para más datos reales
        fecha_fin = datetime(2025, 12, 31)
        
        producto = self.productos[producto_id]
        precio_base = np.random.uniform(costo_unitario_fijo * 1.5, costo_unitario_fijo * 3.5)  # Margen realista
        
        stock_actual = int(self.get_max_stock(producto['demanda_base']) / 2)
        demandas_7dias = []
        canal = np.random.choice(self.canales)
        
        fecha_actual = fecha_inicio
        while fecha_actual <= fecha_fin:
            campaña = np.random.choice(self.campanas, p=[0.65, 0.15, 0.12, 0.08])
            descuento, factor_campana = self.get_campaign_discount(fecha_actual, campaña)
            precio_con_descuento = max(1, precio_base * (1 - descuento/100))
            
            # MÚLTIPLES VENTAS POR DÍA (2-5 transacciones)
            num_ventas_diarias = np.random.randint(1, 6) if np.random.random() > 0.3 else 1
            
            for _ in range(num_ventas_diarias):
                venta, demanda_esperada = self.generar_demanda_diaria(
                    producto_id, fecha_actual, stock_actual, precio_con_descuento
                )
                
                demandas_7dias.append(demanda_esperada)
                if len(demandas_7dias) > 7:
                    demandas_7dias.pop(0)
                
                if venta > 0:
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
                        'Stock_posterior': max(0, stock_actual - venta),
                        'Precio_unitario': round(precio_con_descuento, 2),
                        'Descuento_pct': round(descuento, 2),
                        'Valor_total': round(venta * precio_con_descuento, 2),
                        'Campana': campaña,
                        'Costo_unitario': None,
                    })
                    
                    stock_actual = max(0, stock_actual - venta)
            
            # Producción
            demanda_promedio_7dias = np.mean(demandas_7dias) if demandas_7dias else 0
            cantidad_produccion = self.generar_produccion(producto_id, stock_actual, demanda_promedio_7dias)
            
            if cantidad_produccion > 0:
                # Costo de producción = costo_unitario × factor aleatorio (ineficiencia)
                costo_total = cantidad_produccion * costo_unitario_fijo * np.random.uniform(0.9, 1.1)
                
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
                    'Valor_total': round(costo_total, 2),
                    'Campana': None,
                    'Costo_unitario': costo_unitario_fijo,  # ← CONSTANTE POR PRODUCTO
                })
                
                stock_actual += cantidad_produccion
            
            fecha_actual += timedelta(days=1)
        
        return datos
    
    def generate_full_dataset(self):
        """Genera dataset completo"""
        all_data = []
        
        total_items = len(self.productos) * len(self.empresas)
        current = 0
        
        for producto_id in list(self.productos.keys())[:200]:  # Usar primeros 200
            costo_unitario_fijo = self.costo_unitario_por_producto[producto_id]
            
            for empresa in self.empresas:
                current += 1
                departamento = self.departamentos[hash(empresa) % len(self.departamentos)]
                
                if current % 10 == 0:
                    print(f"  [{current}/1000] {producto_id} - {empresa}")
                
                datos_series = self.generate_dataset_por_producto_empresa(
                    producto_id, empresa, departamento, costo_unitario_fijo
                )
                all_data.extend(datos_series)
        
        df = pd.DataFrame(all_data)
        df = df.sort_values(['Fecha', 'Producto_id']).reset_index(drop=True)
        
        return df
    
    def save_dataset(self, df, filepath='Inventario_ML_Completo_v4.csv'):
        """Guarda el dataset"""
        df.to_csv(filepath, index=False, encoding='utf-8')
        print(f"\n[GUARDADO] {filepath}")
        print(f"Tamaño: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        print("\n" + "="*70)
        print("[ESTADÍSTICAS DEL DATASET]")
        print("="*70)
        
        print(f"\nPeríodo: {df['Fecha'].min().date()} a {df['Fecha'].max().date()}")
        print(f"Total de transacciones: {len(df):,}")
        print(f"Transacciones de VENTA: {len(df[df['Tipo_movimiento']=='Venta']):,}")
        print(f"Transacciones de PRODUCCIÓN: {len(df[df['Tipo_movimiento']=='Producción']):,}")
        print(f"Productos únicos: {df['Producto_id'].nunique()}")
        
        print(f"\nUnidades vendidas totales: {df[df['Tipo_movimiento']=='Venta']['Cantidad'].sum():,.0f}")
        print(f"Unidades producidas totales: {df[df['Tipo_movimiento']=='Producción']['Cantidad'].sum():,.0f}")
        
        print(f"\nValor total de ventas: ${df[df['Tipo_movimiento']=='Venta']['Valor_total'].sum():,.2f}")
        print(f"Costo total de producción: ${df[df['Tipo_movimiento']=='Producción']['Valor_total'].sum():,.2f}")
        
        print("\n" + "="*70)


if __name__ == '__main__':
    print("\n" + "="*70)
    print("[GENERANDO] Dataset Inventario ML v4 - 200+ Productos")
    print("="*70)
    print("Período: 2022-01-01 a 2025-12-31")
    print("Costo_unitario: CONSTANTE POR PRODUCTO")
    print("Transacciones: Múltiples por día")
    print("="*70 + "\n")
    
    generator = InventarioMLDataGenerator()
    df = generator.generate_full_dataset()
    
    print("\n" + "="*70)
    print("[DATASET COMPLETADO]")
    print("="*70)
    
    generator.save_dataset(df)
    
    print("\n✅ Listo para ML!")
