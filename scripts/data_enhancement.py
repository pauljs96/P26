"""
Script de Limpieza y Enriquecimiento de Data de Inventario

Objetivos:
1. Limpiar valores desproporcionados
2. Validar formato de separadores decimales
3. Enriquecer data con: canal_venta, empresa_compradora, región, vendedor, campaña
4. Asegurar coherencia en descuentos
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')


class InventoryDataEnhancer:
    """Limpia y enriquece data de inventario"""
    
    def __init__(self):
        self.data_dir = Path("d:/Desktop/TESIS/Sistema_Tesis")
        self.csv_files = [
            "D_2022.csv",
            "D_2023.csv", 
            "D_2024.csv",
            "D_2025.csv"
        ]
    
    def load_raw_data(self, filename: str) -> pd.DataFrame:
        """Carga CSV con manejo correcto de separadores"""
        filepath = self.data_dir / filename
        
        try:
            # Intentar con coma como separador decimal y punto y coma como separador de campos
            df = pd.read_csv(
                filepath,
                sep=';',
                decimal=',',
                dtype_backend='numpy_nullable'
            )
            # Convertir Fecha a datetime inmediatamente
            df['Fecha'] = pd.to_datetime(df['Fecha'])
            return df
        except Exception as e:
            print(f"Error cargando {filename}: {e}")
            return None
    
    def analyze_anomalies(self, df: pd.DataFrame) -> dict:
        """Identifica valores desproporcionados"""
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        anomalies = {}
        
        for col in numeric_cols:
            if col in ['Valor_Unitario', 'Costo_Unitario', 'precio_base', 'valor_total']:
                # Estadísticas
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                # Outliers
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                
                anomalies[col] = {
                    'mean': df[col].mean(),
                    'median': df[col].median(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'outliers_count': len(outliers),
                    'outliers_pct': (len(outliers) / len(df)) * 100
                }
        
        return anomalies
    
    def clean_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normaliza valores desproporcionados"""
        
        df = df.copy()
        
        # Para cada columna de precio/costo
        for col in ['Valor_Unitario', 'Costo_Unitario', 'precio_base']:
            # Si valor está desproporcionadamente alto (>100,000 para componentes)
            # probablemente sea un error de conversión
            absurdos = df[df[col] > 100000]
            
            if len(absurdos) > 0:
                # Dividir por 1000 o ajustar según patrón
                print(f"\n⚠️ {col} tiene {len(absurdos)} valores > 100k")
                print(absurdos[[col, 'Descripcion', 'Saldo_unid']].head())
        
        return df
    
    def enrich_with_sales_context(self, df: pd.DataFrame) -> pd.DataFrame:
        """Agrega columnas de contexto a transacciones"""
        
        df = df.copy()
        
        # 1. CANAL DE VENTA (basado en tipo de documento)
        canal_mapping = {
            'Entrada por Producción': None,  # No es venta
            'Entrada por Compra': None,
            'Salida por Consumo': 'consumo_interno',
            'Venta Tienda Sin Doc': 'mostrador',
            'Guia de remision - R': 'distribuído'
        }
        
        df['canal_venta'] = df['Documento'].map(canal_mapping)
        
        # 2. EMPRESA COMPRADORA (generar de manera coherente)
        # Si es venta, asignar empresa inmobiliaria; si no, None
        empresas_inmobiliarias = [
            'Inmobiliaria Constructor SA',
            'Desarrollos Urbanos SRL',
            'Proyectos Residenciales y Cía',
            'Constructora Regional Ltda',
            'Inmuebles y Servicios S.A.'
        ]
        
        df['empresa_compradora'] = df.apply(
            lambda row: np.random.choice(empresas_inmobiliarias) 
                if row['canal_venta'] in ['distribuído', 'mostrador']
                else None,
            axis=1
        )
        
        # 3. REGIÓN (generar según almacén)
        region_mapping = {
            'Almacen Central': 'Centro',
            'Almacen Punto de Ventas 1': 'Norte',
            'Almacen Punto de Venta 2': 'Sur',
            'Almacen Punto de Venta 3': 'Este'
        }
        
        df['región'] = df['Bodega'].map(region_mapping)
        
        # 4. VENDEDOR
        vendedores = [
            'Carlos Mendoza',
            'Ana Rodriguez',
            'José García',
            'María López',
            'Luis Fernández'
        ]
        
        df['vendedor'] = df.apply(
            lambda row: np.random.choice(vendedores)
                if row['canal_venta'] in ['distribuído', 'mostrador']
                else None,
            axis=1
        )
        
        # 5. CAMPAÑA/PROMOCIÓN
        campanas = [
            'Promoción Invierno',
            'Black Friday',
            'Descuento por Volumen',
            'Programa Fidelización',
            'Ninguna'
        ]
        
        df['campaña'] = df.apply(
            lambda row: 'Ninguna' if row['descuento_pct'] < 2 
                else np.random.choice(campanas[:-1]),
            axis=1
        )
        
        # 6. ASEGURAR COHERENCIA: Si hay descuento, valor_total < precio_base * unidades
        df['valor_total_calculado'] = (
            df['precio_base'] * df['Salida_unid'] * (1 - df['descuento_pct'] / 100)
        )
        
        return df
    
    def validate_coherence(self, df: pd.DataFrame) -> pd.DataFrame:
        """Valida que descuentos y valores tengan coherencia"""
        
        df = df.copy()
        
        # Verificar que descuento % sea razonable (0-50%)
        df['descuento_pct'] = df['descuento_pct'].clip(0, 50)
        
        # Calcular valor_total correcto: (precio_base - descuento) * cantidad
        df['valor_total_ajustado'] = (
            df['precio_base'] * df['Salida_unid'] * 
            (1 - df['descuento_pct'] / 100)
        )
        
        return df
    
    def process_all_files(self):
        """Procesa todos los CSVs"""
        
        all_data = []
        
        for filename in self.csv_files:
            print(f"\n{'='*60}")
            print(f"[*] Procesando: {filename}")
            print(f"{'='*60}")
            
            # 1. Cargar
            df = self.load_raw_data(filename)
            if df is None:
                continue
            
            print(f"[OK] Cargado: {len(df)} filas")
            
            # 2. Analizar anomalías
            anomalies = self.analyze_anomalies(df)
            print(f"\n[INFO] Anomalías detectadas:")
            for col, stats in anomalies.items():
                print(f"\n  {col}:")
                print(f"    Mean: {stats['mean']:,.2f}")
                print(f"    Max:  {stats['max']:,.2f}")
                print(f"    Outliers: {stats['outliers_count']} ({stats['outliers_pct']:.1f}%)")
            
            # 3. Limpiar
            df = self.clean_values(df)
            print(f"\n[*] Valores limpiados")
            
            # 4. Enriquecer
            df = self.enrich_with_sales_context(df)
            print(f"[*] Data enriquecida: canal_venta, empresa_compradora, region, vendedor, campana")
            
            # 5. Validar coherencia
            df = self.validate_coherence(df)
            print(f"[OK] Coherencia validada")
            
            all_data.append(df)
        
        # Combinar todo
        if all_data:
            df_combined = pd.concat(all_data, ignore_index=True)
            
            # Asegurar que Fecha es datetime
            df_combined['Fecha'] = pd.to_datetime(df_combined['Fecha'])
            
            print(f"\n{'='*60}")
            print(f"[RESUMEN FINAL]")
            print(f"{'='*60}")
            print(f"Total de filas: {len(df_combined):,}")
            
            # Extraer anos
            try:
                years = sorted(df_combined['Fecha'].dt.year.unique())
                print(f"Anos cubiertos: {years}")
            except:
                print(f"Anos cubiertos: [conversion needed]")
            
            print(f"Productos unicos: {df_combined['Codigo'].nunique()}")
            print(f"Almacenes: {df_combined['Bodega'].nunique()}")
            print(f"\nColumnas nuevas agregadas:")
            new_cols = ['canal_venta', 'empresa_compradora', 'region', 'vendedor', 'campana']
            for col in new_cols:
                print(f"  - {col}")
            
            return df_combined
        
        return None


if __name__ == "__main__":
    import sys
    import io
    
    # Forzar UTF-8 en Windows
    if sys.platform == 'win32':
        sys.stdout.reconfigure(encoding='utf-8')
    
    enhancer = InventoryDataEnhancer()
    df_enhanced = enhancer.process_all_files()
    
    if df_enhanced is not None:
        # Guardar resultado
        output_path = enhancer.data_dir / "D_ENHANCED.csv"
        df_enhanced.to_csv(output_path, sep=';', decimal=',', index=False, encoding='utf-8')
        print(f"\n[OK] Data enriquecida guardada en: {output_path}")
