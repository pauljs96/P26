"""
Enriquecimiento Inteligente de CSV de Inventario - Por Archivo

Uso: python enrich_single_csv.py ARCHIVO.csv

Procesa:
1. Corrige valores desproporcionados (columnas J-N)
2. Agrega contexto realista peruano:
   - canal_venta: Online / Tienda Física
   - empresa_compradora: Solo para ventas
   - departamento: Departamentos del Perú
   - vendedor: Solo para ventas
   - campana: Coherente con descuentos y ramo manufacturero
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import warnings

warnings.filterwarnings('ignore')


class InventoryEnricher:
    """Enriquecedor de data de inventario por CSV"""
    
    # Datos realistas para Perú
    EMPRESAS_COMPRADORAS = [
        'Sistemas Constructivos SA',
        'Inmobiliaria Andina SRL',
        'Constructora Pacífico Ltda',
        'Desarrollos Urbanos del Sur',
        'Proyectos Inmobiliarios Nacional',
        'Empresa Constructora Regional',
        'Infraestructura Peruana SAC',
        'Grupo Constructor Andino'
    ]
    
    DEPARTAMENTOS_PERU = [
        'Lima',
        'Arequipa',
        'Cusco',
        'Tacna',
        'Junín',
        'Callao',
        'La Libertad',
        'Piura',
        'Lambayeque',
        'Ancash'
    ]
    
    VENDEDORES = [
        'Juan Quispe',
        'María Sánchez',
        'Carlos Romero',
        'Rosa Valdez',
        'Diego Flores',
        'Andrea Morales',
        'Roberto Guzmán',
        'Sofía Hernández'
    ]
    
    CAMPANAS_MANUFACTURERO = {
        'alto': [  # descuentos >= 6%
            'Promo Constructor 2024',
            'Descuento por Volumen',
            'Alianza Constructora',
            'Programa Fidelización',
            'Oferta Especial Proyectos'
        ],
        'medio': [  # descuentos 3-5%
            'Promo Mensual',
            'Descuento Referencia',
            'Campaña Primavera',
            'Oferta Temporada'
        ],
        'bajo': [  # descuentos < 3%
            'Precio Regular',
            'Venta Normal',
            'Precio Lista'
        ]
    }
    
    def __init__(self, filepath: str):
        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {filepath}")
        self.df = self.load_csv()
    
    def load_csv(self) -> pd.DataFrame:
        """Carga CSV con configuración correcta"""
        df = pd.read_csv(
            self.filepath,
            sep=';',
            decimal=',',
            dtype_backend='numpy_nullable'
        )
        df['Fecha'] = pd.to_datetime(df['Fecha'])
        return df
    
    def extract_product_type(self, descripcion: str) -> str:
        """Extrae tipo de producto para determinar rango de precios"""
        desc_lower = descripcion.lower()
        
        if 'caja' in desc_lower:
            return 'caja'
        elif 'bandeja' in desc_lower:
            return 'bandeja'
        elif 'tuberia' in desc_lower or 'conduit' in desc_lower:
            return 'tuberia'
        elif 'gabinete' in desc_lower:
            return 'gabinete'
        elif 'tapa' in desc_lower:
            return 'tapa'
        elif 'curva' in desc_lower or 'union' in desc_lower or 'tee' in desc_lower:
            return 'accesorios'
        else:
            return 'otros'
    
    def get_price_range(self, product_type: str) -> tuple:
        """Retorna (costo_min, costo_max, precio_min, precio_max) en USD/unidad
        
        Rango: Costo 0.50-5, Precio 1-10
        """
        ranges = {
            'caja': (1.50, 3.50, 4.00, 8.00),
            'bandeja': (2.00, 4.00, 5.00, 9.00),
            'tuberia': (1.00, 2.50, 2.50, 7.00),
            'gabinete': (2.50, 4.50, 6.00, 10.00),
            'tapa': (0.80, 2.00, 2.00, 5.00),
            'accesorios': (0.60, 1.50, 1.50, 4.50),
            'otros': (1.50, 3.00, 3.50, 8.50)
        }
        return ranges.get(product_type, ranges['otros'])
    
    def fix_and_enrich_prices(self, row: pd.Series) -> dict:
        """Corrige precios a rango realista (1-10 USD/unidad)
        
        Estructura:
        - Valor_Unitario: Precio venta sin descuento ($1-10)
        - Costo_Unitario: Costo compra/producción ($0.50-5)
        - precio_base: Mismo que Valor_Unitario (antes de descuento)
        - descuento_pct: Descuento en % (0-40%)
        - valor_total: Valor_Unitario × cantidad × (1 - descuento%/100)
        """
        
        # Determinar tipo de producto
        product_type = self.extract_product_type(row['Descripcion'])
        costo_min, costo_max, precio_min, precio_max = self.get_price_range(product_type)
        
        # Generar costo unitario realista ($0.50-5)
        costo_unitario = np.random.uniform(costo_min, costo_max)
        costo_unitario = round(costo_unitario, 2)
        
        # Generar precio venta unitario ($1-10)
        # Debe ser > costo (margen de ganancia mínimo 30%)
        precio_minimo = max(precio_min, costo_unitario * 1.3)
        valor_unitario = np.random.uniform(precio_minimo, precio_max)
        valor_unitario = round(valor_unitario, 2)
        
        # precio_base = valor sin descuento = Valor_Unitario
        precio_base = valor_unitario
        
        # Validar/ajustar descuento (0-40%)
        descuento_pct = float(row['descuento_pct']) if pd.notna(row['descuento_pct']) else 0
        descuento_pct = np.clip(descuento_pct, 0, 40)
        descuento_pct = round(descuento_pct, 2)
        
        # Cantidad: usar Salida_unid o Entrada_unid
        cantidad = float(row['Salida_unid']) if float(row['Salida_unid']) > 0 else float(row['Entrada_unid'])
        cantidad = max(int(cantidad), 1)
        
        # valor_total = precio_base × cantidad × (1 - descuento_pct/100)
        valor_total = precio_base * cantidad * (1 - descuento_pct / 100)
        valor_total = round(valor_total, 2)
        
        return {
            'Valor_Unitario': valor_unitario,
            'Costo_Unitario': costo_unitario,
            'precio_base': precio_base,
            'descuento_pct': descuento_pct,
            'valor_total': valor_total,
            'product_type': product_type
        }
    
    def add_enriched_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Agrega columnas de contexto realista"""
        
        df = df.copy()
        
        # 1. CANAL DE VENTA
        def get_canal(row):
            documento = row['Documento']
            if 'Entrada' in documento:
                return None  # No es venta
            elif 'Salida por Consumo' in documento:
                return 'Tienda Física'  # Consumo interno/mostrador
            elif 'Venta Tienda' in documento:
                return 'Tienda Física'
            elif 'Guia de remision' in documento:
                return 'Online'  # Distribución
            else:
                return None
        
        df['canal_venta'] = df.apply(get_canal, axis=1)
        
        # 2. EMPRESA COMPRADORA - Solo para ventas
        def get_empresa(row):
            if pd.notna(row['canal_venta']):  # Es venta
                return np.random.choice(self.EMPRESAS_COMPRADORAS)
            return None
        
        df['empresa_compradora'] = df.apply(get_empresa, axis=1)
        
        # 3. DEPARTAMENTO (según almacén o aleatorio)
        def get_departamento(row):
            bodega = row['Bodega']
            if 'Central' in bodega:
                return 'Lima'
            elif 'Ventas 1' in bodega:
                return np.random.choice(['Arequipa', 'Callao', 'La Libertad'])
            elif 'Ventas 2' in bodega:
                return np.random.choice(['Cusco', 'Junín', 'Piura'])
            else:
                return np.random.choice(self.DEPARTAMENTOS_PERU)
        
        df['departamento'] = df.apply(get_departamento, axis=1)
        
        # 4. VENDEDOR - Solo para ventas
        def get_vendedor(row):
            if pd.notna(row['canal_venta']):  # Es venta
                return np.random.choice(self.VENDEDORES)
            return None
        
        df['vendedor'] = df.apply(get_vendedor, axis=1)
        
        # 5. CAMPAÑA - Coherente con descuentos
        def get_campana(row):
            if pd.notna(row['canal_venta']):  # Es venta
                descuento = float(row['descuento_pct'])
                if descuento >= 6:
                    return np.random.choice(self.CAMPANAS_MANUFACTURERO['alto'])
                elif descuento >= 3:
                    return np.random.choice(self.CAMPANAS_MANUFACTURERO['medio'])
                else:
                    return np.random.choice(self.CAMPANAS_MANUFACTURERO['bajo'])
            return None
        
        df['campana'] = df.apply(get_campana, axis=1)
        
        return df
    
    def process(self) -> pd.DataFrame:
        """Procesa completo: corrige precios + enriquece datos"""
        
        print(f"\n{'='*70}")
        print(f"[PROCESANDO] {self.filepath.name}")
        print(f"{'='*70}\n")
        
        # 1. Corregir valores de precios
        print("[1/2] Corrigiendo valores de precios (1-10 USD/unidad)...")
        price_fixes = self.df.apply(self.fix_and_enrich_prices, axis=1)
        price_df = pd.DataFrame(list(price_fixes))
        
        # Aplicar correcciones
        self.df['Valor_Unitario'] = price_df['Valor_Unitario']
        self.df['Costo_Unitario'] = price_df['Costo_Unitario']
        self.df['precio_base'] = price_df['precio_base']
        self.df['descuento_pct'] = price_df['descuento_pct']
        self.df['valor_total'] = price_df['valor_total']
        
        n_registros = len(self.df)
        print(f"  [OK] {n_registros} registros con precios coherentes")
        print(f"  [OK] Valor Unitario: ${self.df['Valor_Unitario'].min():.2f} - ${self.df['Valor_Unitario'].max():.2f}")
        print(f"  [OK] Costo Unitario: ${self.df['Costo_Unitario'].min():.2f} - ${self.df['Costo_Unitario'].max():.2f}")
        print(f"  [OK] Valor Total: ${self.df['valor_total'].min():.2f} - ${self.df['valor_total'].max():.2f}")
        
        # 2. Enriquecer con contexto
        print("\n[2/2] Enriqueciendo con datos peruanos...")
        self.df = self.add_enriched_columns(self.df)
        
        ventas = self.df['canal_venta'].notna().sum()
        print(f"  [OK] Registros de ventas: {ventas}")
        print(f"  [OK] Columna 'canal_venta': {list(self.df['canal_venta'].unique())}")
        print(f"  [OK] Empresas compradoras asignadas: {self.df['empresa_compradora'].notna().sum()}")
        print(f"  [OK] Departamentos: {self.df['departamento'].nunique()} unicos")
        print(f"  [OK] Vendedores asignados: {self.df['vendedor'].notna().sum()}")
        print(f"  ✓ Campañas: {self.df['campana'].unique()}")
        
        return self.df
    
    def save(self, df: pd.DataFrame):
        """Guarda CSV enriquecido con punto decimal"""
        output_name = self.filepath.stem + "_ENHANCED.csv"
        output_path = self.filepath.parent / output_name
        
        df.to_csv(
            output_path,
            sep=';',
            decimal='.',  # Punto decimal, no coma
            index=False,
            encoding='utf-8'
        )
        
        print(f"\n{'='*70}")
        print(f"[GUARDADO] {output_path}")
        print(f"{'='*70}\n")
        
        return output_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\nUso: python enrich_single_csv.py <ARCHIVO.csv>")
        print("\nEjemplo: python enrich_single_csv.py D_2023.csv")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    
    try:
        enricher = InventoryEnricher(csv_file)
        df_enriched = enricher.process()
        enricher.save(df_enriched)
        
        print("[OK] Enriquecimiento completado exitosamente!")
        
    except Exception as e:
        print(f"\n[ERROR] {str(e)}")
        sys.exit(1)
