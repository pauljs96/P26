#!/usr/bin/env python3
"""
Demostración: Cómo usar el sistema v4 desde el dashboard

Este script documentadescribe los pasos que un usuario seguiría:
1. Subir Inventario_ML_Completo_v4.csv 
2. El dashboard detecta automáticamente que es v4
3. Se muestra información específica de v4
4. Se pueden analizar productos con datos reales
"""

print("""
╔════════════════════════════════════════════════════════════════════════════╗
║            GUÍA DE USO: SISTEMA V4 EN EL DASHBOARD                        ║
╚════════════════════════════════════════════════════════════════════════════╝

1. INICIAR EL DASHBOARD
─────────────────────
   $ cd d:/Desktop/TESIS/Sistema_Tesis
   $ streamlit run src/ui/dashboard.py

2. AUTENTICARSE
─────────────────────
   - Ingresa con email de admin
   - (El system está integrado con Supabase)

3. SUBIR DATASET V4
─────────────────────
   - En el sidebar: "📤 Subir Datos"
   - Selecciona: Inventario_ML_Completo_v4.csv
   - El sistema procesará automáticamente (~1-2 min por 1.9M registros)

4. DETECCIÓN AUTOMÁTICA
─────────────────────
   El dashboard automáticamente:
   ✓ Detecta que es Dataset v4 (verifica columnas)
   ✓ Muestra: "✅ Dataset v4 (Inventario ML Completo)"
   ✓ Carga pipeline v4 optimizado
   ✓ Procesa con coherencia de stock validation

5. KPIs ESPECÍFICOS PARA V4
─────────────────────
   La seción "📊 Resumen de Datos" muestra:
   
   PARA V4:
   ├─ 📦 Productos: 200
   ├─ 🛒 Ventas: 1,622,540
   ├─ 🏭 Producciones: 299,070
   ├─ 📋 Movimientos: 1,921,610
   └─ CARACTERÍSTICAS:
       ├─ 📍 Canales: 3 (Online, Tienda Física, Producción)
       ├─ 🏢 Clientes: ~50-100 empresas
       ├─ 📢 Campañas: N campañas activas
       └─ 📌 Granularidad: Diaria (2022-2025)

6. ANÁLISIS DISPONIBLES
─────────────────────
   Con v4 tienes acceso a:
   
   A) Análisis Individual (por Producto)
      ├─ Demanda mensual (agregada de ventas diarias)
      ├─ Stock final del mes
      ├─ Comparación de modelos (Baseline vs ETS vs Random Forest)
      └─ Diagnóstico de series

   B) Análisis de Grupo
      ├─ Resumen comparativo global
      ├─ Clasificación ABC automática
      ├─ Recomendaciones masivas
      └─ Retrospective backtest de modelos

   C) Análisis Avanzado (v4 habilita estos)
      ├─ Demanda por cliente (Empresa_cliente)
      ├─ Demanda por canal (Online vs Tienda Física)
      ├─ Análisis de impacto de campañas
      └─ Correlación descuento vs cantidad vendida

7. TABLA COMPARATIVA: LEGACY vs V4
──────────────────────────────────────

   ASPECTO                  LEGACY              V4 (v4)
   ─────────────────────────────────────────────────────────
   Fuente                   ERP (Mayor-Aux)    Accountng + Sales
   Granularidad             Mensual agregado   Diaria (múltiple x día)
   Productos                ~50 synthéticos    200 reales
   Período                  2022-2025          2022-2025
   Movimientos              Entrada/Salida     Venta/Producción  
   Columnas Contexto        Bodega             Canal, Punto, Cliente
   Stock Coherencia         Manual (guías)     Automática validada
   Canales de Venta         No                 Si (3 tipos)
   Datos de Cliente         No                 Si (empresa, depto)
   Campañas                 No                 Si
   Descuentos               No                 Si
   
   TIEMPO PROCESAMIENTO
   ───────────────────
   Legacy: ~15 seg (pequeño)
   v4:     ~90 seg (1.9M registros)

8. CARACTERÍSTICAS ESPECIALES DE V4
──────────────────────────────────────

   ✓ Stock Coherencia Automática:
     - Valida que Venta: anterior - cantidad = posterior
     - Valida que Prod: anterior + cantidad = posterior
     - Reporta incoherencias (tolerancia 1%)

   ✓ Múltiples Movimientos/Día:
     - Permite análisis diario fino
     - ML models pueden aprender patrones intra-mes

   ✓ Contexto Rico:
     - Demanda por canal (Online vs Tienda)
     - Demanda por cliente real
     - Impacto de campañas
     - Análisis de descuentos

   ✓ 200 Productos Reales:
     - Códigos del sistema contable real
     - Precios y costos realistas
     - Distribución de demanda realista (ABC)

9. PRÓXIMOS PASOS RECOMENDADOS
──────────────────────────────────────

   1. Entrenar modelos ML con v4:
      ├─ ETS con serie diaria agregada mensual
      ├─ Random Forest con features de cliente/canal
      └─ SARIMA si lo prefieres

   2. Crear reportes por canal:
      ├─ "Demanda Online vs Tienda Física"
      ├─ Análisis de fill rate por canal
      └─ Pronósticos específicos

   3. Análisis de campañas:
      ├─ Impacto de campaña en demanda
      ├─ ROI por campaña
      └─ Timing óptimo de campañas

   4. Sistema de alertas:
      ├─ Stock bajo threshold
      ├─ Demanda anormal
      └─ Incoherencias en datos

10. TROUBLESHOOTING
──────────────────────

    P: El dashboard tarda mucho en procesar v4
    R: Es normal (1.9M registros). Cache después de primera carga.

    P: Veo "Dataset Legacy" en lugar de v4
    R: Verifica que todas las columnas v4 estén presentes en el CSV.
       run: python quick_test_pipeline.py

    P: Error en cálculo de stock
    R: Algunos registros pueden fallar validación (tolerancia 1%).
       Aparecerán warning en logs.

    P: Cómo exportar resultado
    R: Dashboard tiene opciones de download en cada gráfico.

═════════════════════════════════════════════════════════════════════════════
Documentación completa: CAMBIOS_v4_RESUMEN.md
═════════════════════════════════════════════════════════════════════════════
""")
