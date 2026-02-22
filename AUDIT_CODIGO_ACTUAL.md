# AUDITORIA DE CODIGO - Estado actual del proyecto

## 1. ESTRUCTURA ACTUAL

### A. MODULO DE DATOS (src/data/)
**Objetivo:** Procesar CSVs crudos y construir datasets listos para análisis/pronósticos

```
data_loader.py        → Carga CSV con detección automática de headers
                        (maneja múltiples separadores y encodings)

data_cleaner.py       → Limpieza de datos (tipado, eliminación duplicados)

guide_reconciliation.py → Reconcilia guías de remisión

demand_builder.py     → Construye demanda MENSUAL agregada
                        (input: movimientos diarios → output: demanda/mes)

ProductStockBuilder.py → Construye stock MENSUAL por empresa
                        (último saldo del mes)

series_completion.py  → Completa meses faltantes con 0

pipeline.py          → Orquesta todo lo anterior en secuencia
                        (DataLoader → DataCleaner → GuideReconciler 
                         → DemandBuilder → ProductStockBuilder 
                         → series_completion)
```

**Entrada:** Archivos CSV subidos por usuario
**Salida:** 
- `movements`: DataFrame con movimientos reconciliados
- `demand_monthly`: DataFrame con demanda mensual agregada
- `stock_monthly`: DataFrame con stock mensual

---

### B. MODULO ML (src/ml/)
**Objetivo:** Modelos de pronóstico y backtesting

```
baselines.py         → 3 métodos simple: Naive, Seasonal12, MA

ets_model.py         → ETSForecaster (Holt-Winters)
                        - forecast_1step(): predice t+1

rf_model.py          → RFForecaster (Random Forest)  
                        - forecast_1step(): predice t+1

rf_features.py       → Genera features para RF (lags, rolling, calendar)

backtest.py          → Backtesting genérico para todos los modelos

backtest_ets.py      → Backtesting específico para ETS

backtest_rf.py       → Backtesting específico para RF
```

**Entrada:** Series de demanda histórica (pd.Series o DataFrame)
**Output:** Pronósticos + métricas (MAE, MAPE)

---

### C. MODULO SERVICIOS (src/services/)
**Objetivo:** Orquestar ML models (NO duplicar lógica)

```
ml_service.py        → compare_models() - Compara Baselines + ETS + RF
                                          Returns: modelo ganador + métricas
                      → forecast_next_month() - Pronóstico t+1 con modelo ganador
```

---

### D. MODULO UI (src/ui/)
**Objetivo:** Dashboard Streamlit

```
dashboard.py         → APP PRINCIPAL
                        - 12 tabs con análisis complete
                        - Upload CSV a S3
                        - Visualización de demanda/stock
                        - Backtesting modelos
                        - Simulación de políticas (con stock disponible)
                        - ABC analysis
                        - Optimización de inventario
                        → Tab nuevo: API Pronósticos (llamar al API)

forecast_tab.py      → Tab NEW para llamar API /forecasts/generate
                        - Seleccionar producto
                        - Input períodos
                        - Mostrar resultado
```

---

### E. MODULO BASES DE DATOS (src/db/)
**Objetivo:** Interactuar con Supabase

```
supabase.py          → SupabaseDB class
                        - save_upload(): Guardar metadata de upload
                        - get_upload(): Obtener metadata
                        - update_upload_status(): Actualizar status
```

---

### F. MODULO STORAGE (src/storage/)  
**Objetivo:** Gestionar S3

```
s3_manager.py        → S3Manager class
                        - upload_file_bytes(): Subir archivo a S3
                        - download_file(): Descargar de S3
                        - delete_file(): Eliminar de S3
                        - get_presigned_url(): URL temporal
```

---

### G. MODULO API (src/api/) - NUEVO/INCOMPLETO
**Objetivo:** Backend FastAPI

```
main.py              → App FastAPI (básico)

models.py            → Pydantic schemas
                        - UploadRequest
                        - ForecastRequest  
                        - ForecastResult
                        - ProcessingStatus

client.py            → HTTPClient para dashboard llamar al API
                        - process_upload()
                        - generate_forecast()
                        - health_check()

dependencies.py      → Inyección de dependencias
                        - get_db(): SupabaseDB
                        - get_storage_manager(): S3Manager

routers/
  ├─ uploads.py      → POST /uploads/process
  │                     (descarga CSV de S3, ejecuta DataPipeline)
  │
  └─ forecasts.py    → POST /forecasts/generate  
                        (ejecuta compare_models + forecast_next_month)

utils/
  └─ pipeline_adapter.py  → Adaptador para DataPipeline con bytes
```

---

## 2. QUE FUNCIONA EN DASHBOARD YA

✅ **Carga de archivos:**
- CSV upload a S3
- Metadata en Supabase

✅ **Procesamiento:**
- Ejecuta DataPipeline completa
- Genera demanda_monthly + stock_monthly

✅ **Visualización:**
- Gráficos de demanda histórica
- Decomposición (trend, seasonal, residual)
- Stock disponible por mes

✅ **ML - Backtesting:**
- Compara Baselines vs ETS vs RF
- Muestra MAE, MAPE de cada modelo
- Selecciona ganador (menor MAE)

✅ **Pronósticos:**
- Pronóstico con modelo ganador
- Iterative forecasting (15+ períodos)

✅ **Recomendaciones:**
- Stock óptimo (EOQ)
- Reorden basado en demanda + stock actual
- ABC classification
- Policy simulation

✅ **Datos multiempresa:**
- Demanda agregada por mes
- Stock por empresa

---

## 3. QUE NECESITA FASTAPI (SOLO)

El API debe REUTILIZAR lo que ya existe en dashboard:

### **ENDPOINT 1: POST /uploads/process**
**Qué hace:**
```
1. Recibe upload_id + s3_path
2. Descarga CSV de S3
3. Ejecuta DataPipeline (ya existe)
4. Actualiza status en Supabase
```

**QUÉ NO DUPLICAR:**
- ❌ NO crear parse_csv nuevo
- ❌ NO crear validate_csv nuevo  
- ❌ NO crear clean_csv nuevo
- ✅ USAR: DataPipeline.run() tal como está

**Estado:** IMPLEMENTADO (pero con adaptador)

---

### **ENDPOINT 2: POST /forecasts/generate**
**Qué hace:**
```
1. Recibe upload_id + product + model_type + periods
2. Ejecuta DataPipeline para obtener demanda_monthly
3. Filtra por producto
4. Ejecuta compare_models() (ya existe en src/services/ml_service.py)
5. Ejecuta forecast_next_month() (ya existe)
6. Retorna pronóstico
```

**QUÉ NO DUPLICAR:**
- ❌ NO crear forecast_ets nuevo
- ❌ NO crear forecast_rf nuevo
- ❌ NO crear compare_models nuevo
- ✅ USAR: src/services/ml_service.py como está

**Estado:** IMPLEMENTADO (pero con adaptador)

---

## 4. ARCHIVOS QUE SE PUEDEN ELIMINAR

❌ **src/api/utils/csv_processor.py** 
   Razón: Duplica lógica de DataPipeline/DataLoader

❌ **src/api/utils/ml_service.py** (la del API)
   Razón: Es versión simplificada del original en src/services/ml_service.py

---

## 5. ARCHIVOS QUE SE MANTIENEN SIN CAMBIO

✅ **src/data/** - Pipeline completa (no toca)
✅ **src/ml/** - Modelos (no toca)
✅ **src/services/ml_service.py** - Orquestador ML (no toca)
✅ **src/ui/dashboard.py** - Dashboard completo (no toca)
✅ **src/db/supabase.py** - Database (no toca)
✅ **src/storage/s3_manager.py** - S3 (no toca)

---

## 6. ARCHIVOS QUE NECESITAN CREARSE/AJUSTARSE

✅ **src/api/utils/pipeline_adapter.py** - NUEVO
   Objetivo: Adaptar DataPipeline para trabajar con bytes (desde S3)
   Sin replantear lógica, solo adaptar el interfaz

✅ **src/api/routers/uploads.py** - NUEVO (pero simple)
   Hace: DataPipeline.run() + update_upload_status()
   
✅ **src/api/routers/forecasts.py** - NUEVO (pero simple)
   Hace: compare_models() + forecast_next_month() + iterative forecast

---

## 7. PLAN FINAL (SIN DUPLICADOS)

```
FASE 0: Auditoría y Plan ← ESTAMOS AQUÍ
  └─ Verificar qué existe
  └─ Listar funcionalidades actuales
  └─ Definir InterfAZ del API (sin cambiar lógica)

FASE 1: FastAPI Minimum (SOLO exposición de lógica existente)
  └─ POST /uploads/process
      - Copia bytes de S3
      - Ejecuta DataPipeline (sin cambios)
      - Marca como procesado
      
  └─ POST /forecasts/generate
      - Extrae demanda + producto
      - Ejecuta compare_models (sin cambios)
      - Ejecuta forecast_next_month iterativamente (sin cambios)
      - Retorna pronóstico

FASE 2: Documentación + Tests
  └─ Documentar cada endpoint
  └─ Test E2E (upload → forecast)
  └─ Verificar que lógica dashboard = API

FASE 3: Producción
  └─ Deployment a nube (AWS/Heroku/GCP)
  └─ Monitoreo
```

---

## 8. RESUMEN EJECUTIVO

| Aspecto | Estado | Acción |
|---------|--------|--------|
| **DataPipeline** | ✅ Completa | Mantener, reutilizar en API |
| **ML Models** | ✅ Completos | Mantener, reutilizar en API |
| **Dashboard** | ✅ Funcional | No tocar |
| **S3 Manager** | ✅ Funcional | No tocar |
| **Supabase** | ✅ Funcional | No tocar |
| **FastAPI routers** | ⚠️ Existen pero con duplicado | Limpiar, simplificar |
| **Lógica duplicada** | ❌ Existe | **ELIMINAR** |
| **Plan claro** | ⚠️ No muy claro | ← **ESTO** |

