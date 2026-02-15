# Contexto del Proyecto Sistema_Tesis

**Última actualización:** 15 de Febrero, 2026  
**Estado:** MVP funcional, planificando transición a cloud multi-tenant  
**Sesiones completadas:** 1

---

## 📋 Resumen Ejecutivo

Sistema web de **planificación de producción e inventario** que:
- Carga datos transaccionales de ERP (archivos CSV)
- Aplica pipeline: limpieza → reconciliación → demanda mensual → pronóstico
- Compara 3 tipos de modelos: Baselines, ETS, Random Forest
- Simula políticas de inventario (Safety Stock + Forecast)
- Visualiza recomendaciones de producción y permite simulaciones retrospectivas

**Objetivo final:** Desplegar como **SaaS multi-tenant en cloud** para que múltiples empresas accedan vía web con credenciales.

---

## 🏗️ Arquitectura Actual (Local MVP)

### Stack tecnológico:
- **Frontend:** Streamlit (Python)
- **Backend (Data):** Python (Pandas, NumPy, Scikit-learn, Statsmodels)
- **ML Models:** 
  - Baselines: Naive Last, Seasonal Naive 12, Moving Average
  - ETS: Holt-Winters (exponential smoothing)
  - RF: Random Forest Regressor (400 árboles, features engineered)
- **Visualización:** Plotly Express
- **Ambiente:** Python 3.12.5 (venv local)

### Estructura de carpetas:
```
Sistema_Tesis/
├── main.py                              # Entrada (Dashboard)
├── README.md                            # Documentación (a actualizar)
├── requirements.txt                     # Dependencias
├── src/
│   ├── data/                           # Pipeline de datos
│   │   ├── data_loader.py             # Carga CSV ERP
│   │   ├── data_cleaner.py            # Normalizacion columnas
│   │   ├── guide_reconciliation.py    # Separar transferencias vs ventas
│   │   ├── demand_builder.py          # Demanda mensual empresa
│   │   ├── stock_builder.py           # Stock por bodega
│   │   ├── ProductStockBuilder.py     # Stock consolidado
│   │   ├── series_completion.py       # Completa meses con 0
│   │   └── pipeline.py                # Orquesta todo
│   ├── ml/                             # Modelos de pronóstico
│   │   ├── baselines.py               # Naive, Seasonal, MA
│   │   ├── ets_model.py               # Holt-Winters
│   │   ├── rf_model.py                # Random Forest
│   │   ├── rf_features.py             # Engineering de features para RF
│   │   ├── backtest.py                # Backtesting 1-step (baselines)
│   │   ├── backtest_ets.py            # Backtesting ETS
│   │   ├── backtest_rf.py             # Backtesting RF
│   │   └── logger.py
│   ├── ui/
│   │   └── dashboard.py               # Interfaz Streamlit (1929 líneas, 11 tabs)
│   └── utils/
│       ├── config.py                  # Constantes (DOC_VENTA, SL_A, etc)
│       └── logger.py
```

### Base de datos:
- **Actualmente:** None (todo en memoria de sesión)
- **En futuro:** PostgreSQL (Supabase) para persistencia multi-tenant

---

## 📖 Última Sesión: Análisis Profundo del Proyecto (15-Feb-2026)

### Qué se hizo:
1. ✅ **Análisis arquitectura completa** (1929 líneas dashboard)
2. ✅ **Machine Learning módulo:**
   - Baselines: 3 métodos simples para referencia
   - ETS: Holt-Winters con fallback robusto (min 24 meses)
   - RF: 400 árboles + features: lags (1,2,3,6,12), rolling (3,6,12), calendario cíclico
   - Backtesting: walk-forward 1-step, métricas MAE/RMSE/sMAPE/MAPE_safe
3. ✅ **UI/Dashboard módulo:**
   - 11 pestañas: demanda, baselines, ETS, RF, comparativa, resumen, stock, recomendación, masiva, validación, comparativa retrospectiva
   - Funciones clave: ABC builder, policy simulation, cost comparison
4. ✅ **Evaluación cloud readiness:**
   - ❌ Problemas actuales: sin auth, sin BD persistente, sin multi-tenant, session state compartido
   - Recomendación: Fase 1 (Streamlit + Supabase), Fase 2 (FastAPI backend)

### Decisiones tomadas:
- **No** usar Streamlit Cloud solo (insuficiente para multi-tenant)
- **Sí** refactorizar con Supabase para MVP cloud en 3-4 semanas
- **Plan 2 fases:** MVP rápido (S1-S3) → Backend robusto (S4-S8)

---

## 🗺️ Roadmap: Transformación a Cloud (Semanas 1-8)

### **FASE 1: MVP Cloud (Semanas 1-3)** ✅ Próximo
**Objetivo:** Despliegue público con autenticación básica y BD persistente

#### Semana 1: Refactoring código (2-3 días)
- [ ] Agregar `streamlit-authenticator` (login simple)
- [ ] Crear módulo `src/db/supabase.py` (queries CRUD)
- [ ] Separar lógica de negocio de Streamlit (crear `src/services/`)
- [ ] Integrar S3 para almacenar CSVs cargados

#### Semana 2: Infraestructura cloud (2-3 días)
- [ ] Crear proyecto Supabase (PostgreSQL + Auth)
- [ ] Schema DB: `users`, `projects`, `uploads`, `backtests`, `recommendations`
- [ ] AWS S3 bucket para CSVs (o usar storage Supabase)
- [ ] GitHub Actions para CI/CD

#### Semana 3: Deploy MVP (2-3 días)
- [ ] Empaquetizar en Docker
- [ ] Streamlit Cloud (push a GitHub automático)
- [ ] Testing: múltiples usuarios simultáneos
- [ ] Documentación: guía de usuario

**Costo:** ~$100/mes (Streamlit free, Supabase starter $50, S3 ~$5)

---

### **FASE 2: Backend separado (Semanas 4-8)** 🔮 Futuro
**Objetivo:** Escalabilidad, entrenamientos asincronos, APIs REST

#### Semana 4-5: FastAPI backend
- [ ] Crear `api/` folder con endpoints FastAPI
- [ ] Mover `src/ml/*`, `src/data/*` a servicios reutilizables
- [ ] Celery + Redis para background jobs (entrenamientos RF)
- [ ] Tests unitarios

#### Semana 6-7: Persistencia + caché
- [ ] Modelo de datos: Organizations, Projects, Forecasts, Recommendations
- [ ] Redis para caché de backtests (evita recalcular)
- [ ] API tokens (OAuth2 o API keys por empresa)

#### Semana 8: Deployment robusto
- [ ] Cloud Run (GCP) o ECS (AWS)
- [ ] PostgreSQL managed (RDS/Cloud SQL)
- [ ] Redis managed (ElastiCache/Memorystore)
- [ ] Monitoring (Cloud Logging, Sentry)

**Costo:** ~$300-500/mes (compute, DB, Redis)

---

## 🎯 Plan de Hoy en Adelante

### Cómo recordaré nuestras conversaciones:

1. **PROJECT_CONTEXT.md** (este archivo)
   - Resumen ejecutivo
   - Arquitectura actual
   - Historial de sesiones
   - Roadmap
   
2. **DEVELOPMENT_LOG.md** (separado)
   - Detalles de cada sesión
   - Decisiones y por qué
   - Blockers / problemas identificados
   - código commits asociados

3. **.conversation_state.json** (estado actual)
   - fase actual del desarrollo
   - pasos completados
   - próximos pasos inmediatos
   - referencias a archivos modificados

4. **Commits de Git**
   - Mensajes descriptivos: `[PHASE-1] Add Supabase auth integration`
   - Link a este contexto en PR descriptions

### Cómo continuaremos:

**Cuando reabras una sesión:**
1. Leo `PROJECT_CONTEXT.md` + `.conversation_state.json`
2. Pregunto: "¿Desde dónde continuamos?"
3. Mostrarte qué falta hacer
4. Continuamos sin perder contexto

**Ejemplo próxima sesión:**
```
Yo: "Veo que en la sesión anterior completamos análisis.
    Estamos en FASE 1, Semana 1.
    Plan de hoy: Agregar autenticación Streamlit.
    ¿Correcto? ¿Empezamos con los cambios a dashboard.py?"
```

---

## 🔗 Referencias Clave

### Arquitectura ML (resumida):
- **Entrada:** DataFrame [Mes, Demanda_Unid] completo (meses con 0)
- **Baselines:** O(n), instant
- **ETS:** O(n²), requiresmin 24 meses
- **RF:** O(n*m*d), requiere min 24 meses, 400 árboles
- **Salida:** float (pronóstico t+1, ≥0)
- **Fallback:** Baselines si RF/ETS falla

### Política de Inventario:
```
Q_recomendada = max(0, Forecast_t+1 + SS - Stock_actual)

Donde:
  SS = Z * σ * √(lead_time)
  Z = z_score(service_level_por_ABC)
  σ = MAE del modelo ganador (proxy)
  
  ABC → Service Level → Z:
    A → 95% → 1.65
    B → 90% → 1.28
    C → 85% → 1.04
```

### Dashboard tabs (11 total):
1. 🧩 Demanda y Componentes (exploración)
2. 🔮 Baselines (referencia)
3. 📈 ETS (trend + seasonal)
4. 🤖 RF (no-lineales)
5. 🏆 Comparativa (elegir ganador)
6. 📊 Resumen global (portafolio)
7. 🏢 Stock (diagnóstico)
8. 🔄 Recomendación 1 SKU
9. 📑 Recomendación masiva (todos)
10. ✅ Validación retrospectiva
11. 📉 Comparativa sin-sistema vs con-sistema

---

## 📝 Notas Importantes

### Sobre el proyecto:
- Es una **tesis** (académica) pero con potencial SaaS
- Datos de **ERP transaccionales** (kardex) → agregan a demanda mensual
- Reconciliación de guías: distingue ventas (externa) vs transferencias (interna)
- Varios productos (SKUs) en paralelo, clasificación ABC dinámica

### Tecnológico:
- Python 3.12.5 (asegúrate que venv esté activo)
- Todas las dependencias en `requirements.txt` (agregar: streamlit-authenticator, supabase, boto3 en Fase 1)
- Sin tests unitarios aún (añadir en Fase 2)
- Sin logging productivo (agregar en Fase 1)

### Riesgos identificados:
- 🔴 RF con 400 árboles es lento para 1000+ SKUs (≈30 min backtest completo)
  - Mitiga: Caché Redis, limitar SKUs por sesión, usar muestreo
- 🔴 Cuello de botella: entrenamientos bloqueantes en Streamlit
  - Mitiga: Celery + background jobs en Fase 2
- 🟡 Supabase gratis tiene límites (100k rows/mes), monitor usar en producción

---

## ✅ Checklist para comenzar Sesión 2

- [ ] Instalar dependencias nuevas: `pip install streamlit-authenticator supabase python-dotenv`
- [ ] Crear cuenta Supabase (5 min, gratis)
- [ ] Crear `.env` con credenciales Supabase
- [ ] Crear schema DB SQL base (5 tablas)
- [ ] Agregar módulo `src/db/supabase.py`
- [ ] Integrar login a `src/ui/dashboard.py`

**Estimado:** 4-5 horas si todo fluye bien

---

## 📞 Preguntas Pendientes

Durante sesión 1 no se decidió:
1. ¿Usar AWS S3 o Supabase Storage para CSVs?
2. ¿OAuth (Google/GitHub) o user/password simple?
3. ¿Precio final del SaaS? (subscription model?)
4. ¿Cuántos empresas target para Fase 1?

(Discutir cuando continúes)

---

**Siguiente paso:** Leer esto, confirmar que está correcto, y decir cuándo quieres continuar con Fase 1 Week 1.
