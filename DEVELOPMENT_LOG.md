# Development Log - Sistema_Tesis

Historial detallado de sesiones de desarrollo. Cada sesión debe documentarse aquí para continuidad.

---

## 📅 Sesión 1: Viernes 14-15 de Febrero, 2026

### Contexto inicial:
- Workspace encontrado con proyecto de tesis (MVP funcional)
- Estado: Errores de ejecución (módulos faltantes)
- Usuario: Nuevo en proyecto, requiere entender arquitectura completa

### Problemas resueltos:

#### 1. Error `run: El término 'run' no se reconoce` (Terminal)
- **Causa:** Usuario intentaba `run main.py` (sintaxis Linux/Mac)
- **Solución:** `python main.py` (Windows PowerShell)
- **Archivo:** N/A (sin cambios de código)

#### 2. ModuleNotFoundError: `streamlit` (y otras)
- **Causa:** requirements.txt no estaba instalado
- **Pasos:**
  - Leído `requirements.txt` (5 dependencias base)
  - Instaladas todas con `pip install -r requirements.txt`
  - Agregado `scikit-learn` (faltante en requirements.txt)
- **Archivos tocados:** requirements.txt (debería actualizarse)

### Análisis profundo:

#### 📊 Análisis del módulo ML (src/ml/)
Documentado:
- **Baselines:** Naive Last, Seasonal Naive 12, Moving Average 3/6
  - O(n), fallback automático, muy estables
- **ETS:** Holt-Winters (Exponential Smoothing)
  - Modela: Nivel + Tendencia + Estacionalidad (12 meses)
  - Min 24 meses data, se activa cuando falla, fallback a naive_last
  - Métricas: MAE, RMSE, sMAPE, MAPE_safe
- **Random Forest:** 400 árboles, features engineered
  - Lags: 1,2,3,6,12 meses
  - Rolling: mean, std, nonzero (3,6,12)
  - Intermitencia: zero_count_6/12
  - Calendario: month_sin, month_cos (cíclico)
  - Min 24 meses, fallback a seasonal_naive_12 → naive_last
  - Robustez: maneja ceros, trunca negativos

#### 💾 Análisis del módulo Data (src/data/)
Documentado:
- **Pipeline:** Orquesta carga → limpieza → reconciliación → demanda → stock
- **DataLoader:** Lee múltiples CSVs, soporta varios separadores/encodings
- **DataCleaner:** Normaliza columnas, convierte tipos (int/float), maneja fechas
- **GuideReconciliation:** Separa transferencias internas de ventas externas (guías)
- **DemandBuilder:** 3 componentes (Venta Tienda, Consumo, Guía externa) → demanda mensual empresa
- **StockBuilder:** Stock consolidado por mes (último Saldo_unid del mes)
- **SeriesCompletion:** Completa meses faltantes con 0 (para que series sean uniformes)

#### 🎨 Análisis del módulo UI (src/ui/dashboard.py)
Documentado (1929 líneas):
- **11 pestañas Streamlit:** Flujo exploración → decisión
  1. Demanda y componentes (validación datos)
  2. Baselines (referencia simple)
  3. ETS (backtest + parámetros)
  4. RF (backtest + parámetros)
  5. Comparativa (elegir ganador por MAE/RMSE/etc)
  6. Resumen global (todos SKUs × ABC)
  7. Stock (diagnóstico)
  8. Recomendación 1 SKU (forecast + SS)
  9. Recomendación masiva (todos SKUs)
  10. Validación retrospectiva (simulación histórica)
  11. Comparativa costo (base vs sistema)
- **Funciones clave:**
  - `compare_models_metrics()`: unifica 3 backtests
  - `select_winner_and_backtests_for_product()`: elige ganador
  - `simulate_policy_backtest_1step()`: simula política mes a mes
  - `simulate_compare_policy_vs_baseline()`: baseline vs propuesta
  - `run_portfolio_cost_comparison_abcA()`: agregado portafolio
- **Policy:** Q = Forecast + Z×σ×√LT - Stock, con Z por ABC

### Decisiones arquitectónicas:

✅ **Decisión 1:** No usar Streamlit Cloud solo
- Razón: Sin multi-tenant, session_state compartido, caché global problemas
- Alternativa: Supabase (Fase 1) + FastAPI backend (Fase 2)

✅ **Decisión 2:** Mantener modelos ML como están (estables)
- Razón: Fallbacks robustos, maneja edge cases bien
- En Fase 2: mover a background jobs (Celery)

✅ **Decisión 3:** Refactorizar en 2 fases
- Fase 1 (3 semanas): MVP cloud + auth básica
- Fase 2 (8 semanas total): FastAPI backend escalable

### Bloqueadores identificados:

🔴 **RF es lento con muchos SKUs**
- Backtest 1 SKU × 12 meses = ~2s (aceptable)
- Backtest 100 SKUs × 12 meses = ~3+ min (lento pero tolerable)
- Backtest 1000 SKUs = ~30+ min (requiere async)
- **Mitiga:** Redis caché, limitar SKUs por sesión, Celery en Fase 2

🟡 **Sin persistencia data actualmente**
- Cada sesión = cálculos desde 0
- Sin histórico recomendaciones
- **Mitiga:** Supabase en Fase 1

🟡 **Sin autenticación/multi-tenant**
- Cualquiera puede acceder a todos datos
- **Mitiga:** streamlit-authenticator en Fase 1

### Commits relacionados:
- Con git ya funciona: usuario hizo `git push` exitoso
- Próxima sesión: commits con mensajes estructurados ([PHASE-1], etc)

### Archivos analizados (no modificados):
- main.py (1 línea, entrada)
- README.md (técnicamente, pero desactualizado)
- src/ml/* (6 archivos, ~400 líneas)
- src/data/* (6 archivos, ~600 líneas)
- src/ui/dashboard.py (1929 líneas, analizado completo)
- requirements.txt (5 líneas, precisa actualización)

### Archivos que necesitan actualización:

| Archivo | Cambio | Prioridad |
|---------|--------|-----------|
| requirements.txt | Agregar scikit-learn, stauth, supabase, boto3, python-dotenv | 🔴 S1W1 |
| README.md | Actualizar con diagrama arquitectura, roadmap cloud | 🟡 S1W2 |
| src/ui/dashboard.py | Integrar auth, s3_upload, db_queries | 🔴 S1W1 |
| src/db/supabase.py | CREAR nuevo módulo | 🔴 S1W1 |
| src/services/* | CREAR módulo servicios (reutilizar logic sin st) | 🟡 S1W2 |

### Conclusiones:

✅ **MVP funcional:** Dashboard con 11 tabs, comparación 3 modelos, simulaciones de política
✅ **Código limpio:** Modular, bien separado por responsabilidad
✅ **ML robusto:** Fallbacks automáticos, manejo edge cases

⚠️ **No listo para producción:** Sin auth, no persistente, no escalable
⚠️ **Próximo:** Refactor Fase 1 (cloud MVP)

---

## Plan Sesión 2 (Próxima)

### Objetivo: MVP Cloud - Week 1

**Tareas:**
1. [ ] Actualizar requirements.txt (agregar 4 dependencias)
2. [ ] Crear `src/db/supabase.py` (módulo BD)
3. [ ] Crear `src/services/ml_service.py` (reutilizar lógica sin st)
4. [ ] Integrar auth streamlit-authenticator en dashboard.py
5. [ ] Crear `.env.example` (template credenciales)
6. [ ] Actualizar `.gitignore` (excluir .env, __pycache__)
7. [ ] Crear Supabase project (5 min, almacenar credenciales en .env)
8. [ ] Schema DB inicial (SQL script)

**Tiempo estimado:** 5-6 horas
**Punto de break:** Auth funciona, Supabase conecta, se puede loggear

---

## Historial Futuro

(Se completará en siguientes sesiones)
