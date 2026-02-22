# 🚀 Guía de Prueba del Sistema - End-to-End

## Estado Actual: ✅ LISTA PARA PROBAR

```
✅ API Backend:       http://localhost:8000
✅ Dashboard:         http://localhost:8501
✅ API Docs (Swagger): http://localhost:8000/docs
```

---

## 📋 Flujo de Prueba Paso a Paso

### **PASO 1: Acceder al Dashboard**

1. Abre tu navegador en: **http://localhost:8501**
2. Verás la pantalla de autenticación

---

### **PASO 2: Crear tu Cuenta**

1. Click en pestañ **"Registrarse"**
2. Ingresa:
   - **Email:** `test@example.com` (o el que prefieras)
   - **Password:** `TestPassword123!`
   - **Empresa:** `Mi Empresa Test`
3. Click en **"Registrarse"**
4. ✅ Se creará tu cuenta automáticamente en Supabase

---

### **PASO 3: Iniciar Sesión**

1. Cambia a la pestaña **"Iniciar Sesión"**
2. Usa las credenciales que acabas de crear
3. ✅ Se te creará un proyecto automáticamente

---

### **PASO 4: Preparar Archivo CSV**

Necesitas un CSV con estructura válida. Aquí está un ejemplo mínimo:

**`test_data.csv`** (copiar y guardar):
```csv
Codigo,Mes,Demanda_Unid,Stock_Unid
PROD001,202401,150,50
PROD001,202402,165,45
PROD001,202403,180,40
PROD001,202404,175,60
PROD001,202405,190,30
PROD001,202406,200,25
PROD001,202407,185,45
PROD001,202408,195,35
PROD001,202409,210,20
PROD001,202410,220,15
PROD001,202411,230,10
PROD001,202412,250,5
PROD002,202401,300,100
PROD002,202402,320,95
PROD002,202403,310,105
PROD002,202404,330,90
PROD002,202405,340,85
PROD002,202406,360,75
PROD002,202407,350,80
PROD002,202408,370,70
PROD002,202409,390,60
PROD002,202410,410,50
PROD002,202411,430,40
PROD002,202412,450,30
```

### **PASO 5: Subir el CSV**

1. En el dashboard, ve a la pestaña **🧩 Demanda y Componentes**
2. Busca la sección **"Subir archivo CSV"**
3. Arrastra o selecciona el archivo `test_data.csv`
4. ✅ Verás mensajes de:
   - "Subiendo a S3..."
   - "✅ file.csv - Subido a S3"
   - "✅ file.csv - Procesado por backend" (esto es nuevo - API integration)

---

### **PASO 6: Generar Pronóstico (NUEVO)**

1. Ve a la pestaña **🚀 API Pronósticos** (la nueva)
2. En "Selecciona un archivo para procesar":
   - Verás el archivo que acabas de subir
   - Seleccionalo
3. En "Configura la predicción":
   - **Nombre del producto:** `PROD001` (o `PROD002`)
   - **Períodos a pronosticar:** 12 (default, puedes cambiar 1-24)
   - **Modelo ML:** Elige uno:
     - "ETS (Holt-Winters)" - Recomendado para series con estacionalidad
     - "Random Forest" - Recomendado si hay muchas variables
     - "Automático (mejor de ambos)" - El más seguro
4. Click en **🚀 Generar Pronóstico**

---

### **PASO 7: Interpretar Resultados**

Verás:

#### Métricas:
```
┌─────────────────────────┐
│ Producto: PROD001       │
│ Modelo: ETS             │
│ MAPE (Error %): 5.23%   │
└─────────────────────────┘
```

- **MAPE < 10%** = Excelente
- **MAPE 10-20%** = Bueno
- **MAPE > 20%** = Revisar datos

#### Gráfico:
- Línea roja = Pronóstico para próximos 12 períodos
- Marca los puntos de cada valor

#### Tabla:
```
Período  | Demanda Pronosticada
T+1      | 245.3
T+2      | 260.1
...      | ...
```

#### Estadísticas:
```
Promedio:  265.5
Mínimo:    245.3
Máximo:    280.2
Desv. Est: 12.4
```

---

## 🧪 Testing Detallado de Endpoints

### **Test 1: Health Check**
```bash
curl http://localhost:8000/health
```

**Respuesta esperada:**
```json
{
  "status": "ok",
  "version": "0.1.0",
  "timestamp": "2026-02-21T14:23:42.626180"
}
```

---

### **Test 2: Ver Documentación API**
Abre: **http://localhost:8000/docs**

Verás:
- ✅ POST /uploads/process
- ✅ GET /uploads/{upload_id}/status
- ✅ POST /forecasts/generate
- ✅ GET /forecasts/{upload_id}/{product}

Puedes hacer "Try it out" con cada endpoint.

---

### **Test 3: Generar Pronóstico Manualmente**

```bash
curl -X POST "http://localhost:8000/forecasts/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "upload_id": "tu-upload-id",
    "product": "PROD001",
    "model_type": "ets",
    "forecast_periods": 12
  }'
```

---

## 📊 Flujo Técnico Completo

```
Usuario en Dashboard
         │
         ├─→ Sube CSV
         │   └─→ S3 (AWS)
         │   └─→ Supabase metadata
         │
         ├─→ Click "Generar Pronóstico"
         │   └─→ POST /forecasts/generate (API)
         │       └─→ Obtiene datos de Supabase
         │       └─→ Descarga CSV de S3
         │       └─→ Filtra por producto
         │       └─→ Entrena modelo ETS/RF
         │       └─→ Genera 12 valores futuros
         │       └─→ Calcula MAPE
         │
         └─→ Ve resultados
             └─→ Gráfico Plotly
             └─→ Tabla con valores
             └─→ Estadísticas

```

---

## ✅ Checklist de Verificación

### Debe pasar:
- [ ] Acceder a http://localhost:8501
- [ ] Registrarse e iniciar sesión
- [ ] Subir un CSV válido
- [ ] Ver "✅ Procesado por backend" en upload
- [ ] Ir a pestaña "🚀 API Pronósticos"
- [ ] Seleccionar archivo cargado
- [ ] Generar pronóstico con ETS
- [ ] Ver gráfico y tabla de resultados
- [ ] Generar pronóstico con RF
- [ ] Cambiar períodos (5, 12, 24) y probar

### Logs a esperar:

**En Dashboard:**
```
✅ API Backend disponible
✅ Pronóstico generado exitosamente
```

**En API Terminal:**
```
[UPLOAD xxxxx] Iniciando procesamiento...
[UPLOAD xxxxx] Descargando de S3: ...
[UPLOAD xxxxx] CSV parseado: X filas, X columnas
[UPLOAD xxxxx] COMPLETADO!
```

---

## 🛠️ Solución de Problemas

### Si no se sube el CSV:
- Revisa que el formato sea válido (CSV con columnas: Codigo, Mes, Demanda_Unid, Stock_Unid)
- Revisa que haya datos de mínimo 6 meses
- Mira la consola del API para ver el error exacto

### Si falla la predicción:
- Asegúrate de ingresar el nombre del producto EXACTO (ej: "PROD001")
- El producto debe existir en el CSV
- Debe haber mínimo 12 registros históricos

### Si dice "API no disponible":
- Verifica que http://localhost:8000/health retorna 200
- Reinicia: `python -m uvicorn src.api.main:app --reload --port 8000`

---

## 📈 Qué Esperar de Cada Modelo

### **ETS (Holt-Winters)**
- Mejor para: Series con tendencia clara y estacionalidad
- MAPE típico: 5-15%
- Velocidad: Rápida (~1 seg)
- Pros: Interpretable, manejo de estacionalidad

### **Random Forest**
- Mejor para: Datos con múltiples patrones complejos
- MAPE típico: 8-20%
- Velocidad: Moderada (~3 seg)
- Pros: Robusto, maneja outliers

### **Automático (Best)**
- Entrena ambos y elige el de menor error
- MAPE típico: Lo mejor de ambos
- Velocidad: Lenta (~4 seg)
- Pros: Máxima precisión

---

## 🎯 Éxito = Cuando:

✅ Subes un CSV correctamente  
✅ Backend lo procesa (ves "Procesado por backend")  
✅ Generas un pronóstico sin errores  
✅ Ves un gráfico con 12 puntos predichos  
✅ Las métricas muestran MAPE razonable (<20%)  
✅ Los valores predichos tienen sentido (continúan la tendencia)  

---

## 📞 Logs en Tiempo Real

**Terminal API:**
```
uvicorn running on http://0.0.0.0:8000
[UPLOAD info logs aquí]
```

**Terminal Dashboard:**
```
streamlit running on http://localhost:8501
[dashboard logs aquí]
```

---

¡**Listos para probar!** Sigue estos pasos y tendrás el sistema completo en funcionamiento. 🎉
