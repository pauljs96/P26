"""
ARQUITECTURA CORRECTA DEL SISTEMA EN LA NUBE
=============================================

PROBLEMA ACTUAL:
================
Dashboard → S3 → Pipeline (local) → Supabase

¿Por qué falla?
- El dashboard intenta procesar múltiples archivos localmente
- Problema de concurrencia con S3
- Sin error handling robusto

SOLUCIÓN CORRECTA:
==================

OPCION 1: API Backend Separado (RECOMENDADO)
---------------------------------------------

    Dashboard (Streamlit)
          ↓ (POST archivos)
    API Backend (Flask/FastAPI)  ← Se ejecuta en:
          ↓ (procesa con pipeline)    - Servidor propio
    Resultado JSON                   - Heroku / Render / Railway
          ↓ (guarda)                  - Google Cloud Run
    Supabase PostgreSQL              - AWS Lambda

    Ventajas:
    ✓ Separación de responsabilidades
    ✓ Escalable (API puede procesar múltiples requests)
    ✓ Sin límites de tiempo de Streamlit (30 min)
    ✓ Fácil de debuggear
    ✓ Perfecta para futuros microservicios

OPCION 2: Cloud Function (Supabase Edge Functions)
---------------------------------------------------

    Dashboard → Supabase Edge Function
                      ↓ (procesa con node.js o deno)
                 PostgreSQL

    Ventajas:
    ✓ Sin servidor separado
    ✓ Todo en Supabase
    ✗ Lenguaje limitado (Node.js/Deno, no Python)
    ✗ No puedes reutilizar el código Python del pipeline

OPCION 3: Carga Local + Sincronización
---------------------------------------

    Desarrollador (local):
    - python cargar_incremental.py 2020 2021 2022
    - Datos procesados se guardan en Supabase
    
    Dashboard (lee de Supabase):
    - No necesita procesar, solo visualiza

    Ventajas:
    ✓ Control total
    ✓ Fácil debugging
    ✗ Manual - debe hacerlo el desarrollador
    ✗ No escalable para múltiples usuarios


IMPLEMENTACION DE LA OPCION 1 (RECOMENDADA)
============================================

PASO 1: Tener el API Backend corriendo
--------------------------------------

El archivo 'api_backend.py' contiene todo lo necesario.

Correr localmente:
  $ python api_backend.py
  Corriendo en: http://localhost:5000

PASO 2: Modificar dashboard.py
-------------------------------

Ver archivo: DASHBOARD_MODIFICACION.py

Básicamente cambiar esta sección (línea ~1647):
  
  ANTES (incorrecto):
  res = pipeline.run(saved_files)  ← Intenta procesar localmente
  
  AHORA (correcto):
  response = requests.post(
      f'{API_URL}/process-files',  ← Envía al backend
      files=api_files,
      data={'org_id': org_id}
  )

PASO 3: Deployar en la nube
---------------------------

Opción A: HEROKU (más fácil)
  1. Crear archivo 'Procfile':
     web: python api_backend.py

  2. Crear archivo 'requirements.txt':
     flask
     pandas
     python-dotenv
     supabase
     boto3

  3. Push a Heroku:
     $ heroku create predicast-api
     $ git push heroku main

  4. En dashboard, actualizar:
     API_URL = "https://predicast-api.herokuapp.com"

Opción B: RENDER (recomendado - gratis)
  1. Conectar GitHub repo a Render.com
  2. Crear nuevo "Web Service"
  3. Build: python -m pip install -r requirements.txt
  4. Start: python api_backend.py
  5. Dar URL del servicio al dashboard

Opción C: RAILWAY (muy simple)
  1. railway.app
  2. Conectar GitHub
  3. Deploy automático
  4. URL se asigna automáticamente


FLUJO COMPLETO CON ESTA ARQUITECTURA
====================================

Usuario sube 6 archivos en Dashboard:

1️⃣  Dashboard lee archivos en memoria
2️⃣  Envía multipart POST a http://api-backend.com/process-files
3️⃣  API Backend recibe {archivos, org_id, user_id}
4️⃣  Backend procesa:
     - DataLoader (carga todos)
     - DataCleaner (limpia)
     - GuideReconciler (reconcilia)
     - DemandBuilder (construye demanda)
     - StockBuilder (construye stock)
5️⃣  Backend retorna JSON con resultados
6️⃣  Dashboard guarda en Supabase
7️⃣  Dashboard visualiza en tiempo real

⚡ TODO esto en ~30 segundos para 164K registros


ESCALABILIDAD FUTURA (2027, 2028...)
====================================

Año 2026 llega:

OPCION 1: Sistema automático
  - Dashboard tiene botón "Cargar nuevo año"
  - Usuario carga D_2026.csv
  - API procesa y suma a datos existentes
  - Automático

OPCION 2: Script incremental
  $ python cargar_incremental.py 2026
  - Procesa localmente
  - Guarda en Supabase
  - Dashboard actualiza automáticamente


SEGURIDAD Y AUTENTICACIÓN
=========================

En producción, el API debe estar protegido:

1. Agregar API Key:
   @app.route('/process-files', methods=['POST'])
   def process_files():
       api_key = request.headers.get('Authorization')
       if api_key != os.getenv('API_SECRET_KEY'):
           return {'error': 'Unauthorized'}, 401
       ...

2. Dashboard envía:
   headers = {'Authorization': os.getenv('API_SECRET_KEY')}
   response = requests.post(..., headers=headers)

3. EN ENV:
   API_BACKEND_URL=https://predicast-api.herokuapp.com
   API_SECRET_KEY=your-secret-key-12345


VARIABLES DE ENTORNO (.env)
============================

Para API Backend:
  SUPABASE_URL=https://...supabase.co
  SUPABASE_KEY=eyJ...
  AWS_ACCESS_KEY_ID=...
  AWS_SECRET_ACCESS_KEY=...
  API_SECRET_KEY=your-secret-123
  FLASK_ENV=production
  PORT=5000

Para Dashboard:
  API_BACKEND_URL=https://predicast-api.herokuapp.com
  API_SECRET_KEY=your-secret-123
  (más las que ya tienes)


MONITOREO Y LOGS
================

El API guarda logs útiles:
  ✓ Cuántos archivos procesó
  ✓ Cuántos registros limpió
  ✓ Cuántos errores tuvo
  ✓ Tiempo de ejecución

Puedes agregar:
  - CloudWatch (AWS)
  - Sentry (error tracking)
  - DataDog (monitoring)
  - New Relic


RESUMIENDO:
===========

✗ ANTES (incorrecto):
  Dashboard → procesa localmente → falla con múltiples archivos

✓ AHORA (correcto):
  Dashboard → API Backend → procesa en la nube → Supabase → visualiza

¿Por qué funciona?
- API está optimizado para procesar
- Sin límites de tiempo como Streamlit (30 min)
- Sin problemas de concurrencia
- Escalable para múltiples usuarios
- Fácil de monitorear

¿Usar esto ahora o después?
- Si tu dashboard corre en Streamlit Cloud → HAZLO AHORA
- Si lo ejecutas localmente → puedes esperar
- Si esperaras mantenimiento futuro → hazlo ya


PRÓXIMOS PASOS
==============

1. Instalar dependencias: pip install requests flask boto3 supabase
2. Correr API bash: python api_backend.py
3. Probar en http://localhost:5000/health
4. Modificar dashboard como en DASHBOARD_MODIFICACION.py
5. Luego deployar en Heroku/Render/Railway

¿Necesitas ayuda con alguno de estos pasos?
"""
