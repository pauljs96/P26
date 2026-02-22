# WEEK 4: Cloud Deployment to Streamlit Cloud ☁️

## 📋 Overview

Este documento guía el deployment de la aplicación **Sistema de Planificación** a Streamlit Cloud. La app es multi-tenant con:
- Autenticación via Supabase Auth
- Base de datos Supabase (PostgreSQL)
- Almacenamiento S3 (AWS)
- Caché JSONB para performance multi-user
- RLS policies para aislamiento de datos

---

## 🚀 Pre-requisitos

Antes de deployar, asegúrate de tener:

✅ **GitHub Repository**
- [ ] Repo creado en GitHub
- [ ] Código pushado con todos los cambios
- [ ] `.streamlit/secrets.toml` en `.gitignore` (no commitear credenciales)

✅ **Supabase Project**
- [ ] Proyecto activo con tablas multi-tenant
- [ ] RLS policies habilitadas para `org_cache`
- [ ] Users + Organizations cargadas para testing

✅ **AWS S3 Bucket**
- [ ] Bucket creado y accesible
- [ ] IAM credentials con acceso S3
- [ ] CORS habilitado si necesario

✅ **Streamlit Cloud Account**
- [ ] Cuenta gratuita en https://streamlit.io
- [ ] Github conectada a Streamlit Cloud

---

## 📝 PASO 1: Preparar el Repositorio

### 1.1 Asegurar que `.streamlit/secrets.toml` NO está en git

```bash
# Verificar que no está committed
git status
# Debería mostrar: nothing to commit (o archivos que SÍ quieres commitear)

# Si fue committed accidentalmente, removerlo:
git rm --cached .streamlit/secrets.toml
git commit -m "Remove secrets.toml from git (should be local only)"
git push
```

### 1.2 Agregar/Actualizar `requirements.txt`

El archivo debe incluir TODAS las dependencias:

```bash
# Verificar versiones actuales
pip freeze > requirements.txt
```

**Mínimas requeridas:**
```
streamlit>=1.36
pandas>=2.0
numpy>=1.24
plotly>=5.18
python-dateutil>=2.8
scikit-learn>=1.3
statsmodels>=0.14
python-dotenv>=1.0
supabase>=1.0
requests>=2.30
boto3>=1.26
```

### 1.3 Crear `.streamlit/config.toml` para producción

Ya existe, pero verificar que tenga:

```toml
[client]
showErrorDetails = false  # No mostrar detalles en producción
toolbarMode = "viewer"    # Solo usuarios ven la UI, no la toolbar de dev

[logger]
level = "info"

[server]
headless = true
port = 8501
enableXsrfProtection = true
maxUploadSize = 200       # Máximo 200 MB para CSV uploads

[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#31333f"
```

### 1.4 Actualizar `main.py` para diferencias cloud

El `main.py` actual es correcto, pero agregar un check de ambiente:

```python
import os
import streamlit as st

# Detectar si está en Streamlit Cloud o local
IS_CLOUD = "STREAMLIT_SERVER_HEADLESS" in os.environ or "streamlitcloud" in os.environ.get("HOME", "")

if not IS_CLOUD:
    print("🚀 Running locally")
else:
    print("☁️ Running on Streamlit Cloud")

from src.ui.dashboard import Dashboard

def main():
    Dashboard().render()

if __name__ == "__main__":
    main()
```

---

## 🔑 PASO 2: Configurar Secrets en Streamlit Cloud

###  2.1 Ir a Streamlit Cloud Dashboard

1. Abre https://share.streamlit.io
2. Conecta tu cuenta GitHub
3. Click "**New app**"

### 2.2 Seleccionar Repository

```
Repository: tu-usuario/Sistema_Tesis
Branch: main
Main file path: main.py
```

### 2.3 Configurar Secrets

**IMPORTANTE:** Streamlit Cloud maneja secrets en panel de configuración.

Después de crear la app:

1. Click ⚙️ **Settings** (esquina superior derecha)
2. Click **Secrets**
3. Pega el contenido (SIN comentarios):

```toml
SUPABASE_URL = "https://xxx.supabase.co"
SUPABASE_KEY = "your-public-anon-key"
AWS_ACCESS_KEY_ID = "your-access-key"
AWS_SECRET_ACCESS_KEY = "your-secret-key"
AWS_REGION = "us-east-1"
AWS_BUCKET_NAME = "sistema-tesis-paul-20260215"
```

4. Click **Save**
5. **App se redeployed automáticamente** ✨

---

## 📦 PASO 3: Variables de Entorno - Cómo se cargan

El código actual en `src/utils/config.py` o similar carga variables así:

```python
import os
from dotenv import load_dotenv

# En LOCAL (development): lee .env
load_dotenv()

# En CLOUD (Streamlit Cloud): lee secrets.toml automáticamente
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
AWS_BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")
```

**Streamlit Cloud automáticamente convierte `secrets.toml` en variables de entorno** ✨

---

## ✅ PASO 4: Deployment Checklist

### Pre-deployment

- [ ] Código está en GitHub (main branch)
- [ ] `.streamlit/secrets.toml` NO está en git
- [ ] `requirements.txt` actualizado con todas las dependencias
- [ ] `.streamlit/config.toml` tiene settings correctas
- [ ] Probaste localmente: `streamlit run main.py`
- [ ] Git está limpio: `git status` no muestra cambios

### Deployment

- [ ] Crea app nueva en https://share.streamlit.io
- [ ] Configura Secrets (SUPABASE_URL, SUPABASE_KEY, AWS_*)
- [ ] Espera a que compile (2-3 minutos)
- [ ] Verifica que no hay errores en Logs

### Post-deployment

- [ ] Accede al URL público de la app
- [ ] Loguéate con credenciales de una organización
- [ ] Verifica que carga data (admin upload)
- [ ] Verifica que otro usuario puede ver data cacheada
- [ ] Prueba con 2 organizaciones diferentes (isolación)

---

## 🔍 Testing en Cloud

### Test 1: Authentication
```
1. Abre: https://your-app.streamlit.app
2. Loguéate con email/password
3. Debería mostrar tu organización en sidebar
✅ Si funciona, Supabase está correctamente configurado
```

### Test 2: Data Loading
```
1. Loguéate como admin
2. Upload un CSV (sidebar)
3. Espera a que procese (puede ser lento en FREE tier)
4. Verifica que muestra data en tabs
✅ Si funciona, AWS S3 y DataPipeline están OK
```

### Test 3: Cache Check
```
1. Loguéate como admin
2. Sidebar debería mostrar: ✅ Datos Cacheados
3. Loguéate como viewer (otra sesión/navegador)
4. Debería cargar INSTANTÁNEAMENTE desde cache
✅ Si funciona, caché multi-user está OK
```

### Test 4: Multi-org Isolation
```
1. Loguéate con usuario de ORG A
2. Ver datos de ORG A
3. Loguéate como usuario de ORG B (incognito)
4. Ver datos de ORG B (diferentes)
✅ Si datos son diferentes, RLS está funcionando
```

---

## 🐛 Troubleshooting

### "ModuleNotFoundError: No module named 'xxx'"

**Causa:** Falta dependencia en `requirements.txt`

**Solución:**
```bash
# Locally:
pip freeze > requirements.txt
git add requirements.txt
git commit -m "Update requirements.txt"
git push

# En Streamlit Cloud:
- Click: Deploy → Rerun
- O espera 5 min y recarga la página
```

### "Error: Email not confirmed"

**Causa:** Usuarios en Supabase no tienen email confirmado

**Solución:**
```sql
-- En Supabase SQL Editor:
UPDATE auth.users 
SET email_confirmed_at = now()
WHERE email_confirmed_at IS NULL;
```

### "Error loading organization data"

**Causa:** Organization_id no coincide

**Solución:**
```sql
-- Verificar que usuarios tienen org_id:
SELECT email, organization_id, is_admin FROM public.users;

-- Crear/asignar org si falta:
UPDATE public.users 
SET organization_id = '550e8400-e29b-41d4-a716-446655440000'
WHERE email = 'user@email.com';
```

### "AWS S3 Access Denied"

**Causa:** IAM credentials no válidos o sin permiso S3

**Solución:**
1. Verifica credenciales en Streamlit Cloud Secrets
2. Verifica que IAM user tiene policy `s3:*` o `s3:GetObject, s3:PutObject`
3. Verifica que bucket existe y es accesible

### "App very slow or times out"

**Causa:** 
- Streamlit Cloud free tier (limited CPU)
- GPU load alto en ML inference
- Muchos usuarios simultáneamente

**Solución:**
```python
# En dashboard.py, cachear más agresivamente
@st.cache_data(ttl=3600)  # Cache por 1 hora
def expensive_computation():
    ...
```

---

## 📊 Monitoring & Logs

### Ver logs en Streamlit Cloud

```
1. Abre https://share.streamlit.io
2. Click tu app
3. Click "Logs" (inferior derecha)
4. Verifica si hay errores
```

### Logs locales (para debugg)

```bash
# En LOCAL:
streamlit run main.py --logger.level=debug

# En CLOUD:
- No puedes editar code desde Cloud
- Todo cambio requiere push a GitHub
- Cloud redeploy automáticamente
```

---

## 🎯 Próximos pasos (Opcional)

### Custom Domain
```
1. Streamlit Cloud → App Settings → Custom Domain
2. Apunta tu dominio: app.tuempresa.com
3. Certificado SSL auto-generado
```

### Pro Tier (si necesitas)
```
- Menos limitations en recursos
- Analytics y monitoring mejorado
- Soporte prioritario
```

### CI/CD Improvements
- Agregar GitHub Actions para tests antes de deploy
- Auto-deploy al hacer push a `main`

---

## ✨ Summary

Tu app está lista para producción:

✅ **Multi-tenant architecture** - Cada org ve solo sus datos
✅ **Cache strategy** - Viewers cargan al instante
✅ **Authentication** - Supabase Auth
✅ **File storage** - AWS S3
✅ **Production-ready** - Config + secrets management

**Tiempo estimado de deployment:** 15-30 minutos (esperar compilación)

**Costo estimado:** 
- Streamlit Cloud FREE tier: $0
- Supabase: ~$5-15/mes (prod tier)
- AWS S3: ~$1-5/mes (low usage)
- **Total: ~$6-20/mes**

¡Listo para ir a producción! 🚀
