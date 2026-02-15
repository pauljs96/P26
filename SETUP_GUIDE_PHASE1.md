# SETUP GUIDE - Fase 1: MVP Cloud

**Objetivo:** Desplegar Sistema_Tesis con autenticación y Supabase en 30 minutos.

---

## 📌 Requisitos Previos

- Python 3.12+
- pip
- Cuenta gratuita en [Supabase](https://supabase.com) (5 minutos)
- Git (ya lo tienes)

---

## 🚀 PASO 1: Instalar Dependencias (5 min)

```bash
cd d:\Desktop\TESIS\Sistema_Tesis

# Activar venv si no lo está
venv\Scripts\activate

# Instalar todo
pip install -r requirements.txt
```

**Verificación:**
```bash
python -c "import streamlit, supabase, streamlit_authenticator; print('✅ OK')"
```

---

## 🚀 PASO 2: Crear Proyecto Supabase (10 min)

### 2.1 Crear proyecto gratis

1. Ir a [Supabase Dashboard](https://app.supabase.com)
2. Click **"New Project"**
3. Rellenar:
   - **Project Name:** `Sistema-Tesis` (o tu nombre)
   - **Database Password:** Guardar en lugar seguro (lo usarás en .env)
   - **Region:** `us-east-1` (o la más cercana)
4. Click **"Create new project"**
5. **Esperar 2-3 minutos** mientras se provisionea

### 2.2 Obtener credenciales

Una vez listo, en el dashboard:
1. Click **Settings** (abajo izquierda)
2. Ir a **API**
3. Copiar:
   - **Project URL** → `SUPABASE_URL=`
   - **anon public** (bajo API KEYS) → `SUPABASE_KEY=`

Ejemplo:
```
SUPABASE_URL=https://abcdefg123456.supabase.co
SUPABASE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

### 2.3 Crear schema SQL

1. En Supabase Dashboard, click **SQL Editor** (izquierda)
2. Click **"New Query"**
3. Copiar TODO el contenido de [SETUP_SUPABASE.sql](SETUP_SUPABASE.sql)
4. Pegar en el editor
5. Click **"Run"** (o Ctrl+Enter)
6. Esperar a que complete (no debe dar errores)

**Si funciona:** Verás "Success" y la consulta SQL se ejecutó.

---

## 🚀 PASO 3: Configurar `.env` Local (5 min)

### 3.1 Crear archivo `.env`

**Copiar** de `.env.example` al proyecto raíz:

```bash
# Windows
copy .env.example .env

# Mac/Linux
cp .env.example .env
```

### 3.2 Editar `.env`

Abre `.env` con tu editor y reemplaza:

```bash
# Antes:
SUPABASE_URL=https://xxxxx.supabase.co
SUPABASE_KEY=xxxxx

# Después: (con TUS valores)
SUPABASE_URL=https://abcdefg123456.supabase.co
SUPABASE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...

# Deja el resto igual
ENVIRONMENT=development
STREAMLIT_SERVER_HEADLESS=false
```

**⚠️ Importante:**
- NO commits .env a Git (está en .gitignore)
- .env.example es solo template
- Mantén SUPABASE_KEY secreto

---

## 🚀 PASO 4: Test Local (5 min)

### 4.1 Ejecutar dashboard

```bash
cd d:\Desktop\TESIS\Sistema_Tesis
streamlit run main.py
```

**Debe abrir:** `http://localhost:8501`

### 4.2 Probar login/registro

En la pantalla de autenticación:

**Opción A: Modo Producción (con Supabase):**
- Si .env está correcto → puedes registrarte real
- Email debe ser válido
- Contraseña ≥6 caracteres
- Los datos se guardan en Supabase

**Opción B: Modo Demo (sin Supabase):**
- Si SUPABASE_URL/KEY no están configurados
- O si hay error de conexión
- Aparecerá: "⚠️ Modo demo: sin conexión a Supabase"
- Puedes loggeartecon cualquier email/password
- Los datos quedan SOLO en session (se pierden al cerrar)

### 4.3 Validar que funciona

```
✅ Hago click en "Registrarse"
✅ Lleno formulario (empresa, email, password)
✅ Click "Registrarse"
✅ Mensaje "Registro exitoso"
✅ Vuelvo a login
✅ Uso las mismas credenciales
✅ ¡Entro al dashboard!
✅ Veo botón "Cerrar Sesión" en sidebar
```

---

## 🚀 PASO 5: Primer Commit (2 min)

```bash
cd d:\Desktop\TESIS\Sistema_Tesis

git add .
git commit -m "[PHASE-1-W1] Add authentication, Supabase DB module, services layer"
git push origin main
```

---

## 📋 Checklist de Verificación

- [ ] requirements.txt actualizado (scikit-learn, supabase, streamlit-authenticator, python-dotenv)
- [ ] src/db/supabase.py creado ✅
- [ ] src/services/ml_service.py creado ✅
- [ ] SETUP_SUPABASE.sql ejecutado en Supabase
- [ ] .env configurado con credenciales reales
- [ ] dashboard.py tiene autenticación ✅
- [ ] `streamlit run main.py` abre login screen
- [ ] Puedo registrarme y loguearme
- [ ] Botón logout funciona
- [ ] Commit hecho con git

---

## 🐛 Troubleshooting

### "ModuleNotFoundError: supabase"
```bash
pip install -q supabase
```

### "Supabase credenciales no encontradas"
- Verificar .env existe en raíz: `ls .env`
- Verificar SUPABASE_URL y SUPABASE_KEY están completos (no xxxxx)
- Restart Streamlit: Ctrl+C + `streamlit run main.py`

### "No puedo registrarme/login"

**Opción A: Modo demo**
- Si deseas probar sin Supabase
- Deja .env sin configurar
- Dashboard entra en "Modo Demo"
- Funciona perfectamente para testing

**Opción B: Validar Supabase**
```bash
# Probar conexión desde terminal
python -c "
import os
from dotenv import load_dotenv
load_dotenv()
url = os.getenv('SUPABASE_URL')
key = os.getenv('SUPABASE_KEY')
print(f'URL: {url}')
print(f'KEY: {key[:30]}...')
from supabase import create_client
client = create_client(url, key)
print('✅ Supabase conectó exitosamente')
"
```

---

## 🎯 Próximos Pasos (Semana 2-3)

- [ ] Conectar S3 para uploads de CSV
- [ ] Crear página de proyectos/historial
- [ ] Deploy a Streamlit Cloud
- [ ] Testing multi-usuario
- [ ] Documentación usuarios finales

---

## 📚 Referencias

- [Documentación Supabase](https://supabase.com/docs)
- [Streamlit Authenticator](https://github.com/mkhorasani/streamlit-authenticator)
- [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md) - plan general
- [DEVELOPMENT_LOG.md](DEVELOPMENT_LOG.md) - historial técnico

---

**¿Preguntas?**  
Revisa [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md#open-questions) para preguntas sin resolver de la Sesión 1.

