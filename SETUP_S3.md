# S3 Configuration Guide - Sistema Tesis

## Overview

Esta guía te ayudará a:
1. Crear cuenta AWS
2. Crear bucket S3
3. Generar credenciales IAM
4. Configurar `.env` para Sistema Tesis

**Tiempo estimado:** 15-20 minutos

---

## Paso 1: Crear Cuenta AWS (si no tienes)

### Opción A: Cuenta nueva

1. Ir a [aws.amazon.com](https://aws.amazon.com)
2. Click en **"Create an AWS Account"**
3. Llenar datos:
   - Email
   - Password
   - Account name
   - Address
   - Credit card (requiere verificación, pero free tier no cobra)
4. Elegir **"Personal"** como tipo de cuenta
5. Completar verificación de identidad
6. Seleccionar plan **"AWS Free Tier"** (free for 12 months)

### Opción B: Cuenta existente

Si ya tienes AWS, proceed to Paso 2.

---

## Paso 2: Crear Bucket S3

1. Ir a [AWS S3 Console](https://s3.console.aws.amazon.com)
2. Click en **"Create bucket"**
3. Rellenar:
   - **Bucket name:** `sistema-tesis-<tu-nombre>-<timestamp>`
     - Ej: `sistema-tesis-juan-20260215`
     - ⚠️ Debe ser único globalmente
   - **Region:** Selecciona la más cercana (ej: `us-east-1` si estás en LATAM, `sa-east-1` si usas Brasil)
4. **ACL settings:**
   - ☐ Unchecked: "Block public access" (para mantener archivos privados)
   - ✅ Checked: "Block all public access"
5. **Encryption:** Default (AES-256)
6. Click **"Create bucket"**

✅ Bucket creado. Deberías ver su nombre en la lista.

---

## Paso 3: Generar Credenciales IAM

### 3.1 Crear usuario IAM

1. Ir a [IAM Dashboard](https://console.aws.amazon.com/iam)
2. Click en **"Users"** (sidebar izquierdo)
3. Click en **"Create user"**
4. **User name:** `sistema-tesis-app`
5. Click **"Next"**

### 3.2 Asignar permisos

1. Click **"Attach policies directly"**
2. Buscar y seleccionar policies:
   - ✅ `AmazonS3FullAccess` (o más restrictivo: crear policy custom)
3. Click **"Next"**
4. Review y click **"Create user"**

### 3.3 Generar Access Keys

1. En la página del usuario, ir a **"Security credentials"** tab
2. Scroll a **"Access keys"**
3. Click **"Create access key"**
4. Elegir **"Application running outside AWS"**
5. Click **"Next"**
6. (Opcional) Agregar descripción: "Sistema Tesis S3 Upload"
7. Click **"Create access key"**

⚠️ **IMPORTANTE:** Guardar estos valores en un lugar seguro:
- **Access Key ID**
- **Secret Access Key**

No podrás volver a ver el Secret Key, así que cópialo ahora.

---

## Paso 4: Configurar `.env`

1. En proyecto, abrir/crear `.env`:

```bash
# En d:\Desktop\TESIS\Sistema_Tesis\.env
```

2. Llenar con credenciales de Paso 3:

```dotenv
# Supabase Configuration
SUPABASE_URL=https://xxxxx.supabase.co
SUPABASE_KEY=xxxxx

# AWS S3 Configuration
AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE
AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
AWS_S3_BUCKET_NAME=sistema-tesis-juan-20260215
AWS_S3_REGION=us-east-1

# App Environment
ENVIRONMENT=development
STREAMLIT_SERVER_HEADLESS=false
```

### Valores a completar:

| Variable | Dónde encontrar |
|----------|-----------------|
| `AWS_ACCESS_KEY_ID` | IAM Users → tu usuario → Security credentials → Access key ID |
| `AWS_SECRET_ACCESS_KEY` | IAM Users → tu usuario → Security credentials → Secret access key |
| `AWS_S3_BUCKET_NAME` | S3 Console → tu bucket name |
| `AWS_S3_REGION` | S3 Console → tu bucket → Properties → AWS Region |

---

## Paso 5: Probar Conexión

### Test 1: Verificar imports

```python
import sys
sys.path.append("d:\\Desktop\\TESIS\\Sistema_Tesis")

from src.storage import get_storage_manager

storage = get_storage_manager()
print(f"S3 configurado: {storage.is_configured}")
print(f"Bucket: {storage.bucket_name}")
print(f"Region: {storage.region}")
```

### Test 2: Upload de archivo de prueba

```python
# Crear archivo de prueba
with open("test_file.txt", "w") as f:
    f.write("Hello S3!")

# Upload
result = storage.upload_file(
    "test_file.txt",
    user_id="test-user",
    project_id="test-project"
)

print(f"Success: {result['success']}")
if result['success']:
    print(f"S3 URL: {result['s3_url']}")
    print(f"Presigned URL: {result['presigned_url']}")
```

### Test 3: Usar en Streamlit

1. Iniciar app: `python -m streamlit run main.py`
2. Subir un CSV desde el sidebar
3. Verificar que no haya errores en la consola
4. Verificar en S3 Console que el archivo aparezca

---

## Uso en el Dashboard

Una vez configurado, el dashboard automáticamente:

1. ✅ Sube CSVs a S3 cuando el usuario los carga
2. ✅ Guarda URLs en Supabase (tabla `uploads`)
3. ✅ Genera URLs presignadas (válidas 7 días)
4. ✅ Procesa datos localmente
5. ✅ Permite descargas desde S3 posteriormente

**Fallback:** Si S3 no está configurado, los archivos se guardan en memoria/session (no persisten).

---

## Costs (Pricing)

**AWS Free Tier (12 meses):**
- ✅ S3 Storage: 5 GB
- ✅ Data transfer: 1 GB/mes
- ✅ Unlimited uploads/downloads

**Después de free tier (~USD):**
- Storage: $0.023 per GB/month
- Download: $0.09 per GB/month
- Uploads: Free

**Estimado para Sistema Tesis:**
- 1000 CSVs × 100 KB = 100 MB → ~$0.002/mes

---

## Troubleshooting

### Error: "NoCredentialsError"

```
botocore.exceptions.NoCredentialsError: Unable to locate credentials
```

**Solución:** Verificar `.env` tiene:
- `AWS_ACCESS_KEY_ID` (no vacío)
- `AWS_SECRET_ACCESS_KEY` (no vacío)

```bash
# Test que .env está siendo leído
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print(os.getenv('AWS_ACCESS_KEY_ID'))"
```

### Error: "NoSuchBucket"

```
s3.exceptions.NoSuchBucket: An error occurred (NoSuchBucket) when calling the HeadBucket operation
```

**Solución:** Verificar nombre del bucket en S3 Console y .env:
```bash
# S3 Console → Buckets → copiar exact name
# Pegar en .env como AWS_S3_BUCKET_NAME
```

### Error: "InvalidAccessKeyId"

```
botocore.exceptions.InvalidAccessKeyId: The AWS Access Key Id you provided does not exist
```

**Solución:** Credenciales expiradas o incorrectas. Generar nuevas en IAM.

### Error: "AccessDenied"

```
s3.exceptions.EndpointConnectionError: Could not connect to the endpoint URL
```

**Solución:** El usuario IAM no tiene permisos. Verificar en IAM Dashboard que tiene `AmazonS3FullAccess`.

---

## Security Best Practices

### ✅ DO (Buenas prácticas)

1. **Nunca commitear `.env` a Git**
   - Verificar `.gitignore`:
   ```
   .env
   .env.local
   *.key
   ```

2. **Usar IAM users específicos** (no root AWS account)

3. **Restringir permisos** (no usar FullAccess en prod)
   ```json
   {
     "Version": "2012-10-17",
     "Statement": [
       {
         "Effect": "Allow",
         "Action": [
           "s3:PutObject",
           "s3:GetObject",
           "s3:DeleteObject"
         ],
         "Resource": "arn:aws:s3:::your-bucket-name/*"
       }
     ]
   }
   ```

4. **Rotar credenciales regularmente** (3-6 meses)

5. **Usar bucket encryption** (default AES-256 es OK)

### ❌ DON'T (Evitar)

1. ❌ Compartir AWS_SECRET_ACCESS_KEY
2. ❌ Commitear credenciales a GitHub
3. ❌ Usar root AWS account para apps
4. ❌ Hacer buckets completamente públicos
5. ❌ Guardar credenciales en código fuente

---

## Próximos Pasos

### Phase 1 Week 2-3:
- [ ] S3 Configuration (⬅️ Estás aquí)
- [ ] GitHub Actions CI/CD (próximo)
- [ ] Preparar para Streamlit Cloud deployment

### Phase 2:
- [ ] FastAPI backend con S3 integration
- [ ] Cloud Run / ECS deployment
- [ ] Auto-scaling

---

## Referencias

- [AWS S3 Documentación](https://docs.aws.amazon.com/s3/)
- [boto3 Documentación](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3.html)
- [AWS IAM Best Practices](https://docs.aws.amazon.com/IAM/latest/UserGuide/best-practices.html)
- [S3 Security Best Practices](https://docs.aws.amazon.com/AmazonS3/latest/userguide/security-best-practices.html)

---

## Support

Si tienes problemas:

1. Ver **Troubleshooting** section arriba
2. Verificar `.env` con `python -c "import os; from dotenv import load_dotenv; load_dotenv(); print(os.getenv('AWS_S3_BUCKET_NAME'))"`
3. Revisar AWS S3 Console → tu bucket → Object → verificar archivos se guardaron
4. Revisar logs de Streamlit en terminal
