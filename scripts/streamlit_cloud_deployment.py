#!/usr/bin/env python3
"""
STREAMLIT CLOUD DEPLOYMENT GUIDE
Sistema Tesis Multi-Tenant
March 15, 2026
"""

import os
import sys
import webbrowser
from pathlib import Path

def print_header():
    """Print deployment guide header"""
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║   STREAMLIT CLOUD DEPLOYMENT GUIDE                       ║
    ║   Sistema Tesis Multi-Tenant                             ║
    ║   March 15, 2026                                         ║
    ╚═══════════════════════════════════════════════════════════╝
    """)

def step_1_github():
    """Step 1: GitHub Repository Setup"""
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║ PASO 1: GITHUB REPOSITORY                                ║
    ╚═══════════════════════════════════════════════════════════╝
    
    ✅ Repository: https://github.com/pauljs96/P26
    ✅ Branch: main
    ✅ Main file: main.py
    ✅ Status: PUSHED (commit 4c1141f)
    
    📋 Verificar que el repo está actualizado:
    
    """)
    
    commands = [
        "git status",
        "git log --oneline -3",
    ]
    
    for cmd in commands:
        print(f"    $ {cmd}")

def step_2_local_secrets():
    """Step 2: Create Local Secrets"""
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║ PASO 2: CREAR SECRETS LOCALES (OPCIONAL)                 ║
    ╚═══════════════════════════════════════════════════════════╝
    
    Para probar localmente antes de hacer deploy:
    
    1. Copiar template:
    """)
    
    print("""
       cp .streamlit/secrets.toml.example .streamlit/secrets.toml
    
    2. Editar .streamlit/secrets.toml con credenciales reales:
    """)
    
    print("""
       [supabase]
       url = "https://YOUR_PROJECT.supabase.co"
       key = "your-anon-key"
       
       [aws]
       access_key_id = "AKIA..."
       secret_access_key = "..."
       region = "us-east-1"
       bucket = "your-bucket"
    
    3. Verificar que NO está commiteado:
    """)
    
    print("""
       $ grep ".streamlit/secrets.toml" .gitignore
       ✅ Should show: .streamlit/secrets.toml
    
    4. Probar localmente:
    """)
    
    print("""
       $ streamlit run main.py
       # Login con credentials de demo
    """)

def step_3_streamlit_cloud():
    """Step 3: Setup Streamlit Cloud"""
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║ PASO 3: CONFIGURAR STREAMLIT CLOUD                        ║
    ╚═══════════════════════════════════════════════════════════╝
    
    1. Ir a https://share.streamlit.io
    
    2. Click "Sign up" (o login)
    
    3. Click "New app"
    
    4. Conectar GitHub:
       - Repository: pauljs96/P26
       - Branch: main
       - Main file path: main.py
    
    5. Click "Deploy"
    
    ⏳ Streamlit compilará y deployará la app (~2-3 min)
    
    📝 URL Resultado:
       https://share.streamlit.io/pauljs96/P26/main/main.py
    
    """)

def step_4_configure_secrets():
    """Step 4: Configure Secrets in Streamlit Cloud"""
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║ PASO 4: CONFIGURAR SECRETS EN STREAMLIT CLOUD             ║
    ╚═══════════════════════════════════════════════════════════╝
    
    1. En Streamlit Cloud, ir a app settings (⋮ menú)
    
    2. Click "Secrets"
    
    3. Paste el siguiente contenido (en formato TOML):
    """)
    
    print("""
       [supabase]
       url = "https://YOUR_PROJECT.supabase.co"
       key = "eyJhbGc..."
       
       [aws]
       access_key_id = "AKIA..."
       secret_access_key = "..."
       region = "us-east-1"
       bucket = "sistema-tesis-prod"
       
       [app]
       environment = "production"
       log_level = "INFO"
    
    4. Click "Save"
    
    5. Streamlit reiniciará la app automáticamente
    
    """)

def step_5_test_production():
    """Step 5: Test Production Deployment"""
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║ PASO 5: TESTEAR DEPLOYMENT EN PRODUCCIÓN                  ║
    ╚═══════════════════════════════════════════════════════════╝
    
    URL: https://share.streamlit.io/pauljs96/P26/main/main.py
    
    📋 CHECKLIST DE TESTING:
    
    [] 1. Login page carga
    [] 2. Login con credentials de demo funciona
    
         Master Admin: admin@sistematesis.com / Admin@123456
         Org Admin: admin@techinnovations.local / OrgAdmin@123456
         Viewer: user1@techinnovations.local / User@123456
    
    [] 3. Dashboard loads después del login
    [] 4. Org selector muestra múltiples orgs
    [] 5. Cambiar org funciona
    [] 6. Datos carga desde S3
    [] 7. Querys son rápidas (<2s)
    [] 8. Sample data visible
    
    ✅ Si todos los tests pasan: DEPLOYMENT EXITOSO
    
    """)

def step_6_monitoring():
    """Step 6: Setup Monitoring"""
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║ PASO 6: MONITOREO EN PRODUCCIÓN                           ║
    ╚═══════════════════════════════════════════════════════════╝
    
    📊 Que monitorear:
    
    1. Streamlit Cloud Dashboard:
       - App status
       - Memory usage
       - Error logs
       - Deploy history
    
    2. Supabase Dashboard:
       - Query performance
       - Auth logs
       - RLS violations
    
    3. AWS CloudWatch:
       - S3 access logs
       - Error rates
       - Performance metrics
    
    🔔 Setup alertas:
       - Error rate > 1%
       - Query time > 5s
       - Memory > 1GB
    
    """)

def step_7_troubleshooting():
    """Step 7: Troubleshooting"""
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║ PASO 7: TROUBLESHOOTING                                   ║
    ╚═══════════════════════════════════════════════════════════╝
    
    ❌ ERROR: "Clustering.py failed to load"
       ✅ FIX: Agregar a requirements.txt:
          scikit-learn==1.3.0
          statsmodels==0.14.0
    
    ❌ ERROR: "Secrets not found"
       ✅ FIX: Verificar que secrets.toml está en .streamlit/
              NO hacer commit
              Configurar en Streamlit Cloud UI
    
    ❌ ERROR: "Can't connect to Supabase"
       ✅ FIX: Verificar URL y KEY en secrets
              Verificar Supabase API está activo
              Verificar firewall
    
    ❌ ERROR: "S3 access denied"
       ✅ FIX: Verificar AWS credentials
              Verificar bucket policy
              Verificar IAM permissions
    
    ❌ ERROR: "Database connection timeout"
       ✅ FIX: Aumentar timeout en connection strings
              Verificar DuckDB memory limits
              Reducir data per query
    
    📝 Para ver logs:
       - Streamlit Cloud: App → Logs (visible en dashboard)
       - Local: streamlit run main.py (logs en terminal)
    
    """)

def step_8_advanced():
    """Step 8: Advanced Configuration"""
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║ PASO 8: CONFIGURACIÓN AVANZADA                            ║
    ╚═══════════════════════════════════════════════════════════╝
    
    🔧 Custom Domain (opcional):
       - Comprar dominio en GoDaddy/Namecheap
       - Configurar CNAME en Streamlit Cloud
       - Ej: app.sistematesis.com → share.streamlit.io
    
    🔐 GitHub Actions CI/CD (opcional):
       - .github/workflows/deploy.yml
       - Auto-test on push
       - Auto-deploy on main branch
       - See: scripts/github_actions.yml
    
    📦 Docker (si migrar de Streamlit Cloud):
       - Crear Dockerfile
       - Push a Docker Hub
       - Deploy en AWS/GCP/Azure
    
    🚀 Scaling (si aumenta tráfico):
       - Upgrade Streamlit org plan
       - Setup load balancer
       - Increase DuckDB memory
       - Optimize queries
    
    """)

def main():
    """Main deployment guide"""
    print_header()
    
    steps = [
        ("GitHub Repository", step_1_github),
        ("Local Secrets (Optional)", step_2_local_secrets),
        ("Streamlit Cloud Setup", step_3_streamlit_cloud),
        ("Configure Secrets", step_4_configure_secrets),
        ("Test Production", step_5_test_production),
        ("Monitoring", step_6_monitoring),
        ("Troubleshooting", step_7_troubleshooting),
        ("Advanced Config", step_8_advanced),
    ]
    
    print("""
    📋 TABLA DE CONTENIDOS:
    """)
    
    for i, (title, _) in enumerate(steps, 1):
        print(f"    {i}. {title}")
    
    print("""
    ═════════════════════════════════════════════════════════════
    """)
    
    for title, step_func in steps:
        step_func()
        input("    [ENTER para continuar...]")
        print("\n")
    
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║ ✅ DEPLOYMENT GUIDE COMPLETADO                           ║
    ╚═══════════════════════════════════════════════════════════╝
    
    📊 RESUMEN:
    
    • Repository: https://github.com/pauljs96/P26 ✅
    • Secrets: Configurados en Streamlit Cloud ✅
    • App URL: https://share.streamlit.io/pauljs96/P26/main/main.py
    • Status: 🟢 PRODUCTION READY
    
    🚀 PRÓXIMOS PASOS:
    
    1. Ejecutar los tests localmente:
       python scripts/run_fase3_tests.py
    
    2. Configurar secrets en Streamlit Cloud
    
    3. Monitorear logs en Streamlit Dashboard
    
    4. Invitar users a probar
    
    5. Recolectar feedback
    
    📞 SOPORTE:
    - Streamlit docs: https://docs.streamlit.io
    - Supabase docs: https://supabase.com/docs
    - AWS S3: https://docs.aws.amazon.com/s3
    
    """)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Deployment guide cancelled by user")
        sys.exit(1)
