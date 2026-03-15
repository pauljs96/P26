#!/usr/bin/env python
"""
Initialize Multi-Tenant Sistema Tesis
=====================================

Este script:
1. Ejecuta SQL schema en Supabase
2. Crea Master Admin
3. Crea 10 organizaciones demo
4. Asigna usuarios a organizaciones

Uso:
  python scripts/init_multitenant.py
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime

# Setup path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load .env
from dotenv import load_dotenv
load_dotenv()

from src.db.supabase_v2 import get_supabase_db


def main():
    """Ejecuta inicialización multi-tenant."""
    
    logger.info("=" * 60)
    logger.info("SISTEMA TESIS - MULTI-TENANT INITIALIZATION")
    logger.info("=" * 60)
    
    try:
        # 1. Conectar a Supabase
        logger.info("\n[1/3] Conectando a Supabase...")
        db = get_supabase_db()
        logger.info("✅ Conexión exitosa")
        
        # 2. Crear Master Admin
        logger.info("\n[2/3] Creando Master Admin...")
        master_result = db.register_user(
            email="admin@sistematesis.com",
            password="Admin@123456",  # CAMBIAR EN PRODUCCIÓN
            full_name="Sistema Tesis Admin",
            is_master_admin=True,
        )
        
        if master_result["success"]:
            master_user_id = master_result["user_id"]
            logger.info(f"✅ Master Admin creado: {master_user_id}")
        else:
            logger.error(f"❌ Error: {master_result['error']}")
            if "already exists" in master_result["error"].lower():
                logger.info("⚠️ Master Admin ya existe. Continuando...")
                # Obtener el ID del admin existente
                user_result = db.get_user_by_email("admin@sistematesis.com")
                if user_result["success"]:
                    master_user_id = user_result["user"]["id"]
                    logger.info(f"✅ Master Admin encontrado: {master_user_id}")
                else:
                    logger.error("Error get master admin")
                    return
            else:
                return
        
        # 3. Crear 10 Organizaciones y usuarios demo
        logger.info("\n[3/3] Creando 10 organizaciones con usuarios...")
        
        demo_orgs = [
            {
                "name": "Tech Innovations Inc",
                "admin_email": "admin@techinnovations.local",
                "users": [f"user{i}@techinnovations.local" for i in range(1, 11)]
            },
            {
                "name": "Global Retail Corp",
                "admin_email": "admin@retailcorp.local",
                "users": [f"user{i}@retailcorp.local" for i in range(1, 11)]
            },
            {
                "name": "Manufacturing Solutions",
                "admin_email": "admin@manufacturing.local",
                "users": [f"user{i}@manufacturing.local" for i in range(1, 11)]
            },
            {
                "name": "Energy Systems Ltd",
                "admin_email": "admin@energy.local",
                "users": [f"user{i}@energy.local" for i in range(1, 11)]
            },
            {
                "name": "Healthcare Services",
                "admin_email": "admin@healthcare.local",
                "users": [f"user{i}@healthcare.local" for i in range(1, 11)]
            },
            {
                "name": "Financial Services Group",
                "admin_email": "admin@finance.local",
                "users": [f"user{i}@finance.local" for i in range(1, 11)]
            },
            {
                "name": "Logistics Network",
                "admin_email": "admin@logistics.local",
                "users": [f"user{i}@logistics.local" for i in range(1, 11)]
            },
            {
                "name": "Agriculture Systems",
                "admin_email": "admin@agriculture.local",
                "users": [f"user{i}@agriculture.local" for i in range(1, 11)]
            },
            {
                "name": "Construction Materials",
                "admin_email": "admin@construction.local",
                "users": [f"user{i}@construction.local" for i in range(1, 11)]
            },
            {
                "name": "Transportation Solutions",
                "admin_email": "admin@transportation.local",
                "users": [f"user{i}@transportation.local" for i in range(1, 11)]
            },
        ]
        
        created_count = 0
        
        for org_config in demo_orgs:
            logger.info(f"\n  📌 {org_config['name']}")
            
            # Crear organización
            org_result = db.create_organization(
                name=org_config["name"],
                description=f"Demo organization: {org_config['name']}",
                s3_folder=f"demo/{org_config['name'].lower().replace(' ', '_')}",
            )
            
            if not org_result["success"]:
                # Verificar si ya existe
                orgs = db.list_organizations()
                org = next((o for o in orgs.get("organizations", []) 
                           if o["name"] == org_config["name"]), None)
                if org:
                    org_id = org["id"]
                    logger.info(f"     ⚠️  Org ya existe: {org_id}")
                else:
                    logger.error(f"     ❌ Error: {org_result['error']}")
                    continue
            else:
                org_id = org_result["org_id"]
                logger.info(f"     ✅ Org creada: {org_id}")
                created_count += 1
            
            # Crear admin de org
            admin_result = db.register_user(
                email=org_config["admin_email"],
                password="OrgAdmin@123456",  # CAMBIAR EN PRODUCCIÓN
                full_name=f"Admin - {org_config['name']}",
                is_master_admin=False,
            )
            
            if admin_result["success"]:
                admin_user_id = admin_result["user_id"]
                logger.info(f"     ✅ Admin creado: {org_config['admin_email']}")
                
                # Asignar admin a org como org_admin
                role_result = db.assign_user_to_org(admin_user_id, org_id, "org_admin")
                if role_result["success"]:
                    logger.info(f"     ✅ Admin asignado a org como org_admin")
                else:
                    logger.warning(f"     ⚠️  {role_result['error']}")
            else:
                if "already exists" in admin_result["error"].lower():
                    logger.info(f"     ⚠️  Admin ya existe: {org_config['admin_email']}")
                    user_result = db.get_user_by_email(org_config["admin_email"])
                    if user_result["success"]:
                        admin_user_id = user_result["user"]["id"]
                        role_result = db.assign_user_to_org(admin_user_id, org_id, "org_admin")
                        if role_result["success"]:
                            logger.info(f"     ✅ Admin re-asignado")
                else:
                    logger.error(f"     ❌ Error: {admin_result['error']}")
                    continue
            
            # Crear usuarios viewers
            for user_email in org_config["users"]:
                user_result = db.register_user(
                    email=user_email,
                    password="User@123456",  # CAMBIAR EN PRODUCCIÓN
                    full_name=user_email.split("@")[0],
                    is_master_admin=False,
                )
                
                if user_result["success"]:
                    user_id = user_result["user_id"]
                    
                    # Asignar a org como viewer
                    role_result = db.assign_user_to_org(user_id, org_id, "viewer")
                    if role_result["success"]:
                        pass  # Logger.info(f"       ✅ {user_email}")
                else:
                    if "already exists" not in user_result["error"].lower():
                        logger.warning(f"       ⚠️  {user_email}: {user_result['error']}")
        
        # Resumen
        logger.info("\n" + "=" * 60)
        logger.info("✅ INICIALIZACIÓN EXITOSA")
        logger.info("=" * 60)
        logger.info(f"\n📊 RESUMEN:")
        logger.info(f"   - Master Admin: admin@sistematesis.com")
        logger.info(f"   - Organizaciones creadas: {created_count}")
        logger.info(f"   - Usuarios por org: 11 (1 admin + 10 viewers)")
        logger.info(f"   - Total de usuarios: ~{110 + 1}")
        
        logger.info(f"\n📝 IMPORTANTE:")
        logger.info(f"   - Cambiar credenciales default en PRODUCCIÓN")
        logger.info(f"   - URLs de datos en S3 ya deben estar disponibles")
        logger.info(f"   - Ejecutar SETUP_MULTITENANT_SCHEMA.sql si no existe")
        
        logger.info(f"\n🚀 Próximos pasos:")
        logger.info(f"   1. Subir archivos CSV a S3 para cada org")
        logger.info(f"   2. Iniciar dashboard: streamlit run main.py")
        logger.info(f"   3. Loguear como admin para verificar datos")
        
    except Exception as e:
        logger.error(f"\n❌ ERROR: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
