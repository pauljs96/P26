-- ========================================
-- WEEK 1: Multi-Tenant Schema Setup
-- Ejecutar en Supabase SQL Console
-- ========================================

-- 1. CREATE TABLE: organizations
CREATE TABLE organizations (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  nombre VARCHAR(255) NOT NULL UNIQUE,
  description TEXT,
  admin_user_id UUID REFERENCES auth.users(id) ON DELETE RESTRICT,
  data_loaded BOOLEAN DEFAULT FALSE,
  created_at TIMESTAMP DEFAULT now(),
  updated_at TIMESTAMP DEFAULT now()
);

-- 2. CREATE TABLE: org_cache (almacena resultados procesados)
CREATE TABLE org_cache (
  organization_id UUID PRIMARY KEY REFERENCES organizations(id) ON DELETE CASCADE,
  demand_monthly JSONB,  -- Serialized DataFrame con demanda mensual
  stock_monthly JSONB,   -- Serialized DataFrame con stock mensual
  movements JSONB,       -- Movimientos originales (opcional)
  updated_at TIMESTAMP DEFAULT now(),
  processed_by UUID REFERENCES auth.users(id),
  csv_files_count INTEGER DEFAULT 0  -- Cuántos CSVs fueron procesados
);

-- 3. CREATE TABLE: org_csv_schema (mapeo de columnas por org)
CREATE TABLE org_csv_schema (
  organization_id UUID PRIMARY KEY REFERENCES organizations(id) ON DELETE CASCADE,
  csv_separator CHAR(1) DEFAULT ',',
  csv_encoding VARCHAR(20) DEFAULT 'utf-8',
  column_mapping JSONB DEFAULT '{
    "product": "producto",
    "date": "fecha", 
    "quantity": "cantidad",
    "company": "empresa"
  }'::jsonb,
  created_at TIMESTAMP DEFAULT now(),
  updated_by UUID REFERENCES auth.users(id),
  notes TEXT
);

-- 4. ALTER TABLE: users (agregar campos para multi-tenant)
ALTER TABLE users 
ADD COLUMN organization_id UUID REFERENCES organizations(id) ON DELETE SET NULL,
ADD COLUMN is_admin BOOLEAN DEFAULT FALSE,
ADD COLUMN created_by UUID REFERENCES auth.users(id) ON DELETE SET NULL,
ADD COLUMN status VARCHAR(20) DEFAULT 'active';  -- active, invited, disabled

-- 5. ALTER TABLE: uploads (agregar organization_id)
ALTER TABLE uploads 
ADD COLUMN organization_id UUID REFERENCES organizations(id) ON DELETE CASCADE;

-- 6. Crear índices para queries rápidas
CREATE INDEX idx_users_org_id ON users(organization_id);
CREATE INDEX idx_uploads_org_id ON uploads(organization_id);
CREATE INDEX idx_org_cache_org_id ON org_cache(organization_id);

-- 7. Crear política RLS para org_cache (opcional pero recomendado)
ALTER TABLE org_cache ENABLE ROW LEVEL SECURITY;

-- Política: users solo ven cache de su org
CREATE POLICY org_cache_org_isolation ON org_cache 
FOR SELECT 
USING (
  organization_id IN (
    SELECT organization_id FROM users 
    WHERE id = auth.uid()
  )
);

-- Insertados solo por admins
CREATE POLICY org_cache_admin_insert ON org_cache 
FOR INSERT 
WITH CHECK (
  (SELECT is_admin FROM users WHERE id = auth.uid()) = TRUE
);

-- ========================================
-- Datos de prueba (OPCIONAL)
-- Descomenta para crear 2 orgs de prueba
-- ========================================

/*
-- Org 1: Tech Company
INSERT INTO organizations (nombre, description) 
VALUES ('Tech Company S.A.', 'Primera organización de prueba');

-- Org 2: Retail Company
INSERT INTO organizations (nombre, description) 
VALUES ('Retail Company Ltd.', 'Segunda organización de prueba');
*/

-- ========================================
-- Verificación
-- ========================================

-- Ver nuevas tablas
-- SELECT table_name FROM information_schema.tables 
-- WHERE table_schema = 'public' AND table_name LIKE 'org%';

-- Ver columnas nuevas en users
-- SELECT column_name, data_type FROM information_schema.columns 
-- WHERE table_name = 'users';
