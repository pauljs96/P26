-- ========================================
-- AGREGAR COLUMNAS Y TABLAS FALTANTES AL SCHEMA MULTI-TENANT
-- ========================================

-- Agregar columna data_loaded a organizations si no existe
ALTER TABLE organizations ADD COLUMN IF NOT EXISTS data_loaded BOOLEAN DEFAULT FALSE;

-- Agregar columna s3_folder si no existe (para org-aware S3 storage)
ALTER TABLE organizations ADD COLUMN IF NOT EXISTS s3_folder VARCHAR DEFAULT 'data/{org_id}/';

-- ========== ORG CACHE TABLE ==========
-- Tabla para cachear datos procesados de organizaciones
CREATE TABLE IF NOT EXISTS org_cache (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  organization_id UUID NOT NULL UNIQUE REFERENCES organizations(id) ON DELETE CASCADE,
  demand_monthly TEXT,
  stock_monthly TEXT,
  movements TEXT,
  processed_by UUID REFERENCES users(id),
  csv_files_count INT DEFAULT 0,
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_org_cache_org ON org_cache(organization_id);

-- Habilitar RLS en org_cache
ALTER TABLE org_cache ENABLE ROW LEVEL SECURITY;

-- Policy: Users can see cache for their orgs
CREATE POLICY org_cache_select_policy ON org_cache FOR SELECT
  USING (
    organization_id IN (
      SELECT org_id FROM user_org_assignments WHERE user_id = auth.uid()
    )
    OR (SELECT is_master_admin FROM users WHERE id = auth.uid())
  );

-- Policy: Users can update cache for their orgs (org_admin only)
CREATE POLICY org_cache_update_policy ON org_cache FOR UPDATE
  USING (
    organization_id IN (
      SELECT org_id FROM user_org_assignments WHERE user_id = auth.uid() AND role_id IN (1, 2)
    )
    OR (SELECT is_master_admin FROM users WHERE id = auth.uid())
  );

-- Policy: Users can insert cache for their orgs (org_admin only)  
CREATE POLICY org_cache_insert_policy ON org_cache FOR INSERT
  WITH CHECK (
    EXISTS (
      SELECT 1 FROM user_org_assignments 
      WHERE user_id = auth.uid() AND org_id = organization_id AND role_id IN (1, 2)
    )
  );

PRINT 'Missing columns and org_cache table added!';

