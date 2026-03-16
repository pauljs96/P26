-- ========================================
-- AGREGAR TABLA PROJECTS AL SCHEMA MULTI-TENANT
-- ========================================

-- Crear tabla de proyectos (vinculada a organizaciones)
CREATE TABLE IF NOT EXISTS projects (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  org_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
  created_by UUID NOT NULL REFERENCES users(id),
  name VARCHAR NOT NULL,
  description TEXT,
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW(),
  UNIQUE(org_id, name)
);

-- Crear índices
CREATE INDEX idx_projects_org ON projects(org_id);
CREATE INDEX idx_projects_creator ON projects(created_by);

-- Habilitar RLS en projects
ALTER TABLE projects ENABLE ROW LEVEL SECURITY;

-- Policy: Users can see projects from their orgs
CREATE POLICY projects_select_policy ON projects FOR SELECT
  USING (
    org_id IN (
      SELECT org_id FROM user_org_assignments WHERE user_id = auth.uid()
    )
    OR (SELECT is_master_admin FROM users WHERE id = auth.uid())
  );

-- Policy: Users can create projects in their orgs (org_admin only)
CREATE POLICY projects_insert_policy ON projects FOR INSERT
  WITH CHECK (
    EXISTS (
      SELECT 1 FROM user_org_assignments 
      WHERE user_id = auth.uid() AND org_id = org_id AND role_id IN (1, 2)  -- master_admin or org_admin
    )
  );

-- Policy: Users can update projects in their orgs
CREATE POLICY projects_update_policy ON projects FOR UPDATE
  USING (
    org_id IN (
      SELECT org_id FROM user_org_assignments WHERE user_id = auth.uid() AND role_id IN (1, 2)
    )
    OR (SELECT is_master_admin FROM users WHERE id = auth.uid())
  );

PRINT 'Projects table added to multi-tenant schema!';
