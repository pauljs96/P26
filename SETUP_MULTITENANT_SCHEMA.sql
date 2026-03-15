-- ========================================
-- MULTI-TENANT SCHEMA SETUP
-- Sistema_Tesis - Demanda Management
-- ========================================

-- ========== ORGANIZATIONS ==========
CREATE TABLE IF NOT EXISTS organizations (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name VARCHAR NOT NULL UNIQUE,
  description TEXT,
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW(),
  s3_folder VARCHAR NOT NULL DEFAULT 'data/{org_id}/',
  is_active BOOLEAN DEFAULT TRUE
);

-- ========== USERS (autenticados) ==========
-- Se crean via Supabase Auth, este tabla es metadata
CREATE TABLE IF NOT EXISTS users (
  id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
  email VARCHAR NOT NULL,
  full_name VARCHAR,
  is_master_admin BOOLEAN DEFAULT FALSE,
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
);

-- ========== ROLES ==========
CREATE TABLE IF NOT EXISTS roles (
  id SERIAL PRIMARY KEY,
  name VARCHAR NOT NULL UNIQUE CHECK (name IN ('master_admin', 'org_admin', 'viewer')),
  description TEXT,
  permissions JSONB DEFAULT '{}'
);

INSERT INTO roles(name, description, permissions) VALUES
  ('master_admin', 'Master administrator - full access', '{"view_all_orgs": true, "manage_orgs": true, "view_all_users": true}'),
  ('org_admin', 'Organization administrator', '{"upload_csv": true, "manage_org_users": true, "view_org_data": true}'),
  ('viewer', 'Data viewer - read only', '{"view_org_data": true}')
ON CONFLICT(name) DO NOTHING;

-- ========== USER-ORG MAPPING ==========
CREATE TABLE IF NOT EXISTS user_org_assignments (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  org_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
  role_id INT NOT NULL REFERENCES roles(id),
  assigned_at TIMESTAMP DEFAULT NOW(),
  UNIQUE(user_id, org_id)
);

-- ========== DATA UPLOADS (Auditoría) ==========
CREATE TABLE IF NOT EXISTS data_uploads (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  org_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
  uploaded_by UUID NOT NULL REFERENCES users(id),
  file_name VARCHAR NOT NULL,
  s3_path VARCHAR NOT NULL,
  year INT,
  file_size_mb FLOAT,
  rows_processed INT,
  status VARCHAR CHECK (status IN ('pending', 'processing', 'success', 'failed')),
  error_message TEXT,
  created_at TIMESTAMP DEFAULT NOW(),
  processed_at TIMESTAMP
);

-- ========== ANALYSES & CONFIGURATIONS ==========
CREATE TABLE IF NOT EXISTS org_configurations (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  org_id UUID NOT NULL UNIQUE REFERENCES organizations(id) ON DELETE CASCADE,
  config_data JSONB DEFAULT '{}',
  date_range_start DATE,
  date_range_end DATE,
  updated_at TIMESTAMP DEFAULT NOW()
);

-- ========== ANALYSIS RESULTS ==========
CREATE TABLE IF NOT EXISTS analysis_results (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  org_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
  performed_by UUID REFERENCES users(id),
  analysis_type VARCHAR NOT NULL,
  results_summary JSONB,
  s3_results_path VARCHAR,
  created_at TIMESTAMP DEFAULT NOW()
);

-- ========== INDEXES ==========
CREATE INDEX idx_userorg_user ON user_org_assignments(user_id);
CREATE INDEX idx_userorg_org ON user_org_assignments(org_id);
CREATE INDEX idx_uploads_org ON data_uploads(org_id);
CREATE INDEX idx_uploads_status ON data_uploads(status);
CREATE INDEX idx_analysis_org ON analysis_results(org_id);

-- ========== ROW-LEVEL SECURITY (RLS) ==========

-- Enable RLS on all tables
ALTER TABLE organizations ENABLE ROW LEVEL SECURITY;
ALTER TABLE users ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_org_assignments ENABLE ROW LEVEL SECURITY;
ALTER TABLE data_uploads ENABLE ROW LEVEL SECURITY;
ALTER TABLE org_configurations ENABLE ROW LEVEL SECURITY;
ALTER TABLE analysis_results ENABLE ROW LEVEL SECURITY;

-- Policy: Users can only see their own user info
CREATE POLICY user_select_own ON users FOR SELECT 
  USING (auth.uid() = id OR (SELECT is_master_admin FROM users WHERE id = auth.uid()));

-- Policy: Users can see orgs they're assigned to (or all if master)
CREATE POLICY org_select_policy ON organizations FOR SELECT
  USING (
    id IN (
      SELECT org_id FROM user_org_assignments WHERE user_id = auth.uid()
    )
    OR (SELECT is_master_admin FROM users WHERE id = auth.uid())
  );

-- Policy: Users can see assignments for their orgs
CREATE POLICY org_assignment_policy ON user_org_assignments FOR SELECT
  USING (
    org_id IN (
      SELECT org_id FROM user_org_assignments WHERE user_id = auth.uid()
    )
    OR (SELECT is_master_admin FROM users WHERE id = auth.uid())
  );

-- Policy: Users can see uploads for their orgs
CREATE POLICY data_upload_policy ON data_uploads FOR SELECT
  USING (
    org_id IN (
      SELECT org_id FROM user_org_assignments WHERE user_id = auth.uid()
    )
    OR (SELECT is_master_admin FROM users WHERE id = auth.uid())
  );

-- Policy: Users can see configs for their orgs
CREATE POLICY org_config_policy ON org_configurations FOR SELECT
  USING (
    org_id IN (
      SELECT org_id FROM user_org_assignments WHERE user_id = auth.uid()
    )
    OR (SELECT is_master_admin FROM users WHERE id = auth.uid())
  );

-- Policy: Users can see analyses for their orgs
CREATE POLICY analysis_policy ON analysis_results FOR SELECT
  USING (
    org_id IN (
      SELECT org_id FROM user_org_assignments WHERE user_id = auth.uid()
    )
    OR (SELECT is_master_admin FROM users WHERE id = auth.uid())
  );

-- ========== FUNCTIONS ==========

-- Helper: Get user's role in an org
CREATE OR REPLACE FUNCTION get_user_org_role(p_user_id UUID, p_org_id UUID)
RETURNS VARCHAR AS $$
  SELECT r.name FROM user_org_assignments uoa
  JOIN roles r ON uoa.role_id = r.id
  WHERE uoa.user_id = p_user_id AND uoa.org_id = p_org_id;
$$ LANGUAGE SQL;

-- Helper: Check if user is master admin
CREATE OR REPLACE FUNCTION is_master_admin(p_user_id UUID)
RETURNS BOOLEAN AS $$
  SELECT COALESCE((SELECT is_master_admin FROM users WHERE id = p_user_id), FALSE);
$$ LANGUAGE SQL;

-- Helper: Check if user has org access
CREATE OR REPLACE FUNCTION has_org_access(p_user_id UUID, p_org_id UUID)
RETURNS BOOLEAN AS $$
  SELECT is_master_admin(p_user_id) OR
    EXISTS (SELECT 1 FROM user_org_assignments WHERE user_id = p_user_id AND org_id = p_org_id);
$$ LANGUAGE SQL;

PRINT 'Multi-tenant schema setup complete!';
