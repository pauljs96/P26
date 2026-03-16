-- ========================================
-- DESACTIVAR RLS TEMPORALMENTE
-- Esto permite que las queries funcionen mientras arreglamos las policies
-- ========================================

-- Deshabilitar RLS en todas las tablas
ALTER TABLE organizations DISABLE ROW LEVEL SECURITY;
ALTER TABLE users DISABLE ROW LEVEL SECURITY;
ALTER TABLE user_org_assignments DISABLE ROW LEVEL SECURITY;
ALTER TABLE data_uploads DISABLE ROW LEVEL SECURITY;
ALTER TABLE org_configurations DISABLE ROW LEVEL SECURITY;
ALTER TABLE analysis_results DISABLE ROW LEVEL SECURITY;

-- Verificar que está deshabilitado
SELECT tablename, rowsecurity FROM pg_tables WHERE schemaname='public' AND tablename IN ('organizations', 'users', 'user_org_assignments', 'data_uploads', 'org_configurations', 'analysis_results');
