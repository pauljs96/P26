-- Schema inicial para Sistema_Tesis en Supabase
-- Ejecutar esto en: Supabase Dashboard > SQL Editor

-- 1. Tabla de usuarios (empresas)
CREATE TABLE IF NOT EXISTS users (
  id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
  email TEXT UNIQUE NOT NULL,
  company_name TEXT NOT NULL,
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
);

-- 2. Tabla de proyectos (análisis por empresa)
CREATE TABLE IF NOT EXISTS projects (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  name TEXT NOT NULL,
  description TEXT,
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW(),
  UNIQUE(user_id, name)
);

-- 3. Tabla de uploads (CSVs cargados)
CREATE TABLE IF NOT EXISTS uploads (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  filename TEXT NOT NULL,
  s3_path TEXT,
  file_size INTEGER,
  uploaded_at TIMESTAMP DEFAULT NOW(),
  processed_at TIMESTAMP,
  status TEXT DEFAULT 'pending' -- pending, processed, failed
);

-- 4. Tabla de backtests (resultados de validación de modelos)
CREATE TABLE IF NOT EXISTS backtests (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  product_code TEXT NOT NULL,
  model_name TEXT NOT NULL, -- "Naive", "Seasonal12", "MA3", "ETS(Holt-Winters)", "RandomForest"
  metrics JSONB, -- { "MAE": 12.5, "RMSE": 15.3, "sMAPE_%": 8.2, "MAPE_safe_%": 9.1, "N": 12 }
  test_months INTEGER DEFAULT 12,
  created_at TIMESTAMP DEFAULT NOW(),
  UNIQUE(project_id, product_code, model_name)
);

-- 5. Tabla de recomendaciones (producción sugerida)
CREATE TABLE IF NOT EXISTS recommendations (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  product_code TEXT NOT NULL,
  abc_class TEXT, -- A, B, C
  forecast FLOAT,
  stock_safety FLOAT,
  stock_actual FLOAT,
  quantity_recommended INTEGER,
  model_used TEXT,
  created_at TIMESTAMP DEFAULT NOW(),
  UNIQUE(project_id, product_code, created_at)
);

-- 6. Tabla de simulaciones (histórico de política)
CREATE TABLE IF NOT EXISTS simulations (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  product_code TEXT NOT NULL,
  model_used TEXT,
  eval_months INTEGER DEFAULT 12,
  fill_rate FLOAT, -- (0-100)
  months_with_stockout INTEGER,
  avg_inventory FLOAT,
  total_units_lost INTEGER,
  created_at TIMESTAMP DEFAULT NOW()
);

-- Índices para performance
CREATE INDEX IF NOT EXISTS idx_projects_user_id ON projects(user_id);
CREATE INDEX IF NOT EXISTS idx_uploads_project_id ON uploads(project_id);
CREATE INDEX IF NOT EXISTS idx_backtests_project_product ON backtests(project_id, product_code);
CREATE INDEX IF NOT EXISTS idx_recommendations_project ON recommendations(project_id);
CREATE INDEX IF NOT EXISTS idx_simulations_project ON simulations(project_id);

-- Row-level security (RLS)
ALTER TABLE users ENABLE ROW LEVEL SECURITY;
ALTER TABLE projects ENABLE ROW LEVEL SECURITY;
ALTER TABLE uploads ENABLE ROW LEVEL SECURITY;
ALTER TABLE backtests ENABLE ROW LEVEL SECURITY;
ALTER TABLE recommendations ENABLE ROW LEVEL SECURITY;
ALTER TABLE simulations ENABLE ROW LEVEL SECURITY;

-- Policies: usuarios solo ven sus propios datos
CREATE POLICY "Users can view own data" ON users
  FOR SELECT USING (auth.uid() = id);

CREATE POLICY "Users can insert own record" ON users
  FOR INSERT WITH CHECK (auth.uid() = id);

CREATE POLICY "Users can view own projects" ON projects
  FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can view own uploads" ON uploads
  FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can view own backtests" ON backtests
  FOR SELECT USING (
    project_id IN (
      SELECT id FROM projects WHERE user_id = auth.uid()
    )
  );

CREATE POLICY "Users can view own recommendations" ON recommendations
  FOR SELECT USING (
    project_id IN (
      SELECT id FROM projects WHERE user_id = auth.uid()
    )
  );

CREATE POLICY "Users can view own simulations" ON simulations
  FOR SELECT USING (
    project_id IN (
      SELECT id FROM projects WHERE user_id = auth.uid()
    )
  );

-- Insert policies
CREATE POLICY "Users can insert own projects" ON projects
  FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can insert own uploads" ON uploads
  FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can insert own backtests" ON backtests
  FOR INSERT WITH CHECK (
    project_id IN (
      SELECT id FROM projects WHERE user_id = auth.uid()
    )
  );

-- Responder a cambios para actualizar timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_users_updated_at
  BEFORE UPDATE ON users
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_projects_updated_at
  BEFORE UPDATE ON projects
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at_column();
