-- ========================================
-- CREAR USUARIO MASTER_ADMIN
-- ========================================

-- 1. Verificar que paulmaste@gmail.com exista en auth.users
SELECT id FROM auth.users WHERE email = 'paulmaste@gmail.com';

-- 2. Crear entry en tabla users como master_admin
-- Reemplaza el UUID con el que obtuviste arriba
INSERT INTO users (id, email, full_name, is_master_admin, created_at, updated_at) 
VALUES (
    'REEMPLAZA_CON_UUID_DE_ARRIBA',  -- Reemplaza con el ID real de auth.users
    'paulmaste@gmail.com',
    'Paul Master Admin',
    TRUE,
    NOW(),
    NOW()
) ON CONFLICT (id) DO UPDATE SET 
    is_master_admin = TRUE,
    updated_at = NOW();

-- 3. Verificar creación
SELECT id, email, is_master_admin FROM users WHERE email = 'paulmaste@gmail.com';
