-- ============================================================
-- DRA MLOps  –  MySQL one-time setup script
-- Run as MySQL root:  mysql -u root -p < mysql_setup.sql
-- ============================================================

-- 1. Create the database
CREATE DATABASE IF NOT EXISTS dra_mlops
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;

-- 2. Create a dedicated application user
--    Change the password if deploying to production.
CREATE USER IF NOT EXISTS 'dra_user'@'localhost'
    IDENTIFIED BY 'dra_password_2026';

-- 3. Grant all privileges on dra_mlops only
GRANT ALL PRIVILEGES ON dra_mlops.* TO 'dra_user'@'localhost';
FLUSH PRIVILEGES;

-- 4. Switch to the new database and create tables
USE dra_mlops;

CREATE TABLE IF NOT EXISTS ingested_files (
    id          INT AUTO_INCREMENT PRIMARY KEY,
    filepath    VARCHAR(1024)   NOT NULL,
    ingested_at DATETIME        NOT NULL,
    INDEX idx_filepath (filepath(255))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS model_scores (
    id          INT AUTO_INCREMENT PRIMARY KEY,
    f1_score    DOUBLE          NOT NULL,
    scored_at   DATETIME        NOT NULL,
    model_path  VARCHAR(512),
    data_path   VARCHAR(512),
    INDEX idx_scored_at (scored_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS diagnostics_snapshots (
    id               INT AUTO_INCREMENT PRIMARY KEY,
    recorded_at      DATETIME        NOT NULL,
    ingestion_time   DOUBLE,
    training_time    DOUBLE,
    avg_missing_pct  DOUBLE,
    summary_stats    LONGTEXT,
    missing_data     LONGTEXT,
    evidently_drift  LONGTEXT,
    INDEX idx_recorded_at (recorded_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 5. Verify
SHOW TABLES;
SELECT 'MySQL setup complete.' AS status;
