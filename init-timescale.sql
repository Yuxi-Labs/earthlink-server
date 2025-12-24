-- Initialize TimescaleDB and PostGIS extensions
CREATE EXTENSION IF NOT EXISTS timescaledb;
CREATE EXTENSION IF NOT EXISTS postgis;

-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE earthlink TO earthlink;
