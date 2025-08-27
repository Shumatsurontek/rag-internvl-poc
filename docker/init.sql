-- Exécuté seulement lors de la première initialisation du volume de données
-- Active l’extension pgvector dans la base par défaut ($POSTGRES_DB)
CREATE EXTENSION IF NOT EXISTS vector;
