# RAG + InternVL3.5-8B POC (Python 3.11, uv)

Ce projet est un petit POC d’une pipeline RAG sur des PDF contenant du texte et des images :
- Ingestion des PDF avec extraction texte par page et rendu image des pages (PyMuPDF)
- Indexation dans Postgres avec pgvector (embeddings denses) + FTS (tsvector) pour une recherche hybride (cosine + ts_rank ~ BM25-like)
- Récupération des meilleurs contextes et, en option, appel du modèle multimodal InternVL3.5-8B (Hugging Face) pour générer une réponse basée sur les contextes et les images de pages correspondantes

Important
- Le modèle InternVL3.5-8B nécessite du matériel GPU conséquent (voir la page Hugging Face). Ce POC prépare l’interface, mais l’inférence locale est optionnelle via un flag `--run-model`. Sans GPU adapté, utilisez `--run-model=false` pour simplement valider la chaîne RAG et visualiser les contextes récupérés.
- Python 3.11 est requis (préférence utilisateur).

Prérequis
- Python 3.11
- uv (https://docs.astral.sh/uv/) installé
- Postgres ≥ 15 avec l’extension `pgvector` installée dans la base cible
- Accès à la base de données (DSN)

Installation
1) Créer et activer un environnement avec uv

   - Créer l’environnement (Python 3.11) :
     uv venv -p 3.11

   - Installer les dépendances :
     uv sync

2) Configurer la base de données

   - Créez une base et activez les extensions nécessaires (en tant que propriétaire de la base) :
     CREATE EXTENSION IF NOT EXISTS vector;

   - Initialiser le schéma de ce POC (depuis le projet) :
     uv run rag-internvl init-db --dsn "postgresql://user:pass@host:5432/dbname"

3) Préparer des PDF

   - Déposez vos PDF dans data/pdfs

4) Ingestion

   - Ingestion (extraction texte + rendu images, embeddings, indexation DB) :
     uv run rag-internvl ingest --dsn "postgresql://user:pass@host:5432/dbname" --pdf-dir data/pdfs

5) Requête RAG (hybride)

   - Sans inférence du modèle (dry-run), affiche les contextes :
     uv run rag-internvl query --dsn "postgresql://user:pass@host:5432/dbname" --question "Votre question en français" --top-k 4 --alpha 0.5

   - Avec tentative d’inférence InternVL3.5-8B (nécessite GPU adapté) :
     uv run rag-internvl query --dsn "postgresql://user:pass@host:5432/dbname" --question "Votre question" --top-k 4 --alpha 0.5 --run-model

Notes sur la recherche hybride
- FTS : `ts_rank_cd(content_tsv, plainto_tsquery('french', :query))`
- Densité : similarité cosinus via `pgvector` (on utilise 1 - (embedding <=> query_vec) pour obtenir un score croissant)
- Score hybride = alpha * dense + (1 - alpha) * fts

Embeddings
- Modèle par défaut : BAAI/bge-m3 (dimension 1024), multilingue, robuste pour le dense.
- La table utilise `vector(1024)`. Si vous changez de modèle d’embedding, adaptez la dimension.

Modèle InternVL3.5-8B
- Référence : https://huggingface.co/OpenGVLab/InternVL3_5-2B
- Version transformers minimale : >= 4.52.1 (d’après la page du modèle)
- Chargement type :

  ```python path=null start=null
  from transformers import AutoTokenizer, AutoModel
  model_id = "OpenGVLab/InternVL3_5-2B"
  tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
  model = AutoModel.from_pretrained(
      model_id,
      torch_dtype=torch.bfloat16,
      low_cpu_mem_usage=True,
      use_flash_attn=True,
      trust_remote_code=True,
  ).eval().cuda()
  ```

- L’API exacte d’inférence multimodale dépend du code personnalisé du modèle (`trust_remote_code=True`). La classe fournie ici tente d’appeler une méthode `chat` courante ; adaptez au besoin selon l’implémentation actuelle du repo Hugging Face.

ENV/Secrets
- Vous pouvez définir `DATABASE_URL` dans un fichier .env (voir .env.example) au lieu de fournir `--dsn`.
- Ne mettez pas vos secrets en clair dans le shell ; chargez-les via variables d’environnement.

Licences
- Voir les licences des modèles tiers (Hugging Face), pgvector, etc. Ce dépôt est fourni à titre de POC.

