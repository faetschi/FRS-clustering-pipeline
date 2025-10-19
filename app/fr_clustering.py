import os
import json
import hdbscan
import numpy as np
import plotly.express as px
import pandas as pd
import warnings
import argparse
import sys
import logging
import threading
import time

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance, Filter, FieldCondition, MatchValue
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

# ==============================
# CONFIG FROM ENV
# ==============================
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_HTTP_PORT", "6333"))
OUTPUT_DIR = "/app/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================
# 1. FUNCTIONAL REQUIREMENTS
# ==============================
# Configure logging to stdout so container logs capture it. Use LOG_LEVEL env var to control verbosity.
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(stream=sys.stdout,
                    level=getattr(logging, LOG_LEVEL, logging.INFO),
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

def load_functional_requirements(path=None):
    """Load functional requirements from a file.

    Supports JSON files that contain a list of strings or plain text files with one requirement per line.
    The path can be overridden by the FR_FILE environment variable. Returns a list of strings.
    """
    if path is None:
        path = os.getenv("FR_FILE", os.path.join(os.path.dirname(__file__), "functional_requirements.txt"))
    if not os.path.exists(path):
        raise FileNotFoundError(f"Functional requirements file not found: {path}")
    try:
        if path.lower().endswith('.json'):
            with open(path, 'r', encoding='utf-8') as fh:
                data = json.load(fh)
            if not isinstance(data, list):
                raise ValueError("JSON FR file must contain a list of strings")
            return [str(x) for x in data]
        else:
            with open(path, 'r', encoding='utf-8') as fh:
                lines = [line.strip() for line in fh if line.strip()]
            if not lines:
                # Graceful, descriptive error for empty FR files
                raise ValueError(f"Functional requirements file is empty: {path}")
            return lines
    except Exception:
        # Re-raise with context
        raise

# Load functional requirements from external file
FUNCTIONAL_REQUIREMENTS = load_functional_requirements()

# =========================================
# 2. HTTP SERVER FOR RESULTS AND EMBEDDINGS
# =========================================

class FRHTTPRequestHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/embeddings":
            self.handle_embeddings(parsed)
        else:
            super().do_GET()
                
    def handle_embeddings(self, parsed):
        try:
            query = parse_qs(parsed.query)
            limit = int(query.get("limit", [5])[0])
            vector_len = int(query.get("vector_len", [50])[0])  # Default: 50 dimensions

            client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
            points, _ = client.scroll(
                collection_name="earlybird_fr",
                limit=limit,
                with_vectors=True
            )

            snippet = []
            for pt in points:
                vec = pt.vector
                snippet.append({
                    "id": pt.id,
                    "text": pt.payload.get("text", ""),
                    "cluster": pt.payload.get("cluster", -1),
                    "vector_sample": vec[:vector_len] if isinstance(vec, list) else []
                })

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(snippet, indent=2).encode("utf-8"))

        except Exception as e:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(f"Error fetching embeddings: {e}".encode("utf-8"))

# ==============================
# 3. FR CLUSTERING PIPELINE
# ==============================

def run_clustering_pipeline(frs=None, projection='umap', perplexity=30):
    logger.info("🚀 Starting FR Clustering Pipeline...")
    
    # Suppress known deprecation/future warnings from sklearn and qdrant helper
    warnings.filterwarnings("ignore", category=FutureWarning, message=".*force_all_finite.*")
    warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*recreate_collection.*")

    # Embed
    frs_list = frs if frs is not None else FUNCTIONAL_REQUIREMENTS
    model = SentenceTransformer('all-MiniLM-L6-v2')
    vectors = model.encode(frs_list).astype(np.float32)

    # Cluster
    clusterer = hdbscan.HDBSCAN(min_cluster_size=2, metric='euclidean', cluster_selection_method='leaf')
    cluster_labels = clusterer.fit_predict(vectors)

    # Store in Qdrant
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    # Newer qdrant client: check if collection exists, delete or create
    collection_name = "earlybird_fr"
    vectors_config = VectorParams(size=384, distance=Distance.COSINE)
    try:
        if client.collection_exists(collection_name=collection_name):
            # delete then create to ensure a fresh collection
            client.delete_collection(collection_name=collection_name)
        client.create_collection(collection_name=collection_name, vectors_config=vectors_config)
    except Exception:
        # Fallback for older client versions that may not have collection_exists
        # Attempt to delete and recreate; if delete fails, try create directly
        try:
            client.delete_collection(collection_name=collection_name)
        except Exception:
            pass
        client.create_collection(collection_name=collection_name, vectors_config=vectors_config)
    points = [
        PointStruct(
            id=i,
            vector=vectors[i].tolist(),
            payload={"text": frs_list[i], "cluster": int(cluster_labels[i])}
        )
        for i in range(len(frs_list))
    ]
    client.upsert(collection_name="earlybird_fr", points=points)

    # 2D Visualization — choose projection method
    if projection == 'tsne':
        # t-SNE focuses on local neighborhoods; slower but sometimes clearer for small datasets
        from sklearn.manifold import TSNE
        embeddings_2d = TSNE(n_components=2, random_state=42, perplexity=perplexity).fit_transform(vectors)
    else:
        import umap
        # UMAP warns when n_jobs is overridden by random_state; set n_jobs=1 explicitly
        reducer = umap.UMAP(n_components=2, random_state=42, n_jobs=1)
        embeddings_2d = reducer.fit_transform(vectors)
    # Remap cluster labels (which may include -1 for noise) to sequential cluster IDs
    unique_labels = sorted(set(cluster_labels))
    label_map = {old: idx for idx, old in enumerate(unique_labels)}
    mapped_labels = [label_map[l] for l in cluster_labels]

    df = pd.DataFrame({
        "x": embeddings_2d[:, 0],
        "y": embeddings_2d[:, 1],
        "cluster": mapped_labels,
        "text": [f"FR-{i+1}: {fr[:60]}..." for i, fr in enumerate(frs_list)]
    })
    # Use discrete cluster names so legend shows 'Cluster N' with color mapping
    df['cluster_name'] = [f"Cluster {c}" for c in df['cluster']]
    fig = px.scatter(
        df,
        x="x",
        y="y",
        color="cluster_name",
        hover_data=["text"],
        title="Requirement Clusters (t-SNE Visualization)",
        color_discrete_sequence=px.colors.qualitative.Plotly,
        width=900,
        height=600
    )
    # Make markers a bit larger and slightly transparent for readability
    fig.update_traces(marker=dict(size=10, opacity=0.85))
    # Axis titles: horizontal = Dimension 1, vertical = Dimension 2
    fig.update_xaxes(title_text="Dimension 1 (Reduced)")
    fig.update_yaxes(title_text="Dimension 2 (Reduced)")
    fig.update_layout(legend_title_text="Cluster")
    # Write HTML using Plotly CDN to avoid embedding the full plotly.js bundle
    # This makes the file much smaller and speeds up loading in the browser.
    fig.write_html(f"{OUTPUT_DIR}/clusters.html", include_plotlyjs='cdn', full_html=True, auto_open=False)

    # Save cluster summary as JSON using user-friendly cluster names
    cluster_summary = {}
    # Build a reverse map from mapped id -> original label values
    reverse_map = {new: old for old, new in label_map.items()}
    for new_cid in sorted(set(mapped_labels)):
        members = [
            {"id": i, "text": frs_list[i]}
            for i, ml in enumerate(mapped_labels) if ml == new_cid
        ]
        cluster_name = f"Cluster {new_cid}"
        cluster_summary[cluster_name] = members
    with open(f"{OUTPUT_DIR}/clusters.json", "w") as f:
        json.dump(cluster_summary, f, indent=2)

    logger.info(f"✅ Pipeline complete. Results saved to {OUTPUT_DIR}")

def start_http_server():
    os.chdir(OUTPUT_DIR)
    server = HTTPServer(("0.0.0.0", 8000), FRHTTPRequestHandler)
    logger.info("🚀 Serving results at http://localhost:8000")
    logger.info("📌 Embeddings snippet available at http://localhost:8000/embeddings?limit=5")
    server.serve_forever()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FR clustering pipeline")
    parser.add_argument('--print-fr', action='store_true', help='Print loaded functional requirements and exit')
    parser.add_argument('--fr-file', type=str, help='Path to functional requirements file (overrides FR_FILE env var)')
    parser.add_argument('--projection', choices=['umap', 'tsne'], default='tsne', help='2D projection method')
    parser.add_argument('--perplexity', type=float, default=30.0, help='Perplexity for t-SNE (if chosen)')
    args = parser.parse_args()

    # If user supplied a custom FR file, reload from it
    if args.fr_file:
        try:
            FUNCTIONAL_REQUIREMENTS = load_functional_requirements(args.fr_file)
        except Exception as e:
            print(f"Error loading functional requirements: {e}", file=sys.stderr)
            sys.exit(2)

    if args.print_fr:
        print("Loaded Functional Requirements:")
        for i, fr in enumerate(FUNCTIONAL_REQUIREMENTS, start=1):
            print(f"{i}. {fr}")
        sys.exit(0)

    # Run clustering once
    run_clustering_pipeline(projection=args.projection, perplexity=args.perplexity)
    
    # Start HTTP server in background
    server_thread = threading.Thread(target=start_http_server, daemon=True)
    server_thread.start()
    
    # Keep container alive
    try:
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        logger.info("Shutting down")
