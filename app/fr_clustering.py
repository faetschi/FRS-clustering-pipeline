"""
Functional Requirements Clustering Pipeline

This script:
1. Loads functional requirements (FRs) from a file (JSON or plain text).
2. Embeds them using SentenceTransformer.
3. Clusters embeddings using HDBSCAN.
4. Stores results in a Qdrant vector database.
5. Generates a 2D visualization (UMAP or t-SNE).
6. Serves results via a built-in HTTP server with JSON/HTML endpoints.

Environment variables:
- LOG_LEVEL: Logging verbosity (default: INFO)
- FR_FILE: Path to FR input file (default: ./functional_requirements.txt)
- QDRANT_HOST: Qdrant server host (default: localhost)
- QDRANT_HTTP_PORT: Qdrant HTTP port (default: 6333)
"""


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
from qdrant_client.models import (
    PointStruct,
    VectorParams,
    Distance,
    Filter,
    FieldCondition,
    MatchValue,
)
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

# =============================================================================
# 1. CONFIGURATION & LOGGING SETUP
# =============================================================================

# Configure logging to stdout for container compatibility.
# Verbosity controlled via LOG_LEVEL environment variable.
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    stream=sys.stdout,
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# Qdrant connection settings
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_HTTP_PORT", "6333"))

# Output directory for generated artifacts
OUTPUT_DIR = "/app/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# 2. FUNCTIONAL REQUIREMENTS LOADING
# =============================================================================

def load_functional_requirements(path: str | None = None) -> list[str]:
    """
    Load functional requirements from a file.

    Supports:
        - JSON files containing a list of strings.
        - Plain text files with one requirement per line.

    Args:
        path (str | None): Path to the requirements file.
            If None, uses the FR_FILE environment variable.
            Defaults to 'functional_requirements.txt' in the script directory.

    Returns:
        list[str]: Non-empty, stripped requirement strings.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is empty or JSON is malformed/not a list.
    """
    if path is None:
        path = os.getenv(
            "FR_FILE",
            os.path.join(os.path.dirname(__file__), "functional_requirements.txt")
        )

    if not os.path.exists(path):
        raise FileNotFoundError(f"Functional requirements file not found: {path}")

    try:
        if path.lower().endswith('.json'):
            with open(path, 'r', encoding='utf-8') as fh:
                data = json.load(fh)
            if not isinstance(data, list):
                raise ValueError("JSON FR file must contain a top-level list of strings.")
            return [str(item).strip() for item in data if str(item).strip()]
        else:
            with open(path, 'r', encoding='utf-8') as fh:
                lines = [line.strip() for line in fh if line.strip()]
            if not lines:
                raise ValueError(f"Functional requirements file is empty: {path}")
            return lines
    except Exception as e:
        logger.error(f"Failed to load functional requirements from {path}: {e}")
        raise


# Load requirements at module startup (can be overridden via CLI)
FUNCTIONAL_REQUIREMENTS = load_functional_requirements()


# =============================================================================
# 3. HTTP SERVER FOR SERVING RESULTS
# =============================================================================

class FRHTTPRequestHandler(SimpleHTTPRequestHandler):
    """HTTP request handler that serves static files and dynamic embeddings."""

    def do_GET(self):
        """Route GET requests: static files by default, /embeddings dynamically."""
        parsed = urlparse(self.path)
        if parsed.path == "/embeddings":
            self.handle_embeddings(parsed)
        else:
            super().do_GET()

    def handle_embeddings(self, parsed):
        """
        Serve a JSON snippet of stored embeddings from Qdrant.

        Query parameters:
            - limit (int): Number of points to return (default: 5).
            - vector_len (int): Number of vector dimensions to include (default: 50).

        Response includes:
            - id: Point ID
            - text: Original requirement text
            - cluster: Assigned cluster label
            - vector_sample: First N dimensions of the embedding
        """
        try:
            query = parse_qs(parsed.query)
            limit = int(query.get("limit", [5])[0])
            vector_len = int(query.get("vector_len", [50])[0])

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
            logger.error(f"Error handling /embeddings request: {e}")
            self.send_response(500)
            self.end_headers()
            self.wfile.write(f"Error fetching embeddings: {e}".encode("utf-8"))


# =============================================================================
# 4. CLUSTERING PIPELINE
# =============================================================================

def run_clustering_pipeline(
    frs: list[str] | None = None,
    projection: str = 'umap',
    perplexity: float = 30.0,
    cluster_distance: float = 0.32
) -> None:
    """
    Execute the full clustering pipeline using Agglomerative Clustering in embedding space.
    """
    logger.info("🚀 Starting FR Clustering Pipeline...")

    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    frs_list = frs if frs is not None else FUNCTIONAL_REQUIREMENTS

    # Step 1: Generate embeddings (use a stronger model if desired)
    logger.info("Embedding functional requirements using SentenceTransformer...")
    model = SentenceTransformer('all-MiniLM-L6-v2') # all-mpnet-base-v2
    vectors = model.encode(frs_list, normalize_embeddings=True).astype(np.float32)

    # Step 2: Cluster in FULL 384D SPACE using COSINE DISTANCE
    logger.info(f"Clustering in embedding space (distance_threshold={cluster_distance})...")
    from sklearn.metrics import pairwise_distances
    from sklearn.cluster import AgglomerativeClustering

    # Compute cosine distance matrix (range: 0 to 2)
    cosine_distances = pairwise_distances(vectors, metric='cosine')

    clusterer = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=cluster_distance,
        linkage='average',
        metric='precomputed'
    )
    cluster_labels = clusterer.fit_predict(cosine_distances)

    n_clusters = len(np.unique(cluster_labels))
    logger.info(f"Formed {n_clusters} clusters. No noise points.")

    # Step 3: Generate 2D projection JUST FOR VISUALIZATION
    logger.info(f"Generating 2D projection using {projection.upper()}...")
    if projection == 'tsne':
        from sklearn.manifold import TSNE
        embeddings_2d = TSNE(
            n_components=2,
            random_state=42,
            perplexity=perplexity,
            metric='cosine',  # important for text
            init='pca',
            learning_rate='auto'
        ).fit_transform(vectors)
    else:  # UMAP
        import umap
        umap_reducer = umap.UMAP(
            n_components=2,
            random_state=42,
            n_neighbors=15,
            min_dist=0.1,
            metric='cosine'
        )
        embeddings_2d = umap_reducer.fit_transform(vectors)

    # Step 4: Store in Qdrant
    logger.info("Storing results in Qdrant vector database...")
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    collection_name = "earlybird_fr"
    vectors_config = VectorParams(size=384, distance=Distance.COSINE)

    try:
        if client.collection_exists(collection_name=collection_name):
            client.delete_collection(collection_name=collection_name)
    except AttributeError:
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
    client.upsert(collection_name=collection_name, points=points)

    # Step 5: Create visualization DataFrame
    plot_labels = [f"Cluster {label}" for label in cluster_labels]  # no noise!

    df = pd.DataFrame({
        "x": embeddings_2d[:, 0],
        "y": embeddings_2d[:, 1],
        "cluster_name": plot_labels,
        "text": [f"FR-{i+1}: {fr[:60]}..." for i, fr in enumerate(frs_list)]
    })

    fig = px.scatter(
        df,
        x="x",
        y="y",
        color="cluster_name",
        hover_data=["text"],
        title=f"Requirement Clusters ({n_clusters} groups)",
        color_discrete_sequence=px.colors.qualitative.Plotly,
        width=900,
        height=600
    )
    fig.update_traces(marker=dict(size=10, opacity=0.85))
    fig.update_layout(legend_title_text="Cluster")

    output_html = os.path.join(OUTPUT_DIR, "clusters.html")
    fig.write_html(output_html, include_plotlyjs='cdn')
    logger.info(f"Interactive plot saved to {output_html}")

    # Step 6: Save cluster summary
    cluster_summary = {}
    for label in sorted(set(cluster_labels)):
        members = [{"id": i, "text": frs_list[i]} for i, l in enumerate(cluster_labels) if l == label]
        cluster_summary[f"Cluster {label}"] = members

    output_json = os.path.join(OUTPUT_DIR, "clusters.json")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(cluster_summary, f, indent=2)
    logger.info(f"Cluster summary saved to {output_json}")

    logger.info("✅ Clustering pipeline completed successfully.")

# =============================================================================
# 5. HTTP SERVER LAUNCHER
# =============================================================================

def start_http_server():
    """Start a simple HTTP server to serve output files on port 8000."""
    os.chdir(OUTPUT_DIR)
    server = HTTPServer(("0.0.0.0", 8000), FRHTTPRequestHandler)
    logger.info("🚀 Serving results at http://localhost:8000")
    logger.info("📌 Embeddings endpoint: http://localhost:8000/embeddings?limit=5")
    server.serve_forever()


# =============================================================================
# 6. MAIN EXECUTION BLOCK
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Functional Requirements Clustering Pipeline")
    parser.add_argument(
        '--print-fr',
        action='store_true',
        help='Print loaded functional requirements and exit.'
    )
    parser.add_argument(
        '--fr-file',
        type=str,
        help='Path to functional requirements file (overrides FR_FILE env var).'
    )
    parser.add_argument(
        '--projection',
        choices=['umap', 'tsne'],
        default='tsne',
        help='2D projection method for visualization (default: tsne).'
    )
    parser.add_argument(
        '--perplexity',
        type=float,
        default=30.0,
        help='Perplexity parameter for t-SNE (ignored if using UMAP).'
    )
    parser.add_argument(
        '--cluster-distance',
        type=float,
        default=0.65,
        help='Distance threshold for Agglomerative Clustering in UMAP space. '
            'Lower values create more clusters, higher values merge clusters. '
            'Default: 0.65'
    )
    args = parser.parse_args()

    # Optionally override FR source
    if args.fr_file:
        try:
            FUNCTIONAL_REQUIREMENTS = load_functional_requirements(args.fr_file)
        except Exception as e:
            logger.error(f"Failed to load FR file: {e}")
            sys.exit(2)

    # Debug: print requirements and exit
    if args.print_fr:
        logger.info("Loaded Functional Requirements:")
        for i, fr in enumerate(FUNCTIONAL_REQUIREMENTS, start=1):
            print(f"{i}. {fr}")
        sys.exit(0)

    # Run the full pipeline
    run_clustering_pipeline(
        projection=args.projection,
        perplexity=args.perplexity,
        cluster_distance=args.cluster_distance
    )

    # Start HTTP server in background thread
    server_thread = threading.Thread(target=start_http_server, daemon=True)
    server_thread.start()
    logger.info("HTTP server started in background.")

    # Keep main thread alive to prevent container exit
    try:
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        logger.info("Received interrupt signal. Shutting down gracefully.")