"""
cluster_tags.py — cluster vault tags by co-occurrence using Louvain community detection.

Usage:
    python3 cluster_tags.py --tag-db vital_articles.db --vault-db vault.db
    python3 cluster_tags.py --tag-db vital_articles.db --vault-db vault.db --unit conversation
    python3 cluster_tags.py --tag-db vital_articles.db --vault-db vault.db --top-k 15 --max-depth 4

Assigns top-k tags to each chunk via embedding similarity, builds a weighted
co-occurrence graph, and runs recursive Louvain clustering. Results stored in
the vault DB alongside the chunks they were derived from.
"""

from __future__ import annotations

import argparse
import itertools
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_TOOLS_DIR = str(Path(__file__).resolve().parent)
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)

from embed_tags import load_tag_matrix, _load_model
import numpy as np
import networkx as nx
import community as community_louvain


def _ensure_schema(con: sqlite3.Connection):
    """Create clustering tables if they don't exist."""
    con.execute("""
        CREATE TABLE IF NOT EXISTS chunk_tags (
            chunk_id   INTEGER NOT NULL REFERENCES chunks(id),
            tag_title  TEXT NOT NULL,
            rank       INTEGER NOT NULL,
            score      REAL NOT NULL,
            PRIMARY KEY (chunk_id, tag_title)
        )
    """)
    con.execute("CREATE INDEX IF NOT EXISTS idx_ct_tag ON chunk_tags(tag_title)")

    con.execute("""
        CREATE TABLE IF NOT EXISTS cluster_versions (
            version_id    INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at    TEXT NOT NULL,
            unit          TEXT NOT NULL,
            top_k         INTEGER NOT NULL,
            max_depth     INTEGER NOT NULL,
            num_clusters  INTEGER NOT NULL
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS clusters (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            version_id    INTEGER NOT NULL REFERENCES cluster_versions(version_id),
            cluster_id    TEXT NOT NULL,
            depth         INTEGER NOT NULL,
            parent_id     TEXT,
            name          TEXT,
            description   TEXT,
            tag_count     INTEGER NOT NULL,
            UNIQUE (version_id, cluster_id)
        )
    """)
    con.execute("CREATE INDEX IF NOT EXISTS idx_cl_depth ON clusters(version_id, depth)")

    con.execute("""
        CREATE TABLE IF NOT EXISTS cluster_tags (
            version_id    INTEGER NOT NULL,
            cluster_id    TEXT NOT NULL,
            tag_title     TEXT NOT NULL,
            weight        REAL NOT NULL,
            PRIMARY KEY (version_id, cluster_id, tag_title)
        )
    """)
    con.execute("CREATE INDEX IF NOT EXISTS idx_clt_tag ON cluster_tags(tag_title)")


def assign_tags(vault_con: sqlite3.Connection, tag_db: Path, top_k: int, min_score: float = 0.0) -> int:
    """Assign top-k tags to each chunk by embedding similarity. Returns row count."""
    matrix, meta = load_tag_matrix(tag_db)
    if len(meta) == 0:
        raise SystemExit("Error: no tag embeddings found — run embed_tags.py --embed first")

    rows = vault_con.execute(
        "SELECT id, embedding FROM chunks ORDER BY id"
    ).fetchall()

    if not rows:
        raise SystemExit("Error: no chunks found in vault DB")

    print(f"[cluster] assigning top-{top_k} tags to {len(rows)} chunks…", file=sys.stderr)

    chunk_ids = [r[0] for r in rows]
    chunk_matrix = np.stack([np.frombuffer(r[1], dtype=np.float32) for r in rows])

    # Batch similarity: (N_chunks, 384) @ (384, N_tags) -> (N_chunks, N_tags)
    scores = chunk_matrix @ matrix.T

    # Truncate and repopulate
    vault_con.execute("DELETE FROM chunk_tags")

    batch = []
    for i, chunk_id in enumerate(chunk_ids):
        top_indices = np.argsort(scores[i])[::-1][:top_k]
        for rank, idx in enumerate(top_indices, 1):
            s = float(scores[i][idx])
            if s < min_score:
                break
            batch.append((chunk_id, meta[idx][0], rank, s))

    vault_con.executemany(
        "INSERT INTO chunk_tags (chunk_id, tag_title, rank, score) VALUES (?, ?, ?, ?)",
        batch,
    )
    vault_con.commit()

    print(f"[cluster] assigned {len(batch)} tag associations", file=sys.stderr)
    return len(batch)


def build_graph(vault_con: sqlite3.Connection, unit: str) -> nx.Graph:
    """Build a weighted tag co-occurrence graph from chunk_tags."""
    G = nx.Graph()

    if unit == "chunk":
        rows = vault_con.execute(
            "SELECT chunk_id, tag_title FROM chunk_tags ORDER BY chunk_id"
        ).fetchall()

        for chunk_id, group in itertools.groupby(rows, key=lambda r: r[0]):
            tags = [r[1] for r in group]
            for t1, t2 in itertools.combinations(sorted(tags), 2):
                if G.has_edge(t1, t2):
                    G[t1][t2]["weight"] += 1
                else:
                    G.add_edge(t1, t2, weight=1)

    elif unit == "conversation":
        rows = vault_con.execute("""
            SELECT c.file_path, ct.tag_title
            FROM chunk_tags ct
            JOIN chunks c ON c.id = ct.chunk_id
            ORDER BY c.file_path
        """).fetchall()

        for file_path, group in itertools.groupby(rows, key=lambda r: r[0]):
            tags = list(set(r[1] for r in group))
            for t1, t2 in itertools.combinations(sorted(tags), 2):
                if G.has_edge(t1, t2):
                    G[t1][t2]["weight"] += 1
                else:
                    G.add_edge(t1, t2, weight=1)

    # Add isolated nodes
    all_tags = vault_con.execute("SELECT DISTINCT tag_title FROM chunk_tags").fetchall()
    for (tag,) in all_tags:
        if tag not in G:
            G.add_node(tag)

    print(f"[cluster] graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges",
          file=sys.stderr)
    return G


def cluster_graph(G: nx.Graph, max_depth: int, min_cluster_size: int = 5) -> list[dict]:
    """Recursive Louvain clustering. Returns flat list of cluster dicts."""
    all_clusters: list[dict] = []
    dropped = 0

    def _recurse(subgraph: nx.Graph, parent_id: str | None, depth: int):
        nonlocal dropped
        if depth >= max_depth or subgraph.number_of_nodes() < 2:
            return

        partition = community_louvain.best_partition(subgraph, weight="weight")

        communities: dict[int, list[str]] = {}
        for node, comm_id in partition.items():
            communities.setdefault(comm_id, []).append(node)

        for comm_idx, (comm_id, nodes) in enumerate(sorted(communities.items())):
            if len(nodes) < min_cluster_size:
                dropped += 1
                continue

            cluster_id = str(comm_idx) if parent_id is None else f"{parent_id}.{comm_idx}"

            # Compute per-tag weights (sum of edge weights in subgraph)
            sub = subgraph.subgraph(nodes)
            tag_weights = {}
            for node in nodes:
                w = sum(d.get("weight", 0) for _, _, d in sub.edges(node, data=True))
                tag_weights[node] = w

            all_clusters.append({
                "cluster_id": cluster_id,
                "depth": depth,
                "parent_id": parent_id,
                "tags": tag_weights,
            })

            _recurse(sub, cluster_id, depth + 1)

    _recurse(G, None, 0)

    print(f"[cluster] {len(all_clusters)} clusters across {max_depth} depth levels "
          f"({dropped} dropped below min size {min_cluster_size})",
          file=sys.stderr)
    return all_clusters


def store_clusters(
    vault_con: sqlite3.Connection,
    clusters: list[dict],
    unit: str,
    top_k: int,
    max_depth: int,
) -> int:
    """Store clustering results. Returns version_id."""
    vault_con.execute(
        """INSERT INTO cluster_versions (created_at, unit, top_k, max_depth, num_clusters)
           VALUES (?, ?, ?, ?, ?)""",
        (datetime.now(timezone.utc).isoformat(), unit, top_k, max_depth, len(clusters)),
    )
    version_id = vault_con.execute("SELECT last_insert_rowid()").fetchone()[0]

    for c in clusters:
        vault_con.execute(
            """INSERT INTO clusters
               (version_id, cluster_id, depth, parent_id, tag_count)
               VALUES (?, ?, ?, ?, ?)""",
            (version_id, c["cluster_id"], c["depth"], c["parent_id"], len(c["tags"])),
        )
        vault_con.executemany(
            """INSERT INTO cluster_tags (version_id, cluster_id, tag_title, weight)
               VALUES (?, ?, ?, ?)""",
            [(version_id, c["cluster_id"], tag, weight) for tag, weight in c["tags"].items()],
        )

    vault_con.commit()
    return version_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cluster vault tags by co-occurrence using Louvain community detection"
    )
    parser.add_argument("--tag-db", required=True, type=Path,
                        help="Path to vital_articles.db (shared tag embeddings)")
    parser.add_argument("--vault-db", required=True, type=Path,
                        help="Path to vault.db (corpus chunks, where results are stored)")
    parser.add_argument("--unit", choices=["chunk", "conversation"], default="chunk",
                        help="Clustering unit (default: chunk)")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Tags per unit (default: 10)")
    parser.add_argument("--min-score", type=float, default=0.25,
                        help="Minimum cosine similarity for tag assignment (default: 0.25)")
    parser.add_argument("--min-cluster-size", type=int, default=5,
                        help="Drop clusters smaller than this (default: 5)")
    parser.add_argument("--max-depth", type=int, default=3,
                        help="Louvain recursion depth (default: 3)")
    args = parser.parse_args()

    tag_db = args.tag_db.resolve()
    vault_db = args.vault_db.resolve()

    if not tag_db.exists():
        raise SystemExit(f"Error: '{tag_db}' not found")
    if not vault_db.exists():
        raise SystemExit(f"Error: '{vault_db}' not found")

    t0 = time.time()

    vault_con = sqlite3.connect(vault_db)
    _ensure_schema(vault_con)

    assign_tags(vault_con, tag_db, args.top_k, args.min_score)
    G = build_graph(vault_con, args.unit)
    clusters = cluster_graph(G, args.max_depth, args.min_cluster_size)
    version_id = store_clusters(vault_con, clusters, args.unit, args.top_k, args.max_depth)

    vault_con.close()

    elapsed = time.time() - t0
    print(f"[cluster] done in {elapsed:.1f}s — version {version_id}", file=sys.stderr)
