# Tag Clustering Pipeline Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a two-stage pipeline that clusters Wikipedia Vital Articles tags by co-occurrence across vault chunks, then names the clusters with an LLM.

**Architecture:** `cluster_tags.py` assigns top-k tags per chunk via embedding similarity, builds a co-occurrence graph, and runs recursive Louvain clustering — all stored in the vault's SQLite DB. `name_clusters.py` reads unnamed clusters, gathers representative context, and spawns Claude CLI (Haiku) to name each one.

**Tech Stack:** Python 3, SQLite, numpy, sentence-transformers, networkx, python-louvain, Claude CLI

**Spec:** `docs/superpowers/specs/2026-03-16-tag-clustering-pipeline-design.md`

---

## File Structure

| File | Responsibility |
|------|---------------|
| `cluster_tags.py` (create) | Tag assignment, co-occurrence graph, Louvain clustering, SQLite storage |
| `name_clusters.py` (create) | Context gathering, Claude CLI cluster naming |
| `embed_tags.py` (read only) | Import `load_tag_matrix`, `_load_model` for tag embeddings |

No modifications to existing files. Both new scripts follow existing patterns from `embed_vault.py` and `tag_text.py`: argparse, `type=Path`, `.resolve()`, stderr logging.

---

## Chunk 1: `cluster_tags.py`

### Task 1: Schema and skeleton

**Files:**
- Create: `cluster_tags.py`

- [ ] **Step 1: Create `cluster_tags.py` with imports, constants, argparse, and schema**

Write the file with:
- Imports: `argparse`, `sqlite3`, `sys`, `time`, `itertools`, `pathlib.Path`
- Imports from siblings: `embed_tags.load_tag_matrix`, `embed_tags._load_model`, `numpy`
- `_ensure_schema(con)` function that creates `chunk_tags`, `cluster_versions`, `clusters`, and `cluster_tags` tables with all indexes per the spec
- `main()` with argparse: `--tag-db` (required, Path), `--vault-db` (required, Path), `--unit` (chunk/conversation, default chunk), `--top-k` (int, default 10), `--max-depth` (int, default 3)
- Validate both DB paths exist; check `tag_embeddings` table is non-empty

```python
"""
cluster_tags.py — cluster vault tags by co-occurrence using Louvain community detection.

Usage:
    python3 cluster_tags.py --tag-db vital_articles.db --vault-db vault.db
    python3 cluster_tags.py --tag-db vital_articles.db --vault-db vault.db --unit conversation
    python3 cluster_tags.py --tag-db vital_articles.db --vault-db vault.db --top-k 15 --max-depth 4
"""

from __future__ import annotations

import argparse
import itertools
import json
import sqlite3
import sys
import time
from pathlib import Path

_TOOLS_DIR = str(Path(__file__).resolve().parent)
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)

from embed_tags import load_tag_matrix, _load_model
import numpy as np


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
```

- [ ] **Step 2: Verify schema creates cleanly**

Run:
```bash
python3 -c "
import sqlite3
con = sqlite3.connect('/tmp/test_cluster.db')
con.execute('CREATE TABLE chunks (id INTEGER PRIMARY KEY, text TEXT)')
import sys; sys.path.insert(0, '.')
from cluster_tags import _ensure_schema
_ensure_schema(con)
print([r[1] for r in con.execute(\"SELECT * FROM sqlite_master WHERE type='table'\").fetchall()])
con.close()
import os; os.remove('/tmp/test_cluster.db')
"
```
Expected: list containing `chunk_tags`, `cluster_versions`, `clusters`, `cluster_tags`

- [ ] **Step 3: Commit**

```bash
git add cluster_tags.py
git commit -m "feat: cluster_tags.py skeleton with schema and argparse"
```

---

### Task 2: Tag assignment (`assign_tags`)

**Files:**
- Modify: `cluster_tags.py`

- [ ] **Step 1: Write `assign_tags` function**

This function:
1. Loads tag matrix from `tag_db` via `load_tag_matrix()`
2. Loads all chunks from `vault_db` (id, text, embedding)
3. For each chunk, computes cosine similarity against tag matrix, takes top-k
4. Truncates and repopulates `chunk_tags` table

```python
def assign_tags(vault_con: sqlite3.Connection, tag_db: Path, top_k: int, model) -> int:
    """Assign top-k tags to each chunk by embedding similarity. Returns row count."""
    matrix, meta = load_tag_matrix(tag_db)
    if len(meta) == 0:
        raise SystemExit("Error: no tag embeddings found — run embed_tags.py --embed first")

    # Load chunk embeddings from vault DB
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
            batch.append((chunk_id, meta[idx][0], rank, float(scores[i][idx])))

    vault_con.executemany(
        "INSERT INTO chunk_tags (chunk_id, tag_title, rank, score) VALUES (?, ?, ?, ?)",
        batch,
    )
    vault_con.commit()

    print(f"[cluster] assigned {len(batch)} tag associations", file=sys.stderr)
    return len(batch)
```

- [ ] **Step 2: Verify against real data**

Run:
```bash
python3 -c "
import sqlite3, sys
sys.path.insert(0, '.')
from cluster_tags import _ensure_schema, assign_tags
from embed_tags import _load_model
from pathlib import Path

vault_con = sqlite3.connect('chatgpt-md/.vault-index/vault.db')
_ensure_schema(vault_con)
model = _load_model()
count = assign_tags(vault_con, Path('vital_articles.db'), top_k=10, model=model)
print(f'Assigned {count} tags')
# Spot check
for row in vault_con.execute('SELECT tag_title, rank, score FROM chunk_tags WHERE chunk_id = 1 ORDER BY rank LIMIT 5'):
    print(f'  rank {row[1]}: {row[0]} ({row[2]:.4f})')
vault_con.close()
"
```
Expected: ~300k tags assigned, spot check shows reasonable tags ranked by score

- [ ] **Step 3: Commit**

```bash
git add cluster_tags.py
git commit -m "feat: tag assignment via batch embedding similarity"
```

---

### Task 3: Co-occurrence graph (`build_graph`)

**Files:**
- Modify: `cluster_tags.py`

- [ ] **Step 1: Write `build_graph` function**

Two modes: chunk-level (each chunk's tags form co-occurrence pairs) and conversation-level (group chunks by file_path, union their tags).

```python
import networkx as nx


def build_graph(vault_con: sqlite3.Connection, unit: str) -> nx.Graph:
    """Build a weighted tag co-occurrence graph from chunk_tags."""
    G = nx.Graph()

    if unit == "chunk":
        # Group tags by chunk_id
        rows = vault_con.execute(
            "SELECT chunk_id, tag_title FROM chunk_tags ORDER BY chunk_id"
        ).fetchall()

        from itertools import groupby
        for chunk_id, group in groupby(rows, key=lambda r: r[0]):
            tags = [r[1] for r in group]
            for t1, t2 in itertools.combinations(sorted(tags), 2):
                if G.has_edge(t1, t2):
                    G[t1][t2]["weight"] += 1
                else:
                    G.add_edge(t1, t2, weight=1)

    elif unit == "conversation":
        # Group tags by file_path (via chunks table)
        rows = vault_con.execute("""
            SELECT c.file_path, ct.tag_title
            FROM chunk_tags ct
            JOIN chunks c ON c.id = ct.chunk_id
            ORDER BY c.file_path
        """).fetchall()

        from itertools import groupby
        for file_path, group in groupby(rows, key=lambda r: r[0]):
            tags = list(set(r[1] for r in group))  # dedupe within conversation
            for t1, t2 in itertools.combinations(sorted(tags), 2):
                if G.has_edge(t1, t2):
                    G[t1][t2]["weight"] += 1
                else:
                    G.add_edge(t1, t2, weight=1)

    # Add isolated nodes (tags with no co-occurrence)
    all_tags = vault_con.execute("SELECT DISTINCT tag_title FROM chunk_tags").fetchall()
    for (tag,) in all_tags:
        if tag not in G:
            G.add_node(tag)

    print(f"[cluster] graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges",
          file=sys.stderr)
    return G
```

- [ ] **Step 2: Verify graph construction**

Run:
```bash
python3 -c "
import sqlite3, sys
sys.path.insert(0, '.')
from cluster_tags import build_graph
vault_con = sqlite3.connect('chatgpt-md/.vault-index/vault.db')
G = build_graph(vault_con, 'chunk')
print(f'Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}')
# Show heaviest edges
edges = sorted(G.edges(data=True), key=lambda e: -e[2]['weight'])[:10]
for t1, t2, d in edges:
    print(f'  {t1} <-> {t2}: {d[\"weight\"]}')
vault_con.close()
"
```
Expected: thousands of nodes, hundreds of thousands of edges, heaviest edges between semantically related tags

- [ ] **Step 3: Commit**

```bash
git add cluster_tags.py
git commit -m "feat: co-occurrence graph construction (chunk and conversation modes)"
```

---

### Task 4: Recursive Louvain clustering (`cluster_graph`)

**Files:**
- Modify: `cluster_tags.py`

- [ ] **Step 1: Install dependencies**

```bash
pip3 install networkx python-louvain
```

- [ ] **Step 2: Write `cluster_graph` function**

```python
import community as community_louvain


def cluster_graph(G: nx.Graph, max_depth: int) -> list[dict]:
    """Recursive Louvain clustering. Returns flat list of cluster dicts."""
    all_clusters = []

    def _recurse(subgraph: nx.Graph, parent_id: str | None, depth: int):
        if depth >= max_depth or subgraph.number_of_nodes() < 2:
            return

        partition = community_louvain.best_partition(subgraph, weight="weight")

        communities: dict[int, list[str]] = {}
        for node, comm_id in partition.items():
            communities.setdefault(comm_id, []).append(node)

        for comm_idx, (comm_id, nodes) in enumerate(sorted(communities.items())):
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

    print(f"[cluster] {len(all_clusters)} clusters across {max_depth} depth levels",
          file=sys.stderr)
    return all_clusters
```

- [ ] **Step 3: Verify clustering**

Run:
```bash
python3 -c "
import sqlite3, sys
sys.path.insert(0, '.')
from cluster_tags import build_graph, cluster_graph
vault_con = sqlite3.connect('chatgpt-md/.vault-index/vault.db')
G = build_graph(vault_con, 'chunk')
clusters = cluster_graph(G, max_depth=3)
vault_con.close()

from collections import Counter
depth_counts = Counter(c['depth'] for c in clusters)
for d in sorted(depth_counts):
    print(f'  depth {d}: {depth_counts[d]} clusters')

# Show a top-level cluster's tags
top = [c for c in clusters if c['depth'] == 0][0]
sorted_tags = sorted(top['tags'].items(), key=lambda x: -x[1])[:10]
print(f'\\nCluster {top[\"cluster_id\"]} top tags:')
for tag, w in sorted_tags:
    print(f'  {tag}: {w}')
"
```
Expected: depth 0 has ~5-15 clusters, depth 1 has more, depth 2 has most. Top tags in a cluster should be thematically related.

- [ ] **Step 4: Commit**

```bash
git add cluster_tags.py
git commit -m "feat: recursive Louvain clustering"
```

---

### Task 5: Storage and orchestrator (`store_clusters`, `main`)

**Files:**
- Modify: `cluster_tags.py`

- [ ] **Step 1: Write `store_clusters` and wire up `main`**

```python
from datetime import datetime, timezone


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
```

Then the `main()` block:

```python
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

    model = _load_model()
    vault_con = sqlite3.connect(vault_db)
    _ensure_schema(vault_con)

    assign_tags(vault_con, tag_db, args.top_k, model)
    G = build_graph(vault_con, args.unit)
    clusters = cluster_graph(G, args.max_depth)
    version_id = store_clusters(vault_con, clusters, args.unit, args.top_k, args.max_depth)

    vault_con.close()

    elapsed = time.time() - t0
    print(f"[cluster] done in {elapsed:.1f}s — version {version_id}", file=sys.stderr)
```

- [ ] **Step 2: Full end-to-end test**

Run:
```bash
python3 cluster_tags.py --tag-db vital_articles.db --vault-db chatgpt-md/.vault-index/vault.db
```
Expected: completes in ~30-60s, prints cluster counts by depth

- [ ] **Step 3: Verify stored data**

Run:
```bash
sqlite3 chatgpt-md/.vault-index/vault.db "
  SELECT 'chunk_tags', COUNT(*) FROM chunk_tags
  UNION ALL
  SELECT 'clusters', COUNT(*) FROM clusters
  UNION ALL
  SELECT 'cluster_tags', COUNT(*) FROM cluster_tags;
"
```
Expected: ~300k chunk_tags, ~50-200 clusters, many cluster_tags

```bash
sqlite3 chatgpt-md/.vault-index/vault.db "
  SELECT depth, COUNT(*) FROM clusters WHERE version_id = (SELECT MAX(version_id) FROM cluster_versions) GROUP BY depth;
"
```
Expected: hierarchical cluster counts (fewer at depth 0, more at depth 2)

- [ ] **Step 4: Test re-run creates new version**

Run:
```bash
python3 cluster_tags.py --tag-db vital_articles.db --vault-db chatgpt-md/.vault-index/vault.db
sqlite3 chatgpt-md/.vault-index/vault.db "SELECT version_id, unit, num_clusters FROM cluster_versions"
```
Expected: two version rows

- [ ] **Step 5: Commit**

```bash
git add cluster_tags.py
git commit -m "feat: complete cluster_tags.py — tag assignment, graph, Louvain, storage"
```

---

## Chunk 2: `name_clusters.py`

### Task 6: Skeleton and context gathering

**Files:**
- Create: `name_clusters.py`

- [ ] **Step 1: Create `name_clusters.py` with context gathering**

```python
"""
name_clusters.py — name tag clusters using Claude CLI (Haiku).

Usage:
    python3 name_clusters.py --vault-db chatgpt-md/.vault-index/vault.db
    python3 name_clusters.py --vault-db ... --depth 0
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

CLAUDE_BIN = os.environ.get("CLAUDE_BIN", "claude")
MODEL = os.environ.get("TAG_MODEL", "claude-haiku-4-5-20251001")


def get_unnamed_clusters(
    con: sqlite3.Connection, version_id: int, depth: int | None = None
) -> list[dict]:
    """Get clusters that haven't been named yet."""
    query = """
        SELECT cluster_id, depth, tag_count
        FROM clusters
        WHERE version_id = ? AND name IS NULL
    """
    params: list = [version_id]
    if depth is not None:
        query += " AND depth = ?"
        params.append(depth)
    query += " ORDER BY depth, cluster_id"

    return [
        {"cluster_id": r[0], "depth": r[1], "tag_count": r[2]}
        for r in con.execute(query, params).fetchall()
    ]


def get_cluster_context(
    con: sqlite3.Connection, version_id: int, cluster_id: str
) -> tuple[list[str], list[str]]:
    """Get tag names and representative chunk excerpts for a cluster.

    Returns (tag_names, chunk_excerpts).
    """
    # Get tags (top 20 by weight)
    tags = con.execute(
        """SELECT tag_title FROM cluster_tags
           WHERE version_id = ? AND cluster_id = ?
           ORDER BY weight DESC LIMIT 20""",
        (version_id, cluster_id),
    ).fetchall()
    tag_names = [r[0] for r in tags]

    # Find representative chunks: most tag overlap with this cluster
    chunks = con.execute(
        """SELECT c.text, COUNT(*) as overlap
           FROM chunk_tags ct
           JOIN chunks c ON c.id = ct.chunk_id
           WHERE ct.tag_title IN (
               SELECT tag_title FROM cluster_tags
               WHERE version_id = ? AND cluster_id = ?
           )
           GROUP BY ct.chunk_id
           ORDER BY overlap DESC
           LIMIT 5""",
        (version_id, cluster_id),
    ).fetchall()
    excerpts = [r[0][:500] for r in chunks]

    return tag_names, excerpts
```

- [ ] **Step 2: Verify context gathering against real data**

Run (after Task 5 has populated clusters):
```bash
python3 -c "
import sqlite3, sys
sys.path.insert(0, '.')
from name_clusters import get_unnamed_clusters, get_cluster_context

con = sqlite3.connect('chatgpt-md/.vault-index/vault.db')
vid = con.execute('SELECT MAX(version_id) FROM cluster_versions').fetchone()[0]
clusters = get_unnamed_clusters(con, vid, depth=0)
print(f'{len(clusters)} unnamed clusters at depth 0')

tags, excerpts = get_cluster_context(con, vid, clusters[0]['cluster_id'])
print(f'\\nCluster {clusters[0][\"cluster_id\"]}:')
print(f'  Tags: {\", \".join(tags[:10])}')
print(f'  Excerpts: {len(excerpts)} chunks')
if excerpts:
    print(f'  First excerpt: {excerpts[0][:200]}...')
con.close()
"
```
Expected: clusters found, tags are thematically related, excerpts are relevant

- [ ] **Step 3: Commit**

```bash
git add name_clusters.py
git commit -m "feat: name_clusters.py skeleton with context gathering"
```

---

### Task 7: LLM naming and CLI

**Files:**
- Modify: `name_clusters.py`

- [ ] **Step 1: Write `name_cluster` and `main`**

```python
def _extract_result(stdout: str) -> str:
    """Parse claude CLI JSON output to get result text."""
    try:
        parsed = json.loads(stdout)
    except json.JSONDecodeError:
        parsed = []
        for line in stdout.strip().split("\n"):
            if line.strip():
                try:
                    parsed.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if isinstance(parsed, dict):
        return parsed.get("result", "")
    elif isinstance(parsed, list):
        for event in parsed:
            if isinstance(event, dict) and event.get("type") == "result":
                return event.get("result", "")
    return ""


def name_cluster(tag_names: list[str], excerpts: list[str]) -> tuple[str, str]:
    """Use Claude CLI to name a cluster. Returns (name, description)."""
    tags_str = ", ".join(tag_names)

    prompt = f"Name this topic cluster in 2-4 words, then a one-sentence description. Format: name | description\n\nTags: {tags_str}"
    if excerpts:
        prompt += "\n\nSample text:\n" + "\n---\n".join(excerpts)

    env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}

    result = subprocess.run(
        [
            CLAUDE_BIN,
            "-p", prompt,
            "--output-format", "json",
            "--model", MODEL,
            "--max-turns", "1",
            "--permission-mode", "dontAsk",
            "--strict-mcp-config",
        ],
        capture_output=True,
        text=True,
        env=env,
    )

    if result.returncode != 0:
        print(f"[name] claude error: {result.stderr}", file=sys.stderr)
        return "Unknown", "Naming failed"

    result_text = _extract_result(result.stdout)
    if not result_text:
        return "Unknown", "No result from claude"

    # Parse "name | description"
    if "|" in result_text:
        parts = result_text.split("|", 1)
        return parts[0].strip(), parts[1].strip()
    return result_text.strip(), ""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Name tag clusters using Claude CLI (Haiku)"
    )
    parser.add_argument("--vault-db", required=True, type=Path,
                        help="Path to vault.db (where clusters live)")
    parser.add_argument("--depth", type=int, default=None,
                        help="Only name clusters at this depth (default: all unnamed)")
    args = parser.parse_args()

    vault_db = args.vault_db.resolve()
    if not vault_db.exists():
        raise SystemExit(f"Error: '{vault_db}' not found")

    con = sqlite3.connect(vault_db)
    version_id = con.execute(
        "SELECT MAX(version_id) FROM cluster_versions"
    ).fetchone()[0]

    if version_id is None:
        raise SystemExit("Error: no cluster versions found — run cluster_tags.py first")

    clusters = get_unnamed_clusters(con, version_id, args.depth)
    if not clusters:
        print("[name] all clusters already named", file=sys.stderr)
        con.close()
        sys.exit(0)

    print(f"[name] naming {len(clusters)} clusters (version {version_id})…",
          file=sys.stderr)

    for i, cluster in enumerate(clusters, 1):
        cid = cluster["cluster_id"]
        tag_names, excerpts = get_cluster_context(con, version_id, cid)

        print(f"[name] [{i}/{len(clusters)}] cluster {cid} "
              f"({cluster['tag_count']} tags, depth {cluster['depth']})…",
              file=sys.stderr)

        name, description = name_cluster(tag_names, excerpts)

        con.execute(
            "UPDATE clusters SET name = ?, description = ? WHERE version_id = ? AND cluster_id = ?",
            (name, description, version_id, cid),
        )
        con.commit()

        print(f"[name]   → {name}", file=sys.stderr)

    con.close()
    print("[name] done", file=sys.stderr)
```

- [ ] **Step 2: Test naming depth-0 clusters only**

Run:
```bash
python3 name_clusters.py --vault-db chatgpt-md/.vault-index/vault.db --depth 0
```
Expected: names ~5-15 top-level clusters, each with a 2-4 word name

- [ ] **Step 3: Verify names stored**

Run:
```bash
sqlite3 chatgpt-md/.vault-index/vault.db "
  SELECT cluster_id, depth, name, description, tag_count
  FROM clusters
  WHERE version_id = (SELECT MAX(version_id) FROM cluster_versions) AND depth = 0
  ORDER BY tag_count DESC;
"
```
Expected: named clusters with reasonable theme names

- [ ] **Step 4: Test idempotency — re-running skips already-named clusters**

Run:
```bash
python3 name_clusters.py --vault-db chatgpt-md/.vault-index/vault.db --depth 0
```
Expected: "all clusters already named"

- [ ] **Step 5: Commit**

```bash
git add name_clusters.py
git commit -m "feat: complete name_clusters.py — LLM-powered cluster naming"
```
