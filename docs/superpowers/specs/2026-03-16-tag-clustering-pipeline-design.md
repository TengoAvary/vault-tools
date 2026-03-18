# Tag Clustering Pipeline Design

## Problem

We have ~10k Wikipedia Vital Articles tags and a corpus of 3,276 ChatGPT conversations (30,687 chunks). We want to discover emergent themes in the corpus by clustering tags based on how they co-occur, then naming those clusters with an LLM.

The GlobeGov project did this with Weaviate + PostgreSQL + Celery. We want the same result with a lightweight, local-first approach: SQLite + numpy + networkx + Claude CLI.

## Key Insight

For co-occurrence clustering, precise per-chunk tag assignment isn't needed. The top 10 tags by embedding similarity are "good enough" — noise washes out in aggregate, and what matters is the co-occurrence signal across thousands of data points. This eliminates the need for LLM calls during tag assignment (the expensive step), shifting LLM cost to cluster naming only (~50-100 calls instead of ~30,000).

## Architecture: Two-Stage Split

### Stage 1: `cluster_tags.py` (fast, pure Python)

Assigns tags, builds co-occurrence graph, runs Louvain clustering. No LLM cost. Safe to iterate on parameters. Expected runtime ~30-60s for the full corpus (dominated by embedding lookups and graph construction).

### Stage 2: `name_clusters.py` (slow, LLM-powered)

Reads unnamed clusters, gathers context, spawns Claude CLI (Haiku) to name each one. Run separately once clustering looks good.

## Stage 1: `cluster_tags.py`

### Tag Assignment

For each unit (chunk or conversation), take the top-k tags (default 10) by cosine similarity from the `tag_embeddings` table in `vital_articles.db`.

- `--unit chunk` (default): Each chunk in `vault.db` is a unit. Embed each chunk's text and compare against the tag matrix. 30,687 data points.
- `--unit conversation`: Chunks are grouped by `file_path`. Use the mean of the chunk embeddings (centroid) as the conversation vector — avoids truncation issues with long conversations. 3,276 data points.

### Co-occurrence Graph

For each unit, every pair of its assigned tags gets an edge weight increment of 1. Uses `networkx.Graph` with weighted edges. Tags that repeatedly co-occur across many units accumulate heavy edges.

With top-10 per unit and 30k chunks, this produces up to 30k x C(10,2) = ~1.35M edge increments. The resulting graph will have ~10k nodes and on the order of hundreds of thousands of unique edges. NetworkX handles this comfortably in memory.

### Recursive Louvain Clustering

Uses `community` (python-louvain) package. Configurable `--max-depth` (default 3).

1. Run `community_louvain.best_partition(G, weight='weight')` on the full graph
2. Group nodes by partition ID into communities
3. For each community, extract its subgraph and recurse
4. Stop when `current_depth >= max_depth` or community has < 2 nodes

Produces a tree with dot-separated IDs: `"0"` (top), `"0.1"` (mid), `"0.1.3"` (leaf).

### Storage Schema

Results stored in the vault DB (`vault.db`) — clusters are corpus-specific, not part of the shared tag taxonomy in `vital_articles.db`.

#### Tag assignments (persisted)

Per-chunk tag assignments are stored so both scripts can use them and for future MCP tools:

```sql
CREATE TABLE chunk_tags (
    chunk_id   INTEGER NOT NULL REFERENCES chunks(id),
    tag_title  TEXT NOT NULL,
    rank       INTEGER NOT NULL,    -- 1-10, by similarity
    score      REAL NOT NULL,       -- cosine similarity
    PRIMARY KEY (chunk_id, tag_title)
);
```

Index on `(tag_title)`. ~300k rows (30k chunks x 10 tags).

Re-running `cluster_tags.py` truncates and repopulates this table (tag assignments are deterministic given the same embeddings, so versioning adds no value).

#### Cluster tables

```sql
CREATE TABLE cluster_versions (
    version_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at    TEXT NOT NULL,        -- ISO 8601
    unit          TEXT NOT NULL,        -- 'chunk' or 'conversation'
    top_k         INTEGER NOT NULL,
    max_depth     INTEGER NOT NULL,
    num_clusters  INTEGER NOT NULL
);

CREATE TABLE clusters (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    version_id    INTEGER NOT NULL REFERENCES cluster_versions(version_id),
    cluster_id    TEXT NOT NULL,        -- dot-separated path: "0", "0.1", "0.1.3"
    depth         INTEGER NOT NULL,     -- 0=top, 1=mid, 2=leaf
    parent_id     TEXT,                 -- parent cluster_id, NULL for top-level
    name          TEXT,                 -- NULL until name_clusters.py runs
    description   TEXT,                 -- NULL until name_clusters.py runs
    tag_count     INTEGER NOT NULL,
    UNIQUE (version_id, cluster_id)
);

CREATE TABLE cluster_tags (
    version_id    INTEGER NOT NULL,
    cluster_id    TEXT NOT NULL,        -- matches clusters.cluster_id
    tag_title     TEXT NOT NULL,
    weight        REAL NOT NULL,        -- sum of edge weights for this tag within the cluster subgraph
    PRIMARY KEY (version_id, cluster_id, tag_title)
);
```

Indexes on `clusters(version_id, depth)` and `cluster_tags(tag_title)`.

Each run creates a new `cluster_versions` row. Old versions are kept for comparison.

### CLI

`--tag-db` points to the shared tag embeddings, `--vault-db` points to the corpus being clustered (where results are also stored).

```
python3 cluster_tags.py --tag-db vital_articles.db --vault-db chatgpt-md/.vault-index/vault.db
python3 cluster_tags.py --tag-db vital_articles.db --vault-db ... --unit conversation
python3 cluster_tags.py --tag-db vital_articles.db --vault-db ... --top-k 15 --max-depth 4
```

Fails with a clear message if `tag_embeddings` table is empty or missing.

## Stage 2: `name_clusters.py`

### Context Gathering

For each unnamed cluster in the latest version:

1. **Tag titles**: All tags in the cluster (top 20 by weight if the cluster is large)
2. **Representative chunks**: Join `chunk_tags` with `cluster_tags` on `tag_title`, count how many of a chunk's tags overlap with the cluster, take the top 5 chunks by overlap count. Truncate each to ~500 chars.

### LLM Prompt

Lean format to minimize output tokens:

```
Name this topic cluster in 2-4 words, then a one-sentence description. Format: name | description

Tags: Ancient Rome, Roman Empire, Byzantine Empire, Julius Caesar, Augustus, ...

Sample text:
[chunk excerpts]
```

Uses Claude CLI with `--strict-mcp-config`, `--max-turns 1`, Haiku model.

### Storage

Writes `name` and `description` back to the `clusters` table in the vault DB.

### CLI

```
python3 name_clusters.py --vault-db chatgpt-md/.vault-index/vault.db
python3 name_clusters.py --vault-db ... --depth 0      # only name depth-0 clusters
```

`--depth N` filters to only name clusters at that depth. Without it, names all unnamed clusters.

## Dependencies

- `networkx` — graph construction and subgraph extraction
- `python-louvain` (`community` package) — Louvain modularity clustering
- Both pip-installable, pure Python

Existing: `numpy`, `sentence-transformers`, `torch` (already installed for vault embedding)

## Not Building

- MCP server tools (future work)
- Visualization (query SQLite directly)
- Automatic re-clustering on vault changes (manual CLI)
- Cross-version cluster diffing (versioning supports it if needed later)

## Cost Estimate

- Tag assignment: $0 (embedding only)
- Clustering: $0 (pure Python)
- Naming: ~50-100 Haiku calls at ~$0.01 each = ~$0.50-1.00 per run
- Total: under $1 per full pipeline run
