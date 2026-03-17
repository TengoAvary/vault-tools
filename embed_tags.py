"""
embed_tags.py — embed Wikipedia Vital Articles tags for semantic candidate matching.

Usage:
    python3 embed_tags.py --vault /path/to/vault           # embed latest scrape version
    python3 embed_tags.py --db /path/to/vital_articles.db  # custom DB path
    python3 embed_tags.py --vault /path/to/vault --query "the fall of the Roman Empire"

Reads vital article titles from vital_articles.db (created by scrape_tags.py),
embeds them with all-MiniLM-L6-v2, and stores embeddings in a tag_embeddings table.

At query time, embed input text and return top-k tags by cosine similarity.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np


def _load_model(device: str | None = None):
    """Load the sentence transformer model."""
    import torch
    from sentence_transformers import SentenceTransformer

    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    return SentenceTransformer("all-MiniLM-L6-v2", device=device)


def _ensure_schema(con: sqlite3.Connection):
    """Create the tag_embeddings table if it doesn't exist."""
    con.execute("""
        CREATE TABLE IF NOT EXISTS tag_embeddings (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            version_id INTEGER NOT NULL REFERENCES scrape_versions(version_id),
            title      TEXT NOT NULL,
            full_path  TEXT NOT NULL,
            level      INTEGER NOT NULL,
            embedding  BLOB NOT NULL
        )
    """)
    con.execute("CREATE INDEX IF NOT EXISTS idx_te_version ON tag_embeddings(version_id)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_te_title ON tag_embeddings(title)")


def _latest_version(con: sqlite3.Connection) -> int | None:
    """Return the most recent scrape version_id, or None."""
    row = con.execute(
        "SELECT version_id FROM scrape_versions ORDER BY version_id DESC LIMIT 1"
    ).fetchone()
    return row[0] if row else None


def _latest_embedded_version(con: sqlite3.Connection) -> int | None:
    """Return the most recent embedded version_id, or None."""
    row = con.execute(
        "SELECT version_id FROM tag_embeddings ORDER BY version_id DESC LIMIT 1"
    ).fetchone()
    return row[0] if row else None


def embed_tags(db_path: Path, model=None) -> int:
    """Embed all tags from the latest scrape version. Returns count embedded."""
    con = sqlite3.connect(db_path)
    _ensure_schema(con)

    version_id = _latest_version(con)
    if version_id is None:
        print("[embed_tags] no scrape versions found — run scrape_tags.py first",
              file=sys.stderr)
        con.close()
        return 0

    # Skip if already embedded
    embedded_version = _latest_embedded_version(con)
    if embedded_version == version_id:
        count = con.execute(
            "SELECT COUNT(*) FROM tag_embeddings WHERE version_id = ?", (version_id,)
        ).fetchone()[0]
        print(f"[embed_tags] version {version_id} already embedded ({count} tags)",
              file=sys.stderr)
        con.close()
        return count

    # Get unique tags — deduplicate by title, keep highest level + deepest path
    rows = con.execute("""
        SELECT title, full_path, level
        FROM vital_articles
        WHERE version_id = ?
        ORDER BY title, level DESC, LENGTH(full_path) DESC
    """, (version_id,)).fetchall()

    # Deduplicate: one entry per title, prefer highest level (most specific)
    seen: dict[str, tuple[str, int]] = {}
    for title, full_path, level in rows:
        if title not in seen:
            seen[title] = (full_path, level)

    titles = list(seen.keys())
    paths = [seen[t][0] for t in titles]
    levels = [seen[t][1] for t in titles]

    print(f"[embed_tags] embedding {len(titles)} unique tags from version {version_id}",
          file=sys.stderr)

    if model is None:
        print("[embed_tags] loading model…", file=sys.stderr)
        model = _load_model()

    # Embed titles
    embeddings = model.encode(
        titles,
        batch_size=256,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    # Clear any partial embeddings for this version and insert
    con.execute("DELETE FROM tag_embeddings WHERE version_id = ?", (version_id,))
    con.executemany(
        """INSERT INTO tag_embeddings (version_id, title, full_path, level, embedding)
           VALUES (?, ?, ?, ?, ?)""",
        [
            (version_id, titles[i], paths[i], levels[i],
             embeddings[i].astype(np.float32).tobytes())
            for i in range(len(titles))
        ],
    )

    con.commit()
    con.close()
    print(f"[embed_tags] done — {len(titles)} tag embeddings stored", file=sys.stderr)
    return len(titles)


def load_tag_matrix(db_path: Path, version_id: int | None = None):
    """Load tag embeddings into a numpy matrix for search.

    Returns (matrix, metadata) where metadata is list of (title, full_path, level).
    """
    con = sqlite3.connect(db_path)
    if version_id is None:
        version_id = _latest_embedded_version(con)
    if version_id is None:
        con.close()
        return np.empty((0, 384), dtype=np.float32), []

    rows = con.execute(
        "SELECT title, full_path, level, embedding FROM tag_embeddings WHERE version_id = ?",
        (version_id,),
    ).fetchall()
    con.close()

    if not rows:
        return np.empty((0, 384), dtype=np.float32), []

    meta = [(r[0], r[1], r[2]) for r in rows]
    matrix = np.stack([np.frombuffer(r[3], dtype=np.float32) for r in rows])
    return matrix, meta


def query_tags(text: str, db_path: Path, top_k: int = 150, model=None):
    """Given input text, return top-k matching tags with scores.

    Returns list of (title, full_path, level, score).
    """
    matrix, meta = load_tag_matrix(db_path)
    if len(meta) == 0:
        return []

    if model is None:
        model = _load_model()

    vec = model.encode(
        [text],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0].astype(np.float32)

    scores = matrix @ vec
    top_indices = np.argsort(scores)[::-1][:top_k]

    return [
        (meta[i][0], meta[i][1], meta[i][2], float(scores[i]))
        for i in top_indices
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Embed Wikipedia Vital Articles tags for semantic matching"
    )
    parser.add_argument("--vault", type=Path, help="Path to the Obsidian vault root")
    parser.add_argument("--db", type=Path, help="Custom database path (overrides --vault)")
    parser.add_argument("--embed", action="store_true",
                        help="Embed (or re-embed) tags from the latest scrape version")
    parser.add_argument("--query", type=str, help="Test query — show top matching tags")
    parser.add_argument("--top-k", type=int, default=150, help="Number of candidates (default: 150)")
    args = parser.parse_args()

    if not args.embed and not args.query:
        raise SystemExit("Error: provide --embed, --query, or both")

    if args.db:
        db_path = args.db.resolve()
    elif args.vault:
        vault = args.vault.resolve()
        if not vault.is_dir():
            raise SystemExit(f"Error: '{vault}' is not a directory")
        db_path = vault / ".vault-index" / "vital_articles.db"
    else:
        raise SystemExit("Error: provide --vault or --db")

    if not db_path.exists():
        raise SystemExit(f"Error: '{db_path}' not found — run scrape_tags.py first")

    model = None

    if args.embed:
        model = _load_model()
        embed_tags(db_path, model=model)

    if args.query:
        if model is None:
            model = _load_model()
        results = query_tags(args.query, db_path, top_k=args.top_k, model=model)
        print(f"\nTop {len(results)} tags for: '{args.query}'\n")
        for i, (title, full_path, level, score) in enumerate(results, 1):
            print(f"  [{i:3d}] {score:.4f}  L{level}  {title}  ({full_path})")
