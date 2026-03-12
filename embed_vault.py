"""
embed_vault.py — index an Obsidian vault into a SQLite vector store.

Usage:
    python3 embed_vault.py --vault /path/to/vault                # incremental (default)
    python3 embed_vault.py --vault /path/to/vault --full         # full rebuild
    python3 embed_vault.py --vault /path/to/vault --chunk-mode delimiter

Creates <vault>/.vault-index/vault.db with a `chunks` table:
    id          INTEGER PRIMARY KEY
    file_path   TEXT     — relative to vault root
    chunk_index INTEGER  — position within the file
    text        TEXT     — raw chunk text
    embedding   BLOB     — float32 numpy array (384 dims, all-MiniLM-L6-v2)

Chunk modes:
    sliding (default)  — overlapping character-window chunks
    delimiter          — split on <!-- chunk --> markers (for chatgpt_to_md.py output)
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from pathlib import Path

import numpy as np

CHUNK_CHARS    = 1800   # ~450 tokens  (4 chars ≈ 1 token)
OVERLAP_CHARS  = 200    # ~50 tokens overlap
SKIP_DIRS      = {".vault-index", ".obsidian", ".git", ".claude", ".trash", "_tools"}
CHUNK_DELIMITER = "<!-- chunk -->"


def iter_md_files(vault: Path):
    for path in sorted(vault.rglob("*.md")):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        yield path


def chunk_sliding(text: str) -> list[str]:
    """Split text into overlapping character-based chunks."""
    pieces, start = [], 0
    while start < len(text):
        end = start + CHUNK_CHARS
        pieces.append(text[start:end].strip())
        start += CHUNK_CHARS - OVERLAP_CHARS
    return [p for p in pieces if len(p) > 60]   # drop tiny tail fragments


def chunk_delimiter(text: str) -> list[str]:
    """Split text on <!-- chunk --> markers."""
    pieces = text.split(CHUNK_DELIMITER)
    return [p.strip() for p in pieces if len(p.strip()) > 60]


def _load_model(device: str | None = None):
    """Load the sentence transformer model."""
    import torch
    from sentence_transformers import SentenceTransformer

    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    return SentenceTransformer("all-MiniLM-L6-v2", device=device)


def _ensure_schema(con: sqlite3.Connection):
    """Create tables if they don't exist."""
    con.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id          INTEGER PRIMARY KEY,
            file_path   TEXT,
            chunk_index INTEGER,
            text        TEXT,
            embedding   BLOB
        )
    """)
    con.execute("CREATE INDEX IF NOT EXISTS idx_file ON chunks(file_path)")
    con.execute("""
        CREATE TABLE IF NOT EXISTS file_meta (
            file_path   TEXT PRIMARY KEY,
            mtime       REAL,
            chunk_mode  TEXT
        )
    """)


def _embed_files(files: list[Path], vault: Path, model, chunk_mode: str) -> list[tuple]:
    """Chunk and embed a list of files. Returns rows for INSERT."""
    chunker = chunk_delimiter if chunk_mode == "delimiter" else chunk_sliding

    all_rows = []
    for path in files:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for i, piece in enumerate(chunker(text)):
            all_rows.append((str(path.relative_to(vault)), i, piece))

    if not all_rows:
        return []

    texts = [r[2] for r in all_rows]
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    return [
        (all_rows[i][0], all_rows[i][1], all_rows[i][2],
         embeddings[i].astype(np.float32).tobytes())
        for i in range(len(all_rows))
    ]


def build_db(vault: Path, chunk_mode: str = "sliding", model=None):
    """Full rebuild — drops and recreates the chunks and file_meta tables."""
    index_dir = vault / ".vault-index"
    index_dir.mkdir(exist_ok=True)
    db_path = index_dir / "vault.db"

    if model is None:
        print("Loading model...", file=sys.stderr)
        model = _load_model()

    con = sqlite3.connect(db_path)
    con.execute("DROP TABLE IF EXISTS chunks")
    con.execute("DROP TABLE IF EXISTS file_meta")
    _ensure_schema(con)

    files = list(iter_md_files(vault))
    print(f"Indexing {len(files)} markdown files (chunk mode: {chunk_mode})...",
          file=sys.stderr)

    rows = _embed_files(files, vault, model, chunk_mode)

    con.executemany(
        "INSERT INTO chunks (file_path, chunk_index, text, embedding) VALUES (?,?,?,?)",
        rows,
    )

    # Populate file_meta so incremental runs have a baseline
    for path in files:
        rel = str(path.relative_to(vault))
        mtime = os.stat(path).st_mtime
        con.execute(
            "INSERT OR REPLACE INTO file_meta (file_path, mtime, chunk_mode) VALUES (?,?,?)",
            (rel, mtime, chunk_mode),
        )

    con.commit()
    con.close()
    print(f"Done. {len(rows)} chunks stored in {db_path}", file=sys.stderr)


def incremental_update(vault: Path, chunk_mode: str = "sliding", model=None):
    """Re-embed only new, modified, or deleted files."""
    index_dir = vault / ".vault-index"
    index_dir.mkdir(exist_ok=True)
    db_path = index_dir / "vault.db"

    con = sqlite3.connect(db_path)
    _ensure_schema(con)

    # Load existing file metadata
    stored = {}
    for row in con.execute("SELECT file_path, mtime, chunk_mode FROM file_meta"):
        stored[row[0]] = (row[1], row[2])

    # Walk current vault files
    current_files = {}
    for path in iter_md_files(vault):
        rel = str(path.relative_to(vault))
        current_files[rel] = path

    # Detect deleted files
    deleted = set(stored.keys()) - set(current_files.keys())

    # Detect new or modified files
    changed = []
    for rel, path in current_files.items():
        mtime = os.stat(path).st_mtime
        if rel not in stored:
            changed.append((rel, path, mtime))
        elif mtime > stored[rel][0] or chunk_mode != stored[rel][1]:
            changed.append((rel, path, mtime))

    if not deleted and not changed:
        print("[embed] vault index is up to date", file=sys.stderr)
        con.close()
        return

    # Remove chunks for deleted files
    if deleted:
        print(f"[embed] removing {len(deleted)} deleted file(s)", file=sys.stderr)
        for rel in deleted:
            con.execute("DELETE FROM chunks WHERE file_path = ?", (rel,))
            con.execute("DELETE FROM file_meta WHERE file_path = ?", (rel,))

    # Re-embed changed files
    if changed:
        print(f"[embed] re-indexing {len(changed)} file(s)...", file=sys.stderr)

        if model is None:
            print("[embed] loading model...", file=sys.stderr)
            model = _load_model()

        # Delete old chunks for changed files
        for rel, _, _ in changed:
            con.execute("DELETE FROM chunks WHERE file_path = ?", (rel,))

        paths = [path for _, path, _ in changed]
        rows = _embed_files(paths, vault, model, chunk_mode)

        con.executemany(
            "INSERT INTO chunks (file_path, chunk_index, text, embedding) VALUES (?,?,?,?)",
            rows,
        )

        for rel, _, mtime in changed:
            con.execute(
                "INSERT OR REPLACE INTO file_meta (file_path, mtime, chunk_mode) VALUES (?,?,?)",
                (rel, mtime, chunk_mode),
            )

    con.commit()
    con.close()

    parts = []
    if deleted:
        parts.append(f"{len(deleted)} removed")
    if changed:
        parts.append(f"{len(changed)} re-indexed")
    print(f"[embed] done — {', '.join(parts)}", file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index an Obsidian vault into a SQLite vector store")
    parser.add_argument("--vault", required=True, type=Path, help="Path to the Obsidian vault root")
    parser.add_argument(
        "--chunk-mode", choices=["sliding", "delimiter"], default="sliding",
        help="Chunking strategy: 'sliding' (overlapping windows) or 'delimiter' (split on <!-- chunk --> markers)"
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Full rebuild (default is incremental update)"
    )
    args = parser.parse_args()

    vault = args.vault.resolve()
    if not vault.is_dir():
        raise SystemExit(f"Error: '{vault}' is not a directory")

    if args.full:
        build_db(vault, chunk_mode=args.chunk_mode)
    else:
        incremental_update(vault, chunk_mode=args.chunk_mode)
