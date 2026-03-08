"""
embed_vault.py — index an Obsidian vault into a SQLite vector store.

Usage:
    python3 embed_vault.py --vault /path/to/vault
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

import argparse
import sqlite3
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
import torch

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


def build_db(vault: Path, chunk_mode: str = "sliding"):
    index_dir = vault / ".vault-index"
    index_dir.mkdir(exist_ok=True)
    db_path = index_dir / "vault.db"

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Loading model on {device}...")
    model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

    con = sqlite3.connect(db_path)
    con.execute("DROP TABLE IF EXISTS chunks")
    con.execute("""
        CREATE TABLE chunks (
            id          INTEGER PRIMARY KEY,
            file_path   TEXT,
            chunk_index INTEGER,
            text        TEXT,
            embedding   BLOB
        )
    """)
    con.execute("CREATE INDEX IF NOT EXISTS idx_file ON chunks(file_path)")

    chunker = chunk_delimiter if chunk_mode == "delimiter" else chunk_sliding

    files = list(iter_md_files(vault))
    print(f"Indexing {len(files)} markdown files (chunk mode: {chunk_mode})...")

    all_rows = []
    for path in files:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for i, piece in enumerate(chunker(text)):
            all_rows.append((str(path.relative_to(vault)), i, piece))

    print(f"Embedding {len(all_rows)} chunks...")
    texts = [r[2] for r in all_rows]

    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    rows = [
        (all_rows[i][0], all_rows[i][1], all_rows[i][2], embeddings[i].astype(np.float32).tobytes())
        for i in range(len(all_rows))
    ]
    con.executemany("INSERT INTO chunks (file_path, chunk_index, text, embedding) VALUES (?,?,?,?)", rows)
    con.commit()
    con.close()

    print(f"\nDone. {len(rows)} chunks stored in {db_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index an Obsidian vault into a SQLite vector store")
    parser.add_argument("--vault", required=True, type=Path, help="Path to the Obsidian vault root")
    parser.add_argument(
        "--chunk-mode", choices=["sliding", "delimiter"], default="sliding",
        help="Chunking strategy: 'sliding' (overlapping windows) or 'delimiter' (split on <!-- chunk --> markers)"
    )
    args = parser.parse_args()

    vault = args.vault.resolve()
    if not vault.is_dir():
        raise SystemExit(f"Error: '{vault}' is not a directory")

    build_db(vault, chunk_mode=args.chunk_mode)
