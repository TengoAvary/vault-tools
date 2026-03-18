"""
vault_mcp.py — MCP server for Obsidian vault
semantic search and navigation.

Exposes semantic search and tag-based navigation over
a vault's SQLite vector index so it can be used directly
in Claude Code.

Tools:
    semantic_search(query, top_k=5)
    search_by_tag(tag, limit=20)
    list_tags(tag_type="all")
    get_related_notes(file_path, top_k=5)
    read_note(file_path)

Usage:
    python3 vault_mcp.py /path/to/vault
"""

from __future__ import annotations

import logging
import sqlite3
import sys
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator

import numpy as np

# Silence sentence-transformers chatter before import
logging.getLogger("sentence_transformers").setLevel(
    logging.WARNING,
)

from mcp.server.fastmcp import (  # noqa: E402
    FastMCP,  # pylint: disable=wrong-import-position
    Context,
)

# Ensure embed_vault is importable
_TOOLS_DIR = str(Path(__file__).resolve().parent)
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)

# ── Paths ─────────────────────────────────────────────


def _resolve_paths() -> tuple[Path, Path]:
    """Parse the vault root from argv and derive DB."""
    if len(sys.argv) < 2:
        print(
            "Usage: python3 vault_mcp.py /path/to/vault",
            file=sys.stderr,
        )
        sys.exit(1)
    vault_root = Path(sys.argv[1]).resolve()
    if not vault_root.is_dir():
        print(
            f"Error: '{vault_root}' is not a directory",
            file=sys.stderr,
        )
        sys.exit(1)
    db_path = vault_root / ".vault-index" / "vault.db"
    # DB may not exist yet — incremental_update creates it
    return vault_root, db_path


VAULT_ROOT, DB_PATH = _resolve_paths()


# ── Shared state ──────────────────────────────────────

@dataclass
class VaultContext:
    model: Any              # SentenceTransformer
    matrix: np.ndarray      # (N, 384) float32
    chunk_meta: list        # [(id, file_path, ci, text)]
    vault_root: Path
    db_path: Path


# ── Lifespan: load model + embeddings once ────────────

@asynccontextmanager
async def lifespan(
    server: FastMCP,  # pylint: disable=unused-argument
) -> AsyncIterator[VaultContext]:
    """Load the embedding model and vault index at startup."""
    import torch
    from sentence_transformers import SentenceTransformer
    from embed_vault import incremental_update

    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(
        f"[vault_mcp] loading model on {device}…",
        file=sys.stderr,
    )
    model = SentenceTransformer(
        "all-MiniLM-L6-v2", device=device,
    )

    # Update index for new/modified/deleted files
    print(
        "[vault_mcp] checking for vault changes…",
        file=sys.stderr,
    )
    incremental_update(
        VAULT_ROOT, chunk_mode="sliding", model=model,
    )

    print(
        "[vault_mcp] loading embeddings from"
        f" {DB_PATH}…",
        file=sys.stderr,
    )
    con = sqlite3.connect(DB_PATH)
    rows = con.execute(
        "SELECT id, file_path, chunk_index,"
        " text, embedding FROM chunks ORDER BY id"
    ).fetchall()
    con.close()

    if rows:
        meta = [
            (r[0], r[1], r[2], r[3]) for r in rows
        ]
        matrix = np.stack([
            np.frombuffer(r[4], dtype=np.float32)
            for r in rows
        ])
    else:
        meta = []
        matrix = np.empty(
            (0, 384), dtype=np.float32,
        )

    n_files = len({r[1] for r in rows})
    print(
        f"[vault_mcp] ready — {len(rows)} chunks"
        f" from {n_files} files",
        file=sys.stderr,
    )
    print(
        f"[vault_mcp] vault: {VAULT_ROOT}",
        file=sys.stderr,
    )

    yield VaultContext(
        model=model, matrix=matrix,
        chunk_meta=meta,
        vault_root=VAULT_ROOT, db_path=DB_PATH,
    )


# ── Server ────────────────────────────────────────────

mcp = FastMCP("vault", lifespan=lifespan)


# ── Tool 1: semantic_search ──────────────────────────

@mcp.tool()
def semantic_search(
    query: str, top_k: int = 5,
    ctx: Context | None = None,
) -> str:
    """
    Find vault chunks most semantically similar to
    the query.  Returns top-k results with file path,
    similarity score, and chunk text.
    Use this as the primary way to explore the vault
    by meaning, not keywords.
    """
    assert ctx is not None
    vc: VaultContext = (
        ctx.request_context.lifespan_context
    )

    vec = vc.model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0].astype(np.float32)

    scores = vc.matrix @ vec
    top_indices = np.argsort(scores)[::-1][:top_k]

    parts = [f"Semantic search: '{query}'\n"]
    for rank, idx in enumerate(top_indices, 1):
        _, file_path, chunk_index, text = (
            vc.chunk_meta[idx]
        )
        score = float(scores[idx])
        parts.append(
            f"[{rank}] {file_path}"
            f"  (chunk {chunk_index},"
            f" score {score:.4f})\n{text}"
        )
    return "\n\n---\n\n".join(parts)


# ── Tool 2: search_by_tag ───────────────────────────

@mcp.tool()
def search_by_tag(
    tag: str, limit: int = 20,
    ctx: Context | None = None,
) -> str:
    """
    Return notes that carry the given tag, ordered by
    how many chunks matched.  Tag names are
    lowercase-hyphenated e.g. 'philosophy',
    'software-engineering'.
    Use list_tags() to discover available tag names.
    """
    assert ctx is not None
    vc: VaultContext = (
        ctx.request_context.lifespan_context
    )
    con = sqlite3.connect(vc.db_path)
    rows = con.execute(
        """
        SELECT file_path, tag_type, chunk_count
        FROM file_tags
        WHERE tag = ?
        ORDER BY chunk_count DESC
        LIMIT ?
        """,
        (tag, limit),
    ).fetchall()

    if not rows:
        fuzzy = con.execute(
            "SELECT name FROM tags WHERE name LIKE ?",
            (f"%{tag}%",),
        ).fetchall()
        con.close()
        if fuzzy:
            suggestions = ", ".join(
                r[0] for r in fuzzy
            )
            return (
                f"Tag '{tag}' not found."
                f" Did you mean: {suggestions}?"
            )
        return (
            f"Tag '{tag}' not found in the vault"
            " index.\n"
            "Use list_tags() to see available tags."
        )
    con.close()

    tag_type = rows[0][1]
    lines = [f"Notes tagged '{tag}' ({tag_type}):\n"]
    for file_path, _, chunk_count in rows:
        suffix = (
            "s" if chunk_count != 1 else ""
        )
        lines.append(
            f"  {file_path}"
            f"  ({chunk_count} chunk{suffix})"
        )
    return "\n".join(lines)


# ── Tool 3: list_tags ───────────────────────────────

@mcp.tool()
def list_tags(
    tag_type: str = "all",
    ctx: Context | None = None,
) -> str:
    """
    List available vault tags with descriptions.
    tag_type: 'broad' (10 top-level themes),
    'specific' (57 granular topics), or 'all'.
    """
    if tag_type not in ("broad", "specific", "all"):
        return (
            "Invalid tag_type."
            " Use 'broad', 'specific', or 'all'."
        )

    assert ctx is not None
    vc: VaultContext = (
        ctx.request_context.lifespan_context
    )
    con = sqlite3.connect(vc.db_path)
    if tag_type == "all":
        rows = con.execute(
            "SELECT name, type, description"
            " FROM tags"
            " ORDER BY type DESC, name"
        ).fetchall()
    else:
        rows = con.execute(
            "SELECT name, type, description"
            " FROM tags WHERE type = ?"
            " ORDER BY name",
            (tag_type,),
        ).fetchall()
    con.close()

    if not rows:
        return "No tags found."

    sections: dict[str, list[str]] = {}
    for name, ttype, description in rows:
        sections.setdefault(ttype, []).append(
            f"  {name}: {description}"
        )

    lines: list[str] = []
    for ttype in ("broad", "specific"):
        if ttype not in sections:
            continue
        lines.append(
            f"{ttype.upper()} TAGS"
            f" ({len(sections[ttype])}):"
        )
        lines.extend(sections[ttype])
        lines.append("")
    return "\n".join(lines)


# ── Tool 4: get_related_notes ────────────────────────

@mcp.tool()
def get_related_notes(
    file_path: str, top_k: int = 5,
    ctx: Context | None = None,
) -> str:
    """
    Find vault notes most similar to the given file.
    Builds a centroid from the file's chunk embeddings,
    then returns the top-k other files by maximum chunk
    similarity.
    file_path is relative to the vault root,
    e.g. 'Philosophy.md'.
    """
    assert ctx is not None
    vc: VaultContext = (
        ctx.request_context.lifespan_context
    )

    indices = [
        i for i, (_, fp, _, _)
        in enumerate(vc.chunk_meta)
        if fp == file_path
    ]

    if not indices:
        return (
            f"File '{file_path}' not found in the"
            " vault index.\n"
            "Use semantic_search() to discover"
            " indexed file paths."
        )

    centroid = vc.matrix[indices].mean(axis=0)
    norm = np.linalg.norm(centroid)
    if norm > 0:
        centroid /= norm

    scores = vc.matrix @ centroid

    file_best: dict[str, float] = {}
    for i, (_, fp, _, _) in enumerate(vc.chunk_meta):
        if fp == file_path:
            continue
        s = float(scores[i])
        if fp not in file_best or s > file_best[fp]:
            file_best[fp] = s

    top_files = sorted(
        file_best.items(), key=lambda x: -x[1],
    )[:top_k]

    lines = [
        f"Notes most related to '{file_path}':\n"
    ]
    for rank, (fp, score) in enumerate(top_files, 1):
        lines.append(
            f"  [{rank}] {fp}"
            f"  (similarity {score:.4f})"
        )
    return "\n".join(lines)


# ── Tool 5: read_note ───────────────────────────────

@mcp.tool()
def read_note(
    file_path: str, ctx: Context | None = None,
) -> str:
    """
    Read the full content of a vault note.
    file_path is relative to the vault root,
    e.g. 'Philosophy.md',
    'corpus/chapter1.txt', or
    'Coding Curriculum/All lessons.md'.
    """
    assert ctx is not None
    vc: VaultContext = (
        ctx.request_context.lifespan_context
    )
    target = (vc.vault_root / file_path).resolve()

    if not str(target).startswith(str(vc.vault_root)):
        return (
            f"Access denied: '{file_path}'"
            " resolves outside the vault."
        )

    if not target.exists():
        return (
            f"File not found: '{file_path}'\n"
            "Paths are relative to the vault root, "
            "e.g. 'Dan Log.md' or"
            " 'Coding Curriculum/All lessons.md'."
        )
    if not target.is_file():
        return (
            f"'{file_path}' is a directory,"
            " not a file."
        )

    try:
        content = target.read_text(
            encoding="utf-8", errors="replace",
        )
    except OSError as e:
        return f"Could not read '{file_path}': {e}"

    return f"# {file_path}\n\n{content}"


# ── Entry point ───────────────────────────────────────

if __name__ == "__main__":
    mcp.run(transport="stdio")
