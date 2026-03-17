"""
tag_text.py — tag input text using Wikipedia Vital Articles taxonomy.

Usage:
    echo "some text" | python3 tag_text.py --db vital_articles.db
    python3 tag_text.py --db vital_articles.db --text "the fall of the Roman Empire"
    python3 tag_text.py --db vital_articles.db --file essay.md
    python3 tag_text.py --db vital_articles.db --file essay.md --candidates 200

Retrieves candidate tags via embedding similarity, then uses a Claude CLI
agent (Haiku) to select the best matches.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Ensure sibling modules are importable
_TOOLS_DIR = str(Path(__file__).resolve().parent)
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)

from embed_tags import load_tag_matrix, _load_model
import numpy as np

CLAUDE_BIN = os.environ.get("CLAUDE_BIN", "claude")
MODEL = os.environ.get("TAG_MODEL", "claude-haiku-4-5-20251001")


def get_candidates(text: str, db_path: Path, top_k: int, model) -> list[dict]:
    """Return top-k candidate tags by embedding similarity."""
    matrix, meta = load_tag_matrix(db_path)
    if len(meta) == 0:
        return []

    vec = model.encode(
        [text],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0].astype(np.float32)

    scores = matrix @ vec
    top_indices = np.argsort(scores)[::-1][:top_k]

    return [
        {
            "title": meta[i][0],
            "path": meta[i][1],
            "level": meta[i][2],
            "score": round(float(scores[i]), 4),
        }
        for i in top_indices
    ]


def _extract_result(stdout: str) -> tuple[str, float | None]:
    """Parse claude CLI JSON output. Returns (result_text, cost_usd)."""
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
        return parsed.get("result", ""), parsed.get("cost_usd") or parsed.get("total_cost_usd")
    elif isinstance(parsed, list):
        for event in parsed:
            if isinstance(event, dict) and event.get("type") == "result":
                return event.get("result", ""), event.get("cost_usd") or event.get("total_cost_usd")
    return "", None


def select_tags(text: str, candidates: list[dict]) -> tuple[list[str], float | None]:
    """Use Claude CLI to select the best tags from candidates."""
    candidate_lines = "\n".join(c["title"] for c in candidates)

    prompt = f"""Select up to 5 tags most relevant to the text. Return ONLY the tag names separated by |. No other text.

TEXT:
{text[:3000]}

TAGS:
{candidate_lines}"""

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
        print(f"[tag_text] claude error: {result.stderr}", file=sys.stderr)
        return [], None

    result_text, cost = _extract_result(result.stdout)
    if not result_text:
        print("[tag_text] no result from claude", file=sys.stderr)
        return [], cost

    tags = [t.strip() for t in result_text.split("|") if t.strip()]
    return tags, cost


def tag_text(text: str, db_path: Path, top_k: int = 150, model=None) -> list[str]:
    """Full pipeline: embed → candidates → LLM selection."""
    if model is None:
        print("[tag_text] loading embedding model…", file=sys.stderr)
        model = _load_model()

    t0 = time.time()
    candidates = get_candidates(text, db_path, top_k, model)
    t_embed = time.time() - t0

    if not candidates:
        print("[tag_text] no candidates found — is the DB embedded?", file=sys.stderr)
        return []

    print(f"[tag_text] {len(candidates)} candidates in {t_embed:.2f}s, asking {MODEL}…",
          file=sys.stderr)

    t1 = time.time()
    tags, cost = select_tags(text, candidates)
    t_llm = time.time() - t1

    cost_str = f"${cost:.4f}" if cost else "unknown"
    print(f"[tag_text] {len(tags)} tags in {t_llm:.1f}s ({cost_str})", file=sys.stderr)
    return tags


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tag input text using Wikipedia Vital Articles taxonomy"
    )
    parser.add_argument("--db", type=Path, required=True,
                        help="Path to vital_articles.db")
    parser.add_argument("--text", type=str, help="Text to tag (inline)")
    parser.add_argument("--file", type=Path, help="File to tag")
    parser.add_argument("--candidates", type=int, default=150,
                        help="Number of embedding candidates (default: 150)")
    args = parser.parse_args()

    db_path = args.db.resolve()
    if not db_path.exists():
        raise SystemExit(f"Error: '{db_path}' not found")

    if args.text:
        text = args.text
    elif args.file:
        text = args.file.resolve().read_text(encoding="utf-8", errors="ignore")
    elif not sys.stdin.isatty():
        text = sys.stdin.read()
    else:
        raise SystemExit("Error: provide --text, --file, or pipe to stdin")

    if not text.strip():
        raise SystemExit("Error: empty input")

    tags = tag_text(text, db_path, top_k=args.candidates)
    print(" | ".join(tags))
