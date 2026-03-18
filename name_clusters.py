"""
name_clusters.py — name tag clusters using Claude CLI (Haiku).

Usage:
    python3 name_clusters.py --vault-db chatgpt-md/.vault-index/vault.db
    python3 name_clusters.py --vault-db ... --depth 0

Reads unnamed clusters from the latest version in vault.db, gathers tag names
and representative chunk excerpts, then spawns Claude CLI to generate a short
name and description for each cluster.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import sys
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
) -> tuple[list[str], list[str], list[str]]:
    """Get tag names, chunk excerpts, and child cluster names.

    Returns (tag_names, chunk_excerpts, child_names).
    """
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

    # Get named child clusters
    children = con.execute(
        """SELECT name FROM clusters
           WHERE version_id = ? AND parent_id = ?
           AND name IS NOT NULL
           ORDER BY tag_count DESC""",
        (version_id, cluster_id),
    ).fetchall()
    child_names = [r[0] for r in children]

    return tag_names, excerpts, child_names


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
            if isinstance(event, dict):
                if event.get("type") == "result":
                    return event.get("result", "")
    return ""


def _call_claude(prompt: str) -> str:
    """Call Claude CLI and return the result text."""
    env = {
        k: v for k, v in os.environ.items()
        if k != "CLAUDECODE"
    }

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
        check=False,
    )

    if result.returncode != 0:
        print(
            f"[name] claude error: {result.stderr}",
            file=sys.stderr,
        )
        return ""

    return _extract_result(result.stdout)


def _parse_name_response(text: str) -> tuple[str, str] | None:
    """Parse 'name | description' response.

    Returns None if format is invalid.
    """
    text = text.replace("**", "").replace("*", "").strip()
    if "|" not in text:
        return None
    parts = text.split("|", 1)
    name = parts[0].strip()
    desc = parts[1].strip()
    if not name or len(name) > 60:
        return None
    return name, desc


def name_cluster(
    tag_names: list[str],
    excerpts: list[str],
    child_names: list[str] | None = None,
) -> tuple[str, str]:
    """Use Claude CLI to name a cluster.

    Returns (name, description).
    """
    tags_str = ", ".join(tag_names)

    if child_names:
        prompt = (
            "Reply with EXACTLY this format, nothing else:"
            " name | description\n\n"
            "- name: 2-4 words naming the dominant theme\n"
            "- description: one sentence\n\n"
            "Name the cluster after the majority theme."
            " Ignore outlier sub-topics that don't fit."
            "\n\nSub-topics: " + ", ".join(child_names)
        )
    else:
        prompt = (
            "Reply with EXACTLY this format, nothing else:"
            " name | description\n\n"
            "- name: 2-4 words naming the topic cluster\n"
            "- description: one sentence\n\n"
            "Tags: " + tags_str
        )
        if excerpts:
            prompt += (
                "\n\nSample text:\n"
                + "\n---\n".join(excerpts)
            )

    # Try up to 2 times
    for attempt in range(2):
        result_text = _call_claude(prompt)
        if not result_text:
            continue
        parsed = _parse_name_response(result_text)
        if parsed:
            return parsed
        if attempt == 0:
            print(
                "[name]   retrying (bad format)…",
                file=sys.stderr,
            )

    return "Unknown", ""


def main():
    """Name tag clusters using Claude CLI (Haiku)."""
    parser = argparse.ArgumentParser(
        description=(
            "Name tag clusters using Claude CLI (Haiku)"
        )
    )
    parser.add_argument(
        "--vault-db", required=True, type=Path,
        help="Path to vault.db (where clusters live)",
    )
    parser.add_argument(
        "--depth", type=int, default=None,
        help=(
            "Only name clusters at this depth"
            " (default: all unnamed)"
        ),
    )
    parser.add_argument(
        "--bottom-up", action="store_true",
        help=(
            "Name deepest clusters first, then use"
            " child names to name parents"
        ),
    )
    args = parser.parse_args()

    vault_db = args.vault_db.resolve()
    if not vault_db.exists():
        raise SystemExit(f"Error: '{vault_db}' not found")

    con = sqlite3.connect(vault_db)
    version_id = con.execute(
        "SELECT MAX(version_id) FROM cluster_versions"
    ).fetchone()[0]

    if version_id is None:
        raise SystemExit(
            "Error: no cluster versions found"
            " — run cluster_tags.py first"
        )

    if args.bottom_up:
        # Get max depth, then name from deepest to shallowest
        max_depth = con.execute(
            "SELECT MAX(depth) FROM clusters"
            " WHERE version_id = ?",
            (version_id,),
        ).fetchone()[0]

        total = 0
        for d in range(max_depth, -1, -1):
            clusters = get_unnamed_clusters(
                con, version_id, depth=d,
            )
            if not clusters:
                continue
            total += len(clusters)
            print(
                f"[name] depth {d}: "
                f"{len(clusters)} clusters to name",
                file=sys.stderr,
            )

        if total == 0:
            print(
                "[name] all clusters already named",
                file=sys.stderr,
            )
            con.close()
            sys.exit(0)

        named = 0
        for d in range(max_depth, -1, -1):
            clusters = get_unnamed_clusters(
                con, version_id, depth=d,
            )
            for cluster in clusters:
                named += 1
                cid = cluster["cluster_id"]
                tag_names, excerpts, child_names = (
                    get_cluster_context(
                        con, version_id, cid,
                    )
                )

                print(
                    f"[name] [{named}/{total}]"
                    f" cluster {cid}"
                    f" ({cluster['tag_count']}"
                    f" tags, depth {d})...",
                    file=sys.stderr,
                )

                name, description = name_cluster(
                    tag_names, excerpts, child_names,
                )

                con.execute(
                    "UPDATE clusters"
                    " SET name = ?, description = ?"
                    " WHERE version_id = ?"
                    " AND cluster_id = ?",
                    (name, description, version_id, cid),
                )
                con.commit()

                print(
                    f"[name]   -> {name}",
                    file=sys.stderr,
                )
    else:
        clusters = get_unnamed_clusters(
            con, version_id, args.depth,
        )
        if not clusters:
            print(
                "[name] all clusters already named",
                file=sys.stderr,
            )
            con.close()
            sys.exit(0)

        print(
            f"[name] naming {len(clusters)}"
            f" clusters (version {version_id})...",
            file=sys.stderr,
        )

        for i, cluster in enumerate(clusters, 1):
            cid = cluster["cluster_id"]
            tag_names, excerpts, child_names = (
                get_cluster_context(
                    con, version_id, cid,
                )
            )

            print(
                f"[name] [{i}/{len(clusters)}]"
                f" cluster {cid}"
                f" ({cluster['tag_count']} tags,"
                f" depth {cluster['depth']})...",
                file=sys.stderr,
            )

            name, description = name_cluster(
                tag_names, excerpts, child_names,
            )

            con.execute(
                "UPDATE clusters"
                " SET name = ?, description = ?"
                " WHERE version_id = ?"
                " AND cluster_id = ?",
                (name, description, version_id, cid),
            )
            con.commit()

            print(
                f"[name]   -> {name}",
                file=sys.stderr,
            )

    con.close()
    print("[name] done", file=sys.stderr)


if __name__ == "__main__":
    main()
