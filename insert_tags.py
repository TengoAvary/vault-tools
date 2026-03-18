#!/usr/bin/env python3
"""
Insert YAML frontmatter tags into vault markdown files
based on vault.db.

For each file in the file_tags view:
  - Collects all tags (broad first, then specific,
    ordered by chunk_count desc)
  - Strips any existing inline #Tag line at the top
  - If the file already has YAML frontmatter
    (starts with ---), merges tags into it
  - Otherwise, prepends a new frontmatter block

Usage:
    python3 insert_tags.py --vault /path/to/vault
    python3 insert_tags.py --vault /path/to/vault \
        --apply
"""

import argparse
import re
import sqlite3
from collections import defaultdict
from pathlib import Path

# Matches a line that is ONLY inline Obsidian tags
# e.g. "#Philosophy #AI"
# Allows optional wikilinks like [[VPD]] mixed in
INLINE_TAG_LINE = re.compile(
    r"^(?:#[A-Za-z][A-Za-z0-9_/-]*"
    r"(?:\s+(?:#[A-Za-z][A-Za-z0-9_/-]*"
    r"|\[\[[^\]]+\]\]))*)\s*$"
)


def get_file_tags(
    db_path: Path,
) -> dict[str, list[dict]]:
    """Return {file_path: [{tag, tag_type, ...}]}."""
    con = sqlite3.connect(db_path)
    rows = con.execute(
        "SELECT file_path, tag, tag_type, chunk_count"
        " FROM file_tags ORDER BY file_path"
    ).fetchall()
    con.close()

    result: dict[str, list[dict]] = defaultdict(list)
    for file_path, tag, tag_type, chunk_count in rows:
        result[file_path].append(
            {
                "tag": tag,
                "tag_type": tag_type,
                "chunk_count": chunk_count,
            }
        )
    return dict(result)


def sort_tags(tag_entries: list[dict]) -> list[str]:
    """Sort: broad first (by count), then specific."""
    broad = sorted(
        [t for t in tag_entries
         if t["tag_type"] == "broad"],
        key=lambda t: -t["chunk_count"],
    )
    specific = sorted(
        [t for t in tag_entries
         if t["tag_type"] == "specific"],
        key=lambda t: -t["chunk_count"],
    )
    return [t["tag"] for t in broad + specific]


def build_frontmatter_block(tags: list[str]) -> str:
    """Build a YAML frontmatter string with tags."""
    lines = ["---", "tags:"]
    for tag in tags:
        lines.append(f"  - {tag}")
    lines.append("---")
    return "\n".join(lines)


def strip_inline_tags(content: str) -> str:
    """Remove a leading inline #Tag line."""
    lines = content.split("\n")
    if lines and INLINE_TAG_LINE.match(lines[0]):
        lines = lines[1:]
        if lines and lines[0].strip() == "":
            lines = lines[1:]
    return "\n".join(lines)


def merge_tags_into_frontmatter(
    content: str, tags: list[str],
) -> str:
    """
    If the file starts with ---, parse the frontmatter,
    add/replace tags, and return the updated content.
    """
    if not content.startswith("---"):
        raise ValueError(
            "Content does not start with frontmatter"
        )

    end_idx = content.index("---", 3)
    fm_body = content[3:end_idx].strip()
    rest = content[end_idx + 3:]

    fm_lines = fm_body.split("\n")
    new_fm_lines = []
    skip = False
    for line in fm_lines:
        if line.strip().startswith("tags:"):
            skip = True
            continue
        if skip and line.startswith("  - "):
            continue
        skip = False
        new_fm_lines.append(line)

    new_fm_lines.append("tags:")
    for tag in tags:
        new_fm_lines.append(f"  - {tag}")

    return (
        "---\n"
        + "\n".join(new_fm_lines)
        + "\n---"
        + rest
    )


def process_file(
    file_path: Path, tags: list[str], apply: bool,
) -> str:
    """Process a single file. Returns a status string."""
    if not file_path.exists():
        return f"SKIP (not found): {file_path}"

    content = file_path.read_text(errors="replace")
    original = content

    if content.startswith("---"):
        new_content = merge_tags_into_frontmatter(
            content, tags,
        )
        action = "merged into existing frontmatter"
    else:
        content = strip_inline_tags(content)
        fm = build_frontmatter_block(tags)
        if content and not content.startswith("\n"):
            new_content = fm + "\n\n" + content
        else:
            new_content = fm + "\n" + content
        action = "added new frontmatter"

    if new_content == original:
        return f"NO CHANGE: {file_path}"

    if apply:
        file_path.write_text(new_content)
        return f"WRITTEN ({action}): {file_path}"
    else:
        preview = new_content.split("\n")[:8]
        return (
            f"WOULD WRITE ({action}): {file_path}\n  "
            + "\n  ".join(preview)
        )


def main():
    """Insert semantic tags into vault Markdown files."""
    parser = argparse.ArgumentParser(
        description="Insert tags into vault files",
    )
    parser.add_argument(
        "--vault", required=True, type=Path,
        help="Path to the Obsidian vault root",
    )
    parser.add_argument(
        "--apply", action="store_true",
        help=(
            "Actually write files"
            " (default is dry run)"
        ),
    )
    args = parser.parse_args()

    vault_root = args.vault.resolve()
    if not vault_root.is_dir():
        raise SystemExit(
            f"Error: '{vault_root}' is not a directory"
        )

    db_path = (
        vault_root / ".vault-index" / "vault.db"
    )
    if not db_path.exists():
        raise SystemExit(
            f"No index found at {db_path}\n"
            "Run: python3 embed_vault.py"
            f" --vault '{vault_root}' first."
        )

    file_tags = get_file_tags(db_path)
    print(
        f"Found tags for {len(file_tags)}"
        f" files in {db_path}"
    )
    mode = "APPLY" if args.apply else "DRY RUN"
    print(f"Mode: {mode}\n")

    stats = {"written": 0, "skipped": 0, "no_change": 0}

    for rel_path, tag_entries in sorted(
        file_tags.items()
    ):
        file_path = vault_root / rel_path
        tags = sort_tags(tag_entries)
        result = process_file(
            file_path, tags, args.apply,
        )
        print(result)

        if "WRITTEN" in result or "WOULD WRITE" in result:
            stats["written"] += 1
        elif "SKIP" in result:
            stats["skipped"] += 1
        else:
            stats["no_change"] += 1

    w = stats["written"]
    s = stats["skipped"]
    nc = stats["no_change"]
    label = "written" if args.apply else "to write"
    print(
        f"\nSummary: {w} files {label},"
        f" {s} skipped, {nc} unchanged"
    )


if __name__ == "__main__":
    main()
