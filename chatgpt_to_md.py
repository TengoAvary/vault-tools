#!/usr/bin/env python3
"""
chatgpt_to_md.py — convert a ChatGPT data export into
markdown files suitable for embedding with embed_vault.py.

Reads all conversations-*.json shards from an export
directory, walks each conversation's message tree, and
writes one .md file per conversation with
<!-- chunk --> delimiters between user/assistant
exchanges.

Usage:
    python3 chatgpt_to_md.py \
        --export ~/Documents/chatgpt-export \
        --out ~/Documents/chatgpt-md

Output structure:
    <out>/2024-03-04 - Django.md
    <out>/2024-03-04 - Water Bill Check.md
    ...
"""

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

# Minimum chunk length (chars) — exchanges shorter than
# this get merged with the next one.
MIN_CHUNK_CHARS = 200

# Maximum chunk length — long assistant replies get split.
MAX_CHUNK_CHARS = 3000

CHUNK_DELIMITER = "\n\n<!-- chunk -->\n\n"


def load_conversations(export_dir: Path) -> list[dict]:
    """Load and concatenate all conversation shards."""
    shards = sorted(
        export_dir.glob("conversations-*.json")
    )
    if not shards:
        raise SystemExit(
            "No conversations-*.json files"
            f" found in {export_dir}"
        )

    conversations = []
    for shard in shards:
        with open(shard, encoding="utf-8") as f:
            conversations.extend(json.load(f))

    print(
        f"Loaded {len(conversations)} conversations"
        f" from {len(shards)} shards"
    )
    return conversations


def walk_linear_path(
    mapping: dict, current_node: str,
) -> list[dict]:
    """
    Walk from current_node back to root via parent
    pointers, then reverse to get the linear
    conversation path.  This gives us the "active"
    branch (the one the user last interacted with).
    """
    path = []
    node_id = current_node
    while node_id is not None:
        node = mapping.get(node_id)
        if node is None:
            break
        path.append(node)
        node_id = node.get("parent")
    path.reverse()
    return path


def extract_text(message: dict | None) -> str | None:
    """Extract displayable text from a message node."""
    if message is None:
        return None

    content = message.get("content", {})
    content_type = content.get("content_type")

    if content_type == "text":
        parts = content.get("parts", [])
        text_parts = [
            p for p in parts
            if isinstance(p, str) and p.strip()
        ]
        return (
            "\n".join(text_parts)
            if text_parts
            else None
        )

    if content_type == "multimodal_text":
        parts = content.get("parts", [])
        text_parts = []
        for p in parts:
            if isinstance(p, str) and p.strip():
                text_parts.append(p)
            elif (
                isinstance(p, dict)
                and p.get("content_type")
                == "audio_transcription"
            ):
                t = p.get("text", "").strip()
                if t:
                    text_parts.append(t)
        return (
            "\n".join(text_parts)
            if text_parts
            else None
        )

    if content_type == "code":
        text = content.get("text", "")
        lang = content.get("language", "")
        return (
            f"```{lang}\n{text}\n```"
            if text.strip()
            else None
        )

    if content_type == "execution_output":
        text = content.get("text", "")
        return (
            f"```\n{text}\n```"
            if text.strip()
            else None
        )

    # Skip system_error, tether_browsing_display, etc.
    return None


def build_exchanges(path: list[dict]) -> list[str]:
    """
    Walk the linear message path and group into
    user/assistant exchanges.  Each exchange is a
    string like:

        **User:** question text

        **Assistant:** response text

    Tool calls and system messages are skipped.
    """
    exchanges: list[str] = []
    current_exchange_parts: list[str] = []

    for node in path:
        msg = node.get("message")
        if msg is None:
            continue

        role = msg["author"]["role"]
        text = extract_text(msg)
        if text is None:
            continue

        if role == "user":
            # If we have a pending exchange, flush it
            if current_exchange_parts:
                exchanges.append(
                    "\n\n".join(current_exchange_parts)
                )
                current_exchange_parts = []
            current_exchange_parts.append(
                f"**User:** {text}"
            )

        elif role == "assistant":
            current_exchange_parts.append(
                f"**Assistant:** {text}"
            )

        # Skip tool, system roles

    # Flush last exchange
    if current_exchange_parts:
        exchanges.append(
            "\n\n".join(current_exchange_parts)
        )

    return exchanges


def split_long_exchange(exchange: str) -> list[str]:
    """Split an exchange exceeding MAX_CHUNK_CHARS."""
    if len(exchange) <= MAX_CHUNK_CHARS:
        return [exchange]

    pieces = []
    remaining = exchange

    while len(remaining) > MAX_CHUNK_CHARS:
        # Prefer paragraph break
        split_at = remaining.rfind(
            "\n\n", 0, MAX_CHUNK_CHARS,
        )
        if split_at < MIN_CHUNK_CHARS:
            split_at = remaining.rfind(
                ". ", 0, MAX_CHUNK_CHARS,
            )
        if split_at < MIN_CHUNK_CHARS:
            split_at = MAX_CHUNK_CHARS

        pieces.append(remaining[:split_at].rstrip())
        remaining = remaining[split_at:].lstrip()

    if remaining.strip():
        pieces.append(remaining.strip())

    return pieces


def merge_short_exchanges(
    exchanges: list[str],
) -> list[str]:
    """Merge consecutive short exchanges."""
    if not exchanges:
        return []

    merged = []
    buffer = exchanges[0]

    for exchange in exchanges[1:]:
        if len(buffer) < MIN_CHUNK_CHARS:
            buffer += "\n\n" + exchange
        else:
            merged.append(buffer)
            buffer = exchange

    merged.append(buffer)
    return merged


def sanitize_filename(name: str) -> str:
    """Make a string safe for use as a filename."""
    name = re.sub(r'[<>:"/\\|?*]', "", name)
    name = re.sub(r"\s+", " ", name).strip()
    if len(name) > 80:
        name = name[:80].rstrip()
    return name or "Untitled"


def format_date(timestamp: float | None) -> str:
    """Format a Unix timestamp as YYYY-MM-DD."""
    if timestamp is None:
        return "unknown-date"
    try:
        return datetime.fromtimestamp(
            timestamp, tz=timezone.utc,
        ).strftime("%Y-%m-%d")
    except (OSError, ValueError):
        return "unknown-date"


def convert_conversation(
    conv: dict,
) -> tuple[str, str] | None:
    """
    Convert a single conversation to
    (filename, markdown_content).
    Returns None if no meaningful content.
    """
    title = conv.get("title") or "Untitled"
    date = format_date(conv.get("create_time"))
    mapping = conv.get("mapping", {})
    current_node = conv.get("current_node")

    if not mapping or not current_node:
        return None

    path = walk_linear_path(mapping, current_node)
    exchanges = build_exchanges(path)

    if not exchanges:
        return None

    # Split long exchanges, then merge short ones
    split = []
    for ex in exchanges:
        split.extend(split_long_exchange(ex))
    chunks = merge_short_exchanges(split)

    if not chunks:
        return None

    # Build the markdown file
    header = f"# {title}\n\n*{date}*"
    body = CHUNK_DELIMITER.join(chunks)
    content = header + CHUNK_DELIMITER + body

    filename = (
        f"{date} - {sanitize_filename(title)}.md"
    )
    return filename, content


def main():
    """Convert a ChatGPT JSON export to Markdown files."""
    parser = argparse.ArgumentParser(
        description=(
            "Convert ChatGPT export to markdown"
            " files for semantic search"
        )
    )
    parser.add_argument(
        "--export", required=True, type=Path,
        help="Path to the ChatGPT export directory",
    )
    parser.add_argument(
        "--out", required=True, type=Path,
        help="Output directory for markdown files",
    )
    args = parser.parse_args()

    export_dir = args.export.resolve()
    out_dir = args.out.resolve()

    if not export_dir.is_dir():
        raise SystemExit(
            f"Error: '{export_dir}' is not a directory"
        )

    out_dir.mkdir(parents=True, exist_ok=True)

    conversations = load_conversations(export_dir)

    written = 0
    skipped = 0
    seen_filenames: dict[str, int] = {}

    for conv in conversations:
        result = convert_conversation(conv)
        if result is None:
            skipped += 1
            continue

        filename, content = result

        # Handle duplicate filenames
        if filename in seen_filenames:
            seen_filenames[filename] += 1
            stem, ext = filename.rsplit(".", 1)
            filename = (
                f"{stem}"
                f" ({seen_filenames[filename]})"
                f".{ext}"
            )
        else:
            seen_filenames[filename] = 0

        (out_dir / filename).write_text(
            content, encoding="utf-8",
        )
        written += 1

    print(f"\nDone. {written} files written to {out_dir}")
    if skipped:
        print(
            f"  ({skipped} conversations skipped"
            " — no text content)"
        )


if __name__ == "__main__":
    main()
