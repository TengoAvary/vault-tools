"""
scrape_tags.py — scrape Wikipedia Vital Articles into a versioned SQLite database.

Usage:
    python3 scrape_tags.py --vault /path/to/vault           # all levels (1-4)
    python3 scrape_tags.py --vault /path/to/vault --level 4  # single level
    python3 scrape_tags.py --db /custom/path.db              # custom DB path

Fetches article lists from the MediaWiki API and stores them in
<vault>/.vault-index/vital_articles.db with full version history.

Stdlib only — no external dependencies.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
import time
import urllib.request
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

API_URL = "https://en.wikipedia.org/w/api.php"
REQUEST_DELAY = 1.0  # seconds between API calls

LEVEL_PAGES = {
    1: "Wikipedia:Vital articles/Level/1",
    2: "Wikipedia:Vital articles/Level/2",
    3: "Wikipedia:Vital articles/Level/3",
    4: "Wikipedia:Vital articles/Level/4",  # hub page
}


def fetch_wikitext(page_title: str) -> str:
    """Fetch raw wikitext for a Wikipedia page via the MediaWiki API."""
    params = urllib.parse.urlencode({
        "action": "parse",
        "page": page_title,
        "prop": "wikitext",
        "format": "json",
        "formatversion": "2",
    })
    url = f"{API_URL}?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": "vault-tools/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
    return data["parse"]["wikitext"]


def discover_level4_subpages(hub_wikitext: str) -> list[str]:
    """Extract Level 4 subpage titles from the hub page wikitext."""
    subpages = []
    for m in re.finditer(
        r"\[\[Wikipedia:Vital articles/Level/4/([^\]|]+)",
        hub_wikitext,
    ):
        subpage = m.group(1).strip()
        full_title = f"Wikipedia:Vital articles/Level/4/{subpage}"
        if full_title not in subpages:
            subpages.append(full_title)
    return subpages


def parse_vital_articles(
    wikitext: str, level: int, source_page: str
) -> list[dict]:
    """Parse wikitext to extract vital article entries with heading context."""
    articles = []
    heading_stack: list[tuple[int, str]] = []  # (depth, title)

    for line in wikitext.split("\n"):
        line = line.strip()

        # Track headings: = H1 =, == H2 ==, === H3 ===, etc.
        heading_match = re.match(r"^(={1,})\s*(.+?)\s*=+\s*$", line)
        if heading_match:
            depth = len(heading_match.group(1))
            # Skip page-level headings (depth 1) — they're just page titles
            if depth < 2:
                continue
            title = heading_match.group(2).strip()
            # Remove any wikilinks from heading text
            title = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", title)
            # Pop headings at same or deeper level
            heading_stack = [(d, t) for d, t in heading_stack if d < depth]
            heading_stack.append((depth, title))
            continue

        # Detect italic sub-labels as pseudo-headings (e.g. "* ''Biology''")
        italic_match = re.match(r"^[*#]+\s*'''?''(.+?)'''?''\s*$", line)
        if italic_match:
            label = italic_match.group(1).strip()
            # Treat as one level deeper than deepest current heading
            pseudo_depth = (heading_stack[-1][0] + 1) if heading_stack else 3
            heading_stack = [(d, t) for d, t in heading_stack if d < pseudo_depth]
            heading_stack.append((pseudo_depth, label))
            continue

        # Extract article links from bullet lines
        if not re.match(r"^[*#]", line):
            continue

        for link_match in re.finditer(r"\[\[([^\]|]+)(?:\|[^\]]*)?\]\]", line):
            title = link_match.group(1).strip()

            # Skip namespace links
            if ":" in title and title.split(":")[0] in (
                "Wikipedia", "Category", "File", "Template", "Help",
                "Portal", "Draft", "User", "Talk", "WP",
            ):
                continue

            # Build path from heading stack
            section = heading_stack[0][1] if heading_stack else ""
            subsection = heading_stack[1][1] if len(heading_stack) > 1 else None
            full_path = "/".join(t for _, t in heading_stack)

            articles.append({
                "level": level,
                "title": title,
                "section": section,
                "subsection": subsection,
                "full_path": full_path,
                "source_page": source_page,
            })

    return articles


def _ensure_schema(con: sqlite3.Connection):
    """Create tables and indexes if they don't exist."""
    con.execute("""
        CREATE TABLE IF NOT EXISTS scrape_versions (
            version_id   INTEGER PRIMARY KEY AUTOINCREMENT,
            scraped_at   TEXT NOT NULL,
            level_counts TEXT NOT NULL,
            duration_s   REAL
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS vital_articles (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            version_id   INTEGER NOT NULL REFERENCES scrape_versions(version_id),
            level        INTEGER NOT NULL,
            title        TEXT NOT NULL,
            section      TEXT NOT NULL,
            subsection   TEXT,
            full_path    TEXT NOT NULL,
            source_page  TEXT NOT NULL
        )
    """)
    con.execute("CREATE INDEX IF NOT EXISTS idx_va_version ON vital_articles(version_id)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_va_title ON vital_articles(title)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_va_version_level ON vital_articles(version_id, level)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_va_version_section ON vital_articles(version_id, section)")


def scrape_all(db_path: Path, levels: list[int] | None = None) -> dict:
    """Orchestrate scraping and store results. Returns level counts."""
    if levels is None:
        levels = [1, 2, 3, 4]

    t0 = time.time()
    all_articles: list[dict] = []

    for level in levels:
        page_title = LEVEL_PAGES[level]
        print(f"[scrape] fetching level {level}: {page_title}", file=sys.stderr)
        wikitext = fetch_wikitext(page_title)
        time.sleep(REQUEST_DELAY)

        if level == 4:
            # Level 4 hub page links to subpages
            subpages = discover_level4_subpages(wikitext)
            print(f"[scrape] level 4 has {len(subpages)} subpages", file=sys.stderr)

            # Also parse the hub page itself (may contain some articles)
            articles = parse_vital_articles(wikitext, level, page_title)
            all_articles.extend(articles)

            for subpage in subpages:
                print(f"[scrape]   fetching {subpage}", file=sys.stderr)
                sub_wikitext = fetch_wikitext(subpage)
                time.sleep(REQUEST_DELAY)
                articles = parse_vital_articles(sub_wikitext, level, subpage)
                all_articles.extend(articles)
        else:
            articles = parse_vital_articles(wikitext, level, page_title)
            all_articles.extend(articles)

    duration = time.time() - t0

    # Count per level
    level_counts = {}
    for a in all_articles:
        lvl = str(a["level"])
        level_counts[lvl] = level_counts.get(lvl, 0) + 1

    # Write to DB
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    _ensure_schema(con)

    con.execute(
        "INSERT INTO scrape_versions (scraped_at, level_counts, duration_s) VALUES (?, ?, ?)",
        (datetime.now(timezone.utc).isoformat(), json.dumps(level_counts), round(duration, 1)),
    )
    version_id = con.execute("SELECT last_insert_rowid()").fetchone()[0]

    con.executemany(
        """INSERT INTO vital_articles
           (version_id, level, title, section, subsection, full_path, source_page)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        [
            (version_id, a["level"], a["title"], a["section"],
             a["subsection"], a["full_path"], a["source_page"])
            for a in all_articles
        ],
    )

    con.commit()
    con.close()

    print(f"[scrape] done in {duration:.1f}s — version {version_id}", file=sys.stderr)
    for lvl in sorted(level_counts):
        print(f"[scrape]   level {lvl}: {level_counts[lvl]} articles", file=sys.stderr)
    print(f"[scrape] total: {len(all_articles)} articles → {db_path}", file=sys.stderr)

    return level_counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scrape Wikipedia Vital Articles into a versioned SQLite database"
    )
    parser.add_argument("--vault", type=Path, help="Path to the Obsidian vault root")
    parser.add_argument("--db", type=Path, help="Custom database path (overrides --vault)")
    parser.add_argument(
        "--level", type=int, choices=[1, 2, 3, 4],
        help="Scrape only this level (default: all)"
    )
    args = parser.parse_args()

    if args.db:
        db_path = args.db.resolve()
    elif args.vault:
        vault = args.vault.resolve()
        if not vault.is_dir():
            raise SystemExit(f"Error: '{vault}' is not a directory")
        db_path = vault / ".vault-index" / "vital_articles.db"
    else:
        raise SystemExit("Error: provide --vault or --db")

    levels = [args.level] if args.level else None
    scrape_all(db_path, levels)
