# Cluster Explorer Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a browser-based explorer for navigating tag cluster hierarchies, viewing tags, and reading source vault files.

**Architecture:** A stdlib-only Python HTTP server (`cluster_server.py`) serves a single HTML file and exposes JSON API endpoints that query SQLite and read vault files on demand. Corpora are managed via the UI and persisted to `~/.vault-tools/corpora.json`.

**Tech Stack:** Python 3 (stdlib only: `http.server`, `sqlite3`, `json`, `pathlib`), vanilla JS, inline CSS

**Spec:** `docs/superpowers/specs/2026-03-17-cluster-explorer-design.md`

---

## File Structure

| File | Responsibility |
|------|---------------|
| `cluster_server.py` (create) | HTTP server, JSON API, corpus config management |
| `cluster-explorer.html` (create) | Frontend: cluster tree, detail panel, file modal, corpus switcher |

---

## Chunk 1: Server

### Task 1: Server skeleton with corpus management

**Files:**
- Create: `cluster_server.py`

- [ ] **Step 1: Create `cluster_server.py` with config management and basic HTTP server**

The server should:
- Load/save corpora config from `~/.vault-tools/corpora.json`
- Serve `cluster-explorer.html` at `/`
- Handle `/api/corpora` GET (list), POST (add), DELETE (remove)
- POST validates that `vault_db` exists and has a `clusters` table, and `vault_dir` exists
- Auto-generate an `id` from the name (slugified)
- Accept `--port` and `--tag-db` CLI flags via argparse

```python
"""
cluster_server.py — HTTP server for the cluster explorer UI.

Usage:
    python3 cluster_server.py
    python3 cluster_server.py --port 8080
    python3 cluster_server.py --tag-db /path/to/vital_articles.db
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, unquote

CONFIG_DIR = Path.home() / ".vault-tools"
CONFIG_FILE = CONFIG_DIR / "corpora.json"
SCRIPT_DIR = Path(__file__).resolve().parent


def _load_config() -> dict:
    if CONFIG_FILE.exists():
        return json.loads(CONFIG_FILE.read_text())
    return {"tag_db": str(SCRIPT_DIR / "vital_articles.db"), "corpora": []}


def _save_config(config: dict):
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(config, indent=2))


def _slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def _validate_corpus(vault_db: str, vault_dir: str) -> str | None:
    """Return error message or None if valid."""
    if not Path(vault_db).exists():
        return f"vault_db not found: {vault_db}"
    if not Path(vault_dir).is_dir():
        return f"vault_dir not found: {vault_dir}"
    try:
        con = sqlite3.connect(vault_db)
        con.execute("SELECT 1 FROM clusters LIMIT 1")
        con.close()
    except Exception:
        return f"No clusters table in {vault_db}"
    return None


class Handler(SimpleHTTPRequestHandler):
    config = None
    tag_db = None

    def do_GET(self):
        parsed = urlparse(self.path)
        path = unquote(parsed.path)

        if path == "/":
            html_path = SCRIPT_DIR / "cluster-explorer.html"
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(html_path.read_bytes())

        elif path == "/api/corpora":
            self._json_response(Handler.config["corpora"])

        elif path.startswith("/api/") and path.endswith("/clusters"):
            corpus_id = path.split("/")[2]
            self._handle_clusters(corpus_id)

        elif "/cluster/" in path:
            parts = path.split("/")
            corpus_id = parts[2]
            cluster_id = "/".join(parts[4:])  # handles dot-separated ids
            self._handle_cluster_detail(corpus_id, cluster_id)

        elif "/file/" in path:
            parts = path.split("/file/", 1)
            corpus_id = parts[0].split("/")[2]
            file_path = parts[1]
            self._handle_file(corpus_id, file_path)

        else:
            self.send_error(404)

    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/api/corpora":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            name = body.get("name", "").strip()
            vault_db = body.get("vault_db", "").strip()
            vault_dir = body.get("vault_dir", "").strip()

            if not name or not vault_db or not vault_dir:
                self._json_response({"error": "name, vault_db, vault_dir required"}, 400)
                return

            error = _validate_corpus(vault_db, vault_dir)
            if error:
                self._json_response({"error": error}, 400)
                return

            corpus = {"id": _slugify(name), "name": name,
                      "vault_db": vault_db, "vault_dir": vault_dir}
            Handler.config["corpora"].append(corpus)
            _save_config(Handler.config)
            self._json_response(corpus, 201)
        else:
            self.send_error(404)

    def do_DELETE(self):
        path = urlparse(self.path).path
        if path.startswith("/api/corpora/"):
            corpus_id = path.split("/")[-1]
            Handler.config["corpora"] = [
                c for c in Handler.config["corpora"] if c["id"] != corpus_id
            ]
            _save_config(Handler.config)
            self._json_response({"ok": True})
        else:
            self.send_error(404)

    def _get_corpus(self, corpus_id: str) -> dict | None:
        for c in Handler.config["corpora"]:
            if c["id"] == corpus_id:
                return c
        return None

    def _handle_clusters(self, corpus_id: str):
        corpus = self._get_corpus(corpus_id)
        if not corpus:
            self._json_response({"error": "corpus not found"}, 404)
            return
        con = sqlite3.connect(corpus["vault_db"])
        version_id = con.execute(
            "SELECT MAX(version_id) FROM cluster_versions"
        ).fetchone()[0]
        if version_id is None:
            con.close()
            self._json_response([])
            return
        rows = con.execute("""
            SELECT cluster_id, depth, parent_id, name, description, tag_count
            FROM clusters WHERE version_id = ?
            ORDER BY depth, cluster_id
        """, (version_id,)).fetchall()
        con.close()
        self._json_response([
            {"cluster_id": r[0], "depth": r[1], "parent_id": r[2],
             "name": r[3], "description": r[4], "tag_count": r[5]}
            for r in rows
        ])

    def _handle_cluster_detail(self, corpus_id: str, cluster_id: str):
        corpus = self._get_corpus(corpus_id)
        if not corpus:
            self._json_response({"error": "corpus not found"}, 404)
            return
        con = sqlite3.connect(corpus["vault_db"])
        version_id = con.execute(
            "SELECT MAX(version_id) FROM cluster_versions"
        ).fetchone()[0]

        # Tags
        tags = con.execute("""
            SELECT tag_title, weight FROM cluster_tags
            WHERE version_id = ? AND cluster_id = ?
            ORDER BY weight DESC LIMIT 50
        """, (version_id, cluster_id)).fetchall()

        # Files: chunks with most tag overlap
        files = con.execute("""
            SELECT c.file_path, COUNT(*) as overlap,
                   SUBSTR(c.text, 1, 200) as excerpt
            FROM chunk_tags ct
            JOIN chunks c ON c.id = ct.chunk_id
            WHERE ct.tag_title IN (
                SELECT tag_title FROM cluster_tags
                WHERE version_id = ? AND cluster_id = ?
            )
            GROUP BY c.file_path
            ORDER BY overlap DESC
            LIMIT 50
        """, (version_id, cluster_id)).fetchall()

        # Cluster info
        info = con.execute("""
            SELECT name, description FROM clusters
            WHERE version_id = ? AND cluster_id = ?
        """, (version_id, cluster_id)).fetchone()

        con.close()
        self._json_response({
            "name": info[0] if info else None,
            "description": info[1] if info else None,
            "tags": [{"title": t[0], "weight": t[1]} for t in tags],
            "files": [{"path": f[0], "overlap": f[1], "excerpt": f[2]} for f in files],
        })

    def _handle_file(self, corpus_id: str, file_path: str):
        corpus = self._get_corpus(corpus_id)
        if not corpus:
            self._json_response({"error": "corpus not found"}, 404)
            return
        full_path = Path(corpus["vault_dir"]) / file_path
        if not full_path.exists():
            self._json_response({"error": "file not found"}, 404)
            return
        # Security: ensure path is within vault_dir
        try:
            full_path.resolve().relative_to(Path(corpus["vault_dir"]).resolve())
        except ValueError:
            self._json_response({"error": "access denied"}, 403)
            return
        content = full_path.read_text(encoding="utf-8", errors="replace")
        self._json_response({"path": file_path, "content": content})

    def _json_response(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass  # Suppress request logging


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cluster Explorer server")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--tag-db", type=Path, default=SCRIPT_DIR / "vital_articles.db")
    args = parser.parse_args()

    Handler.config = _load_config()
    Handler.tag_db = str(args.tag_db.resolve())
    Handler.config["tag_db"] = Handler.tag_db

    print(f"Cluster Explorer: http://localhost:{args.port}", file=sys.stderr)
    HTTPServer(("localhost", args.port), Handler).serve_forever()
```

- [ ] **Step 2: Test server starts and API works**

Run:
```bash
python3 cluster_server.py &
sleep 1
curl -s http://localhost:8000/api/corpora | python3 -m json.tool
kill %1
```
Expected: empty array `[]`

- [ ] **Step 3: Commit**

```bash
git add cluster_server.py
git commit -m "feat: cluster explorer server with corpus management API"
```

---

## Chunk 2: Frontend

### Task 2: Cluster Explorer HTML

**Files:**
- Create: `cluster-explorer.html`

- [ ] **Step 1: Create `cluster-explorer.html`**

Single self-contained HTML file with:
- Neon dark theme (inline CSS)
- Header: corpus dropdown + "Add Corpus" button
- Left panel: expandable cluster tree (depth 0 → 1 → 2)
- Main panel: cluster detail (name, description, tags table, file list)
- Modal: full file content with basic markdown rendering
- Add Corpus modal: name, vault_db path, vault_dir path inputs
- All data fetched from `/api/` endpoints via `fetch()`

The HTML file will be large (~500 lines) since it contains all CSS and JS inline. Key sections:

**CSS:** Dark background `#0a0a0f`, neon accents (`#00f0ff` cyan, `#ff00aa` magenta, `#00ff88` green), glow hover effects, monospace for code/tags, sans-serif for text.

**Tree component:** Recursively builds DOM from cluster data. Click expands/collapses children. Click on a leaf or name loads detail.

**Detail panel:** Fetches `/api/<corpus>/cluster/<id>`, renders tags as a table with weight bars, files as a clickable list with excerpts.

**File modal:** Fetches `/api/<corpus>/file/<path>`, renders markdown with regex-based conversion (headers, bold, italic, lists, code blocks, blockquotes, links).

**Corpus modal:** Form with three inputs, POST to `/api/corpora`, refreshes dropdown on success.

- [ ] **Step 2: Test full flow**

Run:
```bash
python3 cluster_server.py &
```

Then:
1. Open `http://localhost:8000`
2. Click "Add Corpus" — add chatgpt-md with vault_db and vault_dir paths
3. Cluster tree should populate
4. Click a depth-0 cluster → expands children
5. Click a leaf cluster → detail panel shows tags and files
6. Click a file → modal shows rendered markdown
7. Switch corpus via dropdown (if multiple added)

```bash
kill %1
```

- [ ] **Step 3: Commit**

```bash
git add cluster-explorer.html
git commit -m "feat: cluster explorer frontend — neon dark theme with tree/detail/modal"
```
