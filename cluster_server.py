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
    except sqlite3.DatabaseError:
        return f"No clusters table in {vault_db}"
    return None


class Handler(SimpleHTTPRequestHandler):
    config: dict = {}
    tag_db: str = ""

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

        elif path.startswith("/api/") and path.endswith("/cluster-graph"):
            corpus_id = path.split("/")[2]
            self._handle_cluster_graph(corpus_id)

        elif path.startswith("/api/") and path.endswith("/clusters"):
            corpus_id = path.split("/")[2]
            self._handle_clusters(corpus_id)

        elif "/cluster/" in path:
            parts = path.split("/")
            corpus_id = parts[2]
            cluster_id = unquote("/".join(parts[4:]))
            self._handle_cluster_detail(corpus_id, cluster_id)

        elif "/file/" in path:
            parts = path.split("/file/", 1)
            corpus_id = parts[0].split("/")[2]
            file_path = unquote(parts[1])
            self._handle_file(corpus_id, file_path)

        else:
            self.send_error(404)

    def do_POST(self):
        """Handle POST requests to create a new corpus."""
        path = urlparse(self.path).path
        if path == "/api/corpora":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            name = body.get("name", "").strip()
            vault_db = body.get("vault_db", "").strip()
            vault_dir = body.get("vault_dir", "").strip()

            if not name or not vault_db or not vault_dir:
                self._json_response(
                    {"error": "name, vault_db, vault_dir required"},
                    400,
                )
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
        """Handle DELETE requests to remove a corpus."""
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

    def _handle_cluster_graph(self, corpus_id: str):
        """Return nodes + inter-cluster edges for the constellation view."""
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
            self._json_response({"nodes": [], "edges": []})
            return

        # Nodes
        rows = con.execute("""
            SELECT cluster_id, depth, parent_id, name, description, tag_count
            FROM clusters WHERE version_id = ?
            ORDER BY depth, cluster_id
        """, (version_id,)).fetchall()
        nodes = [
            {"cluster_id": r[0], "depth": r[1], "parent_id": r[2],
             "name": r[3], "description": r[4], "tag_count": r[5]}
            for r in rows
        ]

        # Inter-cluster edges: aggregate co-occurrence between tags
        # in different clusters at depth 0 and 1
        edges = []
        for depth in (0, 1):
            depth_clusters = [n for n in nodes if n["depth"] == depth]
            if len(depth_clusters) < 2:
                continue

            # Build tag→cluster_id mapping for this depth
            tag_to_cluster = {}
            for cl in depth_clusters:
                cl_tags = con.execute(
                    "SELECT tag_title FROM cluster_tags "
                    "WHERE version_id = ? AND cluster_id = ?",
                    (version_id, cl["cluster_id"]),
                ).fetchall()
                for (tag,) in cl_tags:
                    tag_to_cluster[tag] = cl["cluster_id"]

            # Count cross-cluster co-occurrences from chunk_tags
            chunk_tags = con.execute(
                "SELECT chunk_id, tag_title FROM chunk_tags ORDER BY chunk_id"
            ).fetchall()

            from itertools import groupby
            pair_weights: dict[tuple[str, str], int] = {}
            for _chunk_id, group in groupby(chunk_tags, key=lambda r: r[0]):
                tags = [r[1] for r in group]
                # Map tags to their clusters at this depth
                cluster_ids = set()
                for t in tags:
                    if t in tag_to_cluster:
                        cluster_ids.add(tag_to_cluster[t])
                # Count pairs
                for c1, c2 in sorted(
                    {(a, b) if a < b else (b, a)
                     for a in cluster_ids for b in cluster_ids if a != b}
                ):
                    pair_weights[(c1, c2)] = pair_weights.get((c1, c2), 0) + 1

            for (c1, c2), weight in pair_weights.items():
                edges.append({
                    "source": c1, "target": c2,
                    "weight": weight, "depth": depth,
                })

        con.close()
        self._json_response({"nodes": nodes, "edges": edges})

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

        tags = con.execute("""
            SELECT tag_title, weight FROM cluster_tags
            WHERE version_id = ? AND cluster_id = ?
            ORDER BY weight DESC LIMIT 50
        """, (version_id, cluster_id)).fetchall()

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

        info = con.execute("""
            SELECT name, description FROM clusters
            WHERE version_id = ? AND cluster_id = ?
        """, (version_id, cluster_id)).fetchone()

        con.close()
        self._json_response({
            "name": info[0] if info else None,
            "description": info[1] if info else None,
            "tags": [{"title": t[0], "weight": t[1]} for t in tags],
            "files": [
                {"path": f[0], "overlap": f[1], "excerpt": f[2]}
                for f in files
            ],
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
        try:
            full_path.resolve().relative_to(
                Path(corpus["vault_dir"]).resolve()
            )
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

    def log_message(self, format, *args):  # pylint: disable=redefined-builtin
        pass


def main():
    """Start the Cluster Explorer HTTP server."""
    parser = argparse.ArgumentParser(description="Cluster Explorer server")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--tag-db", type=Path,
        default=SCRIPT_DIR / "vital_articles.db",
    )
    args = parser.parse_args()

    Handler.config = _load_config()
    Handler.tag_db = str(args.tag_db.resolve())
    Handler.config["tag_db"] = Handler.tag_db

    print(f"Cluster Explorer: http://localhost:{args.port}", file=sys.stderr)
    HTTPServer(("localhost", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
