# vault-tools

Semantic search and navigation tools for an Obsidian vault. Chunks markdown files, embeds them with a sentence transformer, stores everything in SQLite, and exposes it all via an MCP server for use in Claude Code.

---

## Contents

| File | Purpose |
|------|---------|
| `embed_vault.py` | Chunks and embeds all vault markdown files into a SQLite vector index |
| `insert_tags.py` | Writes auto-generated tags into each file's YAML frontmatter |
| `vault_mcp.py` | MCP server exposing semantic search and tag tools to Claude Code |
| `chatgpt_to_md.py` | Converts a ChatGPT data export into markdown files suitable for embedding |
| `chatgpt-md/` | Output directory for converted ChatGPT conversations |

---

## Quick start

### 1. Build the index

```bash
python3 embed_vault.py --vault /path/to/vault
```

This creates `<vault>/.vault-index/vault.db` containing all chunks and their 384-dimensional embeddings (`all-MiniLM-L6-v2`, L2-normalised).

By default, this runs in **incremental mode** — only new or modified files are re-embedded, and chunks for deleted files are removed. Use `--full` for a complete rebuild:

```bash
python3 embed_vault.py --vault /path/to/vault --full
```

**Runtime:** ~2–5 minutes for a full rebuild on Apple Silicon MPS (~300 files). Incremental updates with a few changed files take seconds.

**Chunk modes:**
- `sliding` (default) — overlapping 1800-char windows with 200-char overlap
- `delimiter` — splits on `<!-- chunk -->` markers (for `chatgpt_to_md.py` output)

```bash
# Use delimiter mode for ChatGPT-converted files
python3 embed_vault.py --vault /path/to/vault --chunk-mode delimiter
```

### 2. (Optional) Insert tags into vault files

Tags are derived separately via clustering (UMAP + HDBSCAN + Ward linkage) and stored in the `tags` and `chunk_tags` tables in `vault.db`. Once those tables exist:

```bash
python3 insert_tags.py --vault /path/to/vault              # dry run
python3 insert_tags.py --vault /path/to/vault --apply       # write files
```

This merges a `tags:` block into each file's YAML frontmatter, with broad tags first then specific tags, ordered by relevance (chunk count).

### 3. Register the MCP server

#### Claude Code

Register globally so it's available in every session:

```bash
claude mcp add --transport stdio --scope user vault -- python3 /path/to/vault-tools/vault_mcp.py /path/to/vault
```

This writes the entry to `~/.claude.json`. Restart Claude Code to pick it up. To verify:

```bash
claude mcp list
```

#### Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`.

**Important:** Claude Desktop doesn't inherit shell environment (pyenv, nvm, etc.), so use the absolute path to the Python binary that has the dependencies. Find it with:

```bash
python3 -c "import numpy, sentence_transformers, mcp; print(__import__('sys').executable)"
```

```json
{
  "mcpServers": {
    "vault": {
      "command": "/absolute/path/to/python3",
      "args": ["/path/to/vault-tools/vault_mcp.py", "/path/to/vault"],
      "env": {
        "PYTHONUNBUFFERED": "1"
      }
    }
  }
}
```

Fully quit and reopen Claude Desktop to pick it up. Check `~/Library/Logs/Claude/mcp.log` if it fails to connect.

#### Manual testing

```bash
python3 vault_mcp.py /path/to/vault
# Should print startup messages to stderr and hang waiting for stdin (Ctrl-C to exit)
```

---

## MCP tools

The server exposes five tools:

| Tool | Description |
|------|-------------|
| `semantic_search(query, top_k=5)` | Embed a query and return the most similar chunks |
| `search_by_tag(tag, limit=20)` | Find files carrying a given tag |
| `list_tags(tag_type="all")` | List all tags with descriptions (`broad`, `specific`, or `all`) |
| `get_related_notes(file_path, top_k=5)` | Find notes semantically similar to a given file |
| `read_note(file_path)` | Read the full markdown content of a vault note |

---

## ChatGPT export conversion

Converts a ChatGPT data export (the `conversations-*.json` shards) into one markdown file per conversation, with `<!-- chunk -->` delimiters between exchanges:

```bash
python3 chatgpt_to_md.py --export ~/Documents/chatgpt-export --out chatgpt-md
```

The output files can then be placed in the vault and indexed with `--chunk-mode delimiter`:

```bash
python3 embed_vault.py --vault /path/to/vault --chunk-mode delimiter
```

---

## Database schema

The index lives at `<vault>/.vault-index/vault.db`:

```sql
CREATE TABLE chunks (
    id          INTEGER PRIMARY KEY,
    file_path   TEXT,       -- relative to vault root
    chunk_index INTEGER,    -- 0-based position within the file
    text        TEXT,       -- raw chunk text (~1800 chars)
    embedding   BLOB        -- float32[384], L2-normalised
);

CREATE TABLE file_meta (
    file_path   TEXT PRIMARY KEY,   -- relative to vault root
    mtime       REAL,               -- file modification time (os.stat st_mtime)
    chunk_mode  TEXT                 -- 'sliding' or 'delimiter'
);

CREATE TABLE tags (
    name        TEXT PRIMARY KEY,
    type        TEXT,               -- "broad" or "specific"
    description TEXT
);

CREATE TABLE chunk_tags (
    chunk_id  INTEGER REFERENCES chunks(id),
    tag       TEXT    REFERENCES tags(name),
    PRIMARY KEY (chunk_id, tag)
);

CREATE VIEW file_tags AS
SELECT c.file_path, t.type AS tag_type, ct.tag, COUNT(*) AS chunk_count
FROM chunk_tags ct
JOIN chunks c ON ct.chunk_id = c.id
JOIN tags t   ON ct.tag = t.name
GROUP BY c.file_path, ct.tag, t.type
ORDER BY c.file_path, t.type, chunk_count DESC;
```

Embeddings are raw binary blobs — deserialise with `np.frombuffer(blob, dtype=np.float32)`. Since all vectors are L2-normalised, cosine similarity = dot product.

---

## Dependencies

- `sentence-transformers`, `torch`, `numpy` — for embedding
- `mcp[cli]` — for the MCP server (`pip3 install "mcp[cli]"`)

---

## Automatic index updates

The MCP server runs an incremental reindex every time it starts up (i.e. every time Claude Code launches). It reuses the sentence transformer model it already loads, so this adds minimal startup time — typically seconds if only a few files have changed.

No cron jobs, filesystem watchers, or manual re-runs needed. Just edit your vault and the next Claude Code session will have up-to-date embeddings.

For a manual reindex: `python3 embed_vault.py --vault /path/to/vault`

---

## Maintenance

**To re-derive tags:** tag derivation (UMAP + HDBSCAN + Ward clustering + manual naming) was a one-time semi-manual process. To redo it from scratch, rebuild chunks, run dimensionality reduction and clustering, populate the `tags` and `chunk_tags` tables, then run `insert_tags.py --apply`.
