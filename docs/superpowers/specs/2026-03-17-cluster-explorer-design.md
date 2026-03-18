# Cluster Explorer Design

## Problem

We have a tag clustering pipeline that produces named hierarchical clusters in SQLite, but no way to browse the results beyond raw SQL queries. We need a visual explorer to navigate the cluster tree, inspect tags, and read the source files.

## Architecture

Two files in vault-tools:

- **`cluster_server.py`** (~200 lines) — stdlib-only Python HTTP server. Serves the frontend, exposes a JSON API, reads from SQLite and the vault filesystem.
- **`cluster-explorer.html`** — single self-contained HTML file. Vanilla JS, inline CSS. Dark theme with neon accents.

No npm, no build step, no external dependencies.

### CLI

```
python3 cluster_server.py
python3 cluster_server.py --port 8080
python3 cluster_server.py --tag-db /custom/path/vital_articles.db
```

The tag DB defaults to `vital_articles.db` in the same directory as the script. Corpora are managed within the app, not via CLI flags.

## Corpus Management

Corpora are persisted in `~/.vault-tools/corpora.json`:

```json
{
  "tag_db": "/abs/path/to/vital_articles.db",
  "corpora": [
    {
      "id": "chatgpt",
      "name": "ChatGPT",
      "vault_db": "/abs/path/to/chatgpt-md/.vault-index/vault.db",
      "vault_dir": "/abs/path/to/chatgpt-md"
    }
  ]
}
```

- "Add Corpus" button in the app header — user provides a name, vault DB path, and vault directory path
- Server validates paths exist and that the vault DB has a `clusters` table
- Dropdown in the header to switch between corpora
- First corpus is selected by default

## API Endpoints

| Method | Path | Returns |
|--------|------|---------|
| GET | `/` | Serves `cluster-explorer.html` |
| GET | `/api/corpora` | List of configured corpora |
| POST | `/api/corpora` | Add corpus (body: `{name, vault_db, vault_dir}`) |
| DELETE | `/api/corpora/<id>` | Remove corpus |
| GET | `/api/<corpus_id>/clusters` | Full cluster tree (all depths, names, tag counts, parent_id) |
| GET | `/api/<corpus_id>/cluster/<cluster_id>` | Cluster detail: description, tags (with weights), file list (path + first 200 chars) |
| GET | `/api/<corpus_id>/file/<path>` | Raw markdown content of a vault file |

All cluster endpoints use the latest `cluster_versions` version.

## Frontend

### Layout

```
┌──────────────────────────────────────────────────────────┐
│  [Corpus Dropdown ▾]              [+ Add Corpus]         │
├──────────────────┬───────────────────────────────────────┤
│                  │                                       │
│  Cluster Tree    │  Cluster Detail                       │
│                  │                                       │
│  ▸ Technology    │  Name: Backend Infrastructure         │
│    Infrastructure│  Description: ...                     │
│  ▾ Geopolitics   │                                       │
│    ▸ Middle East │  Tags (20):                           │
│    ▸ Cold War    │  ┌─────────────┬────────┐             │
│    ▸ South Africa│  │ Tag         │ Weight │             │
│    ...           │  ├─────────────┼────────┤             │
│                  │  │ Database    │ 4805   │             │
│                  │  │ Pipeline    │ 4432   │             │
│                  │  └─────────────┴────────┘             │
│                  │                                       │
│                  │  Files (15):                           │
│                  │  📄 2025-01-17 - Prisma SQL Error.md  │
│                  │  📄 2025-03-05 - CE Game Theory.md    │
│                  │  ...                                   │
│                  │                                       │
└──────────────────┴───────────────────────────────────────┘
```

### Cluster Tree (left panel)

- Depth 0 as top-level expandable items
- Depth 1 nested under parents, depth 2 nested under depth 1
- Each node shows: name, tag count in parentheses
- Click to select → loads detail in main panel
- Expand/collapse with arrow icons

### Cluster Detail (main panel)

- Cluster name (large), description
- Child clusters listed if present (clickable, navigates tree)
- Tag table sorted by weight descending, showing top 50
- File list: files whose chunks overlap most with this cluster's tags. Show file path and first 200 chars as preview. Click opens modal.

File list query: join `chunk_tags` with `cluster_tags` on `tag_title`, group by `file_path`, order by overlap count descending, limit 50.

### File Modal

- Full-screen overlay, dark background
- File path as title
- Raw markdown rendered as HTML (basic rendering: headers, bold, italic, lists, code blocks, links, blockquotes)
- Close with X button or Escape key
- Simple client-side markdown rendering — regex-based, no library needed. Doesn't need to be perfect.

### Visual Theme

Dark background (#0a0a0f), neon accent colors:
- Primary: cyan/electric blue (#00f0ff)
- Secondary: magenta/pink (#ff00aa)
- Tertiary: electric green (#00ff88)
- Text: light grey (#e0e0e0)
- Borders/dividers: dark grey (#1a1a2e)
- Hover states: glow effects using box-shadow with neon colors

Monospace font for tags and file paths. Sans-serif for headings and descriptions.

## Not Building

- Search/filter (can add later)
- Editing clusters or tags
- Authentication
- Persistence of UI state (selected cluster, expanded nodes)

## Files

| File | Action |
|------|--------|
| `cluster_server.py` | Create |
| `cluster-explorer.html` | Create |
