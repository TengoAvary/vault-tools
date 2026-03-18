# vault-tools

Semantic search and navigation tools for Obsidian vaults, exposed as an MCP server for Claude Code and Claude Desktop.

## Setup from scratch

When a user asks to set up vault-tools, follow these steps in order:

### 1. Ask for the vault location

Ask the user: **"Where is your Obsidian vault directory?"**

Wait for a response. The path must be an existing directory. Verify it exists before proceeding. Store it as `VAULT_PATH` for the remaining steps.

### 2. Ask which clients to set up

Ask the user: **"Do you want this set up for Claude Code, Claude Desktop, or both?"**

### 3. Install Python dependencies

```bash
pip3 install "mcp[cli]" sentence-transformers torch numpy
```

If the user already has these installed, this will be a no-op.

### 4. Build the initial index

Run the embedder to create the SQLite vector index:

```bash
python3 <this-repo>/embed_vault.py --vault VAULT_PATH --full
```

This creates `VAULT_PATH/.vault-index/vault.db`. It takes ~2-5 minutes on Apple Silicon for ~300 files. Let the user know it will take a few minutes.

### 5. Register the MCP server

Replace `<this-repo>` with the absolute path to this repository in all commands below.

#### For Claude Code

```bash
claude mcp add --transport stdio --scope user vault -- python3 <this-repo>/vault_mcp.py VAULT_PATH
```

This registers the server in `~/.claude.json` so it's available in all projects.

Verify with:

```bash
claude mcp list
```

The `vault` entry should show `Connected`.

#### For Claude Desktop

Add the server to `~/Library/Application Support/Claude/claude_desktop_config.json`. If the file doesn't exist or is empty, create it. Merge the `vault` entry into any existing `mcpServers` object — do not overwrite other servers.

**Important:** Claude Desktop does not inherit shell environment (pyenv, nvm, etc.), so you must use the **absolute path** to the Python binary that has the dependencies installed. Find it with:

```bash
python3 -c "import numpy, sentence_transformers, mcp; print(__import__('sys').executable)"
```

Use that full path as the `command` value:

```json
{
  "mcpServers": {
    "vault": {
      "command": "/absolute/path/to/python3",
      "args": ["<this-repo>/vault_mcp.py", "VAULT_PATH"],
      "env": {
        "PYTHONUNBUFFERED": "1"
      }
    }
  }
}
```

### 6. Tell the user to restart

- **Claude Code:** restart the CLI session
- **Claude Desktop:** fully quit and reopen the app

The MCP server won't be available until after restart.

### 7. Verification after restart

After the user restarts, verify the tools work by running:
- `list_tags(tag_type="broad")` — should return tag categories
- `semantic_search("test query")` — should return vault chunks with similarity scores

If the vault is freshly indexed (no tags/chunk_tags tables yet), `list_tags` will return "No tags found" — that's expected. `semantic_search` should still work.

If it shows `Failed to connect`, check:
- The vault path is correct and exists
- Python dependencies are installed
- `vault.db` was created in step 4
- For Claude Desktop: check logs at `~/Library/Logs/Claude/mcp.log`

## Development

When modifying Python files in this repo, always run pylint and flake8 via the venv before considering the work done:

```bash
.venv/bin/pylint <changed-files>
.venv/bin/flake8 <changed-files>
.venv/bin/mypy <changed-files>
```

Fix any errors or warnings before committing.

## How it works

- `embed_vault.py` chunks markdown files and embeds them with `all-MiniLM-L6-v2` into a SQLite database
- `vault_mcp.py` loads the embeddings at startup and exposes search tools over MCP
- The MCP server automatically runs an incremental reindex at every startup, so new/modified vault files are indexed without manual intervention
- Incremental updates compare file modification times and only re-embed what's changed

## Key paths

- Index database: `VAULT_PATH/.vault-index/vault.db`
- Claude Code MCP config: `~/.claude.json`
- Claude Desktop MCP config: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Claude Desktop logs: `~/Library/Logs/Claude/mcp.log`
- Skipped directories during indexing: `.vault-index`, `.obsidian`, `.git`, `.claude`, `.trash`, `_tools`
