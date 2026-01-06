# Repository Guidelines

## Project Structure
- `llm_trading/`: Python package (CLI, scanners, indicators, plotting, web API). Entry point: `python -m llm_trading ...`
- `llm_trading/web_static/`: Static Web UI (`index.html`, `styles.css`, `app.js`) with no bundler/build step.
- `prompts/`: Prompt templates used by `--llm` and `--narrate`.
- `data/`: Optional CSV exports produced by `fetch`.
- `outputs/`: Generated artifacts (charts/JSON/CSV/logs). Safe to delete; results are time-sensitive.
- `references/`: Reference code only (not runtime dependencies).

## Setup, Run, and Development Commands
- Install dependencies (Python 3.12 recommended): `".venv/bin/python" -m pip install -r "requirements-py312.txt"`
- Run Web UI locally: `".venv/bin/python" -m llm_trading serve --host 127.0.0.1 --port 8000`
- Scan ETFs (BBB shortlist): `".venv/bin/python" -m llm_trading scan-etf --limit 200 --min-weeks 60 --out-dir "outputs/scan_etf"`
- Analyze one symbol (includes institution signal if `--method all`): `".venv/bin/python" -m llm_trading analyze --asset stock --symbol 000725 --method all --out-dir "outputs/analyze_demo"`
- Clean old artifacts: `".venv/bin/python" -m llm_trading clean-outputs --path "outputs" --keep-days 1 --keep-last 20 --apply`

## Coding Style & Naming Conventions
- Python: 4-space indentation, type hints where practical, small pure functions, data-first design; JSON keys use `snake_case`.
- Web: keep plain JS/CSS (no build toolchain). When editing `llm_trading/web_static/app.js`, bump the cache-busting query in `llm_trading/web_static/index.html`.
- CLI: implement new commands as `cmd_<name>` in `llm_trading/cli.py` and register in `build_parser()`.

## Testing & Validation
- No dedicated test suite yet. Use smoke checks:
  - Syntax: `".venv/bin/python" -m py_compile llm_trading/*.py`
  - Run a small scan with `--limit` and verify `outputs/` contains `*.json/*.csv/*.png`.

## Configuration & Security
- Copy `.env.example` → `.env` and fill provider keys (OpenAI/Gemini/compatible proxy). Never commit secrets.
- Data sources (AkShare/Eastmoney) may throttle or lag; debug using “last_date” fields in outputs.

## Commits & Pull Requests
- Git history may not be available in this workspace; use Conventional Commits (`feat:`, `fix:`, `docs:`, `chore:`).
- PRs should include rationale, verification steps, screenshots for UI changes, and any data-source assumptions.
