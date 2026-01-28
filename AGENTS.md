# Repository Guidelines

## Project Structure
- `llm_trading/`: Python package (CLI, scanners, indicators, plotting). Entry point: `python -m llm_trading ...`
- `prompts/`: Prompt templates used by `--llm` and `--narrate`.
- `data/`: Optional CSV exports produced by `fetch`.
- `outputs/`: Generated artifacts (charts/JSON/CSV/logs). Safe to delete; results are time-sensitive.
- `references/`: Reference code only (not runtime dependencies).

## Setup, Run, and Development Commands
- Install dependencies (Python 3.12 recommended): `".venv/bin/python" -m pip install -r "requirements-py312.txt"`
- Scan ETFs (BBB shortlist): `".venv/bin/python" -m llm_trading scan-etf --limit 200 --min-weeks 60 --out-dir "outputs/scan_etf"`
- Analyze one symbol (includes institution signal if `--method all`): `".venv/bin/python" -m llm_trading analyze --asset stock --symbol 000725 --method all --out-dir "outputs/analyze_demo"`
- Clean old artifacts: `".venv/bin/python" -m llm_trading clean-outputs --path "outputs" --keep-days 1 --keep-last 20 --apply`

## Local Data Warehouse (DuckDB / SQL)
- Goal: treat `data/` + `outputs/` as queryable datasets; use SQL for slicing/aggregations instead of ad-hoc scripts.
- Default DB file: `"data/warehouse.duckdb"` (local artifact; safe to recreate).
- Init / refresh:
  - Initialize DB + views + file catalog: `".venv/bin/python" -m llm_trading sql-init`
  - Refresh file catalog only: `".venv/bin/python" -m llm_trading sql-sync`
  - Query (default `limit=50`, `--limit -1` disables): `".venv/bin/python" -m llm_trading sql-query --sql "select count(*) from wh.file_catalog" --limit -1`
- Reserved path: `"outputs/_duckdb_sentinel/"` is a sentinel directory to prevent DuckDB glob queries from failing when `outputs/` is empty; do not delete it.
- Schema conventions (for anything we want to be SQL-friendly):
  - Prefer `csv/json` over free-form text; JSON keys must be `snake_case`.
  - Always include stable IDs/metadata where relevant: `symbol`, `asset`, `as_of`, `source`, `ref_date` (or equivalent).
  - Avoid deeply nested arrays/objects for “top-level” analysis outputs; if unavoidable, provide a flattened summary alongside.
- When adding new data outputs that should be queryable, also add/extend a `wh.v_*` view in `llm_trading/warehouse.py` (keep it KISS; do not over-engineer).
- Common views (for slicing/monitoring):
  - `wh.v_bars`: unified OHLCV bars (etf/stock/index/crypto) from `data/cache/*/*.csv`.
  - `wh.v_analysis_meta` / `wh.v_signal_backtest`: per-symbol analysis outputs (top-level JSON).
  - `wh.v_tushare_factors_flat`: ERP proxy (shibor1y + 10Y alt), HSGT north/south scores, microstructure proxy.
  - `wh.v_etf_holdings_top10_items`: ETF top holdings (quarterly disclosure; top10 rows).
  - `wh.v_signals_items`: unified signals schema items (from `scan-*` / `run` outputs).
  - `wh.v_top_bbb_items`: scan-etf BBB shortlist items + factor_panel_7 flattened fields.
  - `wh.v_holdings_user_holdings`: holdings-user per-position rows + factor_panel_7 flattened fields.
  - `wh.v_orders_next_open_orders` / `wh.v_rebalance_user_orders_next_open`: next-open orders (execution list) flattened.

## Coding Style & Naming Conventions
- Python: 4-space indentation, type hints where practical, small pure functions, data-first design; JSON keys use `snake_case`.
- CLI: implement new commands as `cmd_<name>` in `llm_trading/cli.py` and register in `build_parser()`.

## Avoid Reinventing Wheels (No “Wheel-Building”)
- Before adding indicators, backtests, execution sims, caching, or CLI helpers: search this repo first and reuse existing modules.
- Prefer mature, well-used libraries (e.g. `ta`) and established GitHub implementations over one-off custom code.
- If custom code is still needed (look-ahead safety, A-share quirks, performance, or dependency constraints), document the reason in the PR/commit message and keep the implementation minimal.

## Testing & Validation
- Minimal tests + smoke checks:
  - One-command smoke: `bash scripts/smoke.sh`
  - Or run manually:
    - Syntax: `".venv/bin/python" -m compileall -q llm_trading`
    - Tests (no network): `".venv/bin/python" -m unittest discover -s tests -p "test_*.py"`
    - Run a small scan with `--limit` and verify `outputs/` contains `*.json/*.csv/*.png`.

## Configuration & Security
- Copy `.env.example` → `.env` and fill provider keys (OpenAI/Gemini/compatible proxy). Never commit secrets.
- Data sources (AkShare/Eastmoney) may throttle or lag; debug using “last_date” fields in outputs.

## Commits & Pull Requests
- Git history may not be available in this workspace; use Conventional Commits (`feat:`, `fix:`, `docs:`, `chore:`).
- PRs should include rationale, verification steps, screenshots for UI changes, and any data-source assumptions.

## User Portfolio Context (Local)
- Maintain an up-to-date holdings snapshot in `data/user_holdings.json` (this repo ignores `data/` via `.gitignore`).
- When the user reports position/cash changes, update `data/user_holdings.json` in the same turn.
- Before answering position sizing / add / reduce / rotate questions, read `data/user_holdings.json` and base analysis on it.
- If the file conflicts with the user's latest message, treat the user's latest message as source of truth and then update the file.

## User Preference Memory (Local)
- Keep long-term preferences/constraints in `data/user_profile.json` (auto-managed by `llm_trading memory` and `llm_trading run`).
- Keep daily running context in `data/memory/daily/YYYY-MM-DD.md` (append-only; do not overwrite).
- Before giving any trading recommendation/plan: read `data/user_profile.json` + `data/user_holdings.json`, and if exists read today's + yesterday's daily memory file.
- When the user states a durable preference/constraint/workflow change, update `data/user_profile.json` in the same turn, and append a one-line note to today's daily memory.
- CLI quick ref:
  - Status: `".venv/bin/python" -m llm_trading memory status` (add `--json` for scripts)
  - Sync hard constraints: `".venv/bin/python" -m llm_trading memory sync` (sync `data/user_holdings.json.trade_rules` → `data/user_profile.json`)
  - Write long-term note: `".venv/bin/python" -m llm_trading memory remember --text "..."` (appends to `data/memory/MEMORY.md`)
  - Write daily log: `".venv/bin/python" -m llm_trading memory remember --daily --title "..." --text "..."` (appends to `data/memory/daily/YYYY-MM-DD.md`)
  - Update structured prefs: `".venv/bin/python" -m llm_trading memory remember --set "workflow.preferred_interface=chat" --set "memory.auto_write_preferences=true"`
  - Vector index: `".venv/bin/python" -m llm_trading memory index --force` (needs `EMBEDDINGS_*`)
  - Search: `".venv/bin/python" -m llm_trading memory search "查询" --mode keyword|vector|hybrid`
  - Export prompt context: `".venv/bin/python" -m llm_trading memory export-prompt --max-chars 6000 --daily-days 2`
  - Archive daily: `".venv/bin/python" -m llm_trading memory archive --keep-days 7 --group month` (默认 dry-run；加 `--apply` 才真删旧 daily)
- Auto-write hooks:
  - `llm_trading run` auto-syncs `trade_rules` and appends a `run` snapshot to today's daily memory.
  - `analyze --narrate` injects (profile + long-term + last 2 days daily) into the LLM system prompt.
  - `llm_trading chat` (when `preferences.memory.auto_write_preferences=true`) will:
    - auto-extract durable preference changes from user text (soft prefs only: `workflow.*` / `output.*` / `memory.*`), write to `data/user_profile.json`, and append a note to today's daily memory;
    - generate `outputs/chat_*/coach.md` with “执行前复核问题”（coach mode; does not place real orders）。
- Env overrides (optional):
  - `LLM_TRADING_MEMORY_DIR` (default `data/memory/`)
  - `LLM_TRADING_PROFILE_PATH` (default `data/user_profile.json`)
  - `EMBEDDINGS_API_KEY` / `EMBEDDINGS_MODEL` / `EMBEDDINGS_BASE_URL` / `EMBEDDINGS_HEADERS_JSON` (enable vector search)

## User Trading Playbook (Must Follow, 2026-01)

This section documents the user's current "B-style guerrilla" workflow so new agents don't guess.

- **Goal**: ride 1~2 week "main wave" (right-side momentum), not long-term investing.
- **Execution**:
  - Use **close-only triggers**; execute at **next open (T+1)**.
  - Intraday: allow observation until **10:00~10:30** for *non-stop-loss* actions.
  - Never auto-place real orders; only output a manual execution plan / `orders_next_open.json` draft.
- **Position constraints**:
  - Max total positions: **5**.
  - Risk slots: at most **3 risk positions** (positions whose stop is below entry / not yet protected).
  - "Protected" positions (after TP1, stop raised to breakeven or above) may occupy the extra slots, but still keep total <= 5.
- **Risk management**:
  - Single-position hard loss: default **-6%** from entry (combined with a structure stop like MA20 / AVWAP; use the tighter one).
  - Weekly portfolio max drawdown: **-8%** (from start-of-week equity). If hit: cut to cash or keep only the strongest 1 position; no new entries for the rest of the week.
- **Take profit** (staged, to be cost-aware):
  - Default (two tranches; fits small capital + 5 CNY minimum friction): TP1 at +6% sell 1/2; TP2 at +10% sell remaining 1/2 (exit).
  - If a tranche order is too small, merge tranches to reduce friction (see below).
- **Friction-aware sizing (critical)**:
  - The user has **minimum 5 CNY friction/commission per order** (buy/sell). Treat this as a fixed cost floor.
  - Therefore, when proposing discretionary entries/adds/trims/TP tranches, ensure each order notional is large enough (rule of thumb: >= **2000 CNY**), otherwise **merge** (except forced stop-loss/exits).
  - Stop-loss / forced risk exits override friction considerations.

## Codex Repo Skills（强烈建议：积极调用）

本仓库内置 repo-scope skills，用来把重复脑力活固化成 SOP，避免“戳一下动一下”。

- Skills 目录：`.codex/skills/`（当前：`strategy` / `research` / `backtest`）
- 运行入口（已落地成 CLI）：`".venv/bin/python" -m llm_trading skill strategy|research|backtest ...`；也可由 `llm_trading chat` 在自然语言规划时自动触发。
- chat 强制触发（兜底语法）：在输入里写 `#strategy` / `#research` / `#backtest`（LLM planner 不可用时也能触发）。
- 触发原则：用户没点名也要主动用——只要任务明显匹配，就按对应 `SKILL.md` 的流程执行，并落盘产物到 `outputs/agents/`。
- 输出优先级：先写文件（可复核、可追溯），再在聊天里解读重点；不要只口嗨不落盘。
- 标的口径：聊天/报告里**禁止只写代码**；必须同时写 `symbol + 名称`（例：`sh512980（中证传媒ETF）`）。若名称暂时解析失败，写 `symbol（名称未知）` 并说明原因（例如 universe 缓存缺失/数据源失败）。
- strategy（持仓/候选收敛成“可执行动作+失效条件”）：
  - 优先跑一次 `run` 产出最新 `outputs/run_YYYYMMDD/`（必要时 `--deep-holdings`），再按 `.codex/skills/strategy/SKILL.md` 输出 `outputs/agents/strategy_action.md`。
  - CLI：`".venv/bin/python" -m llm_trading skill strategy --run-dir outputs/run_YYYYMMDD --out outputs/agents/strategy_action.md`
  - 任何“动作”必须给到可验证的价位/结构/失效条件；严禁自动实盘下单。
- research（基本面/行业/新闻线索）：
  - 需要新闻时：优先用 CLI 一键抓取 + 汇总：`".venv/bin/python" -m llm_trading skill research --run-dir outputs/run_YYYYMMDD --out-dir outputs/agents`
    - 产物：`outputs/agents/news_raw.json` / `outputs/agents/news_digest.md` / `outputs/agents/research.md`
  - 媒体稿只能当线索；关键结论必须回到公告/财报/监管披露核验。
- backtest（规则回测/对照实验）：
  - 默认 t 日信号→t+1 开盘成交；必须写成本/滑点；严禁未来函数。
  - 产出 `outputs/agents/backtest_report.md`（或用户指定路径），并写清假设与局限性。
  - CLI：`".venv/bin/python" -m llm_trading skill backtest --asset etf --symbols sh518880,sh159937 --out outputs/agents/backtest_report.md`

## 分析提示词（LLM 解读/下一步动作）

下面这段是给 `--narrate` 或人工调用大模型做“综合解读”的提示词模板。输入是本项目产出的单标的分析目录（`meta.json`、`signal_backtest.json`、各流派结构化输出、`tushare_factors.json` 等）拼成的一份 JSON。

```text
你会收到一份“多流派分析结果(JSON)”，字段可能包含：
- meta：数据源/频率/区间等
- signal_backtest：右侧信号是否触发 + 历史统计 + decision.action
- wyckoff / chan / ichimoku / turtle / momentum / dow / vsa / institution：各流派结构化摘要（可能缺失）
- tushare_factors：ERP(shibor1y；可选10Y国债对照) / HSGT north-south / microstructure(大单+超大单)（可能缺失）
（可选）- user_holdings：用户持仓快照（若提供则结合 trade_rules/max_positions/frozen 等约束）
（可选）- analysis_hints：程序生成的“关注优先级/执行约束”（如果有就必须遵守）

你的任务：把这些结果整合成一段“研究复盘解读 + 下一步动作”，用于执行前复核，不构成投资建议。

硬性约束：
1) 必须中文输出；允许 Markdown；不要输出代码。
2) 必须给出一个“终极动作（胜率优先）”，且只能五选一：
   - 观望 / 试错小仓 / 执行计划 / 减仓 / 退出
   动作优先级（保命优先）：退出 > 减仓 > 执行计划 > 试错小仓 > 观望。
3) 如果 JSON 里存在 signal_backtest.decision.action：必须直接采用该动作（不要反着写）。
   - 若动作是“减仓/退出”，请明确写：这属于持仓风控动作；若当前空仓，则等价于“不新开仓/保持观望”。
4) 周线为主、日线为辅：所有结论尽量落在“可验证的价位/结构”上（MA50/MA200、Donchian、云层、关键中枢/支撑压力等）。
5) 不要堆砌因子：只挑最重要的 5~8 个证据展开，剩下的用一句话说明“不适用/无信号/数据缺失”。

因子侧重点（根据标的类型自动取舍）：
- stock：优先看 signal_backtest + 周线趋势结构 + 日线风险确认（MACD/MA20）；
  可用时再看 tushare_factors.microstructure（大单/超大单净流占比）和 institution（A/D、OBV、资金流）。
- etf：优先看 BBB/趋势结构/流动性；宏观层面用 tushare_factors.ERP + HSGT north/south 做风险温度计；
  不要把“北向/南向”当成确定性买卖理由，只能做解释和风险加权。
- index：优先看趋势/波动/回撤 + ERP/HSGT（风险偏好），强调“环境”而不是“个股买点”。
如果提供了 user_holdings：
- 尊重 trade_rules（例如 max_positions、max_position_pct、min_trade_notional_yuan、调仓节奏）
- 如果某个持仓标记为 frozen=true：默认不建议“加仓/换仓”，除非出现明确的“退出”风控信号。

输出结构（建议照做）：
- 一句话结论：当前更像趋势/震荡/弱反弹/等待确认
- 终极动作（五选一）：写清楚依据、执行窗口、失效条件（2~3条，必须可验证）
- 重点因子清单（Top 5~8）：每条=结论 + 证据（引用 JSON 里的字段/数值/价位）
- 共识 vs 分歧：哪些模块一致、哪些冲突；冲突如何处理（用触发/失效条件解决）
- 风险提示：样本不足/滑点成本/指标滞后/宏观逆风等（2~4条）
- 免责声明：一句话
```
