# LLM辅助交易（多流派对比：脚本化 + 提示词驱动）

艹，`资料/提示词` 现在只是个“给模型看的提示词”，不算系统。老王给你把它落地成一个**可跑的最小骨架**：读取你给的 CSV → 计算 MA50/MA200 →（可选）让 LLM 输出结构化威科夫事件 JSON → 用我们自己的绘图代码生成中文标注图（不执行模型吐出来的任意 Python 代码，省得被自己坑死）。

## 你能得到什么

- 读入 CSV 行情数据（至少要有日期 + 收盘价）
- 计算 MA50 / MA200
- 同一份数据可跑多套方法并排对比（脚本能算的就脚本算；不适合脚本化的就用提示词驱动、强制 JSON 输出）
- 产出（按方法不同会略有区别）：
  - `chart.png`：中文标注图
  - `*.json`：结构/指标/事件的结构化结果（方便你后续回测、筛选、聚合）

> 免责声明：这是分析/复盘工具，不是投资建议，更不是自动下单系统。别拿它去梭哈，亏了别来找老王骂街。

## 快速开始

1) 安装依赖（项目基线：**Python 3.12+**）

你这机子里我已经确认有：`/home/root_zzl/miniconda3/bin/python` = **Python 3.12.2**。  
建议单独建个 venv（别动 conda base，省得又抽风）：

```bash
"/home/root_zzl/miniconda3/bin/python" -m venv ".venv"
# 这环境对 ~/.cache 没写权限：pip 构建轮子会炸，老实用 --no-cache-dir
".venv/bin/pip" install --no-cache-dir -r "requirements.txt"
```
> 旧的 Python 3.8 方案我留了个档：`requirements-legacy-py38.txt`，但别再用，纯折磨。

2)（可选）配置 OpenAI Key（如果你要启用 LLM 分析）：

- 复制 `.env.example` 为 `.env`，填 `OPENAI_API_KEY` 和 `OPENAI_MODEL`
- 如果你用 OpenAI 兼容中转站（new api 之类）：把 `OPENAI_BASE_URL` 改成你的中转站地址（支持 `https://xxx` 或 `https://xxx/v1`）

2.5)（可选）配置 Gemini（如果你要在 Web/UI 里拿它做“综合自然语言解读”）：

- 同样在 `.env` 里填：`GEMINI_API_KEY`、`GEMINI_MODEL`
- `GEMINI_MODEL` 例子：`gemini-2.0-flash` / `gemini-1.5-flash`（以你账号可用为准）
- 如果你用中转站：把 `GEMINI_BASE_URL` 改成你的代理地址，并按需把 `GEMINI_API_KEY_MODE` 改成 `authorization`

3) 运行（先不启用 LLM，保证流程能跑通）：

```bash
".venv/bin/python" -m llm_trading analyze --csv "你的数据.csv"
```

如果你现在手上没 CSV（你就是这种情况），直接用 AkShare 抓 ETF/指数：

```bash
".venv/bin/python" -m llm_trading analyze --asset "etf" --symbol "sh510300"
".venv/bin/python" -m llm_trading analyze --asset "index" --symbol "sh000300"
```

抓个股（支持直接写中文名，但别指望老版本 AkShare 永远不抽风）：

```bash
".venv/bin/python" -m llm_trading analyze --asset "stock" --symbol "京东方A"
".venv/bin/python" -m llm_trading analyze --asset "stock" --symbol "000725"
".venv/bin/python" -m llm_trading analyze --asset "stock" --symbol "sz000725"
```

单独抓数落 CSV：

```bash
".venv/bin/python" -m llm_trading fetch --asset "etf" --symbol "sh510300" --freq "weekly"
".venv/bin/python" -m llm_trading fetch --asset "index" --symbol "sh000300" --freq "weekly"
".venv/bin/python" -m llm_trading fetch --asset "stock" --symbol "京东方A" --freq "weekly"
```

带日期范围（抓一段你关心的区间）：

```bash
".venv/bin/python" -m llm_trading analyze --asset "etf" --symbol "sh510300" --start-date "20190101" --end-date "20251231"
```

4) 启用 LLM（需要环境变量）：

```bash
OPENAI_API_KEY="xxx" OPENAI_MODEL="gpt-4o-mini" \
".venv/bin/python" -m llm_trading analyze --csv "你的数据.csv" --llm
```

输出默认在 `outputs/<csv文件名>_<时间戳>/` 里。

## 缠论（正道：结构先算法算出来）

默认是 `--method wyckoff`；你要缠论就用 `--method chan`：

```bash
".venv/bin/python" -m llm_trading analyze --asset "stock" --symbol "京东方A" --method "chan"
```

输出文件：

- `chart.png`：周K + MA + 笔 + 中枢可视化
- `chan_structure.json`：程序计算出的分型/笔/中枢结构
- `llm_analysis.json`：只有在加 `--llm` 时才会有（LLM 基于结构 JSON 输出“候选机会/触发/失效/风险”，不喊单）

## 威科夫（含“威科夫线”A/D）

威科夫图默认会画一条 `A/D（累积/派发线）` 作为量价强弱参考（需要 high/low/volume）。

```bash
".venv/bin/python" -m llm_trading analyze --asset "stock" --symbol "京东方A" --method "wyckoff"
```

## 两派对比（同一份数据跑两套）

```bash
".venv/bin/python" -m llm_trading analyze --asset "stock" --symbol "京东方A" --method "both" --out-dir "outputs/boe_compare"
```

输出：

- `outputs/boe_compare/wyckoff/chart.png`
- `outputs/boe_compare/chan/chart.png`

## 一目（Ichimoku）

```bash
".venv/bin/python" -m llm_trading analyze --asset "stock" --symbol "京东方A" --method "ichimoku"
```

输出文件：

- `chart.png`
- `ichimoku.json`

## 海龟（Turtle / Donchian）

```bash
".venv/bin/python" -m llm_trading analyze --asset "stock" --symbol "京东方A" --method "turtle"
```

输出文件：

- `chart.png`
- `turtle.json`

## Momentum（RSI/MACD/ADX）

```bash
".venv/bin/python" -m llm_trading analyze --asset "stock" --symbol "京东方A" --method "momentum"
```

输出文件：

- `chart.png`
- `momentum.json`

## Dow（趋势结构：HH/HL/LH/LL）

```bash
".venv/bin/python" -m llm_trading analyze --asset "stock" --symbol "京东方A" --method "dow"
```

输出文件：

- `chart.png`
- `dow.json`

## VSA（量价行为特征 + 可选 LLM 解读）

```bash
".venv/bin/python" -m llm_trading analyze --asset "stock" --symbol "京东方A" --method "vsa"
```

输出文件：

- `chart.png`
- `vsa_features.json`
- `llm_analysis.json`：只有在加 `--llm` 时才会有（基于 `prompts/vsa_json_prompt.md`）

## 一键全跑（同一份数据跑全部方法）

```bash
".venv/bin/python" -m llm_trading analyze --asset "stock" --symbol "京东方A" --method "all" --out-dir "outputs/boe_all"
```

## 一键全跑 + 综合自然语言解读（Gemini/OpenAI）

会在输出目录根下生成一个 `summary.md`（失败会写 `summary_error.txt`，但不影响图表输出）。

```bash
".venv/bin/python" -m llm_trading analyze --asset "stock" --symbol "京东方A" --method "all" --narrate
```

输出目录示例：

- `outputs/boe_all/wyckoff/`
- `outputs/boe_all/chan/`
- `outputs/boe_all/ichimoku/`
- `outputs/boe_all/turtle/`
- `outputs/boe_all/momentum/`
- `outputs/boe_all/dow/`
- `outputs/boe_all/vsa/`

## ETF 全市场扫描（找“可能吃波段”的候选）

> 免责声明：这是扫描/复盘工具，不是投资建议。别梭哈，亏了别找老王。

扫描（默认只扫股票/海外股票 ETF；周线；输出 Top 候选 + 全量表）：

```bash
".venv/bin/python" -m llm_trading scan-etf --freq "weekly" --top-k 30
```

输出目录里会有：

- `top_bbb.json`：**核心三件套（BBB）**候选（周线MACD + 位置 + 日线MACD；更符合“别套山上”的口味）
  - BBB 会过滤**周K太少的新 ETF**（默认要求 `weekly_total >= 60`），否则 MA50/唐奇安/一目这些都是“假指标”
- `top_trend.json`：更偏“趋势突破/延续”的候选（需要你按触发条件确认）
- `top_swing.json`：更偏“回踩支撑/箱体波段”的候选
- `all_results.csv`：全量结果（方便你自己再筛）
- `errors.json`：抓数失败的代码（源站抽风别怪我）

默认榜单会过滤**周K太少**的新 ETF（避免“新ETF霸榜”的离谱情况）：

- `--min-weeks 60`：周K 少于该值就不进 `top_trend/top_swing`（默认 60；填 0 关闭）

如果你嫌太多、想过滤“磨损低/流动性好”的：

```bash
".venv/bin/python" -m llm_trading scan-etf --freq "weekly" --min-amount-avg20 50000000
```

注意：`--min-amount` 过滤的是“最后一根成交额”（优先用数据源 `amount`，缺失才用 `close*volume` 估算），就当过滤器用，别抬杠。
更推荐用 `--min-amount-avg20`（最近20日均成交额）当“流动性门槛”，比单日/单周尖峰更靠谱。

补充说明（避免误读）：

- ETF 日线优先走东财接口（`fund_etf_hist_em`），拿不到才兜底 Sina（Sina 有时会滞后到前一个交易日甚至更久）。
- 周线 `last_date` 用的是“该周最后一个交易日”（不再用 `W-FRI` 的周五标签去“穿越”误导你）。

## 全A 扫描（买入信号 + 胜率/磨损量化）

> 免责声明：这是研究/复盘工具，不是投资建议。胜率是历史统计，不保证未来；别拿它当圣杯。

你说的“能给我赚钱算赢”，老王给你落到**可计算**的口径：

- **周线为主、日线辅助**（默认：日线 `MACD` 过滤掉明显弱势周）
- **基础环境过滤（默认开）**：`trend_template`（周线版 Weinstein/Minervini 风格，宁可少也别乱给“能买”）
  - 只影响“现在能不能上车”的候选池（避免震荡里瞎冲）
  - **不参与历史胜率统计**（否则样本会被砍到太少，统计不稳定）
- 两套“买入信号”（同一份周线数据同时算）：
  - `trend`：周线 **20W Donchian 突破**（`MA200` 改为软过滤，不做一票否决）
  - `swing`：**从 MA50 下方重新站回**（`MA200` 改为软过滤）
  - `dip`：**回踩触碰 MA50 附近后收回，且不远离 MA50**（更偏左侧“捡漏”，`MA200` 软过滤）
- **胜率定义**（默认统计 `4/8/12` 周）：信号触发后按“下一周开盘买入”，持有 N 周，“第 N 周后开盘卖出”，扣除成本后收益 `>0` 记为赢
- **磨损**：持仓期间的 `MAE`（用周线 low 估算最大不利波动）

先小样本跑通（别一上来就全A把源站薅炸）：

```bash
".venv/bin/python" -m llm_trading scan-stock --freq "weekly" --limit 200 --top-k 50 --workers 8 --min-amount 50000000 --base-filters "trend_template"
```

真要全A就把 `--limit` 去掉（会很久，别催）：

```bash
".venv/bin/python" -m llm_trading scan-stock --freq "weekly" --top-k 50 --workers 8 --min-amount 50000000 --base-filters "trend_template"
```

如果你想按“我买得起”的范围筛股价（别买不起还硬看）：

```bash
".venv/bin/python" -m llm_trading scan-stock --min-price 2 --max-price 30
```

想关闭趋势模板，放宽候选（信号会更多，也更容易杂）：

```bash
".venv/bin/python" -m llm_trading scan-stock --base-filters "none"
```

输出目录里会有：

- `top_trend.json`：当前触发 `trend` 信号的榜单（按 8 周口径排序）
- `top_swing.json`：当前触发 `swing` 信号的榜单（按 8 周口径排序）
- `top_dip.json`：当前触发 `dip` 信号的榜单（左侧捡漏：回踩 MA50 不破）
- `all_results.csv`：全量结果（大）
- `errors.json`：抓数失败/样本不足的列表（源站抽风别怪我）

## Web 前端（浏览器操作）

启动服务（默认只监听本机，安全点）：

```bash
cd "/home/root_zzl/LLM辅助交易"
".venv/bin/python" -m llm_trading serve --host "127.0.0.1" --port 8000
```

然后浏览器打开：

- `http://127.0.0.1:8000`

你可以在页面里：

- 一键跑 `analyze`（包括 `--method all`）
- 一键跑 `scan-etf`，并点表格里的“一键分析”把候选 ETF 拉去全跑
- 一键跑 `scan-stock`（全A：买入信号 + 胜率/磨损量化）

## 清理 outputs（结果有时效性，别占空间）

默认只 dry-run（列出将删除什么），真删必须加 `--apply`：

```bash
".venv/bin/python" -m llm_trading clean-outputs --path "outputs" --keep-days 1 --keep-last 20
".venv/bin/python" -m llm_trading clean-outputs --path "outputs" --keep-days 1 --keep-last 20 --apply
```

## 中文字体（重要）

如果你运行时看到 `Glyph xxxx missing from current font` 这种憨批警告，说明系统里没中文字体。解决方式：

- 给 `analyze` 传 `--font-path "/path/to/你的中文字体.ttf"`
- 或者在系统里安装中文字体（如 Noto Sans CJK / 思源黑体 / SimHei）
- 现在程序也会尝试自动下载一份 `NotoSansCJKsc-Regular.otf` 到 `.matplotlib/fonts/`（第一次会慢一点）

## CSV 格式要求（别搞幺蛾子）

至少需要：

- 日期列：`date` / `datetime` / `time` / `timestamp` / `日期`
- 收盘列：`close` / `Close` / `收盘` / `收盘价`

如果你有 OHLC（建议有，后面做缠论/形态离不开）：

- 开盘：`open` / `开盘`
- 最高：`high` / `最高`
- 最低：`low` / `最低`

成交量可选：`volume` / `Volume` / `成交量`

列名对不上就用参数指定：`--date-col`、`--open-col`、`--high-col`、`--low-col`、`--close-col`、`--volume-col`。

## AkShare（你要的免费抓数）

现在项目基线是 Python 3.12+，`akshare` 直接走 PyPI 正常装就行（见 `requirements.txt`）。

## 提示词

- 原版（你给的）：`资料/提示词`
- 系统用的（强制 JSON 输出，便于程序画图）：`prompts/wyckoff_json_prompt.md`

## 进度

- [x] 读取 CSV + 计算 MA50/MA200
- [x] LLM 输出 JSON → 稳定解析
- [x] 按 JSON 画威科夫事件标注图（中文字体自动尝试加载）
- [x] 缠论：分型/笔/中枢（脚本化计算 + 可选 LLM 解读）
- [x] Ichimoku（一目均衡表）
- [x] Turtle / Donchian（趋势突破）
- [x] Momentum（RSI/MACD/ADX）
- [x] Dow（HH/HL/LH/LL 趋势结构）
- [x] VSA（量价特征 + 事件启发式 + 可选 LLM 解读）
- [x] `--method all`：同一份数据全跑并分目录输出
- [x] 全A：趋势模板（Weinstein/Minervini）基础过滤（`--base-filters trend_template`）
- [x] 全A：按股价区间过滤（`--min-price/--max-price`）
- [x] 全A：输出买入信号榜单 + 胜率/磨损量化
- [ ] 后续你要真做“交易系统”（接行情源/回测/风控/执行），再把需求说清楚：做哪个市场、哪家交易所/券商、频率、风控规则、是否允许自动下单。
