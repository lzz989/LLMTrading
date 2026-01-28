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

## 框架升级文档（建议从这里看）

- 升级计划总入口：`docs/框架升级/README.md`
- 兼容入口（仅索引）：`IMPROVEMENT_PLAN.md`

## Codex Repo Skills（strategy / research / backtest）

你嫌“戳一下动一下”太累，这就对了：把重复脑力活固化成 SOP 才像人干的事。

- 位置：`.codex/skills/`
- 用途：
  - `strategy`：把最新 `outputs/run_*` 收敛成“终极动作（五选一）+ 失效条件”（执行导向，保命优先）
  - `research`：基本面/行业/技术调研（证据-不确定性）；可选抓取门户/资讯新闻线索
  - `backtest`：规则回测（严禁未来函数；成本/滑点写清；默认 t+1 开盘成交）
- 最佳实践与样例：看 `.codex/skills/README.md`

## 正确使用姿势（小白版：你提需求，我来追问+跑脚本+给可执行动作）

这仓库不是“点按钮就发财”的玩具，它是**研究/复盘流水线**：脚本负责算出结构化证据（因子/信号/回测/风控阈值），我负责把这些证据翻译成你能执行的动作，并且在你没说清楚时**主动追问**，避免你一边亏钱一边还不知道自己在干啥。

- 你做的事：用一句人话提需求（不限于：分析持仓/选股/调仓/改策略/深挖某个标的）。
- 我做的事（默认自动，不等你下命令）：
  - 先读 `data/user_holdings.json`（你口述变动=最高优先级，我会先更新这个文件再跑分析）。
  - 先问清楚关键约束：目标（保命/搏收益）、持有周期、最大回撤/止损纪律、是否加钱、是否严格按 `trade_rules`（例如 `fri_close_mon_open`）。
  - 再跑合适的命令与 skills：优先 `run` 生成最新 `outputs/run_YYYYMMDD*/`，再按需调用 `strategy/research/backtest` 把信息收敛成“怎么做/什么时候错/错了怎么办”。
  - 最后给你**可执行**的结果：终极动作（五选一）+ 执行窗口 + 2~3 条失效条件 +（可选）`orders_next_open.json` 下单清单（你自己手动下单，别指望自动交易替你背锅）。
- 安全边界（写死，别碰瓷）：
  - 不自动实盘下单、不接券商 API。
  - 不编数据；缺数据就写“缺失/待核验”。
  - 交易求真：回测/分析可以当线索，但执行必须有风控阈值（止损/仓位/现金约束）。

一个典型流程长这样：

1) 你：`“老王，看看我现在持仓该不该减？顺便给下周一开盘的计划。”`  
2) 我：先追问你 3~5 个关键问题（目标/回撤/执行纪律/现金/是否允许换仓）。  
3) 我：跑 `run` / `analyze` / 新闻线索 / 回测对照，落盘到 `outputs/` + `outputs/agents/`。  
4) 我：给结论（五选一动作）+ 价位/结构证据 + 失效条件 + 订单草案（你手动下单）。  

## 交易策略口径（B风格：游击战主升浪；2026-01）

你要的是“1~2周快进快出”，不是价值投资。为了防止新对话里别的 AI 乱猜，这里把默认策略口径写死（可随时再改）：

- **目标**：右侧动量为主，坐“主升浪”；最多同时持仓 **5** 个，但其中**风险仓位**最多 **3** 个。
- **执行**：只用“收盘触发→次日开盘执行（T+1）”；盘中最多允许看盘到 **10:00~10:30** 再决定（止损例外）。
- **止损**：单票默认 -6%（并叠加结构线如 MA20/AVWAP，取更紧者）；收盘跌破 => 次日开盘卖。
- **止盈（分批）**：默认两段（更适配小资金+5元最低磨损）：TP1=+6% 卖 1/2、TP2=+10% 卖剩余 1/2（清仓）（具体阈值可调）。
- **组合保命线**：周度最大回撤默认 **-8%**（从周初权益算）；触发后降到现金/只留最强 1 只，本周不再新开仓。
- **磨损/佣金（必须时刻考虑）**：每笔买/卖存在**最低 5 元磨损**。因此任何“非止损”的分批/加减仓，单次订单金额太小（经验值 <2000 元）就别拆，直接合并；止损/风控卖出不受此限制。

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

为了别每次都全量拉历史数据把源站薅秃，`scan-etf` 默认会把每个 ETF 的日线数据缓存到本地，并做**增量更新**：

- 缓存目录：`data/cache/etf/`（每个标的一份 CSV）
- 下一次再扫：优先读缓存，只补“新K线”，速度会快很多
- 可控参数：
  - `--cache-dir`：换缓存目录
  - `--cache-ttl-hours`：缓存有效期（默认 24；填 `0` 表示完全不使用缓存、每次强制拉取）

另外，为了避免每次都重复计算指标/回测（尤其你反复扫同一批 ETF），`scan-etf` 默认还启用了**派生结果缓存**：

- 缓存目录：`data/cache/analysis/etf/`
- key：`symbol + last_daily_date + 参数hash`
- 可控参数：
  - `--analysis-cache / --no-analysis-cache`
  - `--analysis-cache-dir`

输出目录里会有：

- `top_bbb.json`：**核心三件套（BBB）**候选（周线MACD + 位置 + 日线MACD；更符合“别套山上”的口味）
  - BBB 会过滤**周K太少的新 ETF**（默认要求 `weekly_total >= 60`），否则 MA50/唐奇安/一目这些都是“假指标”
  - 每个候选会附带：
    - `bbb_forward`: `4/8/12w` 的历史胜率/平均收益/MAE（按“下一周开盘买入 -> 持有N周 -> 开盘卖出”，并扣除成本；含 `win_rate_shrunk` 防小样本）
      - 额外给 `implied_ann`：用每笔交易的平均 log 收益推算的“年化”（更贴近你说的“最大年化”，但别迷信，样本/市场环境会变）
      - 同时给 `net_*`（扣成本）和 `gross_*`（不扣成本）；为兼容旧字段，`win_rate/avg_return/implied_ann` 默认指 **net**
    - `bbb_best`: 在多个 horizon 里“分数”最高的那个（给“更像持有多久”一个量化参考；分数取决于 `--bbb-score-mode`）
    - `exit`: 给已持仓的人看的风控提示（`hold/reduce/exit`；周线硬失效 + 日线2日确认的软风控）
    - `bbb_exit_bt`: BBB 的“进场->出场”闭环回测统计（含 `avg_hold_days/median_hold_days`，更贴近“波段大概拿多久”）
      - exit reasons（计数）：`soft/hard/trail/stop_loss/profit_stop/panic`
- `top_bbb.json` 的 `bbb.market_regime` 会给一个**大盘牛/熊/中性**粗判（`scan-etf` 默认用 `sh000300,sz399006` 做“风险优先”合并 + canary；用于 `--bbb-mode auto` 自动调节“激进/保守”）
    - 当前判定口径：**日线 MA50/MA200 + MACD**，并带 **深回撤/大跌(panic) 兜底**（不预测黑天鹅，只做更早的风险识别）
    - 为了减少“震荡反复横跳”，默认做了 **3 日确认（confirm_days=3）**：宁愿慢半拍，也别天天被磨损
    - `--regime-index` 支持逗号分隔多个指数（例如 `sh000300,sz399006`）：
      - **风险优先合并**：`panic` 任一触发=>bear；否则任一 bear=>bear；否则任一 neutral/unknown=>neutral；全 bull 才 bull
      - **canary 降级（可关）**：
        - 兼容旧口径：逗号后面的指数当“风险偏好温度计”（PAA/DAA 思路）
        - 显式口径：用 `;` 把“主指数”和“canary指数”分开，例如 `sh000300,sz399006;sh000852`（中证1000只当 canary，不参与主合并）
        - 若 canary 的 `mom_risk_off=True`（126d&252d 动量都为负）则：
        - `bull -> neutral`（少激进）
        - `neutral -> bear`（别硬做）
        - 想更灵敏就加 `--no-regime-canary`（关闭 canary 的“只降不升”）
    - `factor_panel_7`：BBB 的**7因子可解释面板**（ETF 版；OHLCV可算）：
      - RS 相对强弱（12W/26W：标的收益-基准指数收益）
        - RS 基准默认：`sh000300+sh000905`（沪深300+中证500 等权合成，更中性）；可用 `--bbb-rs-index` 覆盖（`auto`=跟随 `--regime-index` 第一个指数；`off`=关闭）
      - 趋势质量（ADX14）
      - 波动（20D波动率 + ATR%）
      - 回撤/位置（252D回撤、距52周低点）
      - 流动性（20D均成交额 + 量能比）
      - BOLL 带宽/挤压（bandwidth_rel + squeeze）
      - 量能确认（成交额/成交量相对20日均值比）
    - `bbb_factor7`：7因子**只用于候选排序加权**（模式1：面板解释+排序加权），不改变 `bbb.ok/fails` 的硬条件
      - `--bbb-factor7 / --no-bbb-factor7`：开关（默认启用）
      - `--bbb-factor7-weights "rs=0.35,trend=0.15,vol=0.15,drawdown=0.15,liquidity=0.10,boll=0.05,volume=0.05"`：权重（留空=默认；自动归一化）
- `top_trend.json`：更偏“趋势突破/延续”的候选（需要你按触发条件确认）
- `top_swing.json`：更偏“回踩支撑/箱体波段”的候选
- `signals.json`：统一 signals schema（给组合层/回测层/执行层吃的中间产物；不用你再手动拼 JSON）
- `all_results.csv`：全量结果（方便你自己再筛）
- `errors.json`：抓数失败的代码（源站抽风别怪我）

默认榜单会过滤**周K太少**的新 ETF（避免“新ETF霸榜”的离谱情况）：

- `--min-weeks 60`：周K 少于该值就不进 `top_trend/top_swing`（默认 60；填 0 关闭）

如果你嫌太多、想过滤“磨损低/流动性好”的：

```bash
".venv/bin/python" -m llm_trading scan-etf --freq "weekly" --min-amount-avg20 50000000
```

成本口径（默认按你说的“来回 10 块磨损、单笔 3000”算）：

- `--capital-yuan 3000 --roundtrip-cost-yuan 10`（默认值）
- 或者直接覆盖比例成本：`--buy-cost 0.001 --sell-cost 0.001`

BBB 默认 `--bbb-mode auto`（牛/中性更偏 `pullback`，熊市更偏 `strict`）。如果你想手动更激进/更保守：

- `--bbb-mode pullback`：允许周线回踩造成的“周MACD未转多”（更贴近“右侧定方向 + 回踩挑位置”）
- **熊市过滤（默认生效）**：当 `market_regime=bear` 时，BBB **不输出“能买”候选**；你要硬刚熊市就显式加：`--bbb-allow-bear`
- `--bbb-mode strict`：更保守（更容易空仓，但更少“上错车”）
- `--bbb-entry-ma 20`：用 MA20 当“位置线”（比 MA50 更贴近你这种数周波段）
- `--bbb-dist-ma-max` / `--bbb-max-above-20w`：分别控制“离均线多远算追高”和“离20W上轨多远算追高”
- `--bbb-score-mode annualized`：BBB 排名按“年化优先”打分（更适合“小资金+磨损大+追最大年化”；`scan-etf` 现在默认就是 `annualized`）
  - 提醒：`annualized` 对小样本非常敏感；工具内置了“收缩胜率 + 小样本权重 + 年化封顶”来防止 trades=1 这种离谱霸榜，但你依然可以用 `--bbb-min-trades` 再保守一点

BBB 出场/止盈止损（用于 `bbb_exit_bt` 的闭环回测统计；都是研究口径，别当圣经）：

- `--bbb-exit-trail / --no-bbb-exit-trail`：启用/关闭周线锚线（默认启用；更早保护利润也更容易被震荡抖出去）
- `--bbb-exit-trail-ma 20`：周线锚线均线周期（默认 20）
- `--bbb-exit-profit-stop / --no-bbb-exit-profit-stop`：启用/关闭“盈利后回撤止盈”（默认启用）
- `--bbb-exit-profit-min-ret 0.20`：回撤止盈启用最低浮盈（默认 20%）
- `--bbb-exit-profit-dd-pct 0.12`：回撤止盈回撤比例（默认 12%）
- `--bbb-exit-stop-loss-ret 0.00`：最大亏损止损（按收盘触发；默认关闭；例如 0.08=亏8%就触发）
- `--bbb-exit-panic / --no-bbb-exit-panic`：启用/关闭 panic 兜底（大跌/深回撤快速离场；默认启用）
- `--bbb-exit-panic-vol-mult 3.0` / `--bbb-exit-panic-min-drop 0.04` / `--bbb-exit-panic-drawdown-252d 0.25`：panic 阈值参数

注意：`--min-amount` 过滤的是“最后一根成交额”（优先用数据源 `amount`，缺失才用 `close*volume` 估算），就当过滤器用，别抬杠。
更推荐用 `--min-amount-avg20`（最近20日均成交额）当“流动性门槛”，比单日/单周尖峰更靠谱。

补充说明（避免误读）：

- ETF 日线优先走东财接口（`fund_etf_hist_em`），拿不到才兜底 Sina（Sina 有时会滞后到前一个交易日甚至更久）。
- 周线 `last_date` 用的是“该周最后一个交易日”（不再用 `W-FRI` 的周五标签去“穿越”误导你）。

### 仓位计划（ETF）

你这种“总仓 3000 + 来回磨损 10 块”的小资金，仓位太碎就是给券商打工。
建议：先扫出 BBB 候选，再生成一个“明天买多少 + 止损线”的计划：

```bash
".venv/bin/python" -m llm_trading scan-etf --freq "weekly" --top-k 30 --out-dir "outputs/scan_etf"
".venv/bin/python" -m llm_trading plan-etf --scan-dir "outputs/scan_etf"
```

`plan-etf` 会读取 `top_bbb.json` 里的 `market_regime`（默认沪深300）来自动选“激进/保守”的仓位上限、单笔风险和止损参考线；
你也可以手动覆盖（比如更保守）：

```bash
".venv/bin/python" -m llm_trading plan-etf --scan-dir "outputs/scan_etf" --max-exposure-pct 0.5 --risk-per-trade-pct 0.008 --stop-mode daily_ma20
```

如果你想让止损更“波动自适应”（更像风险一致/波动率缩放的思路），可以用 ATR：

```bash
".venv/bin/python" -m llm_trading plan-etf --scan-dir "outputs/scan_etf" --stop-mode atr --atr-mult 2.0
```

默认风控口径（给你个“别拍脑袋”的基准）：

- 单笔风险预算≈“**单笔预期仓位** × 5%”（若总资金 ≤ 2000，则放宽为 × 10%）

你要是习惯用“我这笔最多亏多少元”来定风险（例如 1000 仓位最多亏 50），可以直接用：

```bash
".venv/bin/python" -m llm_trading plan-etf --scan-dir "outputs/scan_etf" --risk-per-trade-yuan 50
```

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

## 周内短线扫描（已精简移除）

`scan-short` / `eval-shortline` / `top_short.json` 已从主框架精简移除：换手高、对小资金固定磨损不友好，且与当前默认主线“趋势回踩低吸（1~数周）”冲突。

BBB 稳健性评估（**walk-forward + 参数敏感性**；研究用途）：

```bash
".venv/bin/python" -m llm_trading eval-bbb --symbol sh510300 --bbb-mode pullback --horizon-weeks 8
```

输出目录包含：
- `bbb_robust_<symbol>.json`：参数扰动对比 + walk-forward 各 fold 的 OOS 表现 + 简单告警
- `summary.json`：多个标的时的汇总（按 `stability_score` 排序）
- `run_meta.json`：运行环境/版本/argv/数据 `as_of`
- `run_config.json`：统一配置文件（只存 argv，用于复跑）
- `report.json`：标准化报告入口（列 artifacts + 简要摘要）

一键复跑同配置（默认写到新的 `outputs/replay_*`，不覆盖老产物）：

```bash
".venv/bin/python" -m llm_trading replay --from "outputs/etf_scan_20260107_120000"
```

## 经典量化策略赛马（牛/熊/震荡分段）

> 免责声明：这是研究/复盘工具，不是投资建议。策略都有适用环境，别指望“一招鲜吃遍天”。

对单个 ETF 跑一套经典策略赛马（并按 `regime-index` 的 bull/bear/neutral 分段统计）：

```bash
".venv/bin/python" -m llm_trading race --asset etf --symbol sh510300 --regime-index sh000300
```

全 ETF 赛马榜（输出 bull/bear/neutral 各自 TopN；默认会过滤样本太短/分段样本太少/交易次数太少的噪声）：

```bash
".venv/bin/python" -m llm_trading race --asset etf --universe etf --workers 8 --min-amount-avg20 50000000
```

只跑部分策略（逗号分隔，默认跑全套）：

```bash
".venv/bin/python" -m llm_trading race --symbol sh510300 --strategies "bbb,tsmom,turtle"
```

策略 key（当前内置）：
- `buyhold`：基线（一直持有）
- `ma_timing`：MA40(≈10个月) 择时
- `tsmom`：TSMOM（52周动量>0）
- `turtle`：海龟（20W 突破入场 / 10W 跌破出场）
- `boll_mr`：BOLL 均值回归（下轨入场 / 中轨或超时退出）
- `bbb`：BBB（周线定方向+回踩，统一周线出场口径）

输出目录包含：
- `summary.json`：每个标的按 CAGR 选出的“当前最强策略”摘要
- `race_<symbol>.json`：每个策略的总览指标 + bull/bear/neutral 分段统计 + 样本尾部曲线
- `leaderboards.json`：全局榜单（bull/bear/neutral 的 TopN；每项含 best_strategy/年化/回撤/交易次数）

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

股票“质量闸门”（硬过滤，默认启用，不跟你商量）：

- 北交所（`bj`）一刀切：不碰
- ST/*ST/退市：不碰
- 低价股/低流动性：默认要求 `close>=2` 且 `近20日均成交额>=5000万`

相关实现：`llm_trading/quality_gate.py`（要改阈值就改这里，别在命令行里瞎传一堆参数把自己绕晕）。

输出目录里会有：

- `top_trend.json`：当前触发 `trend` 信号的榜单（按 8 周口径排序）
- `top_swing.json`：当前触发 `swing` 信号的榜单（按 8 周口径排序）
- `top_dip.json`：当前触发 `dip` 信号的榜单（左侧捡漏：回踩 MA50 不破）
- `signals.json`：统一 signals schema（只输出“当前触发信号”的候选，避免把全A塞爆）
- `top_*.json` 里会附 `market_regime`（默认 `sh000300` 沪深300，可用 `--regime-index` 指定；或 `--regime-index off` 关闭）
- `all_results.csv`：全量结果（大）
- `errors.json`：抓数失败/样本不足的列表（源站抽风别怪我）
- `filtered.json`：被质量闸门硬过滤掉的列表（不是错误，是系统拒绝交易）

## 持仓分析（本地快照）

如果你按仓库约定维护了 `data/user_holdings.json`（字段：`positions=[{symbol,shares,cost_basis}...]`），可以一键跑持仓风控/止盈止损：

左侧试仓支持（可选）：在单个 position 里加 `entry_style: "left"`，就不会把 MA 锚线当硬止损线（避免震荡抖飞），只按“最大亏损兜底/盈利保护线”做离场提示。

冻仓支持（可选）：在单个 position 里加 `frozen: true`，`rebalance-user/run` 不会对该标的自动生成买卖单（你手动下单不受影响）。

输出里会附一段 `portfolio`（组合层汇总）：现金/仓位、单票&主题集中度、粗粒度相关性（周收益）、到“effective_stop”的风险金额等。
（注意：相关性需要本地缓存里有对应标的 CSV；第一次跑会自动落盘。）

```bash
".venv/bin/python" -m llm_trading holdings-user
```

常用参数：

```bash
".venv/bin/python" -m llm_trading holdings-user --regime-index "sh000300" --sell-cost-yuan 5
".venv/bin/python" -m llm_trading holdings-user --out "outputs/holdings_user.json"
```

## 组合调仓建议（研究用途）

给定你自己的 `data/user_holdings.json` + `scan-etf` 的 `signals.json`（或 `top_bbb.json`），生成“目标仓位 + 次日开盘买卖清单”：

```bash
".venv/bin/python" -m llm_trading rebalance-user --signals "outputs/scan_etf_20260114_close/signals.json" --mode "add" --out "outputs/rebalance_user.json"
```

- `--mode add`：按总权益算目标仓位，只用 `cash.amount` 做增量加仓（默认不卖；新钱优先补已有仓，慢慢逼近目标）
- `--mode rotate`：按目标仓位轮动/再平衡（会卖出非目标）
- `--min-trade-notional-yuan`：单笔买入最小成交额门槛（元；默认读 `data/user_holdings.json` 的 `trade_rules.min_trade_notional_yuan`；避免 5 元最低佣金把你磨死）
- `--max-positions`：最多持仓标的数（默认读 `trade_rules.max_positions`；否则随牛熊自动选）
- `--max-position-pct`：单标的最大仓位占比（例如 `0.30=30%`；默认读 `trade_rules.max_position_pct`；为空=不限制）

## 日常跑批（run）

一条命令跑完（默认走“因子库”口径）：`scan-strategy(etf,bbb_weekly) -> holdings-user -> rebalance-user -> report`（产物写到 `outputs/run_YYYYMMDD/`，研究用途）：

```bash
".venv/bin/python" -m llm_trading run --scan-freq "weekly" --scan-limit 200 --rebalance-mode "add"
```

稳健切换（默认开启）：会额外跑一份 legacy `scan-etf` 作为对照，并生成 `strategy_alignment/` 报告；想跑快点就加 `--no-scan-shadow-legacy`。

如果你是“周频~双周、纪律优先”的执行方式：可以用 `--rebalance-schedule "fri_close_mon_open"` 把调仓单锁死到“周五收盘后出单、下周一开盘执行”（止损/止盈 risk 单不受影响）：

```bash
".venv/bin/python" -m llm_trading run --rebalance-schedule "fri_close_mon_open"
```

如果你已经有现成的 `signals.json`，想跳过扫描：

```bash
".venv/bin/python" -m llm_trading run --signals "outputs/scan_etf_20260114_close/signals.json"
```

如果你有多份 `signals.json`（多策略/多来源），可以重复传 `--signals`，会自动 `signals-merge`：

```bash
".venv/bin/python" -m llm_trading run \
  --signals "outputs/run_smoke2/signals.json" \
  --signals "outputs/scan_stock_smoke/signals.json" \
  --signals-merge-conflict "risk_first"
```

输出目录会有（最关心的执行物都在这）：

- `signals.json`：本次候选（统一 schema）
- `signals_merged.json`：当你传多份 `--signals` 时，会生成这个显式文件名（内容与 `signals.json` 一致）
- `signals_strategy.json`：本次 scan-strategy 的原始产物（如果启用/命中）
- `signals_legacy.json`：本次 legacy scan-etf 的对照产物（如果启用/命中）
- `signals_inputs/`：当你传多份 `--signals` 时，会把输入复制一份放这里（便于复现/排查）
- `holdings_user.json`：持仓风控 + 组合层汇总
- `rebalance_user.json`：调仓建议（目标仓位 + orders_next_open）
- `alerts.json`：从持仓里抽的止盈/止损/观察告警
- `orders_next_open.json`：合并后的“次日开盘参考订单清单”（先风控卖出，再执行调仓）
- `report.md` / `report.json` / `run_meta.json` / `run_config.json`
- `scan_strategy/`：因子库扫描原始产物（signals.json）
- `scan_etf/`：legacy 扫描原始产物（top_bbb/top_trend/top_swing 等；用于对照/兜底）
- `strategy_alignment/`：新旧信号对齐报告（TopK overlap + mismatches）

## 多策略信号聚合（signals-merge）

你也可以手动合并多份 `signals.json`（输出 `strategy=signals_merged`，研究用途）：

```bash
".venv/bin/python" -m llm_trading signals-merge \
  --in "outputs/run_smoke2/signals.json" \
  --in "outputs/scan_stock_smoke/signals.json" \
  --conflict "risk_first" \
  --out "outputs/signals_merged.json"
```

说明：

- `signals_merged` 会透传一个“主策略”的 `entry/meta` 给下游（否则组合层算不了止损/仓位）。
- `rebalance-user` 当前对 `signals_merged` 只吃 `asset=etf 且 action=entry` 的候选（别拿 exit 去凑仓位）。

## 对账闭环（reconcile）

你是“手动点确认”的半自动模式：那就 **收盘跑 `run` 生成次日参考单**，第二天你手动下单，**收盘后把真实成交（fills）导出来**，跑 `reconcile` 把 `持仓/现金` 写回 `data/user_holdings.json`，并追加一份审计台账（jsonl）。

默认 dry-run（只生成产物，不写回）：

```bash
".venv/bin/python" -m llm_trading reconcile --fills "examples/fills_example.csv" --orders "outputs/run_YYYYMMDD/orders_next_open.json"
```

真要写回（危险操作，必须显式加 `--apply`）：

```bash
".venv/bin/python" -m llm_trading reconcile --fills "examples/fills_example.csv" --orders "outputs/run_YYYYMMDD/orders_next_open.json" --apply
```

`--fills` 支持 `csv/json/jsonl`，最低字段建议包含：

- `trade_id`（强烈建议有，否则只能退化 hash，反复导出可能会重复）
- `datetime`
- `symbol`（如 `sh510150` / `510150`）
- `side`（buy/sell 或 中文“买/卖”）
- `price`
- `shares`
- `fee`/`tax`（可选，缺失按 0；但现金/盈亏会有偏差）

产物目录：`outputs/reconcile_YYYYMMDD/`，会生成：

- `reconcile.json`：对账摘要（新增成交数、现金前后、warnings、逐笔 changes）
- `user_holdings_next.json`：写回前的“下一版快照预览”
- `ledger_trades_append.jsonl`：本次新增成交的台账追加预览
- `report.md` / `report.json` / `run_meta.json` / `run_config.json`

## 组合级模拟盘/回测（paper-sim）

账户级模拟（共享一笔资金，不是“每个标的各跑各的”），产物：`paper_sim.json` + `report.md`（研究用途）：

```bash
".venv/bin/python" -m llm_trading paper-sim --strategy "bbb_etf" --signals "outputs/scan_etf_20260114_close/signals.json" --start-date "20220101" --end-date "20251231" --capital-yuan 100000
```

想对比“旧排序 vs 7因子排序”（不改BBB硬规则，只改入场优先级），用：

```bash
".venv/bin/python" -m llm_trading paper-sim --strategy "bbb_etf" --signals "outputs/scan_etf_20260114_close/signals.json" --start-date "20220101" --end-date "20251231" --capital-yuan 100000 --bbb-entry-rank-mode "ma20_dist"
".venv/bin/python" -m llm_trading paper-sim --strategy "bbb_etf" --signals "outputs/scan_etf_20260114_close/signals.json" --start-date "20220101" --end-date "20251231" --capital-yuan 100000 --bbb-entry-rank-mode "factor7"
```

`factor7` 的 RS 基准默认 `sh000300+sh000905`（沪深300+中证500 等权合成）；可用 `--bbb-rs-index` 覆盖（`auto`=跟随 `--regime-index`；`off`=关闭）。

想“吃满 beta”（减少现金拖累）：用 `--core` 指定宽基底仓，模拟器会用剩余现金填仓；如果需要给 BBB 腾预算/现金，会在不跌破 `--core-min-pct` 的前提下卖出一部分 core。小资金强烈建议配合 `--min-trade-notional-yuan`（例如 2000）避免碎单磨损。

```bash
".venv/bin/python" -m llm_trading paper-sim --strategy "bbb_etf" --signals "outputs/scan_etf_20260114_close/signals.json" --start-date "20200101" --end-date "20260116" --capital-yuan 10000 --min-fee-yuan 5 --core "sh510300=0.5,sh510500=0.5" --core-min-pct 0.7 --min-trade-notional-yuan 2000
```

组合级最大回撤熔断（研究用途）：触发后会在**次日开盘**尝试清仓到现金，并进入冷却期；可选启用“重启闸门”（冷却结束后需指数连续 N 天收盘 > MA20 才允许重新开仓）。

```bash
".venv/bin/python" -m llm_trading paper-sim --strategy "bbb_etf" --signals "outputs/scan_etf_20260114_close/signals.json" --start-date "20200101" --end-date "20260116" --capital-yuan 10000 --min-fee-yuan 5 --min-trade-notional-yuan 2000 --portfolio-dd-stop 0.5 --portfolio-dd-cooldown-days 20 --portfolio-dd-restart-ma-days 5
```

股票版（研究用途）：用指数成分股做股票池（避免扫全A太慢/太杂），例如沪深300+中证500：

```bash
".venv/bin/python" -m llm_trading paper-sim --strategy "bbb_stock" --universe-index "000300+000905" --start-date "20190101" --end-date "20260116" --capital-yuan 100000 --min-fee-yuan 5 --roundtrip-cost-yuan 0
```

`--universe-index` 支持 `000300`/`sh000300`/`hs300` 这种写法，也支持用 `+`/`,` 组合多个指数；回测摘要会输出 `profit_factor`（总赢/总亏）与 `payoff`（平均赢/平均亏），方便看盈亏比。

## 数据体检（data-doctor）

用来查：`data/cache` 的 OHLCV 异常/缺列、`outputs/*` 的 `run_meta/run_config/signals` 基本字段（只抽样检查，避免 1w+ cache 文件把机器拖死）：

```bash
".venv/bin/python" -m llm_trading data-doctor --cache-recent-days 3 --cache-max-files 200 --outputs-max-dirs 30
```

## DuckDB 数据仓库（SQL）

`data/cache` + `outputs` 越来越多后，你用 SQL 查会舒服很多（筛选/聚合/回看统计都一条语句搞定）。

初始化/刷新（会生成：`data/warehouse.duckdb`）：

```bash
".venv/bin/python" -m llm_trading sql-init
".venv/bin/python" -m llm_trading sql-sync
```

常用视图（都在 `wh` schema 下）：

- `wh.file_catalog`：本地数据文件目录（csv/json 全量索引）
- `wh.v_bars`：统一 OHLCV（etf/stock/index/crypto）
- `wh.v_outputs_json`：`outputs/*/*.json` 一锅端（run/scan/paper/monitor 等）
- `wh.v_analysis_meta` / `wh.v_signal_backtest` / `wh.v_tushare_factors`：单标的分析核心产物

直接跑 SQL：

```bash
".venv/bin/python" -m llm_trading sql-query --sql "select count(*) as n from wh.v_bars_etf" --limit -1
".venv/bin/python" -m llm_trading sql-query --sql "select out_dir, decision.action from wh.v_signal_backtest" --limit 20
```

## 监控/回顾（monitor）

把 `outputs/*/report.json` 汇总成一个“可扫一眼的监控报表”（研究用途）：

```bash
".venv/bin/python" -m llm_trading monitor --outputs-dir "outputs" --max-dirs 200
```

产物在 `outputs/monitor_YYYYMMDD/`：

- `monitor.json`：结构化汇总 + 预警列表
- `summary.csv`：一行一个 report 的摘要表
- `report.md`：给人看的简版摘要

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

## 参考库（对标/别造轮子）

下面这些库/框架是业内常用的“坐标系”，用来回答两个问题：

1) 我们这个仓库现在在哪一层（数据/因子/回测/组合/执行）？  
2) 我们还缺哪些关键能力？（不是为了引依赖，是为了少走弯路）

> 备注：很多框架默认口径是美股/可做空/可 T+0，我们做 A 股日线/周线研究工具时只能“借思想”，别硬抄。

### 数据/研究（因子化框架）

- Qlib：https://github.com/microsoft/qlib  
  一体化研究框架（数据管道/因子/回测/评估）；尤其值得参考它的“研究闭环”和数据抽象层。

### 回测/策略框架（事件驱动/向量化）

- Zipline（原版）：https://github.com/quantopian/zipline  
  老牌事件驱动回测框架（原仓库维护不稳定，主要参考其事件/撮合/成本接口设计）。
- Zipline Reloaded（社区维护）：https://github.com/stefan-jansen/zipline-reloaded  
  Zipline 的社区维护版（更适合当“事件驱动回测”的参考实现）。
- Backtrader：https://github.com/mementum/backtrader  
  事件驱动回测/策略框架；适合作为“撮合与订单状态机”的参考。
- vectorbt：https://github.com/polakowo/vectorbt  
  向量化回测/研究工具；适合做大量参数扰动/稳健性分析（但不是交易系统）。
- backtesting.py：https://github.com/kernc/backtesting.py  
  轻量回测库；适合快速验证一个规则有没有基本胜率。
- QuantConnect Lean：https://github.com/QuantConnect/Lean  
  工程化交易系统（很重，但“交易系统该长啥样”可以对标）。

### 组合优化/风险

- PyPortfolioOpt：https://github.com/robertmartin8/PyPortfolioOpt  
  经典组合优化（均值-方差/风险平价/Black-Litterman 等）；可借鉴“约束表达/风险预算”的口径。
- Riskfolio-Lib：https://github.com/dcajasn/Riskfolio-Lib  
  更全的组合/风险模型库；适合当“风险度量/约束集合”的参考。

### 绩效分析/报告

- empyrical：https://github.com/quantopian/empyrical  
  常用绩效指标计算（Sharpe/Sortino/Calmar/...）。
- pyfolio：https://github.com/quantopian/pyfolio  
  回测报告/归因思路（较旧，但结构值得参考）。
- quantstats：https://github.com/ranaroussi/quantstats  
  快速生成绩效报告（偏“产出给人看”）。

### 我们仓库的“对应关系”（粗对照）

| 领域 | 我们现在（对应模块） | 参考库里对应能力 |
|---|---|---|
| 数据抓取/缓存 | `llm_trading/akshare_source.py` / `llm_trading/data_cache.py` / `llm_trading/tushare_*` | Qlib DataHandler / Zipline 数据源 |
| 因子库 | `llm_trading/factors/` + `llm_trading/factors/research.py` | Qlib 因子研究闭环 |
| 单标分析/出图 | `llm_trading/plotting.py` / `llm_trading/narrative.py` | 各框架都能做，但我们更偏“研究读图” |
| 信号/扫描 | `llm_trading/etf_scan.py` / `llm_trading/stock_scan.py` | vectorbt（批量研究）/ Zipline（事件驱动策略） |
| 回测/模拟盘 | `llm_trading/paper_sim.py` / `llm_trading/eval_*` | Zipline/Backtrader（事件驱动）/ vectorbt（向量化） |
| 组合层/风控/调仓 | `llm_trading/portfolio.py` / `llm_trading/positioning.py` / `rebalance-user` | PyPortfolioOpt/Riskfolio（组合优化/约束表达） |
| SQL 数据仓库 | `llm_trading/warehouse.py`（DuckDB views） | Qlib 的数据仓库/数据抽象思想 |
| 执行/对账/复盘 | `run` / `reconcile` / `monitor` / `data-doctor` | Lean/Zipline 的“订单/成交/审计”思路 |

### 我们缺什么（按优先级，别瞎堆功能）

1) **更严格的“事件驱动撮合/成本模型”**：现在很多统计是研究近似（够用但不够硬）。  
2) **更统一的数据口径**：交易日历、复权/分红、停牌/涨跌停处理需要再系统化（否则回测会骗人）。  
3) **组合优化/风险预算的模块化表达**：目前偏规则，后续可以借鉴 PyPortfolioOpt/Riskfolio 的约束表达。  
4) **更完整的回测报告**：指标/归因/分段统计可以借鉴 empyrical/pyfolio/quantstats。  

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
- [x] 组合层闭环：`holdings-user` / `rebalance-user` / `run` / `reconcile`（半自动对账）
- [x] 多策略候选聚合：统一 `signals.json` schema + `signals-merge`（`strategy=signals_merged`）
- [x] 健康检查/监控：`data-doctor` / `monitor`
- [ ] 后续你要真做“交易系统”（接行情源/回测/风控/执行），再把需求说清楚：做哪个市场、哪家交易所/券商、频率、风控规则、是否允许自动下单。
