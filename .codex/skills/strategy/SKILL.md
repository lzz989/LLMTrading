---
name: strategy
description: 交易/持仓的策略分析与可执行动作输出（保命优先）。优先复用本仓库产物（outputs/run_*/report.md、holdings_user.json、rebalance_user.json、signals*.json、cash_signal.json、opportunity_score.json 等），把“看起来不错”落成“怎么做/什么时候错/错了怎么办”。
---

# 策略分析（执行导向）

## 输入 / 输出约定（别扯皮，先把口径统一）

- 默认输入：最近一次 `outputs/run_*` 跑批目录（没有就先跑一次）。
- 默认输出：`outputs/agents/strategy_action.md`
- 如果用户指定了目录/文件：以用户为准，但必须把“依据/窗口/失效条件”写清楚。

## 你要的结果是什么

- 给出“终极动作（五选一）”：观望 / 试错小仓 / 执行计划 / 减仓 / 退出
- 只讲可验证证据：价位/结构/风控口径/现金约束/执行窗口
- 把“看起来不错”落成“怎么做、什么时候错、错了怎么办”

## 工作流（按顺序，别瞎跳）

### 1) 先确认输入（不然全是瞎扯）

- 如果用户更新持仓/现金：先更新 `data/user_holdings.json`（用户口述为准）。
- 如果用户有最新跑批目录：优先用 `outputs/run_YYYYMMDD/`。
- 如果没有跑批：先跑一次（建议带 stock 扫描与左侧候选）：

```bash
".venv/bin/python" -m llm_trading run --scan-stock --out-dir "outputs/run_YYYYMMDD"
```

### 2) 先看环境（CashSignal/Regime）再看个股（别反着来）

优先读这些：
- `outputs/run_*/holdings_user.json`（组合汇总、as_of、风险提示）
- `outputs/run_*/report.md`（最关键信息已经在这）
- （可选）单标的 `analyze` 的 `cash_signal.json` / `tushare_factors.json`

环境判定规则（保命优先）：
- cash_signal 风险偏好 = risk_off / cash_ratio>=0.7：默认不新开仓；持仓只谈减仓/退出/防守。
- 否则才讨论“执行计划/试错小仓”。

### 3) 再看持仓（先风控，再收益）

对每个持仓，必须回答三件事：
- 现在处在：趋势 / 震荡 / 破位（周线为主，日线作确认）
- 失效位是什么：优先用系统产物的止损口径（MA20/MA50/ATR）
- 账户约束：是否触发最小交易额/最大持仓数/冻结仓位等

### 4) 再看候选（右侧 + 左侧）

候选来源优先级：
1) `signals.json`（右侧，默认 bbb_weekly 的因子库扫描）
2) `signals_left.json`（左侧低吸，高赔率试错池）
3) `signals_stock.json` / `signals_left_stock.json`（stock 观察池）

筛选逻辑（KISS）：
- 右侧：趋势更确定、回撤更可控；适合“执行计划”
- 左侧：赔率高但胜率低；只适合“试错小仓”，必须止损近

### 5) 输出（用模板，不要口播）

按 `references/action_template.md` 输出：
- 一句话结论
- 终极动作（五选一）+ 执行窗口 + 2~3 条失效条件
- 证据清单（Top 5~8，引用 outputs 里的字段/价位）
- 共识 vs 分歧（模块冲突怎么处理）
- 风险提示（2~4 条）

## 禁止事项（别越界）

- 不自动实盘下单、不接券商 API、不做“保证收益/保底回撤”这种鬼话。
- 不伪造数据；缺数据就写“缺失/不确定”，别硬编。
- 结论必须可复核：给到具体文件字段、价位、触发/失效条件。

## 资源

- 输出模板：`references/action_template.md`
