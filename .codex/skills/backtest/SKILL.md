---
name: backtest
description: 量化回测（严格无未来函数、成本口径明确、可复现）。适用于：验证因子/入场出场规则/参数是否有效，或对比不同出场策略（MACD/布林带/止盈等）的收益-回撤-持有期表现；必须输出“可复核”的回测报告（区间、成本、执行假设、核心指标、局限性）。
---

# 量化回测（保命版：先防未来函数）

## 输入 / 输出约定

- 默认输入：`data/cache/*/*.csv`（不够就让框架按需抓取并落缓存）。
- 默认输出：`outputs/agents/backtest_report.md`

## 先立规矩（不然回测全是假的）

- 执行假设：默认 **t 日收盘产生信号 → t+1 开盘成交**（保守、最不容易骗自己）。
- 成本假设必须写：手续费/滑点/最低佣金（小资金不写就是骗人）。
- 严禁未来函数：任何 rolling 统计都必须只用过去数据；突破类信号用 `shift(1)` 排除当天。

## 工作流（按这个走，效率最高）

### 1) 明确实验定义

- 标的（symbols）
- 频率（日/周）
- 区间（start/end）
- 入场规则（必须写清楚“强趋势/震荡/反转”属于哪个 regime）
- 出场规则（至少一个对照组）
- 成本与滑点（bps 或现金口径）

### 2) 先做最小可复现实验（MVP）

先只做：
- 单标的
- 非重叠交易（flat 才能进）
- 单一仓位（全仓/固定仓位）

把“规则对比”跑出来后，再谈组合、轮动、仓位管理。

### 3) 跑脚本（推荐）

本 skill 自带一个“出场对比”的脚本（可改参数，不要魔改逻辑）：

```bash
".venv/bin/python" .codex/skills/backtest/scripts/backtest_exit_signals.py \
  --asset etf \
  --symbols sh518880,sh159937 \
  --start 2015-01-01 --end 2026-01-23 \
  --fee-bps 10 --slippage-bps 5 \
  --out outputs/agents/backtest_report.md
```

### 4) 输出报告（必须可复核）

按 `references/report_template.md` 输出，至少包含：
- CAGR/年化、最大回撤、交易次数、胜率、平均/中位收益
- 平均持有期 + P90 持有期（出场规则的“磨人程度”很关键）
- 局限性（入场定义是否过拟合、样本外风险、成本敏感性）

## 禁止事项（防止自欺欺人）

- 严禁未来函数；看见 “today included” 的 rolling/最高价直接判死刑。
- 不写成本 = 不配谈结论（尤其是高换手）。
- 不把回测当预测：必须写清适用环境与失效条件。

## 资源

- 回测检查清单：`references/backtest_checklist.md`
- 报告模板：`references/report_template.md`
