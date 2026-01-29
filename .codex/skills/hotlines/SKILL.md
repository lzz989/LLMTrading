---
name: hotlines
description: 主线热度/拥挤度识别（偏行情驱动，TuShare 优先+AkShare 兜底）。输出主线 TopN + 风险提示 + 可买ETF映射；落盘 md+json，便于复核与入仓库。
---

# 主线热度（hotlines）

## 输入 / 输出约定

- 输入：`config/hotlines_universe.yaml`（可买 ETF 宇宙 + tags）
- 输出（默认）：
  - `outputs/agents/hotlines.md`
  - `outputs/agents/hotlines.json`

## 你要的结果是什么

- 把“主线热门赛道”从嘴炮变成可复核的量化口径：
  - 哪些 tags 在热（趋势/成交量/强度）
  - 哪些 tags 过热（拥挤度/波动上升/离均线太远）
  - 对应到哪些 **可买 ETF**（代表作）
- 不强行抓热榜网页（不稳定/反爬），默认走结构化行情数据。

补充口径（很关键）：
- `tags`：底座标签（尽量走“官方分类”口径：股票ETF/商品ETF/宽基/跨境…），保持稳定方便复盘。
- `topic_tags`：从 ETF 名称派生的主题标签（半导体/券商/传媒/中概互联…），用于“主线热度”聚合排序。

## 工作流（按顺序）

1) 读取 universe（ETF + tags）
2) 拉取每只 ETF 的日线（TuShare 优先，失败回退 AkShare；内部会做缓存）
3) 计算热度指标（KISS）
- ret_5d/10d/20d
- vol_ratio_20d（近5日均量 / 20日均量）
- atr_pct_14（波动/拥挤度 proxy）
- close_vs_ma20_pct（趋势强度）
- dd_from_20d_high_pct（离高点回撤，判断追高风险）
4) 聚合到 tag 维度，输出 TopN + 风险 flags

## 禁止事项

- 不把不可交易标的输出成“执行票”（可以作为线索，但必须标注 tradable=false）。
- 不做收益承诺；只做“热度识别 + 风险提示”。
