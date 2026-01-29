---
name: five_schools
description: 五派“教主快评”（威科夫/缠论/一目/海龟/VSA）。对拟上车标的先给每派 1 句话 + 1 条失效条件；默认不调用 LLM（你点名深挖才启用）。
---

# 五派“教主快评”（先快评，后深挖）

## 输入 / 输出约定

- 输入：你指定 `asset + symbols`；可选复用已有 `outputs/run_*`（若存在 deep-holdings 产物则更快）。
- 输出（默认）：
  - `outputs/agents/five_schools.md`
  - `outputs/agents/five_schools.json`

## 你要的结果是什么

- 每个标的输出 5 派快评：
  - 每派 **1 句话结论 + 1 条失效条件**（必须可验证：价位/结构锚点）
  - 口吻：A 股江湖“刻薄直给”，但不胡编数据
- 快评只负责“筛子”，不是终极决策；你对某派有兴趣再点名深挖（可选 LLM）。

## 工作流（按顺序）

1) 尽量复用已有产物（省钱省时间）
- 如果你有最新 `outputs/run_YYYYMMDD`，且跑过 `--deep-holdings`：
  - `run_dir/holdings_deep/*` 里已有单标的 `analyze --method all` 的结果
  - five_schools 会优先读取这些文件生成快评

2) 如果没有现成产物
- five_schools 会按需要抓取行情并用脚本计算五派的关键锚点（不依赖 LLM）。

3) 输出落盘（必须可复核）
- 生成 `five_schools.md/json`，任何缺数据必须明确标注“缺失/降级”。

## 禁止事项

- 不自动实盘下单、不接券商 API。
- 不伪造数据；缺数据就写缺数据。
- 不输出“只写代码不写结论”的垃圾报告：必须给可执行的失效条件。

