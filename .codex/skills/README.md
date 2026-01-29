# Codex Repo Skills（strategy / research / backtest）

> 目标：把“每次都要重新解释一遍”的脑力活，固化成可复用 SOP。  
> 这玩意不是后台常驻 agent，也不是自动下单系统；就是一套**写死口径 + 可复核产出**的工作流。

## 目录结构

每个 skill 都在 `.codex/skills/<name>/`：

- `SKILL.md`：这个 skill 的工作流与边界（最重要）
- `references/`（仓库根目录）：模板/检查清单（strategy/research/backtest 共用）
- `scripts/`：（可选）能直接跑的脚本，负责把“线索/实验”落盘

当前内置 3 个：

| skill | 适用场景 | 默认输出 | 入口 |
|---|---|---|---|
| `strategy` | 持仓/候选收敛成“可执行动作 + 失效条件”（保命优先） | `outputs/agents/strategy_action.md` | `.codex/skills/strategy/SKILL.md` |
| `research` | 基本面/行业/技术调研（证据-不确定性）+ 可选抓新闻线索 | `outputs/agents/research.md` | `.codex/skills/research/SKILL.md` |
| `backtest` | 回测某个规则/参数是否靠谱（严禁未来函数） | `outputs/agents/backtest_report.md` | `.codex/skills/backtest/SKILL.md` |
| `five_schools` | 五派“教主快评”：先快评筛子，后点名深挖 | `outputs/agents/five_schools.md` | `.codex/skills/five_schools/SKILL.md` |
| `hotlines` | 主线热度/拥挤度识别（行情驱动，输出主线 TopN） | `outputs/agents/hotlines.md` | `.codex/skills/hotlines/SKILL.md` |

## 怎么用（最稳的触发方式）

你在 Codex CLI 里下任务时，直接把 skill 名字和路径写死，别让模型“猜”：

- strategy：
  - 例：`请使用 repo-skill strategy（读取 .codex/skills/strategy/SKILL.md 严格按流程），基于最新 outputs/run_* 生成 outputs/agents/strategy_action.md。`
- research：
  - 例：`请使用 repo-skill research（读取 .codex/skills/research/SKILL.md），先抓取新闻线索再做证据汇总，输出 outputs/agents/research.md。`
- backtest：
  - 例：`请使用 repo-skill backtest（读取 .codex/skills/backtest/SKILL.md），按 t+1 开盘成交假设跑回测并输出报告。`

如果你的 Codex CLI 已支持自动识别 repo-scope skills，那你只写 `strategy/research/backtest` 也能触发；但我建议你仍然写路径，省得版本差异把你坑了。

## 使用场景（典型工作流）

### 1) 日常“跑一次出报告”（组合/持仓）

1. 跑批：产出 `outputs/run_YYYYMMDD/`  
2. 让 `strategy` 把信息收敛成“终极动作（五选一）+ 失效条件”  

```bash
".venv/bin/python" -m llm_trading run --scan-stock --out-dir "outputs/run_YYYYMMDD"
```

### 2) 有争议的消息/舆情：先抓线索，再谈观点

`research` 支持“先抓新闻线索 → 再按模板归纳”，避免空口扯：

```bash
mkdir -p outputs/agents
".venv/bin/python" .codex/skills/research/scripts/news_digest.py \
  --query "拒绝题材炒作" \
  --pages 3 --page-size 10 \
  --out-json outputs/agents/news_raw.json \
  --out-md outputs/agents/news_digest.md
```

然后再让 `research` 读取 `news_raw.json/news_digest.md`，输出 `research.md`（结论-证据-不确定性）。

### 3) 规则争论：别吵，回测说话

`backtest` 的原则是：先 MVP，后扩展；默认 t+1 开盘成交，写清成本，防止自欺欺人。

```bash
".venv/bin/python" .codex/skills/backtest/scripts/backtest_exit_signals.py \
  --asset etf \
  --symbols sh518880,sh159937 \
  --start 2015-01-01 --end 2026-01-23 \
  --fee-bps 10 --slippage-bps 5 \
  --out outputs/agents/backtest_report.md
```

## 最佳实践（别把自己当韭菜割）

- `strategy`：先环境（cash_signal/regime）再个股；只输出“可验证价位/结构”；永远写失效条件。
- `research`：媒体稿=线索；关键结论必须回到公告/财报/监管披露；不确定就写“不确定”，别硬编。
- `backtest`：严禁未来函数；不写成本=不配谈结论；至少一个基线对照（买入持有/固定止盈止损）。

## 反例（禁止事项）

- 不自动实盘下单、不接券商 API、不做收益承诺。
- 不伪造数据；缺数据就落盘记录缺口，别靠“想象力”补。
- 不把单次回测当预测：结论必须带适用环境与失效条件。
