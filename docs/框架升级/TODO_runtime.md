# TODO（戳一次跑完版）

> 目的：把“日常跑批=一条命令→产出报告→能复盘/能审计”做成肌肉记忆，别每次都手动戳。
>
> 说明：这里不写玄学 roadmap，只写你今天能做、做完就爽的活。

## 已做硬（P0）

- [x] `run` 默认走因子库扫描：`scan-strategy(etf, bbb_weekly)`；失败才回退 legacy `scan-etf`
- [x] 稳健切换：默认 shadow 跑一份 legacy `scan-etf` + 自动 `strategy-align` 对齐报告
- [x] 修复 legacy `scan-etf` 在 ADX 指标上的崩溃（单标的不拖垮整次扫描）
- [x] `fri_close_mon_open` 执行窗：周末也允许输出 rebalance（方便周一开盘前准备）
- [x] `rebalance-user` 支持 `signals(schema_version=1)`，并能为 scan-strategy 候选补齐止损口径（MA20/MA50/ATR）
- [x] `run/report.md` 已补齐：候选清单（signals_top）+ 仓位计划（position_plan）
- [x] diagnostics 不再互相覆盖：除 `diagnostics.json` 外额外写 `diagnostics_<cmd>.json`

## 还没做完（P1）

- [x] stock 的“因子库扫描”闭环（默认 source=auto；支持 whitelist/blacklist；指数成分池有缓存）
- [x] `run` 的多资产编排（etf + stock 可选：`--scan-stock`）
- [x] “持仓深度复盘”一键化：`run --deep-holdings` 逐标的跑 `analyze --method all` 并聚合 `report_holdings.md`
- [x] 风控解释更直观：当 `orders_next_open=0` 时，report 里输出 blockers 摘要（exposure/min_trade_notional 等）

## 你要怎么用（最少动作）

### 一键跑批（推荐）

```bash
".venv/bin/python" -m llm_trading run --out-dir "outputs/run_YYYYMMDD"
```

验收：打开 `outputs/run_YYYYMMDD/report.md`，能看到：
- alerts（持仓风险清单）
- signals_top（ETF 候选Top）
- signals_top_stock（若开启 `--scan-stock`）
- position_plan（目标仓位/止损口径）
- orders_next_open（若为空，看 warnings）
