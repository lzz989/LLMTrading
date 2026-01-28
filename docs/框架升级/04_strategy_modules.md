# 交易执行与策略增强模块（小散特化优先）

> 文档版本: 2026-01-24  
> 模块: 策略功能扩展（从“信号”到“可执行”）  
> 优先级: **P1（评分/过滤器）**，P2（花活/动态权重/组合优化）

本文件不再写“金字塔/资金阶梯/凯利神功”这种容易把系统写成花活堆的东西。  
我们按两份落地版总纲做：**先把小散的执行优势工程化 + 把博弈/流动性做成过滤器**。

必须遵守：`docs/框架升级/00_constraints.md`

---

## 4.1 CashSignal（环境不对就空仓）

对齐：`docs/小散特化框架方案.md`

**目的**：把“忍住不交易”写进系统里，降低系统性回撤。

建议落地文件：`llm_trading/cash_signal.py`（已实现）

**输入（MVP）**：
- `market_regime`（已有：`llm_trading/market_regime.py`）
- 波动/风险温度计（ATR/带宽、可选 ERP proxy）
- 本周候选机会质量（OpportunityScore 的分布：top/mean）

**输出（结构化，必须可 SQL）**：`outputs/<run>/cash_signal.json`
```json
{
  "schema": "llm_trading.cash_signal.v1",
  "as_of": "2026-01-23",
  "should_stay_cash": true,
  "cash_ratio": 0.8,
  "reason": "regime=bear + 波动偏高 + 机会评分不足",
  "expected_duration_days": 10
}
```

执行口径（建议）：
- 用到当日收盘的判断 → 默认 T+1 执行
- cash_ratio 是“建议”，最终仍受持仓/冻结/网格特殊规则约束（如有）

---

## 4.2 OpportunityScore（只做肥球，拒绝垃圾交易）

对齐：`docs/小散特化框架方案.md`

**目的**：把“看起来不错”落成一个可解释的 0~1 分数，用于：
- scan：排序/过滤（`--min-score`）
- analyze：给出“是否值得研究/是否具备交易结构”
- holdings/rebalance：候选质量闸门

建议落地文件：`llm_trading/opportunity_score.py`（已实现）

**评分维度（先 KISS）**：
- `trend`：趋势是否明确（MA/MACD/Ichimoku/ADX）
- `risk_reward`：是否能给出清晰失效位（关键位距离/ATR/pullback）
- `liquidity`：成交额门槛 + 量能确认（volume_ratio/amount）
- `regime`：市场状态是否允许这类交易
- `trap_risk`：流动性陷阱风险（来自 4.4 的 `liquidity_trap`）

输出：`outputs/<run>/opportunity_score.json`（必须可 SQL）
```json
{
  "schema": "llm_trading.opportunity_score.v1",
  "symbol": "sh510300",
  "asset": "etf",
  "as_of": "2026-01-23",
  "total_score": 0.78,
  "min_score": 0.70,
  "verdict": "tradeable",
  "components": { "trend": 0.82, "regime": 0.70, "risk_reward": 0.75, "liquidity": 0.85, "trap_risk": 0.20 },
  "details": { "key_level": "MA50", "invalidation": "close < MA50", "expected_holding_days": 10 }
}
```

> 分数不是为了“让你多交易”，是为了“让你更敢拒绝交易”。

---

## 4.3 PositionSizing（成本敏感仓位：5 元最低佣金是刀）

对齐：`docs/小散特化框架方案.md`

建议落地文件：`llm_trading/position_sizing.py`（已实现）

**现实约束（系统必须显式处理）**：
- 最低佣金 5 元（小额交易会被磨死）
- 最小交易额（仓库既有口径：建议 `>= 2000`）
- T+1（信号/执行要讲清楚，不许用“当日收盘”装成可当天买卖）

输出：`outputs/<run>/position_sizing.json`
```json
{
  "schema": "llm_trading.position_sizing.v1",
  "symbol": "sh510300",
  "as_of": "2026-01-23",
  "max_position_pct": 0.30,
  "suggest_position_pct": 0.18,
  "reason": "score=0.78, confidence=0.60, regime=neutral"
}
```

> 仓位别先碰“凯利”，除非你能拿出样本外胜率/赔率统计；否则就是拿钱给自己上强度。

---

## 4.4 博弈/流动性过滤器（别追在陷阱里）

对齐：`docs/博弈论框架升级方案.md`

候选因子（MVP）：
- `liquidity_trap`：假突破/假跌破（扫流动性）
- `capitulation`：恐慌释放（只表示“进入观察窗口”，不是抄底按钮）
- `fomo`：追涨狂热（只表示“别追/考虑兑现”，不是做空按钮）

建议输出：`outputs/<run>/game_theory_factors.json`（结构见 02/07）

**接入原则（KISS）**：
- 先当过滤器/扣分项：如 `bull_trap` 强时禁止追突破
- 再当触发前置条件：如 `bear_trap` 后出现右侧确认才允许介入

---

## 4.5 现有可复用模块（别重复造轮子）

仓库已有这些“执行相关”能力，先复用、少写新代码：
- 成本模型：`llm_trading/costs.py`
- 可交易性过滤：`llm_trading/tradeability.py`
- 仓位/止盈止损相关：`llm_trading/positioning.py`、`llm_trading/take_profit.py`
- 市场状态：`llm_trading/market_regime.py`

---

## 4.6 后置（P2+，有证据/有需求再做）

以下属于“锦上添花”，别阻塞 P0/P1 的研究与评分落地：
- 动态权重/元策略（必须严格 walk-forward）
- 组合优化（Markowitz/Risk Parity 等）
- 金字塔加仓/资金阶梯/复杂止盈阶梯（先跑通“少交易+低成本+高质量候选”再说）

---

## 4.7 集成示意（先并行输出，降低回归风险）

```
数据(OHLCV/TuShare) ──→ 因子(technical + game_theory + tushare_pack)
             │
             ├──→ OpportunityScore（scan/analyze/holdings 排序与过滤）
             ├──→ CashSignal（账户级风险开关：现金比例建议）
             └──→ PositionSizing（成本敏感仓位建议）

备注：老策略信号先不强制替换，只做“并行输出→对齐验证→逐步启用”。
```

---

## 相关文档

- [因子化架构](./02_factor_architecture.md) - 因子/评分/输出规范
- [实施路线图](./05_roadmap.md) - 里程碑/交付物/验收标准
- [风险提示](./06_risks.md) - 风险与容错
