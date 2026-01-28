# 因子化架构升级（P0 核心）

> 文档版本: 2026-01-24  
> 模块: 架构升级方案  
> 优先级: **P0 - 最高优先级**

本文件是“因子化升级”的工程落地点：**先把现象做成可计算因子/评分，再用研究闭环证明有效，最后才允许进策略。**

必须遵守红线：`docs/框架升级/00_constraints.md`

并与两份“落地版总纲”保持一致：
- `docs/博弈论框架升级方案.md`：博弈/流动性 proxy → 因子化 → 研究验证 → 过滤器/风控接入
- `docs/小散特化框架方案.md`：CashSignal / OpportunityScore / PositionSizing（成本敏感 + 少交易）

---

## 目标（别写成玄学引擎）

把框架从“静态策略堆叠”升级为“**因子/评分/过滤器**驱动”，实现：
- 因子计算一次，多处复用（scan/analyze/holdings）
- 有效性可验证（IC/IR/衰减/成本/样本外）
- 输出可复现、可审计、可 SQL（DuckDB）
- 小散执行优势工程化（空仓/只做肥球/成本敏感仓位）

---

## 2.1 已有：技术因子库骨架 ✅（17 个）

目录结构（已存在）：

```
llm_trading/factors/
├── __init__.py
├── base.py          # Factor基类、FactorResult、FactorRegistry
├── trend.py         # 趋势：MA/MACD/ADX/Ichimoku
├── momentum.py      # 动量：RSI/ROC/Momentum
├── volume.py        # 量能：VolumeRatio/OBV/MFI
├── volatility.py    # 波动：ATR/Bollinger
├── pattern.py       # 形态：ZTType/Pullback/CandlePattern
└── market.py        # 市场：Regime/Breadth
```

> 这些因子是 OpportunityScore 的“可复用输入维度”，但它们本身不等于“可交易信号”。别偷懒把分数当买卖按钮。

---

## 2.2 新增候选因子池（对齐两份总纲）

### 2.2.1 博弈/流动性 proxy（先做成因子，再研究）

放置建议：`llm_trading/factors/game_theory.py`（或拆成 `game_theory/*.py`，先别过度设计）

MVP 候选（P1 进入“研究闭环”跑一遍）：
- `liquidity_trap`：假突破/假跌破（扫流动性）
- `stop_cluster`：止损聚集区 proxy（摆动高低/关键均线/整数位）
- `capitulation`：恐慌释放 proxy（大阴线/ATR 扩张/放量/RSI 低位等组合）
- `fomo`：追涨狂热 proxy（大阳线/ATR 扩张/放量/偏离均线等组合）
- `wyckoff_phase_proxy`：吸筹/派发相似度分数（两个分数比硬分类更稳）

注意：
- 这些多数更适合作为 **过滤器/风控项/触发前置条件**，别当“开仓圣杯”。
- 默认执行口径：若因子用到当日收盘，则 T+1 执行。

### 2.2.2 TuShare 补强因子（宏观/资金/微观结构）

仓库已有输出入口：`llm_trading/tushare_factors.py`（通常在 `analyze --method all` 时生成 `tushare_factors.json`）。

建议纳入“研究闭环”的字段方向（具体以 TuShare 返回为准）：
- **ERP proxy（风险温度计）**：
  - `shibor_1y` vs `cn_10y_yield`：二选一或都存（研究阶段先都保留，别拍脑袋）
  - 输出只做“风险加权/现金比例建议”，不做“买卖按钮”
- **北向/南向资金 proxy**：`moneyflow_hsgt` 这类
  - 只能解释风险偏好/边际变化，不能当确定性买卖理由
- **微观交易结构 proxy**：大单/超大单净流等
  - 只作为“资金确认”维度，必须容错（积分/接口/限流都可能挂）

硬约束（别作死）：
- 必须缓存 + TTL + 降级：TuShare 挂了不能把主流程搞崩（可以输出 `*_error.txt`，但整体流程要继续）
- 字段口径必须写清：单位、复权、对齐日期、缺失处理

---

## 2.3 从“因子”到“小散可执行”的三件套（评分/现金/仓位）

这仨不是“因子”，是把因子翻译成可执行规则的中间层（对齐 `docs/小散特化框架方案.md`）：

1) **CashSignal**：环境不对就空仓（或提高现金比例）  
2) **OpportunityScore**：只做肥球（0~1 的可解释评分，带失效位/预期持有期）  
3) **PositionSizing**：成本敏感的仓位建议（含 5 元最低佣金、最小交易额、T+1 约束提示）

放置建议（先 KISS）：
- `llm_trading/cash_signal.py`
- `llm_trading/opportunity_score.py`
- `llm_trading/position_sizing.py`

集成点建议（先并行输出，不强制改老信号）：
- `analyze`：输出 `opportunity_score.json / cash_signal.json / position_sizing.json`
- `scan-*`：排名结果追加 `total_score` 并支持 `--min-score`
- `holdings-user/rebalance-user`：用 CashSignal 控制“整体风险开关”，用 OpportunityScore 控制“候选质量”

---

## 2.4 P0：因子研究最小闭环（必须先做）

目标：让“这个因子到底有没有用”变成 **可复现 + 可审计 + 可 SQL 查询** 的报告。

文件：`llm_trading/factors/research.py`（已实现）

必须包含：
- forward returns 对齐（`shift(-n)`，严禁未来函数；执行默认 T+1）
- IC/IR + 衰减（1/5/10/20）
- 可交易性过滤/标注（涨跌停/停牌/流动性不足，至少标注剔除比例）
- 成本敏感性（最小佣金 5 元 + 滑点）
- 样本外（walk-forward 或至少时间切分）
- 产出结构化报告（`outputs/factor_reports/` 的 csv/json），并可被 DuckDB 查询

> 研究闭环是“因子化升级”的唯一硬门槛：过不了这关，后面别谈动态权重/元策略。

---

## 2.5 输出规范（必须可 SQL）

### 2.5.1 单因子输出（FactorResult）

因子输出保持 `FactorResult(score/direction/confidence/details)` 的统一格式（见 `llm_trading/factors/base.py`）。

### 2.5.2 分析产物（analyze 目录下的 JSON）

建议新增/规范化这些文件（都要带 schema + 稳定字段）：
- `tushare_factors.json`（已有）
- `game_theory_factors.json`（新增：3.x 博弈因子打包输出）
- `cash_signal.json`（新增）
- `opportunity_score.json`（新增）
- `position_sizing.json`（新增）

并在 `llm_trading/warehouse.py` 增加对应 `wh.v_*` 视图，把 JSON 扁平化成可查表字段（保持 KISS，别嵌套大对象）。

---

## 2.6 后置（有证据再做）

- **动态权重/元策略**：必须基于研究闭环给出的“不同 regime 下稳定差异”，并严格 walk-forward  
- **策略迁移到纯配置**：价值是“去重复/可复用”，但不应阻塞先把评分/过滤器输出跑起来

结论一句话：**先把可验证的东西写出来，再谈优雅架构。**
