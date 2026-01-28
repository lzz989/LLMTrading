# LLM辅助交易框架 - 改进计划

> 文档版本: 2026-01-24
> 目标: **因子化架构升级** + 系统化改进框架

---

## 📋 文档导航

本目录包含框架改进计划的模块化文档，按主题拆分便于维护和查阅。

### 两份“落地版”总纲（先读这个，别上来就写玄学引擎）

- `docs/博弈论框架升级方案.md`：把“博弈/流动性”落成可观测 proxy 因子（先研究验证，再接入策略）
- `docs/小散特化框架方案.md`：把“小散优势”落成 CashSignal / OpportunityScore / PositionSizing（不靠情绪）

### 核心文档

| 文档 | 说明 | 优先级 |
|------|------|--------|
| [00_constraints.md](./00_constraints.md) | 🚨 **红线约束** - 必须遵守的研究规范 | 必读 |
| [01_assessment.md](./01_assessment.md) | 📊 当前框架评估 (8.2/10) | 了解现状 |
| [02_factor_architecture.md](./02_factor_architecture.md) | 🏗️ **因子化架构升级方案** | P0 |
| [03_code_improvements.md](./03_code_improvements.md) | 🔧 代码层面改进 | P1-P3 |
| [04_strategy_modules.md](./04_strategy_modules.md) | 📈 交易策略增强模块 | P1-P2 |
| [05_roadmap.md](./05_roadmap.md) | 🗺️ 实施路线图 | 规划 |
| [06_risks.md](./06_risks.md) | ⚠️ 风险提示和应对 | 必读 |
| [07_appendix.md](./07_appendix.md) | 📎 附录（配置模板、SQL示例等） | 参考 |
| [08_productization_club_style.md](./08_productization_club_style.md) | 🧩 产品化结构对标（俱乐部广告可学点） | P2 |

---

## 🎯 当前状态

### 已完成
- ✅ 因子库骨架（17个因子）
- ✅ 涨停分级因子（ZTTypeFactor）
- ✅ CLI 入口变薄：`cli.py` 仅 argparse + `cli_commands.py` 承载 `cmd_*`
- ✅ 本地 DuckDB：`sql-init/sql-sync/sql-query`（data/ + outputs/ SQL 化）
- ✅ Phase1：因子研究最小闭环（`factor-research` 命令 + `outputs/factor_reports_*` + DuckDB 视图）
- ✅ Phase2：评分/过滤器并行输出（`game_theory_factors/opportunity_score/cash_signal/position_sizing` + scan `--min-score` + DuckDB 视图）
- ✅ Phase3：策略迁移到因子配置（`scan-strategy/strategy-align` + analyze 可选 `strategy_signal.json`）
- ✅ Phase4：动态权重研究闭环（`dynamic-weights` 命令 + `outputs/walk_forward_*` + DuckDB 视图）

### 进行中
- 🔄 代码质量治理（持续）：逐步把 print 换成 logging、补集成测试/回归样本
- ✅ Phase5 验收线已通过：核心模块覆盖率>=80%，且 except Exception 收敛>=50%

### 待启动
- ⏳（可选）更严谨的成本/滑点模型与样本外验证门槛（用于“上线前”风控验收）
- ⏳（可选）多数据源容错/实盘接口/可视化面板（P3，按需再搞）

---

## 🚀 快速开始

### 1. 了解约束（必读）
```bash
cat 00_constraints.md
```
任何代码变更都必须遵守红线约束，违反直接打回。

### 2. 查看当前评估
```bash
cat 01_assessment.md
```
了解框架的优势和局限。

### 3. 理解因子架构
```bash
cat 02_factor_architecture.md
```
这是 P0 优先级的核心改进方向。

### 4. 查看实施计划
```bash
cat 05_roadmap.md
```
了解各阶段的里程碑和交付物。

---

## 🧪 双榜模式（右侧趋势 + 左侧高赔率）

默认 `run` 会同时产出：
- 右侧趋势候选榜（signals_top）
- 左侧高赔率候选榜（signals_top_left）

一键命令（默认已启用左侧榜）：
```bash
".venv/bin/python" -m llm_trading run --out-dir "outputs/run_YYYYMMDD"
```

可选参数（与 CLI 一致）：
- `--scan-left/--no-scan-left`：开关左侧高赔率榜
- `--scan-left-strategy left_dip_rr`：左侧榜策略 key（默认 `left_dip_rr`）
- `--scan-left-top-k 30`：左侧榜 TopK
- `--scan-stock`：额外产出股票候选榜 + 左侧股票榜

验收点（report.md）：
- `signals_top` / `signals_top_left`
- `signals_top_stock` / `signals_top_left_stock`（仅 `--scan-stock`）

主要产物：
- `signals.json`（右侧趋势候选）
- `signals_left.json`（左侧高赔率候选）
- `signals_stock.json` / `signals_left_stock.json`（仅 `--scan-stock`）

---

## 📐 优先级说明

| 优先级 | 含义 | 时间窗口 |
|--------|------|----------|
| **P0** | 必须完成，阻塞后续工作 | 2周内 |
| **P1** | 重要，影响稳定性和可维护性 | 1月内 |
| **P2** | 一般，有价值但不紧急 | 季度内 |
| **P3** | 锦上添花，有空再做 | 无限期 |

---

## 🔗 相关资源

- 主项目 README: `../README.md`
- 代码规范: `../AGENTS.md`
- 因子库代码: `../llm_trading/factors/`
- 落地方案（总纲）:
  - `../docs/博弈论框架升级方案.md`
  - `../docs/小散特化框架方案.md`

---

## 📝 变更日志

| 日期 | 变更 |
|------|------|
| 2026-01-23 | 初始化模块化文档结构 |
| 2026-01-23 | 从单体 IMPROVEMENT_PLAN.md 拆分 |

---

**下次 Review**: 2026-02-23
