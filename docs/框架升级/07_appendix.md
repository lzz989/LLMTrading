# 附录

> 文档版本: 2026-01-24
> 模块: 参考资料

---

## 7.1 新增模块清单

| 模块 | 文件/输出 | 功能 | 优先级 | 状态 |
|------|----------|------|--------|------|
| 技术因子库 | `llm_trading/factors/*.py` | 17个技术因子（趋势/动量/量能/波动/形态/市场） | P0 | ✅ 已完成 |
| 因子研究闭环 | `llm_trading/factors/research.py` → `outputs/factor_reports_*/*` | IC/IR/衰减/成本/样本外（walk-forward） | P0 | ✅ 已完成 |
| TuShare 因子包 | `llm_trading/tushare_factors.py` → `outputs/*/tushare_factors.json` | ERP/HSGT/微观交易结构 proxy | P1 | ✅ 输出入口已存在 |
| 博弈/流动性因子包 | `llm_trading/factors/game_theory.py` → `outputs/*/game_theory_factors.json` | liquidity_trap/stop_cluster/capitulation/fomo/... | P1 | ✅ 已完成 |
| OpportunityScore | `llm_trading/opportunity_score.py` → `outputs/*/opportunity_score.json` | 机会评分（可解释 + 失效位） | P1 | ✅ 已完成 |
| CashSignal | `llm_trading/cash_signal.py` → `outputs/*/cash_signal.json` | 空仓/现金比例建议（风险开关） | P1 | ✅ 已完成 |
| PositionSizing | `llm_trading/position_sizing.py` → `outputs/*/position_sizing.json` | 成本敏感仓位建议（含 5 元最低佣金） | P1 | ✅ 已完成 |
|（可选）策略配置迁移 | `config/strategy_configs.yaml` | 去重复/复用（新旧信号对齐） | P2 | ✅ 已完成 |
| 动态权重 | `llm_trading/factors/dynamic_weights.py` → `outputs/walk_forward_*/*` | regime-aware（严格 walk-forward） | P2 | ✅ 已完成 |
| 命令模块化 | `llm_trading/commands/*` | `cmd_*` 按域拆分（避免 cli_commands.py 继续屎山） | P2 | ✅ 已完成 |

---

## 7.2 CLI 命令（现状 + 规划）

### 已存在（能直接跑）

```bash
# 单标的分析（建议 --method all 同时产出多流派 + tushare_factors.json）
".venv/bin/python" -m llm_trading analyze --asset etf --symbol sh510300 --method all --out-dir "outputs/analyze_demo"

# 扫描（输出候选列表到 outputs/）
".venv/bin/python" -m llm_trading scan-etf --limit 200 --min-weeks 60 --min-score 0.70 --out-dir "outputs/scan_etf"

# Phase1：因子研究闭环（批量 IC/IR/衰减/成本/样本外）
".venv/bin/python" -m llm_trading factor-research --asset etf --freq weekly --limit 200

# Phase3：策略配置化扫描/对齐
".venv/bin/python" -m llm_trading scan-strategy --asset etf --strategy conservative --top-k 30
".venv/bin/python" -m llm_trading strategy-align --base outputs/A/signals.json --new outputs/B/signals.json

# Phase4：动态权重研究闭环（walk-forward）
".venv/bin/python" -m llm_trading dynamic-weights --asset etf --freq weekly --limit 200 --context-index sh000300+sh000905

# 本地 DuckDB（data/ + outputs/ SQL 化）
".venv/bin/python" -m llm_trading sql-init
".venv/bin/python" -m llm_trading sql-sync
".venv/bin/python" -m llm_trading sql-query --sql "select count(*) from wh.file_catalog" --limit -1
```

### 备注

- Phase2 的四类产物（`game_theory_factors/opportunity_score/cash_signal/position_sizing`）目前在 `analyze` 中默认并行输出（不改变原有信号口径）。

---

## 7.3 配置文件模板

> 说明：这部分属于 Phase3/Phase4（可选/后置）。Phase2 先把评分/过滤器并行输出跑起来，别急着上配置化大改。

### 策略配置 (config/strategy_configs.yaml)

```yaml
strategies:
  bbb_weekly:
    factor_weights:
      ma_cross: 0.25
      macd: 0.20
      regime: 0.30
      pullback: 0.15
      atr: 0.10
    entry_threshold: 0.65
    exit_threshold: 0.35
    allowed_regimes: ["bull", "neutral"]
  # shortline_t1t3：已精简移除（超短线/周内短线模块不再维护）

  aggressive:
    factor_weights:
      momentum: 0.30
      roc: 0.25
      volume_ratio: 0.20
      macd: 0.15
      regime: 0.10
    entry_threshold: 0.60
    exit_threshold: 0.35
    allowed_regimes: ["bull"]

  conservative:
    factor_weights:
      ma_cross: 0.20
      regime: 0.40
      atr: 0.20
      pullback: 0.20
    entry_threshold: 0.70
    exit_threshold: 0.45
    allowed_regimes: ["bull", "neutral"]
```

### Regime权重配置 (config/regime_weights.yaml)

```yaml
regime_weights:
  bull:
    ma_cross: 0.30
    momentum: 0.25
    rsi: 0.10
    regime: 0.15
    volume_ratio: 0.20

  bear:
    ma_cross: 0.15
    momentum: 0.10
    rsi: 0.25
    atr: 0.20
    regime: 0.30

  neutral:
    ma_cross: 0.20
    macd: 0.15
    rsi: 0.15
    bollinger: 0.15
    volume_ratio: 0.15
    regime: 0.20
```

### 风控配置 (config/risk_config.yaml)

```yaml
account_risk:
  max_daily_drawdown: 0.05
  max_weekly_drawdown: 0.10
  max_total_drawdown: 0.20
  cooldown_days_after_circuit: 5

position_limits:
  max_single_position: 0.30
  max_sector_concentration: 0.40
  max_correlation_threshold: 0.85

stop_loss:
  default_pct: 0.06
  trailing_activation: 0.10
  trailing_distance: 0.05
```

---

## 7.4 因子清单详情

### 趋势因子

| 因子名 | 文件 | 参数 | 输出 |
|--------|------|------|------|
| MACrossFactor | trend.py | fast=10, slow=50 | bullish/bearish/neutral |
| MACDFactor | trend.py | fast=12, slow=26, signal=9 | 金叉/死叉/强度 |
| ADXFactor | trend.py | period=14 | 趋势强度0-100 |
| IchimokuFactor | trend.py | tenkan=9, kijun=26 | 云上/云下/云中 |

### 动量因子

| 因子名 | 文件 | 参数 | 输出 |
|--------|------|------|------|
| RSIFactor | momentum.py | period=14 | 超买(>70)/超卖(<30) |
| ROCFactor | momentum.py | period=12 | 变化率百分比 |
| MomentumFactor | momentum.py | periods=[5,10,20] | 多周期一致性 |

### 量能因子

| 因子名 | 文件 | 参数 | 输出 |
|--------|------|------|------|
| VolumeRatioFactor | volume.py | period=20 | 量比 |
| OBVFactor | volume.py | - | 背离检测 |
| MFIFactor | volume.py | period=14 | 资金流超买超卖 |

### 波动因子

| 因子名 | 文件 | 参数 | 输出 |
|--------|------|------|------|
| ATRFactor | volatility.py | period=14 | 波动率 |
| BollingerFactor | volatility.py | period=20, std=2 | %B位置 |

### 形态因子

| 因子名 | 文件 | 参数 | 输出 |
|--------|------|------|------|
| ZTTypeFactor | pattern.py | zt_threshold=0.095 | 一字/T字/换手/烂板 |
| PullbackFactor | pattern.py | ma_period=10 | 支撑有效性 |
| CandlePatternFactor | pattern.py | - | 锤子/吞没/星线 |

### 市场因子

| 因子名 | 文件 | 参数 | 输出 |
|--------|------|------|------|
| RegimeFactor | market.py | ma_fast=50, ma_slow=200 | bull/bear/neutral |
| BreadthFactor | market.py | period=20 | 市场参与度 |

---

## 7.5 博弈/流动性因子（候选，Phase2 并行输出）

> 对齐：`docs/博弈论框架升级方案.md`  
> 口径：先做 proxy 因子 + 结构化输出 + 研究闭环，别硬写“意图预判器”。

| 因子名 | 建议文件 | 输入（MVP） | 用法（优先） |
|--------|----------|-------------|--------------|
| liquidity_trap | `llm_trading/factors/game_theory.py` | swing高低 + 关键位 + 成交量/成交额（可选） | 过滤器/扣分项（禁止追在 bull_trap） |
| stop_cluster | 同上 | swing高低/均线/整数位距离 | 距离约束（别贴着“可能的止损堆”进场） |
| capitulation | 同上 | ATR 扩张 + 放量 + RSI/长下影等 | 进入观察窗口（不是抄底按钮） |
| fomo | 同上 | ATR 扩张 + 放量 + 偏离均线/RSI 高位 | 不追/兑现倾向（不是做空按钮） |
| wyckoff_phase_proxy | 同上 | 区间宽度 + 波动收缩 + OBV/A-D 背离 | 只输出“相似度分数”，别硬分类 |

建议输出文件：`outputs/*/game_theory_factors.json`（并在 DuckDB 加 `wh.v_game_theory_factors` 扁平视图）。

---

## 7.6 TuShare 因子包字段（现有输出：tushare_factors.json）

> 输出入口已存在：`llm_trading/tushare_factors.py` → `outputs/*/tushare_factors.json` → DuckDB: `wh.v_tushare_factors`

顶层结构（固定）：
- `erp`：ERP proxy（主口径 shibor_1y；附 `rf_alt_10y/erp_alt_10y` 做对照）
- `hsgt`：沪深港通 north/south（robust z-score + score01）
- `microstructure`：个股微观结构 proxy（大单+超大单净额/成交额；缺成交额则退化为净额）

常用字段（示例，具体以输出为准）：
- `erp.erp` / `erp.equity_yield` / `erp.rf.yield` / `erp.rf_alt_10y.rf.yield`
- `hsgt.north.score01` / `hsgt.south.score01`
- `microstructure.last.net_big_ratio` / `microstructure.score01`

使用原则：
- ERP/HSGT：只做“风险温度计/风险加权”，别当确定性买卖按钮
- microstructure：只做“资金确认维度”，必须容错（积分/限流/缺数据）

---

## 7.7 Phase2 输出 Schema（定稿，后面代码必须按这个来）

> 目标：让 Phase2 的产物 **稳定可 SQL**，后续任何模块只要跟着这四个输出对齐就行。  
> 约束：字段名 snake_case；日期用 ISO（YYYY-MM-DD）；缺失用 null；严禁写一堆不可查询的大段口播。

### 7.7.1 `outputs/*/game_theory_factors.json`

用途：博弈/流动性 proxy 因子打包输出（先并行输出，默认只做过滤器/扣分项）。

```json
{
  "schema": "llm_trading.game_theory_factors.v1",
  "symbol": "sh510300",
  "asset": "etf",
  "as_of": "2026-01-23",
  "ref_date": "2026-01-23",
  "source": "factors",
  "factors": {
    "liquidity_trap": {
      "name": "liquidity_trap",
      "value": null,
      "score": 0.72,
      "direction": "bearish",
      "confidence": 0.80,
      "details": {
        "trap_kind": "bull_trap",
        "level": 1.234,
        "sweep_pct": 0.006,
        "swing_high": 1.226,
        "swing_low": 1.158,
        "high": 1.241,
        "low": 1.198,
        "close": 1.210,
        "volume_ratio": 1.35,
        "amount_ratio": 1.28
      }
    },
    "stop_cluster": {
      "name": "stop_cluster",
      "value": null,
      "score": 0.61,
      "direction": "neutral",
      "confidence": 0.70,
      "details": {
        "nearest_level": 1.200,
        "nearest_kind": "ma20",
        "nearest_distance_pct": 0.008,
        "ma20": 1.200,
        "ma60": 1.260,
        "ma200": 1.450,
        "swing_high": 1.226,
        "swing_low": 1.158,
        "integer_level": 1.200,
        "zones": [
          { "level": 1.200, "kind": "ma20", "distance_pct": 0.008 }
        ]
      }
    },
    "capitulation": {
      "name": "capitulation",
      "value": null,
      "score": 0.10,
      "direction": "neutral",
      "confidence": 0.60,
      "details": {
        "atr": 0.032,
        "move_atr": 0.40,
        "volume_ratio": 0.95,
        "rsi": 41.2
      }
    },
    "fomo": {
      "name": "fomo",
      "value": null,
      "score": 0.18,
      "direction": "neutral",
      "confidence": 0.60,
      "details": {
        "atr": 0.032,
        "move_atr": 0.55,
        "volume_ratio": 1.05,
        "rsi": 58.7
      }
    },
    "wyckoff_phase_proxy": {
      "name": "wyckoff_phase_proxy",
      "value": null,
      "score": 0.58,
      "direction": "neutral",
      "confidence": 0.60,
      "details": {
        "accumulation_like": 0.62,
        "distribution_like": 0.46,
        "range_width_pct": 0.12,
        "vol_contract_score": 0.55,
        "obv_divergence_score": 0.20
      }
    }
  }
}
```

说明（写死口径，别乱发挥）：
- `score` 统一 0~1；`direction` 表达偏多/偏空/中性。
- `wyckoff_phase_proxy.score` 建议定义为：`0.5 + 0.5 * (accumulation_like - distribution_like)`（再 clip 到 0~1），保证可解释与可研究。
- `zones` 可以留，但必须同时给 `nearest_*` 扁平字段，保证 SQL 直接可用。

### 7.7.2 `outputs/*/opportunity_score.json`

用途：机会评分（排序/过滤/解释），默认门槛 `min_score=0.70`。

```json
{
  "schema": "llm_trading.opportunity_score.v1",
  "symbol": "sh510300",
  "asset": "etf",
  "as_of": "2026-01-23",
  "ref_date": "2026-01-23",
  "source": "opportunity_score",
  "total_score": 0.78,
  "min_score": 0.70,
  "verdict": "tradeable",
  "bucket": "probe",
  "components": {
    "trend": 0.82,
    "regime": 0.70,
    "risk_reward": 0.75,
    "liquidity": 0.85,
    "trap_risk": 0.20,
    "fund_flow": null
  },
  "key_level": { "name": "ma50", "value": 1.234 },
  "invalidation": { "rule": "close_below_level", "level": 1.234, "note": "T+1 执行" },
  "expected_holding_days": 10,
  "t_plus_one": true,
  "notes": null
}
```

约束：
- `bucket`：`reject`(<0.70) / `probe`(0.70~0.80) / `plan`(>0.80)
- `key_level`/`invalidation` 必须是“可验证”的价位规则，别写玄学。

### 7.7.3 `outputs/*/cash_signal.json`

用途：账户/市场环境风险开关（现金比例建议）。**它不是买卖按钮**，是“别乱冲”的刹车。

```json
{
  "schema": "llm_trading.cash_signal.v1",
  "as_of": "2026-01-23",
  "ref_date": "2026-01-23",
  "source": "cash_signal",
  "scope": "portfolio",
  "context_index_symbol": "sh000300+sh000905",
  "should_stay_cash": true,
  "cash_ratio": 0.80,
  "risk_mode": "risk_off",
  "expected_duration_days": 10,
  "evidence": {
    "market_regime": "bear",
    "vol_state": "high",
    "erp_proxy": null,
    "north_score01": null,
    "south_score01": null
  },
  "reason": "regime=bear + vol=high + 机会评分不足",
  "notes": null
}
```

### 7.7.4 `outputs/*/position_sizing.json`

用途：成本敏感仓位建议（包含最低佣金 5 元、最小交易额、T+1 提示）。

```json
{
  "schema": "llm_trading.position_sizing.v1",
  "symbol": "sh510300",
  "asset": "etf",
  "as_of": "2026-01-23",
  "ref_date": "2026-01-23",
  "source": "position_sizing",
  "opportunity_score": 0.78,
  "confidence": 0.60,
  "max_position_pct": 0.30,
  "suggest_position_pct": 0.18,
  "equity_yuan": null,
  "cash_yuan": null,
  "price": 1.210,
  "lot_size": 1,
  "min_trade_notional_yuan": 2000,
  "suggest_trade_notional_yuan": 5000,
  "suggest_shares": 4132,
  "est_commission_yuan": 5.0,
  "est_slippage_yuan": 0.0,
  "t_plus_one": true,
  "reason": "score=0.78, confidence=0.60, regime=neutral",
  "notes": null
}
```

### 7.7.5 DuckDB 视图（扁平化字段约定）

四个视图名写死：
- `wh.v_game_theory_factors`
- `wh.v_opportunity_score`
- `wh.v_cash_signal`
- `wh.v_position_sizing`

Join key：统一用 `out_dir`（同一次 run 目录内的 JSON 都能 join）。

建议扁平字段（最小但够用）：
- `wh.v_game_theory_factors`：`liquidity_trap_score/dir/kind/level/stop_nearest_level/.../capitulation_score/fomo_score/wyckoff_score`
- `wh.v_opportunity_score`：`total_score/min_score/verdict/bucket + comp_* + key_level_* + invalidation_*`
- `wh.v_cash_signal`：`should_stay_cash/cash_ratio/risk_mode + evidence.market_regime/vol_state/...`
- `wh.v_position_sizing`：`suggest_position_pct/suggest_trade_notional_yuan/suggest_shares/est_commission_yuan/min_trade_notional_yuan/...`

> 规则：先扁平“常用字段”，其他留 struct，别为了“全量”把视图写成屎山。

---

## 7.8 数据结构参考

### FactorResult

```python
@dataclass
class FactorResult:
    name: str                    # 因子名
    value: float                 # 原始值
    score: float                 # 标准化分数 (0-1)
    direction: Literal["bullish", "bearish", "neutral"]
    confidence: float            # 置信度 (0-1)
    details: dict                # 详细信息

# 示例
FactorResult(
    name="ma_cross",
    value=1.0,
    score=0.75,
    direction="bullish",
    confidence=0.8,
    details={
        "ma_fast": 10.5,
        "ma_slow": 10.2,
        "cross_type": "golden"
    }
)
```

### StrategyConfig

```python
@dataclass
class StrategyConfig:
    name: str                    # 策略名
    factor_weights: dict[str, float]  # 因子权重
    entry_threshold: float = 0.6      # 入场阈值
    exit_threshold: float = 0.4       # 出场阈值
    allowed_regimes: list[str] = field(default_factory=lambda: ["bull", "neutral"])

# 示例
StrategyConfig(
    name="bbb_weekly",
    factor_weights={
        "ma_cross": 0.25,
        "macd": 0.20,
        "regime": 0.30,
        "pullback": 0.15,
        "atr": 0.10,
    },
    entry_threshold=0.65,
    exit_threshold=0.35,
    allowed_regimes=["bull", "neutral"],
)
```

---

## 7.9 常用SQL查询

### 查询候选排序

```sql
-- ETF候选排序 (按胜率降序)
SELECT symbol, name, bbb_score, win_rate_8w
FROM wh.v_signal_backtest
WHERE asset='etf' AND strategy='bbb'
ORDER BY win_rate_8w DESC
LIMIT 30;
```

### 看看 outputs 里到底有哪些 JSON（别靠记忆）

```sql
SELECT out_dir, name, COUNT(*) AS n
FROM wh.v_outputs_json
GROUP BY 1, 2
ORDER BY n DESC, out_dir, name;
```

### 查询 TuShare 因子包（ERP/HSGT/microstructure）

```sql
-- v_tushare_factors 已存在（sql-init 后即可查）
SELECT
  out_dir,
  as_of,
  erp.erp AS erp_proxy,
  hsgt.north.score01 AS north_score01,
  hsgt.south.score01 AS south_score01,
  microstructure.score01 AS micro_score01
FROM wh.v_tushare_factors
WHERE ok = true
ORDER BY as_of DESC
LIMIT 50;
```

### 查询因子有效性

```sql
-- Phase1 之后（计划）：因子报告视图（示例名：wh.v_factor_reports）
-- 这里不直接写死字段名，避免“看起来能跑，实际全报错”的 SB 文档。
-- 你只要记住原则：factor_reports 输出要扁平字段（ic_1d/ic_5d/ir_5d/decay/...），别嵌套一堆对象。
SELECT *
FROM wh.v_factor_reports
ORDER BY as_of DESC, factor_name
LIMIT 50;
```

### 查询持仓状态

```sql
-- 当前仓库把 data/user_holdings.json 纳入 file_catalog（方便定位与审计）
SELECT rel_path, mtime, size_bytes
FROM wh.file_catalog
WHERE kind = 'user_holdings';

-- （可选）直接读 JSON（结构取决于 user_holdings.json schema）
-- SELECT * FROM read_json_auto('data/user_holdings.json');
```

---

## 7.10 常见问题

### Q: 因子研究需要多少数据？

A: 建议至少 3 年日线数据（约 750 个交易日），用于：
- IC 计算需要足够样本
- Walk-forward 需要多个训练/测试周期
- 分组收益需要足够的调仓次数

### Q: 动态权重什么时候开启？

A: 需要满足以下条件：
1. 因子研究报告显示因子在不同 Regime 下有显著差异
2. Walk-forward 验证显示 OOS 改善 > 10%
3. 参数数量控制在 10 个以内

### Q: 如何判断因子失效？

A: 监控以下指标：
- IC < 0.02 连续 3 周 → 降权
- IC < 0 连续 2 周 → 剔除
- 分组收益单调性被破坏 → 重新评估

---

## 相关文档

- [红线约束](./00_constraints.md) - 必须遵守的研究规范
- [因子化架构](./02_factor_architecture.md) - 因子系统详情
- [风险提示](./06_risks.md) - 风险管理

---

**文档维护**: 每次重大改进后更新此文档
