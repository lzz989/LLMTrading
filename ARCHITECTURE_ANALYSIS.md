# LLM辅助交易框架 - 全面架构分析报告

## 执行摘要

这是一个**专业级量化交易研究框架**，专为中国A股/ETF市场设计，集成了多流派技术分析、BBB(三买)策略、持仓风控和组合管理的闭环系统。整个项目约**32,000行Python代码**，采用数据驱动设计，支持离线分析、回测和模拟盘。

**核心特色：**
- 多流派同时分析（缠论/威科夫/一目/海龟/Momentum/Dow/VSA）
- BBB周线波段策略（已验证的胜率统计）
- 全A/全ETF扫描系统（日/周线）
- 组合级持仓风控和仓位计划
- 闭环对账系统（订单→成交→持仓更新）
- DuckDB数据仓库（SQL查询支持）

---

## 一、项目整体结构

```
/LLM辅助交易/
├── llm_trading/                 # 核心Python包（32,063行代码）
│   ├── cli.py                   # 命令行入口（386KB，最大单文件）
│   ├── 数据源和缓存
│   │   ├── akshare_source.py    # AkShare数据接口（A股/ETF/指数）
│   │   ├── tushare_source.py    # TuShare数据接口（因子/财务）
│   │   ├── tushare_kline.py     # TuShare K线（补充来源）
│   │   ├── crypto_source.py     # 加密货币数据（可选）
│   │   ├── data_cache.py        # 本地缓存管理（增量更新）
│   │   ├── csv_loader.py        # CSV导入工具
│   │   └── symbol_names.py      # 代码→名称映射
│   │
│   ├── 技术指标和特征
│   │   ├── indicators.py        # 基础指标（MA/MACD/RSI/ATR/ADX/Ichimoku等）
│   │   ├── chanlun.py           # 缠论结构（分型/笔/中枢）
│   │   ├── dow.py               # 道氏理论（HH/HL/LH/LL结构）
│   │   ├── vsa.py               # 量价特征分析
│   │   └── resample.py          # 时间序列重采样（日→周/月）
│   │
│   ├── 核心策略
│   │   ├── bbb.py               # BBB策略（周线定方向+回踩）
│   │   ├── (removed) shortline.py         # 已精简移除（超短线/周内短线）
│   │   ├── sunrise.py           # 晨星/包线形态
│   │   ├── market_regime.py     # 大盘牛熊判断（日线MA50/MA200+MACD）
│   │   ├── backtest.py          # 前向回测框架（胜率/磨损统计）
│   │   └── paper_sim.py         # 模拟盘（账户级）
│   │
│   ├── 扫描和信号
│   │   ├── etf_scan.py          # ETF全市场扫描（BBB/趋势/波段）
│   │   ├── stock_scan.py        # 全A扫描（信号过滤+环境判断）
│   │   ├── (removed) shortline_scan.py    # 已精简移除（超短线/周内短线）
│   │   ├── (removed) sunrise_scan.py      # 已精简移除（原依赖短线回测口径）
│   │   ├── signals.py           # 统一signals schema
│   │   └── signals_merge.py     # 多策略信号聚合
│   │
│   ├── 持仓和仓位管理
│   │   ├── holdings.py          # 持仓分析（风控/止盈止损）
│   │   ├── positioning.py       # 仓位计划（资金分配/风险预算）
│   │   ├── portfolio.py         # 组合层统计（集中度/相关性）
│   │   ├── take_profit.py       # 止盈逻辑（回撤保护）
│   │   ├── costs.py             # 交易成本模型
│   │   └── tradeability.py      # 流动性判断
│   │
│   ├── 对账和执行
│   │   ├── reconcile.py         # 对账闭环（成交→持仓更新）
│   │   ├── quality_gate.py      # 质量闸门（ST/低价/低流动性过滤）
│   │   └── national_team.py     # 国家队（机构持仓分析）
│   │
│   ├── 绘图和可视化
│   │   ├── plotting.py          # 多流派图表（中文字体支持）
│   │   ├── narrative.py         # 自然语言综合解读
│   │   └── pipeline.py          # LLM分析管道
│   │
│   ├── 数据仓库和监控
│   │   ├── warehouse.py         # DuckDB SQL视图
│   │   ├── monitor.py           # 结果监控汇总
│   │   ├── data_doctor.py       # 数据体检
│   │   ├── analysis_cache.py    # 分析结果缓存
│   │   └── run_meta.py          # 运行元数据
│   │
│   ├── 工具和配置
│   │   ├── llm_client.py        # OpenAI/Gemini客户端
│   │   ├── config.py            # 配置加载
│   │   ├── json_utils.py        # JSON清洗（NaN/Inf处理）
│   │   ├── utils_stats.py       # 统计函数（收缩胜率等）
│   │   ├── utils_time.py        # 时间工具
│   │   ├── institution.py       # 机构资金流指标
│   │   ├── index_composite.py   # 指数合成
│   │   ├── (removed) eval_shortline.py    # 已精简移除（超短线/周内短线）
│   │   ├── robustness_bbb.py    # BBB稳健性测试
│   │   ├── strategy_race.py     # 策略赛马
│   │   ├── strategy_registry.py # 策略注册表
│   │   └── reporting.py         # 报告生成
│   │
│   ├── __main__.py              # CLI入口
│   ├── __init__.py              # 包初始化
│   └── tushare_factors.py       # 因子面板（ERP/HSGT/微观结构）
│
├── prompts/                      # LLM提示词模板
│   ├── synthesis_prompt.md      # 多流派综合解读提示词
│   ├── wyckoff_json_prompt.md   # 威科夫JSON结构提示词
│   ├── vsa_json_prompt.md       # 量价JSON提示词
│   └── chanlun_json_prompt.md   # 缠论JSON提示词
│
├── data/                        # 本地数据（gitignore）
│   ├── cache/                   # 数据缓存
│   │   ├── etf/                # ETF日线缓存
│   │   ├── stock/              # 股票日线缓存
│   │   └── analysis/           # 分析结果缓存
│   ├── user_holdings.json       # 用户持仓快照
│   └── warehouse.duckdb         # SQL数据库
│
├── outputs/                     # 分析结果（gitignore）
│   ├── analyze_<symbol>_*/     # 单标的分析产物
│   ├── scan_etf_*/             # ETF扫描结果
│   ├── run_YYYYMMDD/           # 日常跑批产物
│   ├── paper_sim_*/            # 模拟盘结果
│   └── _duckdb_sentinel/       # DuckDB哨兵目录
│
├── references/                  # 参考代码（不作为运行时依赖）
│   └── chanlun/                # 缠论参考实现
│
├── README.md                    # 使用文档（详细命令行示例）
├── AGENTS.md                    # 项目指南和代码规范
├── requirements.txt             # Python依赖
└── .env.example                 # 环境变量模板

```

---

## 二、核心技术栈和依赖

### 外部依赖
```
pandas>=2.2,<2.3          # 数据处理
matplotlib>=3.9,<3.10     # 绘图
akshare>=1.13             # 免费数据源（A股/ETF）
ta>=0.11,<0.12            # TA-Lib技术指标库
tushare>=1.4,<2.0         # 财务因子数据
duckdb>=1.4,<2.0          # 本地SQL引擎
（可选）openai            # GPT分析
（可选）google-generativeai # Gemini分析
```

### Python基线
- **Python 3.12+**（强制，不再支持3.8）
- 使用`poetry`或`pip`管理依赖

---

## 三、数据流和处理流程

### 3.1 数据获取流程

```
┌─────────────────┐
│  数据源选择      │
└────────┬────────┘
         │
    ┌────┴──────┬─────────┬────────┐
    │            │         │        │
    ▼            ▼         ▼        ▼
 AkShare     TuShare   CSV导入   加密货币
（推荐）      (因子)    (用户数据) (可选)
    │            │         │        │
    └────┬───────┴─────────┴────────┘
         │
    ┌────▼─────────────────┐
    │  data_cache.py       │
    │ - 本地增量缓存       │
    │ - TTL自动清理       │
    │ - 支持CSV导出       │
    └────┬─────────────────┘
         │
    ┌────▼──────────────────┐
    │  CSV标准化           │
    │ - 日期列识别         │
    │ - OHLCV处理          │
    │ - 类型转换           │
    └────┬──────────────────┘
         │
    ┌────▼──────────────────┐
    │  指标计算 (indicators)│
    │ - MA/MACD/RSI/ATR    │
    │ - Ichimoku/ADX       │
    │ - 累积派发线/OBV     │
    └────┬──────────────────┘
         │
    ┌────▼──────────────────┐
    │ 时间序列重采样        │
    │ (resample.py)        │
    │ - 日线 → 周线/月线    │
    │ - 保留OHLC/成交量    │
    └────┬──────────────────┘
         │
    ┌────▼──────────────────────┐
    │  结构化信号生成          │
    │ - BBB策略判断            │
    │ - 短线信号              │
    │ - 技术结构（缠论/Dow等） │
    └────┬──────────────────────┘
         │
    ┌────▼──────────────────────┐
    │  统一signals schema       │
    │ - 标准化输出格式         │
    │ - 支持多策略聚合         │
    └──────────────────────────┘
```

### 3.2 单标的分析流程（analyze命令）

```
输入: symbol + 频率 + 方法
  │
  ├─ 获取日线数据 (akshare_source.fetch_daily)
  │
  ├─ 计算指标 (indicators.*)
  │
  ├─ 多流派并行计算:
  │   ├─ Wyckoff (w/JSON输出)
  │   ├─ Chanlun (分型→笔→中枢)
  │   ├─ Ichimoku
  │   ├─ Turtle (Donchian)
  │   ├─ Momentum (RSI/MACD/ADX)
  │   ├─ Dow (结构判断)
  │   └─ VSA (量价特征)
  │
  ├─ 绘图 (plotting.* → PNG)
  │
  ├─ LLM分析 (可选, --llm标志)
  │   ├─ 流派JSON → 提示词 → LLM
  │   └─ 输出llm_analysis.json
  │
  └─ 输出目录结构:
      outputs/<symbol>_<time>/
      ├─ chart.png (或多个分流派)
      ├─ *.json (结构化结果)
      ├─ llm_analysis.json (如果启用)
      └─ run_meta.json (元数据)
```

### 3.3 扫描流程（scan-etf/scan-stock/scan-strategy）

```
┌─────────────────────────────┐
│  加载宇宙 (所有ETF/股票)      │
└────────┬────────────────────┘
         │
    ┌────▼────────────────────┐
    │ 质量闸门过滤            │
    │ (quality_gate.py)      │
    │ - ST/退市过滤          │
    │ - 低价股过滤           │
    │ - 低流动性过滤         │
    └────┬────────────────────┘
         │
    ┌────▼────────────────────┐
    │ 多线程并行处理          │
    │ (ThreadPoolExecutor)    │
    │ - 数据获取 + 缓存       │
    │ - 指标计算             │
    │ - 信号判断             │
    └────┬────────────────────┘
         │
    ┌────▼────────────────────┐
    │ 大盘牛熊判断            │
    │ (market_regime.py)     │
    │ - 日线MA50/MA200       │
    │ - MACD方向             │
    │ - panic兜底            │
    └────┬────────────────────┘
         │
    ┌────▼────────────────────┐
    │ 候选排序和筛选          │
    │ - BBB分数加权          │
    │ - 7因子排序 (可选)      │
    │ - Top-K选择            │
    └────┬────────────────────┘
         │
    ┌────▼────────────────────┐
    │ 历史胜率统计            │
    │ (backtest.py)          │
    │ - 4/8/12周horizon      │
    │ - 收缩胜率防小样本      │
    │ - MAE/MFE计算          │
    └────┬────────────────────┘
         │
    ┌────▼────────────────────┐
    │ 输出结果                │
    │ - top_bbb.json        │
    │ - top_trend.json      │
    │ - top_swing.json      │
    │ - signals.json        │
    │ - all_results.csv     │
    │ - errors.json         │
    └────────────────────────┘
```

### 3.4 持仓管理闭环

```
Run命令 (日常跑批)
  │
  ├─ scan-etf (获取候选+signals)
  │
  ├─ holdings-user (分析持仓)
  │   └─ 输出: 风控提醒, 组合统计
  │
  ├─ rebalance-user (仓位计划)
  │   └─ 输出: 目标仓位, 次日开盘订单
  │
  ├─ 人工执行 (半自动)
  │   └─ 下单并获得fills
  │
  └─ reconcile (对账)
      ├─ 导入 CSV/JSON fills
      ├─ 对比 orders
      ├─ 更新 user_holdings.json
      └─ 追加 ledger_trades.jsonl (审计)
```

---

## 四、核心模块详解

### 4.1 BBB策略（三买模块）

**核心逻辑**: 周线定方向 + 日线辅助 + 位置过滤

```python
# bbb.py 关键数据结构
@dataclass(frozen=True, slots=True)
class BBBParams:
    entry_ma: int = 50              # 进场参考均线（20或50）
    dist_ma50_max: float = 0.12     # 最高离MA50距离（12%以内）
    max_above_20w: float = 0.05     # 不追高：略高于20W上轨
    min_weekly_bars_total: int = 60 # 样本最低要求（周K）
    require_weekly_macd_bullish: bool = True
    require_weekly_macd_above_zero: bool = True
    require_daily_macd_bullish: bool = True
```

**判断流程**:
1. **周K样本检查**: < 60周K → 不信任
2. **周线定方向**: 
   - MA50 > MA200 ✓
   - MACD > Signal ✓
   - MACD > 0 ✓
3. **位置过滤**:
   - 不追高: `close <= MA50 * (1 + max_above_20w)`
   - 回踩: `MA50 * (1 - dist_ma50_max) <= close <= MA50`
4. **日线确认**:
   - MACD > Signal ✓
5. **输出**: `ok: bool` + 原因说明

**胜率统计** (backtest.py):
- 前向持有测试: 信号次日开盘买 → 持有N周 → 第N+1周开盘卖
- 成本: `sell_cost` 扣除, `buy_cost` 扣除
- 收缩胜率: Beta-Binomial先验防小样本吹牛
- MAE/MFE: 持仓期最大不利/有利波动

### 4.2 市场制度判断（market_regime.py）

**输入**: 日线数据 + 指数代码
**输出**: `label: bull|bear|neutral|unknown` + 详细指标

**计算规则**:
```
日线MA50 > MA200?  日线MACD > 0?  → 倾向 bull
日线MA50 < MA200?  日线MACD < 0?  → 倾向 bear
否则                               → neutral

额外兜底:
- 深回撤 (252日 <= -0.25) → bear
- 单日大跌 (>= 3×20日波动率) → panic → bear
- 3日确认机制 (避免反复横跳)
```

**应用场景**:
- BBB熊市禁入 (除非显式 `--bbb-allow-bear`)
- 仓位规模自适应 (bull 90%, bear 30%, neutral 60%)
- （removed）短线禁入 (bear市)：超短线/周内短线模块已精简移除

### 4.3 短线策略（已精简移除）

原 shortline（T+1~T+3 周内短线）相关模块已从主框架精简移除，避免高换手对小资金磨损不友好、且与默认“趋势回踩低吸（1~数周）”主线冲突。

### 4.4 持仓风控（holdings.py + positioning.py）

**风控层次**:

1. **止损 (hard stop)**:
   - 周线: MA50跌破 → 下周确认后卖
   - 日线: MA20跌破 + MACD死叉 → 次日卖
   - 最大亏损: 可选, 默认关

2. **止盈 (profit stop)**:
   - 前提: 已盈利 >= 20%
   - 触发: 日线回撤 >= 12% 或 MACD失效
   - 动作: 卖出50%保护利润

3. **panic兜底**:
   - 单日大跌 <= -波动率×3 且 >= -4%
   - 一年回撤 <= -25%
   - 动作: 次日开盘全卖

**仓位计划**:
- 总资金拆成 N 笔
- 单笔风险预算: `总资金 × risk_per_trade_pct`
- 止损距离 → 仓位数量 (shares = risk_yuan / (entry - stop))
- 动态调整: 牛市多持, 熊市少持

### 4.5 信号聚合和对账（signals_merge.py + reconcile.py）

**signals schema** (统一):
```json
{
  "schema_version": 1,
  "strategy": "bbb_weekly | trend_pullback_weekly | stock_scan",
  "items": [
    {
      "asset": "etf | stock | index",
      "symbol": "sh510300",
      "action": "entry | watch | avoid | exit",
      "score": 0.75,          // 排序分数
      "confidence": 0.65,     // 胜率参考
      "entry": {
        "price_ref": 2.05,
        "price_ref_type": "close"
      },
      "meta": {...}           // 原始详情
    }
  ]
}
```

**对账流程** (reconcile):
```
orders_next_open.json (理想计划)
           ↓
    人工执行 (半自动)
           ↓
fills.csv/json (真实成交)
           ↓
reconcile命令:
- 逐笔匹配 orders → fills
- 计算 cash / 持仓变化
- 更新 user_holdings.json
- 追加 ledger_trades.jsonl (审计)
```

### 4.6 数据仓库（warehouse.py + DuckDB）

**使用场景**: SQL查询替代ad-hoc脚本

**关键视图**:
- `wh.v_bars`: 统一OHLCV (所有资产)
- `wh.v_bars_etf`: ETF专用
- `wh.v_outputs_json`: outputs/*/*.json索引
- `wh.v_analysis_meta`: 单标的分析元数据
- `wh.v_signal_backtest`: 回测统计

**示例查询**:
```sql
-- ETF候选排序 (按胜率降序)
SELECT symbol, name, bbb_score, win_rate_8w 
FROM wh.v_signal_backtest
WHERE asset='etf' AND strategy='bbb'
ORDER BY win_rate_8w DESC
LIMIT 30;

-- 某日期范围内 top 候选
SELECT * FROM wh.v_analysis_meta
WHERE asset_type='etf' AND as_of >= '2026-01-15';
```

---

## 五、设计亮点

### 1. 多流派共存架构
- **不强行一家独大**: 同一份K线数据可并行跑7种方法 (wyckoff/chan/ichimoku/turtle/momentum/dow/vsa)
- **灵活组合**: `--method all` 全跑, 或 `--method both` 对比
- **LLM增强可选**: 纯脚本 vs LLM解读可选, 不绑定

### 2. 样本量保护机制
- **收缩胜率** (Beta-Binomial): `shrunk_win_rate(wins, trades, prior_strength=20)`
  - trades=1 的极端吹牛被严重惩罚
  - trades=100+ 接近原始胜率
- **年化封顶**: annualized 模式下限制在 200% (防止 trades=1 极端值霸榜)
- **小样本权重**: `trades / (trades + 20)` 做加权

### 3. 日期精确处理
- **周K定义**: 用"该周最后一个交易日"作为周K日期, 而不是 W-FRI (避免穿越误导)
- **日线对齐**: 日线MACD对齐到周K (merge_asof 后向查找)
- **时间戳一致**: 所有输出带 `as_of` 日期标记

### 4. 成本模型严谨
- **买卖分离**: `buy_cost` 和 `sell_cost` 分别计算
- **磨损实测**: MAE (Maximum Adverse Excursion) 实测持仓期间最大回撤
- **小资金适配**: 单笔最小成本可配置 (默认3000起+10块磨损), 防止被手续费搞死

### 5. 组合风险视角
- **不是"各标的分别回测"**: paper_sim 是共享资金的账户级模拟
- **相关性考量**: positioning.py 支持基于历史周收益的相关性过滤 (去掉相似度>0.95的)
- **集中度检查**: portfolio.py 计算单标的/主题集中度, 预警过高

### 6. 闭环对账设计
- **审计链完整**: fills → reconcile → holdings_next + ledger_trades
- **冲突处理**: 多笔order匹配一笔fill, 自动split
- **幂等性**: trade_id 防重复 (若源站未提供则hash降级)

---

## 六、可能的问题和改进方向

### 问题1: 数据源风险
**现象**: AkShare/TuShare 频繁限流/滞后  
**影响**: 定时扫描容易中断, 缓存策略不足  
**建议**:
- 增加数据源容错 (A: AkShare → B: Sina → C: 本地缓存)
- 实现断点续传 (记录上次失败的标的, 下次优先重试)
- 添加数据质量告警 (缺失K线/价格异常自动跳过)

### 问题2: 回测方法学局限
**现象**: 
- 前向持有假设"只按开盘成交", 实际滑点可能更大
- 不支持"部分止盈/分批加仓" (只有全买全卖)
- MAE假设持仓期间没追加操作 (不符合实际T+1规则)

**建议**:
- 实现滑点模型 (基于成交量估算)
- 支持动态仓位管理 (分批加仓/减仓)
- 加入"手续费最低额"约束 (避免极小单笔)

### 问题3: 大盘判断过于简化
**现象**: 
- 只用MA50/MA200+MACD, 没有成交量确认
- Panic判断 (单日大跌) 可能误触 (如节假日跳空)
- 不考虑宏观因素 (央行政策/外汇/融资成本等)

**建议**:
- 加入量能确认 (下跌时量能萎缩→反弹, 量能放大→破位)
- 增加Panic的"3日确认" (当前已有, 但参数可优化)
- 可选集成ERP/融资利率/融券余额 (已有tushare_factors框架)

### 问题4: 短线逻辑不完备
**现象**:
- 只检查"最近10日是否涨停", 不区分"第N日涨停后" (容易晚入)
- 回踩幅度判断 (4%) 是硬编码, 对不同涨幅强度的适应性差
- 不支持"分阶段持仓" (一口气全进全出)

**建议**:
- 改为"涨停后第N日" + "回踩确认" 的两步判断
- 回踩幅度 = f(涨幅, 历史波动率) 自适应
- 支持分批出场 (例如 1/3 @ +5%, 2/3 @ 时间止损)

### 问题5: 持仓数据一致性
**现象**:
- user_holdings.json 是"单点快照", 如果没有定期 reconcile 会漂移
- 没有"未成交订单"的管理 (可能某笔order从未执行, 但系统假设已执行)
- 多设备/多账户同步困难

**建议**:
- 改为event-sourcing模式 (ledger_trades.jsonl 作为真实来源, 持仓由其计算)
- 增加pending orders表 (订单→成交转换的中间态)
- 支持多账户分离 (例如 data/user_1/data/user_2 各自独立)

### 问题6: 参数风险
**现象**:
- BBB参数众多 (entry_ma/dist_ma50_max/max_above_20w 等), 易过度优化
- 仓位风控参数硬编码 (牛市90%/熊市30%), 不同账户需求差异大
- "先验强度 (prior_strength=20)" 选择依据不足

**建议**:
- 实现walk-forward CV (参数扫描 + OOS评估)
- 将风险配置文件化 (trade_rules.json 可配)
- 发布参数敏感性分析 (各参数偏离基准多少%, 胜率变化多少%)

### 问题7: 可执行性和真实成本
**现象**:
- 模拟盘默认成本只有"手续费+佣金", 没有考虑：
  - 融资利息 (若使用融资买入)
  - 融券费用 (若做空)
  - T+1限制下的被迫持仓成本
  - 公告期间停牌损失
- 小资金的"最低佣金 5 块"约束没有严格实施

**建议**:
- 实装融资/融券成本模型
- 模拟盘中加入停牌判断 (读上市公司公告API)
- 对小单笔 (<500块) 的订单进行合并或拒绝

---

## 七、使用场景和推荐工作流

### 场景1: 周线波段 (风险小, 持仓3~8周)
```bash
# 周一晚上运行
".venv/bin/python" -m llm_trading scan-etf --freq weekly --top-k 30
".venv/bin/python" -m llm_trading plan-etf --scan-dir outputs/scan_etf_*

# 周二开盘执行 + 周五收盘对账
# 周六/周日复盘和调整

推荐参数:
- --bbb-mode pullback (允许周线回踩)
- --min-weeks 60 (避免新ETF)
- --min-amount-avg20 50000000 (流动性)
```

### 场景2: 短线T+1~T+3（已精简移除）

短线/周内短线（`scan-short`/`eval-shortline`）已从主框架精简移除；当前默认主线改为“趋势回踩低吸（1~数周）”，以降低换手与固定磨损对收益的侵蚀。

### 场景3: 全A趋势 (选股难度高)
```bash
# 周线分析, 可月度或高风险环境下跳过
".venv/bin/python" -m llm_trading scan-stock --freq weekly --limit 500 \
  --min-amount 50000000 --min-price 5 --max-price 50

# 输出 top_trend / top_swing / top_dip
# 结合fundamental面再选取

推荐参数:
- --base-filters trend_template (Weinstein+Minervini基线)
- --workers 8 (并行加速)
```

### 场景4: 组合管理和持仓监控
```bash
# 日常命令 (一键生成次日参考单)
".venv/bin/python" -m llm_trading run --scan-freq weekly --rebalance-mode add

# 月度回顾
".venv/bin/python" -m llm_trading monitor --outputs-dir outputs --max-dirs 200
".venv/bin/python" -m llm_trading sql-query --sql "select ... from wh.v_signal_backtest"

# 年度评估和参数调整
".venv/bin/python" -m llm_trading eval-bbb --symbol sh510300 --horizon-weeks 8
```

---

## 八、代码质量评估

### 优势
✅ **代码结构清晰**: 模块职责分离, 无过度耦合  
✅ **类型提示完善**: 大量 dataclass / TypedDict 标注  
✅ **错误处理规范**: try-except 覆盖外部API调用  
✅ **文档齐全**: README 1,700+ 行命令示例, AGENTS.md 规范详细  
✅ **可配置性高**: 几乎所有参数都支持CLI覆盖  
✅ **数据安全**: .gitignore 保护 data/ 和 .env  

### 需改进的地方
⚠️ **单文件过大**: cli.py 386KB (32,063行代码中的 1/8), 建议按命令拆分  
⚠️ **缺少单元测试**: 虽然有smoke check建议, 但无自动化测试套件  
⚠️ **文档散乱**: 提示词在 prompts/ 目录, 实现细节藏在代码注释中  
⚠️ **依赖版本锁定不够**: requirements.txt 用 `>=1.13` 这种宽松约束, 可能遇到兼容性问题  
⚠️ **日志缺乏**: 很多关键步骤没有 logging, 只能通过 print 或看输出  

---

## 九、总体架构评分

| 维度 | 评分 | 备注 |
|------|------|------|
| **功能完整性** | 9/10 | 覆盖扫描/回测/风控/对账全流程 |
| **代码质量** | 7.5/10 | 结构好, 但cli.py过大, 缺测试 |
| **易用性** | 8/10 | 命令行丰富, 文档详细, 有一定学习曲线 |
| **可扩展性** | 8/10 | 模块化设计好, 添加新指标/策略相对容易 |
| **稳定性** | 7/10 | 依赖外部数据源, 需人工监控和容错 |
| **性能** | 7.5/10 | 全A扫描会慢(无分布式), 但缓存机制有效 |
| **风险控制** | 8.5/10 | 止损/止盈/panic兜底完善, 但缺宏观约束 |

**总体评价**: **8.2/10** - 一个**生产级别的量化研究框架**, 适合：
- ✅ 专业交易者(学习 + 参考策略)
- ✅ 量化研究团队(基础框架 + 二次开发)
- ✅ 小规模资金管理(<500万, 日常跑批用)
- ❌ 不适合: 高频交易/衍生品/完全自动下单(需额外接入券商API)

---

## 十、快速上手建议

### Step 1: 环境准备
```bash
cd /home/root_zzl/LLM辅助交易
python -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env
# 可选: 填 OPENAI_API_KEY 用于LLM分析
```

### Step 2: 测试数据流
```bash
# 抓一个ETF试试
.venv/bin/python -m llm_trading fetch --asset etf --symbol sh510300 --freq daily

# 单个分析 (不启用LLM)
.venv/bin/python -m llm_trading analyze --asset etf --symbol sh510300

# 查看输出
ls -l outputs/
```

### Step 3: 扫描和仓位计划
```bash
# ETF周线扫描 (首次会慢, 因为缓存为空)
.venv/bin/python -m llm_trading scan-etf --freq weekly --top-k 30

# 生成仓位计划
.venv/bin/python -m llm_trading plan-etf --scan-dir outputs/scan_etf_*/
```

### Step 4: 设置持仓和对账
```bash
# 创建持仓快照
cat > data/user_holdings.json << 'EOF'
{
  "cash": {"amount": 3000},
  "positions": [
    {
      "symbol": "sh510300",
      "shares": 100,
      "cost_basis": 3.2,
      "name": "沪深300"
    }
  ]
}
