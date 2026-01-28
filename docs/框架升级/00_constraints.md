# 红线约束（不遵守 = 看起来很科学，实际全是幻觉）

> **这是"红线"。后面所有研究/优化/动态权重，都必须满足这些约束才允许合并进主流程。**

---

## 0.1 禁止未来函数（Look-ahead）

- 因子/信号计算只能使用 `<= as_of` 的数据；任何 `t` 日因子都只能预测 `t+1..t+n` 的未来收益。
- ETF/指数/个股都要按交易日历对齐；非交易日不"瞎补未来"。

**常见违规场景：**
```python
# ❌ 错误：用了当天收盘价来计算当天的信号
df['signal'] = df['close'] > df['ma50']  # 收盘才知道，但信号假装开盘就知道了

# ✅ 正确：信号延迟一天
df['signal'] = df['close'].shift(1) > df['ma50'].shift(1)
```

---

## 0.2 禁止过拟合（Overfitting）

- 所有"权重优化/参数搜索/ML"必须 walk-forward（训练窗口/测试窗口分离），必须产出样本外结果。
- 参数搜索要"粗粒度+少自由度"：先限制网格大小、限制因子数量、限制正交化/筛选规则；别把自己优化成战神。
- 指标必须纳入交易成本/滑点/最小佣金（5 元）与可交易性约束（涨跌停/停牌/申赎/流动性）。

**常见违规场景：**
```python
# ❌ 错误：全样本优化参数
best_params = optimize(df_all)  # 用全部数据找"最优"

# ✅ 正确：walk-forward
for train, test in walk_forward_split(df, train_window=252, test_window=63):
    params = optimize(train)
    results.append(backtest(test, params))
```

---

## 0.3 输出必须可复现、可审计、可 SQL

- 所有结构化输出优先 `csv/json`；JSON key 统一 `snake_case`。
- 每份输出必须包含稳定字段：`symbol/asset/as_of/source/ref_date`（或同义字段）。
- 所有新输出如果要纳入仓库分析，必须同步补一张 DuckDB 视图（保持 KISS，别过度设计）。

**输出字段示例：**
```json
{
  "symbol": "sh510300",
  "asset": "etf",
  "as_of": "2026-01-23",
  "source": "factor_research",
  "ref_date": "2026-01-23",
  "ic_5d": 0.045,
  "ir_5d": 0.32
}
```

---

## 0.4 数据源与容错（现实世界很脏）

- 数据源（AkShare/Eastmoney/TuShare）不稳定是常态：必须缓存 + TTL + 降级策略（接口挂了不能把主流程搞崩）。
- 任何"更好数据源替换"都要先做到：同口径对齐、缺失值策略一致、字段单位/复权一致；否则宁可不用。

**容错策略示例：**
```python
def fetch_with_fallback(symbol: str) -> pd.DataFrame:
    """优先AkShare，挂了用缓存，再挂抛异常但不崩"""
    try:
        return fetch_akshare(symbol)
    except Exception:
        logger.warning(f"AkShare failed for {symbol}, trying cache")
        cached = load_cache(symbol)
        if cached is not None:
            return cached
        raise DataSourceError(f"No data available for {symbol}")
```

---

## 检查清单（代码审查用）

| 检查项 | 通过条件 |
|--------|----------|
| 未来函数 | 所有因子计算只用 `<= as_of` 数据 |
| 样本外 | 有 walk-forward 或 train/test 切分 |
| 成本纳入 | 回测包含最小佣金 5 元 + 滑点 |
| 可交易性 | 剔除涨跌停/停牌/流动性不足标的 |
| 输出规范 | 包含 `symbol/as_of/source` 字段 |
| 数据容错 | 有缓存 + 降级策略 |

---

**任何 PR 违反以上约束，直接打回。**
