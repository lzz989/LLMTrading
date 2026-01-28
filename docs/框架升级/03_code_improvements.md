# 代码层面改进（P1-P3）

> 文档版本: 2026-01-24
> 模块: 代码质量优化
> 注意: P0/P1 的因子研究与评分落地优先；代码治理可并行做“低回归风险”的部分（比如拆文件/补冒烟）。

---

## 3.1 P1 重要（因子化完成后）

### 3.1.1 拆分cli.py巨型文件

**问题（历史）**: cli.py 一度有 8108 行，维护成本爆炸

**现状**: 已完成 ✅  
- `llm_trading/cli.py`：入口 + argparse（保持薄）
- `llm_trading/cli_commands.py`：轻量 wrapper（延迟导入，避免 `--help` 也把重依赖全 import）
- `llm_trading/commands/*`：按命令域拆分后的实现（scan/analyze/portfolio/sql/strategy/…）

```
llm_trading/
├── cli.py              # 入口 + argparse（已拆）
├── cli_commands.py     # cmd_* wrapper（已拆）
└── commands/           # 按命令域拆分（已完成）
    ├── __init__.py
    ├── scan.py         # scan-etf, scan-stock...
    ├── analyze.py      # analyze, fetch...
    ├── portfolio.py    # holdings-user, rebalance-user...
    ├── eval.py         # eval-*, paper-sim...
    ├── strategy.py     # scan-strategy, strategy-align...
    ├── sql.py          # sql-init, sql-sync, sql-query...
    └── ...
```

---

### 3.1.2 修复异常处理

**问题**: 623处无差别捕获Exception，真正的bug被默默吞掉

**方案**: 改为特定异常类型 + logging

```python
# Before (SB写法)
try:
    result = fetch_data(symbol)
except Exception:
    return None

# After (正确写法)
try:
    result = fetch_data(symbol)
except requests.Timeout as e:
    logger.warning(f"超时重试: {symbol}")
    return retry_fetch(symbol)
except ValueError as e:
    logger.error(f"数据格式错误: {e}")
    raise
```

**工作量**: 2天

---

### 3.1.3 建立单元测试框架

**问题**: 零测试覆盖，改代码就是裸奔

**必须测试的模块**:
- `llm_trading/factors/*.py` - 所有因子
- `forward_holding_backtest()` - 回测核心
- `compute_market_regime()` - 牛熊判断
- `shrunk_win_rate()` - 收缩胜率

**方案**:
```bash
mkdir -p tests
touch tests/__init__.py
touch tests/conftest.py
touch tests/test_factors.py
touch tests/test_backtest.py
touch tests/test_market_regime.py
```

**测试示例**:
```python
# tests/test_factors.py
import pytest
import pandas as pd
from llm_trading.factors import MACrossFactor, RSIFactor

class TestMACrossFactor:
    def test_bullish_cross(self, sample_bullish_df):
        """测试金叉信号"""
        factor = MACrossFactor()
        result = factor.compute(sample_bullish_df)
        assert result.direction == "bullish"
        assert result.score > 0.5

    def test_insufficient_data(self):
        """数据不足时返回中性"""
        factor = MACrossFactor()
        df = pd.DataFrame({"close": [1, 2, 3]})  # 太短
        result = factor.compute(df)
        assert result.confidence == 0.0

@pytest.fixture
def sample_bullish_df():
    """生成金叉测试数据"""
    # ... 构造测试数据
```

**工作量**: 5天

---

## 3.2 P2 一般（一个月内）

| 改进项 | 工作量 | 说明 |
|--------|--------|------|
| backtest.py增强 | 2天 | 滑点模型、分批止盈 |
| market_regime.py参数化 | 2天 | 配置文件化 |
| （removed）shortline.py扩展 | 2天 | 已精简移除：超短线/周内短线模块不再维护 |
| 锁定依赖版本 | 0.5天 | requirements.txt精确锁定 |
| 添加日志系统 | 1天 | 用logging替代print |

### 日志系统示例

```python
# llm_trading/logger.py
import logging
import sys
from pathlib import Path

def setup_logger(name: str, log_file: Path = None) -> logging.Logger:
    """统一日志配置"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    logger.addHandler(ch)

    # File handler (if specified)
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(
            '%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s'
        ))
        logger.addHandler(fh)

    return logger
```

---

## 3.3 P3 锦上添花（有空再搞）

| 改进项 | 工作量 | 说明 |
|--------|--------|------|
| 多数据源容错 | 2天 | AkShare→Sina→本地缓存 |
| Event-sourcing持仓 | 2天 | ledger_trades作为真实来源 |
| 自动下单接口 | 5天 | 接入券商API（QMT/掘金） |
| Web仪表盘 | 5天 | 替代CLI的可视化界面 |

### 多数据源容错示例

```python
# llm_trading/data_source.py
from dataclasses import dataclass
from typing import Protocol, Optional
import pandas as pd

class DataSource(Protocol):
    """数据源协议"""
    def fetch_daily(self, symbol: str, start: str, end: str) -> pd.DataFrame: ...

@dataclass
class FallbackDataSource:
    """多数据源容错"""
    primary: DataSource
    fallback: DataSource
    cache: DataSource

    def fetch_daily(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        # 1. 尝试主数据源
        try:
            df = self.primary.fetch_daily(symbol, start, end)
            self.cache.save(symbol, df)  # 更新缓存
            return df
        except Exception as e:
            logger.warning(f"Primary source failed: {e}")

        # 2. 尝试备用数据源
        try:
            df = self.fallback.fetch_daily(symbol, start, end)
            self.cache.save(symbol, df)
            return df
        except Exception as e:
            logger.warning(f"Fallback source failed: {e}")

        # 3. 使用缓存
        cached = self.cache.load(symbol)
        if cached is not None:
            logger.info(f"Using cached data for {symbol}")
            return cached

        raise DataSourceError(f"All sources failed for {symbol}")
```

---

## 3.4 改进优先级总结

```
Phase 1 (P0): 因子研究闭环 ──→ 必须先做，证明因子有效
    │
    ↓
Phase 2 (P1): 评分/过滤器并行输出 ──→ CashSignal/OpportunityScore/GameTheory（先能跑）
    │
    ↓
Phase 3-4 (P2):（可选）策略迁移/动态权重 ──→ 有证据再做
    │
    ↓
Phase 5 (P2-P3): 代码质量治理 ──→ 测试/异常治理/命令模块化
```

---

## 相关文档

- [框架评估](./01_assessment.md) - 当前问题汇总
- [实施路线图](./05_roadmap.md) - 具体实施计划

---

**状态**: 可并行推进（但不允许拖慢 Phase1/Phase2 的研究与并行输出）
