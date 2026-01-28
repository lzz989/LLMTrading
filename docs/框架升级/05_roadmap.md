# 实施路线图

> 文档版本: 2026-01-24
> 模块: 项目实施计划

---

## 总体原则

1. **先验证后优化**: 先证明因子有效，再谈权重优化
2. **先稳定后扩展**: 先保证现有功能稳定，再添加新功能
3. **小步快跑**: 每个阶段产出可验证的成果

---

## Phase 1（P0）: 因子研究最小闭环

> **目标**: 让“因子是否有效”变成可验证、可复现、能持续跑的报告  
> **范围**: 现有技术因子(17) + 新增候选池（博弈/流动性 proxy、TuShare 因子包字段）

### 里程碑

```
Week 1-2:
├── 定义 forward returns 与交易日对齐口径（严禁未来函数）
├── 实现 IC/IR 计算函数
└── 产出单因子报告模板

Week 3:
├── 跑全因子批量报告（当前 17 个 + 新候选）
├── 加入成本敏感性分析
└── 得到"可用因子白名单/黑名单"
```

### 交付物

- [x] `llm_trading/factors/research.py` - 因子研究模块
- [x] `outputs/factor_reports_*/*` - 因子分析报告目录（按 run 分目录）
- [x] 因子有效性白名单/黑名单（结构化 JSON + 可 SQL 查询）

### 验收标准

| 检查项 | 通过条件 |
|--------|----------|
| 未来函数 | 所有因子计算只用 `<= as_of` 数据 |
| 样本外 | 有 walk-forward 或 train/test 切分 |
| 成本纳入 | 报告包含最小佣金 5 元 + 滑点 |
| 输出规范 | 包含 `symbol/as_of/source` 字段 |

---

## Phase 2（P1）: 评分/过滤器并行输出（先不改老信号）

> **目标**: 把“可执行”的小散三件套跑起来：CashSignal / OpportunityScore / PositionSizing  
> **原则**: 先并行输出，不强制替换现有策略信号（降低回归风险）

### 里程碑

```
Week 4:
├── 输出 game_theory_factors.json（liquidity_trap/stop_cluster/capitulation/fomo/wyckoff_phase_proxy）
├── 输出 opportunity_score.json / cash_signal.json / position_sizing.json
└── 同步 DuckDB 视图（wh.v_game_theory_factors / wh.v_opportunity_score / wh.v_cash_signal / wh.v_position_sizing）

Week 5:
├── scan-* 追加 score 字段 + 支持 --min-score
├── holdings/rebalance 接入（CashSignal 做风控开关；OpportunityScore 控候选质量）
└── 产出“并行输出对齐报告”（不改变原信号，仅增加可解释层）
```

### 交付物

- [x] `outputs/<run>/game_theory_factors.json`
- [x] `outputs/<run>/opportunity_score.json`
- [x] `outputs/<run>/cash_signal.json`
- [x] `outputs/<run>/position_sizing.json`
- [x] DuckDB 视图（可查询样例写进附录）

### 验收标准

| 检查项 | 通过条件 |
|--------|----------|
| 并行输出 | 不改老信号，仅增加评分/过滤器产物 |
| 可解释性 | 评分拆解字段齐全（components + invalidation/holding_days） |
| SQL 可查 | DuckDB 可直接查到这四类输出（扁平字段） |

---

## Phase 3（P2，可选）: 策略迁移到因子配置（新旧信号对齐）

> **目标**: 解决“重复代码/策略复用困难”，把硬编码迁移成配置化权重  
> **注意**: 不应阻塞 Phase2 的评分/过滤器落地

### 里程碑

```
Week 6-7:
├── 补 config/strategy_configs.yaml（固定权重先复刻旧口径）
├── 产出新旧信号对齐对比报告（偏离率可量化）
└── scan/analyze 增加可选路径（不强制替换，先并行验证）
```

### 交付物

- [x] `config/strategy_configs.yaml`
- [x] `outputs/strategy_alignment_*/*`（新旧对齐报告）

### 验收标准

| 检查项 | 通过条件 |
|--------|----------|
| 信号一致性 | 新旧信号偏离可量化（<5% 为目标，先对齐口径再调参） |
| 可复跑 | 输出含 run_meta/run_config，保证可复跑 |

---

## Phase 4（P2）: 动态权重（有研究证据再做）

> **目标**: 只有当 Phase1 证明“因子在不同 regime 下稳定差异”时，才允许做权重切换/搜索。

### 里程碑

```
Week 8-9:
├── 先做手工 regime 权重表（粗粒度、少参数）
├── 严格 walk-forward 验证
└── 只有样本外确实改善，才考虑继续
```

### 交付物

- [x] `llm_trading/factors/dynamic_weights.py`（或同级模块）
- [x] `outputs/walk_forward_*/*`（验证报告）

### 验收标准

| 检查项 | 通过条件 |
|--------|----------|
| 样本外改善 | OOS 指标稳定改善（先不设拍脑袋阈值，报告说话） |
| 参数数量 | 少参数（每个 regime < 10 个自由度） |
| 过拟合检验 | OOS/IS 比率可解释、无“战神曲线” |

---

## Phase 5（P2-P3）: 代码质量改进

> **目标**: 别让项目靠运气活着

### 里程碑

```
Week 10-11:
├── 补测试：因子/回测核心/市场状态
├── 测试覆盖率 > 60%
└── CI 流水线配置

Week 12:
├── 修异常：优先修"吞错导致结果默默不对"的地方
├── 异常处理规范化
└── 日志系统上线

Week 13+（有空再搞）:
├── 进一步拆分 cli_commands.py
└── 按命令分模块（llm_trading/commands/*）
```

### 交付物

- [x] `tests/` - 测试套件
- [x] `.github/workflows/test.yml` - CI 配置
- [x] `llm_trading/commands/` - 拆分后的命令模块

### 验收标准

| 检查项 | 通过条件 |
|--------|----------|
| 测试覆盖 | 核心模块 > 80% |
| 异常处理 | `except Exception` 减少 50% |
| cli.py 行数 | < 1200 行（入口保持薄；重活放到 commands/） |

---

## Phase 6（持续）: 生产验证（真金白银）

> **目标**: 真金白银验证策略有效性

### 持续任务

```
每周:
├── 小资金验证（严格记录交易与信号快照）
├── 信号 vs 实际执行对比
└── 周复盘（用 SQL 抽取 "触发→执行→结果"）

每月:
├── 因子 IC/IR 监控
├── 发现衰减：因子降权/剔除
└── 策略收益统计

每季度:
├── 全面复盘
├── 参数敏感性分析
└── 策略有效性评估
```

### 监控指标

| 指标 | 预警阈值 | 动作 |
|------|----------|------|
| 因子 IC | < 0.02 连续3周 | 降权 |
| 因子 IR | < 0.3 | 观察 |
| 策略胜率 | < 40% 连续2月 | 停用 |
| 最大回撤 | > 15% | 减仓 |

---

## 时间线总览（按“先能跑、再优雅”）

```
2026-01
├── Week 1-2: Phase 1 - 因子研究闭环
├── Week 3: Phase 1 - 因子批量分析
└── Week 4-5: Phase 2 - 评分/过滤器并行输出 + SQL 视图

2026-02
├── Week 6-7: Phase 3(可选) - 策略迁移到因子配置（对齐验证）
└── Week 8-9: Phase 4 - 动态权重（有研究证据再做）

2026-03
├── Week 10-11: Phase 5 - 补测试
└── Week 12-13: Phase 5 - 异常治理/命令模块化

2026-04+
└── Phase 6 - 持续生产验证
```

---

## 资源需求

| 阶段 | 预计工作量 | 依赖 |
|------|------------|------|
| Phase 1 | 2周 | 无 |
| Phase 2 | 2周 | Phase 1 完成 |
| Phase 3（可选） | 2周 | Phase 1/2 |
| Phase 4 | 2周 | Phase 1（+最好有 Phase3） |
| Phase 5 | 4周 | 可并行 |
| 生产验证 | 持续 | 全部完成（或至少 Phase1/2 先跑） |

---

## 相关文档

- [因子化架构](./02_factor_architecture.md) - 技术方案
- [代码改进](./03_code_improvements.md) - 代码优化详情
- [红线约束](./00_constraints.md) - 必须遵守的规范

---

**下次Review**: 2026-02-23
