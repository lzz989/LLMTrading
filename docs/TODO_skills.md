# Skills TODO（角色化：策略分析 / 行业研究 / 量化回测）

> 目标：把“问一次就跑一次”的脑力活，固化成 3 个可复用 Skills（不搞花架子）。

## P0（先能用）

- [x] 建立 3 个 repo-scope Skills 目录：`.codex/skills/*`
- [x] 策略分析 Skill：读 `outputs/run_*` + `data/user_holdings.json`，输出可执行动作模板
- [x] 行业研究 Skill：结论-证据-不确定性模板（可复核）
- [x] 量化回测 Skill：无未来函数/成本假设/报告模板 + 附一个可跑脚本（出场规则对比）

## P1（做硬：边界与验收）

- [x] 统一三类 Skill 的“输入/输出约定”（skills 名字保持英文单词，便于 CLI/跨平台）
  - strategy：默认输入=最近一次 `outputs/run_*`；默认输出= `outputs/agents/strategy_action.md`
  - research：默认输出= `outputs/agents/research.md`（可选：抓新闻输出 `outputs/agents/news_raw.json` + `outputs/agents/news_digest.md`）
  - backtest：默认输出= `outputs/agents/backtest_report.md`
- [x] 增加每个 Skill 的“反例/禁止事项”（避免越界）
  - strategy：不自动实盘下单、不接券商 API、不承诺收益
  - research：媒体稿只做线索、必须二次核验
  - backtest：严禁未来函数；必须写成本；默认 t+1 执行
- [ ] 给 `run` 加一个可选开关：跑完自动生成一份“策略分析报告”（只写文件，不影响 orders 主流程）

## 备注

- repo-scope skills 路径：`.codex/skills/<skill-name>`（当前：strategy/research/backtest）
- 这些 skills 是“方法论/SOP 固化”，不是后台常驻 agent。
