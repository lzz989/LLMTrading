from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

if sys.version_info < (3, 12):
    raise SystemExit(
        "艹，别用 Python 3.8 了，这仓库现在基线是 Python 3.12+。\n"
        "用：\"/home/root_zzl/miniconda3/bin/python\" -m venv \".venv\" 然后装 requirements.txt"
    )
# 这些 cmd_* 实现在 cli_commands.py 里，cli.py 只负责 argparse + main 入口。
from .cli_commands import (
    cmd_analyze,
    cmd_chat,
    cmd_clean_outputs,
    cmd_commodity_chain,
    cmd_data_doctor,
    cmd_daily_brief,
    cmd_dynamic_weights,
    cmd_factor_research,
    cmd_eval_bbb,
    cmd_fetch,
    cmd_holdings_etf,
    cmd_holdings_user,
    cmd_memory,
    cmd_monitor,
    cmd_national_team,
    cmd_national_team_backtest,
    cmd_paper_sim,
    cmd_plan_etf,
    cmd_race_strategies,
    cmd_rebalance_user,
    cmd_reconcile,
    cmd_replay,
    cmd_run,
    cmd_scan_etf,
    cmd_scan_stock,
    cmd_scan_strategy,
    cmd_signals_merge,
    cmd_skill,
    cmd_sql_init,
    cmd_sql_query,
    cmd_sql_sync,
    cmd_strategy_align,
    cmd_verify_prices,
)
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="llm_trading", description="LLM辅助交易：威科夫读图 + 自动标注出图")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_fetch = sub.add_parser("fetch", help="抓行情数据（默认 auto：优先 TuShare；不行再用 AkShare）并落 CSV")
    p_fetch.add_argument("--asset", choices=["etf", "index", "stock"], required=True, help="数据类型：etf / index / stock")
    p_fetch.add_argument(
        "--symbol",
        required=True,
        help="代码或名称：ETF 支持 510300 或 sh510300；指数支持 sh000300 / sz399006；个股支持 000725 / sz000725 / 京东方A",
    )
    p_fetch.add_argument(
        "--source",
        choices=["auto", "akshare", "tushare"],
        default="auto",
        help="行情数据源：auto(优先TuShare，失败回退AkShare) / akshare / tushare（ETF 走 fund_daily，不支持 qfq/hfq）",
    )
    p_fetch.add_argument("--start-date", default=None, help="开始日期（YYYYMMDD 或 YYYY-MM-DD，可选）")
    p_fetch.add_argument("--end-date", default=None, help="结束日期（YYYYMMDD 或 YYYY-MM-DD，可选）")
    p_fetch.add_argument("--freq", choices=["daily", "weekly"], default="weekly", help="输出频率（默认 weekly）")
    p_fetch.add_argument("--adjust", default=None, help="仅个股：复权方式（qfq/hfq/空，可选；默认 qfq）")
    p_fetch.add_argument("--out", default=None, help="输出 CSV 路径（默认 data/<asset>_<symbol>_<freq>.csv）")
    p_fetch.set_defaults(func=cmd_fetch)

    p = sub.add_parser("analyze", help="读取CSV -> 计算均线 -> (可选) LLM 生成标注 -> 出图")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--csv", default=None, help="CSV 文件路径")
    g.add_argument("--symbol", default=None, help="直接抓数分析：ETF/指数代码（如 510300 / sh000300）")
    p.add_argument("--encoding", default=None, help="CSV 编码（可选）")
    p.add_argument("--date-col", default=None, help="日期列名（可选）")
    p.add_argument("--open-col", default=None, help="开盘列名（可选）")
    p.add_argument("--high-col", default=None, help="最高列名（可选）")
    p.add_argument("--low-col", default=None, help="最低列名（可选）")
    p.add_argument("--close-col", default=None, help="收盘列名（可选）")
    p.add_argument("--volume-col", default=None, help="成交量列名（可选）")
    p.add_argument("--asset", choices=["etf", "index", "stock"], default="etf", help="当使用 --symbol 时必须指定资产类型（默认 etf）")
    p.add_argument(
        "--source",
        choices=["auto", "akshare", "tushare"],
        default="auto",
        help="行情数据源：auto(优先TuShare，失败回退AkShare) / akshare / tushare（仅在 --symbol 模式生效；ETF 走 fund_daily，不支持 qfq/hfq）",
    )
    p.add_argument("--start-date", default=None, help="开始日期（YYYYMMDD 或 YYYY-MM-DD，可选）")
    p.add_argument("--end-date", default=None, help="结束日期（YYYYMMDD 或 YYYY-MM-DD，可选）")
    p.add_argument("--adjust", default=None, help="仅个股：复权方式（qfq/hfq/空，可选；默认 qfq）")
    p.add_argument("--freq", choices=["daily", "weekly"], default="weekly", help="分析频率（默认 weekly）")
    p.add_argument("--window", type=int, default=500, help="只取最近 N 行（默认 500；周线≈10年）")
    p.add_argument("--out-dir", default=None, help="输出目录（默认 outputs/<name>_<timestamp>）")
    p.add_argument("--title", default=None, help="图标题（可选）")
    p.add_argument("--font-path", default=None, help="中文字体文件路径(.ttf/.otf)，用于解决缺字问题（可选）")
    p.add_argument(
        "--method",
        choices=["wyckoff", "chan", "ichimoku", "turtle", "momentum", "dow", "vsa", "institution", "both", "all"],
        default="wyckoff",
        help="分析方法（默认 wyckoff）",
    )
    p.add_argument("--chan-min-gap", type=int, default=4, help="缠论：分型成笔的最小间隔（默认 4，越大越稳但越少）")
    p.add_argument("--ichimoku-tenkan", type=int, default=9, help="一目：转换线周期（默认 9）")
    p.add_argument("--ichimoku-kijun", type=int, default=26, help="一目：基准线周期（默认 26）")
    p.add_argument("--ichimoku-spanb", type=int, default=52, help="一目：先行B周期（默认 52）")
    p.add_argument("--ichimoku-disp", type=int, default=26, help="一目：位移周期（默认 26）")
    p.add_argument("--turtle-entry", type=int, default=20, help="海龟：入场 Donchian 周期（默认 20）")
    p.add_argument("--turtle-exit", type=int, default=10, help="海龟：出场 Donchian 周期（默认 10）")
    p.add_argument("--turtle-atr", type=int, default=20, help="海龟：ATR 周期（默认 20）")
    p.add_argument("--turtle-stop-atr", type=float, default=2.0, help="海龟：止损倍数（默认 2.0*ATR）")
    p.add_argument("--rsi-period", type=int, default=14, help="Momentum：RSI 周期（默认 14）")
    p.add_argument("--macd-fast", type=int, default=12, help="Momentum：MACD 快线 EMA（默认 12）")
    p.add_argument("--macd-slow", type=int, default=26, help="Momentum：MACD 慢线 EMA（默认 26）")
    p.add_argument("--macd-signal", type=int, default=9, help="Momentum：MACD 信号线 EMA（默认 9）")
    p.add_argument("--adx-period", type=int, default=14, help="Momentum：ADX 周期（默认 14）")
    p.add_argument("--dow-lookback", type=int, default=2, help="Dow：分型 lookback（默认 2；越大越稳但越少）")
    p.add_argument("--dow-min-gap", type=int, default=2, help="Dow：swing 最小间隔（默认 2）")
    p.add_argument("--vsa-vol-window", type=int, default=20, help="VSA：相对成交量窗口（默认 20）")
    p.add_argument("--vsa-spread-window", type=int, default=20, help="VSA：相对 spread 窗口（默认 20）")
    p.add_argument("--vsa-lookback", type=int, default=120, help="VSA：最近事件回看根数（默认 120）")
    p.add_argument("--llm", action="store_true", help="启用 LLM 结构化分析（需要 OPENAI_API_KEY/OPENAI_MODEL）")
    p.add_argument(
        "--prompt",
        default=str(Path("prompts") / "wyckoff_json_prompt.md"),
        help="威科夫提示词路径（默认 prompts/wyckoff_json_prompt.md）",
    )
    p.add_argument(
        "--chan-prompt",
        default=str(Path("prompts") / "chanlun_json_prompt.md"),
        help="缠论解读提示词路径（默认 prompts/chanlun_json_prompt.md）",
    )
    p.add_argument(
        "--vsa-prompt",
        default=str(Path("prompts") / "vsa_json_prompt.md"),
        help="VSA 解读提示词路径（默认 prompts/vsa_json_prompt.md）",
    )
    p.add_argument("--max-rows-llm", type=int, default=300, help="喂给 LLM 的最大行数（默认 300，会等距抽样）")
    p.add_argument("--narrate", action="store_true", help="生成“多流派自然语言解读”（默认用 Gemini，需要 GEMINI_API_KEY/MODEL）")
    p.add_argument("--narrate-provider", choices=["gemini", "openai"], default="openai", help="自然语言解读的 LLM 提供方（默认 openai）")
    p.add_argument(
        "--narrate-prompt",
        default=str(Path("prompts") / "synthesis_prompt.md"),
        help="自然语言解读提示词路径（默认 prompts/synthesis_prompt.md）",
    )
    p.add_argument(
        "--narrate-schools",
        default="chan,wyckoff,ichimoku,turtle,momentum",
        help="参与综合解读的流派列表，逗号分隔（默认 5 派：chan,wyckoff,ichimoku,turtle,momentum；可加 dow,vsa,institution）",
    )
    p.add_argument("--narrate-temperature", type=float, default=0.2, help="自然语言解读 temperature（默认 0.2）")
    p.add_argument("--narrate-max-output-tokens", type=int, default=1200, help="自然语言解读最大输出 token（默认 1200）")

    # Phase3：可选输出（不影响原有口径）：按 strategy_configs.yaml 额外算一份 strategy_signal.json
    p.add_argument("--strategy-config", default=None, help="策略配置文件路径（YAML；配合 --strategy；默认不启用）")
    p.add_argument("--strategy", default=None, help="策略 key（例 conservative；配合 --strategy-config）")
    p.add_argument("--strategy-regime-index", default="sh000300", help="用于 allowed_regimes 的市场 regime 指数（默认 sh000300；off=关闭）")
    p.add_argument(
        "--strategy-regime-canary",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="strategy-regime 的 canary 降级开关（默认启用）",
    )
    p.set_defaults(func=cmd_analyze)

    p_scan = sub.add_parser("scan-etf", help="扫描场内 ETF/基金，输出波段候选排名（研究用途）")
    p_scan.add_argument("--freq", choices=["daily", "weekly"], default="weekly", help="扫描频率（默认 weekly）")
    p_scan.add_argument("--window", type=int, default=400, help="每个标的取最近 N 根K线（默认 400）")
    p_scan.add_argument("--min-weeks", type=int, default=60, help="周K 少于该值不进榜（默认 60；填 0 关闭）")
    p_scan.add_argument(
        "--min-amount",
        type=float,
        default=0.0,
        help="过滤最后一根成交额小于该值的标的（默认 0=不过滤；优先用数据源 amount，缺失才用 close*volume 估算）",
    )
    p_scan.add_argument(
        "--min-amount-avg20",
        type=float,
        default=0.0,
        help="过滤最近20日均成交额小于该值的标的（默认 0=不过滤；更适合当流动性门槛）",
    )
    p_scan.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="过滤 OpportunityScore(0~1) 低于该值的标的（默认 0=不过滤；胜率/磨损等口径不变）",
    )
    p_scan.add_argument("--limit", type=int, default=0, help="只扫描前 N 个（默认 0=全量；按代码排序）")
    p_scan.add_argument("--top-k", type=int, default=30, help="输出 Top K（默认 30）")
    p_scan.add_argument("--out-dir", default=None, help="输出目录（默认 outputs/etf_scan_<timestamp>）")
    p_scan.add_argument("--workers", type=int, default=8, help="并发线程数（默认 8）")
    p_scan.add_argument("--cache-dir", default=None, help="ETF 日线缓存目录（默认 data/cache/etf）")
    p_scan.add_argument("--cache-ttl-hours", type=float, default=24.0, help="缓存有效期（小时；0=不使用缓存；默认 24）")
    p_scan.add_argument(
        "--analysis-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="启用派生结果缓存（加速重复扫描；默认启用；可用 --no-analysis-cache 关闭）",
    )
    p_scan.add_argument("--analysis-cache-dir", default=None, help="派生结果缓存目录（默认 data/cache/analysis/etf）")
    p_scan.add_argument(
        "--include-all-funds",
        action="store_true",
        help="把 LOF/固收/其它场内基金也算进去（默认只扫股票/海外股票 ETF：15xxxx + 5[1/2/3/6/8/9]xxxx）",
    )
    p_scan.add_argument("--bbb-horizons", default="4,8,12", help="BBB 胜率统计持有周数，逗号分隔（默认 4,8,12）")
    p_scan.add_argument("--bbb-rank-horizon", type=int, default=8, help="BBB 排名使用的 horizon（默认 8）")
    p_scan.add_argument(
        "--bbb-score-mode",
        choices=["win_rate", "annualized"],
        default="annualized",
        help="BBB 排名分数：win_rate=胜率优先（更稳）；annualized=年化优先（小资金+磨损大想冲最大年化就选它）",
    )
    p_scan.add_argument("--bbb-min-trades", type=int, default=0, help="BBB 排名口径 trades 少于该值直接过滤（默认 0=不过滤）")
    p_scan.add_argument("--bbb-min-win-rate", type=float, default=0.0, help="BBB 排名口径 win_rate 低于该值直接过滤（默认 0=不过滤）")
    p_scan.add_argument("--bbb-allow-overlap", action="store_true", help="BBB 胜率统计允许样本重叠（默认不允许，避免假高胜率）")
    p_scan.add_argument("--bbb-include-samples", action="store_true", help="BBB 输出包含收益分布 sample（默认不包含，文件会更大）")
    p_scan.add_argument(
        "--bbb-mode",
        choices=["auto", "strict", "pullback", "early"],
        default="auto",
        help="BBB 模式：auto=根据大盘牛熊自动选（牛/中性=>pullback，熊=>strict）；strict=最保守；pullback=允许周线回踩（更贴近右侧定方向+回踩挑位置）；early=更早但更容易吃回撤",
    )
    p_scan.add_argument(
        "--bbb-allow-bear",
        action="store_true",
        help="允许 BBB 在熊市里也给买入候选（默认不允许：bear 时直接过滤，省得你用右侧策略去硬刚熊市）",
    )
    p_scan.add_argument(
        "--regime-index",
        default="sh000300,sz399006",
        help="大盘指数代码（用于牛熊/风险偏好判断；默认 sh000300,sz399006；支持逗号分隔多指数：sh000300,sz399006；可用 ';' 显式指定 canary：sh000300,sz399006;sh000852；填 off 关闭）",
    )
    p_scan.add_argument(
        "--regime-canary",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="多指数 regime 的 canary 降级开关（默认启用；--no-regime-canary 更灵敏；canary=逗号后的指数）",
    )
    p_scan.add_argument(
        "--bbb-rs-index",
        default="sh000300+sh000905",
        help="BBB 7因子面板 RS 基准指数（支持 '+' 合成；默认 sh000300+sh000905=300+500 等权；auto=用 --regime-index 的第一个指数；off=关闭）",
    )
    p_scan.add_argument("--bbb-entry-ma", type=int, default=50, help="BBB 入场参考均线周期（周线；默认 50；可用 20 更贴近数周波段）")
    p_scan.add_argument("--bbb-dist-ma-max", type=float, default=0.12, help="BBB 允许离入场均线的最大偏离（默认 0.12=12%%）")
    p_scan.add_argument("--bbb-max-above-20w", type=float, default=0.05, help="BBB 允许高于20W上轨的最大比例（默认 0.05=5%%；越小越不追高）")
    p_scan.add_argument(
        "--bbb-factor7",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="BBB 7因子面板排序加权（默认启用；--no-bbb-factor7 关闭；不影响 bbb.ok/fails 硬条件）",
    )
    p_scan.add_argument(
        "--bbb-factor7-weights",
        default="",
        help="BBB 7因子权重：rs=0.35,trend=0.15,vol=0.15,drawdown=0.15,liquidity=0.10,boll=0.05,volume=0.05（留空=默认；会自动归一化）",
    )
    p_scan.add_argument("--capital-yuan", type=float, default=3000.0, help="单笔投入资金（元，用于把固定磨损换算为比例成本；默认 3000）")
    p_scan.add_argument("--roundtrip-cost-yuan", type=float, default=10.0, help="一进一出总磨损（元，默认 10）")
    p_scan.add_argument("--min-fee-yuan", type=float, default=0.0, help="最低佣金（每边，元；默认 0）")
    p_scan.add_argument("--buy-cost", type=float, default=None, help="买入比例成本（例如 0.001=0.10%%；与 roundtrip/min_fee 一起生效）")
    p_scan.add_argument("--sell-cost", type=float, default=None, help="卖出比例成本（例如 0.001=0.10%%；与 roundtrip/min_fee 一起生效）")
    p_scan.add_argument(
        "--bbb-slippage-mode",
        choices=["none", "fixed", "liquidity"],
        default="none",
        help="BBB 额外滑点/冲击成本模型（默认 none；fixed=固定bps；liquidity=按近20日均成交额估算）",
    )
    p_scan.add_argument(
        "--bbb-slippage-bps",
        type=float,
        default=0.0,
        help="滑点 bps（每边）：mode=fixed 时为固定bps；mode=liquidity 时表示在 ref_amount_yuan 下的 bps（默认 0）",
    )
    p_scan.add_argument("--bbb-slippage-ref-amount-yuan", type=float, default=1e8, help="liquidity 模式参考成交额（元，默认 1e8）")
    p_scan.add_argument("--bbb-slippage-bps-min", type=float, default=0.0, help="liquidity 模式最小 bps（默认 0）")
    p_scan.add_argument("--bbb-slippage-bps-max", type=float, default=30.0, help="liquidity 模式最大 bps（默认 30）")
    p_scan.add_argument("--bbb-slippage-unknown-bps", type=float, default=10.0, help="liquidity 模式成交额缺失时的 bps（默认 10）")
    p_scan.add_argument("--bbb-slippage-vol-mult", type=float, default=0.0, help="波动放大系数（0=不按波动放大；默认 0）")
    p_scan.add_argument("--bbb-exit-min-hold-days", type=int, default=5, help="BBB 出场回测：最少持有天数（默认 5，避免日线噪声磨损）")
    p_scan.add_argument("--bbb-exit-cooldown-days", type=int, default=0, help="BBB 出场回测：卖出后冷却天数（默认 0）")
    p_scan.add_argument(
        "--bbb-exit-trail",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="BBB 出场回测：启用周线锚线(trail)（默认启用；--no-bbb-exit-trail 关闭）",
    )
    p_scan.add_argument("--bbb-exit-trail-ma", type=int, default=20, help="BBB 出场回测：周线锚线均线周期（默认 20）")
    p_scan.add_argument(
        "--bbb-exit-profit-stop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="BBB 出场回测：启用盈利回撤止盈(profit_stop)（默认启用；--no-bbb-exit-profit-stop 关闭）",
    )
    p_scan.add_argument("--bbb-exit-profit-min-ret", type=float, default=0.20, help="BBB 出场回测：回撤止盈启用的最低浮盈（默认 0.20=20%%）")
    p_scan.add_argument("--bbb-exit-profit-dd-pct", type=float, default=0.12, help="BBB 出场回测：回撤止盈回撤比例（默认 0.12=12%%）")
    p_scan.add_argument("--bbb-exit-stop-loss-ret", type=float, default=0.0, help="BBB 出场回测：最大亏损止损（按收盘触发；默认 0=关闭）")
    p_scan.add_argument(
        "--bbb-exit-panic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="BBB 出场回测：启用 panic 兜底（大跌/深回撤快速离场；默认启用；--no-bbb-exit-panic 关闭）",
    )
    p_scan.add_argument("--bbb-exit-panic-vol-mult", type=float, default=3.0, help="BBB 出场回测：panic 波动倍数阈值（默认 3.0）")
    p_scan.add_argument("--bbb-exit-panic-min-drop", type=float, default=0.04, help="BBB 出场回测：panic 最小日跌幅阈值（默认 0.04=4%%）")
    p_scan.add_argument("--bbb-exit-panic-drawdown-252d", type=float, default=0.25, help="BBB 出场回测：panic 1年回撤阈值（默认 0.25=25%%）")
    p_scan.add_argument("--verbose", action="store_true", help="打印扫描进度（可选）")
    p_scan.set_defaults(func=cmd_scan_etf)

    p_chain = sub.add_parser("commodity-chain", help="扫描大宗商品链路 ETF 热度（黄金→有色→油化→农产品）")
    p_chain.add_argument("--min-days", type=int, default=80, help="最少历史天数（默认 80）")
    p_chain.add_argument("--top-k", type=int, default=3, help="每段输出 Top K（默认 3）")
    p_chain.add_argument("--out-dir", default=str(Path("outputs") / "agents"), help="输出目录（默认 outputs/agents）")
    p_chain.set_defaults(func=cmd_commodity_chain)

    p_plan = sub.add_parser("plan-etf", help="根据 scan-etf 的 BBB 结果生成仓位/止损计划（研究用途）")
    p_plan.add_argument("--scan-dir", default=None, help="scan-etf 输出目录（默认读取 <scan-dir>/top_bbb.json）")
    p_plan.add_argument("--input", default=None, help="直接指定 top_bbb.json 路径（优先级高于 --scan-dir）")
    p_plan.add_argument("--out", default=None, help="输出路径（默认写到 top_bbb.json 同目录下 position_plan.json）")
    p_plan.add_argument("--capital-yuan", type=float, default=None, help="总资金（默认继承 scan-etf 的 capital_yuan）")
    p_plan.add_argument("--roundtrip-cost-yuan", type=float, default=None, help="来回磨损（元，默认继承 scan-etf 的 roundtrip_cost_yuan）")
    p_plan.add_argument("--lot-size", type=int, default=100, help="最小交易单位（ETF 通常 100；默认 100）")
    p_plan.add_argument("--max-cost-pct", type=float, default=0.02, help="磨损占仓位比例上限（默认 0.02=2%%；超过则跳过）")
    p_plan.add_argument("--risk-min-yuan", type=float, default=None, help="单笔最小风险预算（元；默认 roundtrip_cost_yuan*3）")
    p_plan.add_argument("--risk-per-trade-yuan", type=float, default=None, help="单笔风险预算（元；优先级高于 risk_per_trade_pct）")
    p_plan.add_argument("--max-exposure-pct", type=float, default=None, help="最大总持仓比例（覆盖牛熊默认；例如 0.6=60%%）")
    p_plan.add_argument("--risk-per-trade-pct", type=float, default=None, help="单笔风险比例（覆盖牛熊默认；例如 0.01=1%%）")
    p_plan.add_argument(
        "--stop-mode",
        choices=["weekly_entry_ma", "daily_ma20", "atr"],
        default=None,
        help="硬止损参考：weekly_entry_ma=周线entry_ma；daily_ma20=日线MA20；atr=entry-ATR*mult（默认随牛熊自动选）",
    )
    p_plan.add_argument("--atr-mult", type=float, default=2.0, help="stop-mode=atr 时 ATR 倍数（默认 2.0）")
    p_plan.add_argument("--max-positions", type=int, default=None, help="最多持仓标的数（默认随牛熊自动选）")
    p_plan.add_argument("--returns-cache-dir", default=None, help="相关性/分散用的 ETF 日线缓存目录（默认 data/cache/etf）")
    p_plan.add_argument(
        "--diversify",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="启用分散过滤（相关性去重/同主题限仓；默认启用；可用 --no-diversify 关闭）",
    )
    p_plan.add_argument("--diversify-window-weeks", type=int, default=104, help="相关性计算使用的周收益窗口（默认 104）")
    p_plan.add_argument("--diversify-min-overlap-weeks", type=int, default=26, help="相关性最小重叠周数（默认 26）")
    p_plan.add_argument("--diversify-max-corr", type=float, default=0.95, help="相关性阈值（|corr|>=该值视为同一类，默认 0.95）")
    p_plan.add_argument("--max-per-theme", type=int, default=0, help="同主题最多持有 N 个（0=不限制；默认 0）")
    p_plan.set_defaults(func=cmd_plan_etf)

    p_hold = sub.add_parser("holdings-etf", help="ETF 持仓分析（收盘价止损 + 跟随牛熊）")
    p_hold.add_argument(
        "--item",
        action="append",
        default=[],
        help="单条持仓：symbol,shares,cost（可多次指定）；例：512400,500,1.967",
    )
    p_hold.add_argument("--input", default=None, help="从 JSON 文件读取：{holdings:[{symbol,shares,cost}...]}")
    p_hold.add_argument(
        "--regime-index",
        default="sh000300",
        help="大盘指数代码（用于牛熊；默认 sh000300；支持逗号分隔多指数：sh000300,sz399006；可用 ';' 显式指定 canary：sh000300,sz399006;sh000852；填 off 关闭）",
    )
    p_hold.add_argument(
        "--regime-canary",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="多指数 regime 的 canary 降级开关（默认启用；--no-regime-canary 更灵敏）",
    )
    p_hold.add_argument("--sell-cost-yuan", type=float, default=5.0, help="卖出固定磨损（元，默认 5）")
    p_hold.add_argument("--out", default=None, help="输出 JSON 路径（默认打印到 stdout）")
    p_hold.set_defaults(func=cmd_holdings_etf)

    p_hold_user = sub.add_parser("holdings-user", help="从 data/user_holdings.json 一键分析持仓（ETF/股票）")
    p_hold_user.add_argument(
        "--path",
        "--holdings-path",
        dest="path",
        default=str(Path("data") / "user_holdings.json"),
        help="持仓文件路径（默认 data/user_holdings.json）",
    )
    p_hold_user.add_argument(
        "--regime-index",
        default="sh000300",
        help="大盘指数代码（用于牛熊；默认 sh000300；支持逗号分隔多指数：sh000300,sz399006；可用 ';' 显式指定 canary：sh000300,sz399006;sh000852；填 off 关闭）",
    )
    p_hold_user.add_argument(
        "--regime-canary",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="多指数 regime 的 canary 降级开关（默认启用；--no-regime-canary 更灵敏）",
    )
    p_hold_user.add_argument("--sell-cost-yuan", type=float, default=5.0, help="卖出固定磨损（元，默认 5）")
    p_hold_user.add_argument("--cache-ttl-hours", type=float, default=6.0, help="K 线缓存 TTL（小时；默认 6）")
    p_hold_user.add_argument("--stock-adjust", default="qfq", help="股票复权方式（qfq/hfq；默认 qfq）")
    p_hold_user.add_argument("--out", default=None, help="输出 JSON 路径（默认打印到 stdout）")
    p_hold_user.set_defaults(func=cmd_holdings_user)

    p_reb_user = sub.add_parser("rebalance-user", help="从 user_holdings.json + top_bbb.json 生成组合调仓计划（研究用途）")
    p_reb_user.add_argument("--path", default=str(Path("data") / "user_holdings.json"), help="持仓文件路径（默认 data/user_holdings.json）")
    p_reb_user.add_argument("--signals", required=True, help="signals 文件路径（当前支持 scan-etf 的 top_bbb.json）")
    p_reb_user.add_argument("--mode", choices=["add", "rotate"], default="add", help="add=按总权益算目标仓位、只用现金增量加仓；rotate=按目标轮动/再平衡（会卖出非目标）")
    p_reb_user.add_argument("--capital-yuan", type=float, default=None, help="覆盖 capital_yuan（add 默认用 cash.amount；rotate 默认用 equity）")
    p_reb_user.add_argument("--out", default=None, help="输出 JSON 路径（默认打印到 stdout）")

    p_reb_user.add_argument(
        "--regime-index",
        default="sh000300",
        help="大盘指数代码（用于牛熊；默认 sh000300；支持逗号分隔多指数：sh000300,sz399006；可用 ';' 显式指定 canary：sh000300,sz399006;sh000852；填 off 关闭）",
    )
    p_reb_user.add_argument(
        "--regime-canary",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="多指数 regime 的 canary 降级开关（默认启用；--no-regime-canary 更灵敏）",
    )
    p_reb_user.add_argument("--sell-cost-yuan", type=float, default=5.0, help="卖出固定磨损（元，默认 5）")
    p_reb_user.add_argument("--cache-ttl-hours", type=float, default=6.0, help="K 线缓存 TTL（小时；默认 6）")
    p_reb_user.add_argument("--stock-adjust", default="qfq", help="股票复权方式（qfq/hfq；默认 qfq）")

    # 复用 plan-etf 的核心风控/分散参数（组合调仓也是同一套口径）。
    p_reb_user.add_argument("--roundtrip-cost-yuan", type=float, default=None, help="来回磨损（元，默认继承 signals.bbb.roundtrip_cost_yuan）")
    p_reb_user.add_argument("--min-fee-yuan", type=float, default=None, help="最低佣金（每边；默认继承 signals.config.min_fee_yuan 或 0）")
    p_reb_user.add_argument("--buy-cost", type=float, default=None, help="买入比例成本（默认继承 signals.config.buy_cost 或 0）")
    p_reb_user.add_argument("--sell-cost", type=float, default=None, help="卖出比例成本（默认继承 signals.config.sell_cost 或 0）")
    p_reb_user.add_argument(
        "--slippage-mode",
        choices=["none", "fixed", "liquidity"],
        default=None,
        help="滑点/冲击成本模型（默认继承 signals.config；none/fixed/liquidity）",
    )
    p_reb_user.add_argument("--slippage-bps", type=float, default=None, help="fixed/liquidity 模式基础 bps（默认继承 signals.config）")
    p_reb_user.add_argument("--slippage-ref-amount-yuan", type=float, default=None, help="liquidity 模式参考成交额（元，默认继承 signals.config）")
    p_reb_user.add_argument("--slippage-bps-min", type=float, default=None, help="liquidity 模式最小 bps（默认继承 signals.config）")
    p_reb_user.add_argument("--slippage-bps-max", type=float, default=None, help="liquidity 模式最大 bps（默认继承 signals.config）")
    p_reb_user.add_argument("--slippage-unknown-bps", type=float, default=None, help="成交额缺失时 bps（默认继承 signals.config）")
    p_reb_user.add_argument("--slippage-vol-mult", type=float, default=None, help="波动放大系数（默认继承 signals.config）")
    p_reb_user.add_argument("--lot-size", type=int, default=100, help="最小交易单位（ETF 通常 100；默认 100）")
    p_reb_user.add_argument("--limit-up-pct", type=float, default=0.0, help="涨停阈值（比例；默认 0=不启用；例如 0.10=10%%）")
    p_reb_user.add_argument("--limit-down-pct", type=float, default=0.0, help="跌停阈值（比例；默认 0=不启用；例如 0.10=10%%）")
    p_reb_user.add_argument(
        "--halt-vol-zero",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="把 volume/amount=0 视为停牌/无成交（默认启用；可用 --no-halt-vol-zero 关闭）",
    )
    p_reb_user.add_argument("--max-cost-pct", type=float, default=0.02, help="磨损占仓位比例上限（默认 0.02=2%%；超过则跳过）")
    p_reb_user.add_argument("--risk-min-yuan", type=float, default=None, help="单笔最小风险预算（元；默认 roundtrip_cost_yuan*3）")
    p_reb_user.add_argument("--risk-per-trade-yuan", type=float, default=None, help="单笔风险预算（元；优先级高于 risk_per_trade_pct）")
    p_reb_user.add_argument("--max-exposure-pct", type=float, default=None, help="最大总持仓比例（覆盖牛熊默认；例如 0.6=60%%）")
    p_reb_user.add_argument(
        "--cash-signal",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="启用 CashSignal 风控：用现金比例建议把 max_exposure_pct 做“只降不升”上限（默认关闭；用 --cash-signal 开启）",
    )
    p_reb_user.add_argument("--risk-per-trade-pct", type=float, default=None, help="单笔风险比例（覆盖牛熊默认；例如 0.01=1%%）")
    p_reb_user.add_argument(
        "--vol-target",
        type=float,
        default=0.0,
        help="波动率目标（年化；0=关闭；例如 0.15=15%%）。当前实现=只用指数波动率把 max_exposure_pct 做“只降不升”的缩放（研究用途）。",
    )
    p_reb_user.add_argument("--vol-lookback-days", type=int, default=20, help="vol-target 的日线波动率回看天数（默认 20）")
    p_reb_user.add_argument(
        "--min-trade-notional-yuan",
        type=float,
        default=None,
        help="单笔买入最小成交额门槛（元；默认读 user_holdings.trade_rules.min_trade_notional_yuan；0=不限制）",
    )
    p_reb_user.add_argument(
        "--max-turnover-pct",
        type=float,
        default=0.0,
        help="单次调仓“买入侧”最大换手（占 equity；0=不限制；例如 0.10=10%%）。说明：当前只限制 buy，不限制 sell（KISS）。",
    )
    p_reb_user.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="过滤 OpportunityScore(0~1) 低于该值的候选（默认 0=不过滤；只影响候选质量，不改其他约束）",
    )
    p_reb_user.add_argument(
        "--stop-mode",
        choices=["weekly_entry_ma", "daily_ma20", "atr"],
        default=None,
        help="硬止损参考：weekly_entry_ma=周线entry_ma；daily_ma20=日线MA20；atr=entry-ATR*mult（默认随牛熊自动选）",
    )
    p_reb_user.add_argument("--atr-mult", type=float, default=2.0, help="stop-mode=atr 时 ATR 倍数（默认 2.0）")
    p_reb_user.add_argument(
        "--max-positions",
        type=int,
        default=None,
        help="最多持仓标的数（默认读 user_holdings.trade_rules.max_positions；否则随牛熊自动选）",
    )
    p_reb_user.add_argument(
        "--max-position-pct",
        type=float,
        default=None,
        help="单标的最大仓位占比（例如 0.30=30%%；默认读 user_holdings.trade_rules.max_position_pct；为空=不限制）",
    )
    p_reb_user.add_argument(
        "--core-satellite",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="ETF+stock 混合时启用核心+卫星（默认启用；--no-core-satellite 关闭）",
    )
    p_reb_user.add_argument("--national-team-path", default=None, help="national_team.json 路径（可选；用于国家队风险过滤）")
    p_reb_user.add_argument(
        "--national-team-guard",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="启用国家队 proxy 风险过滤（默认启用；--no-national-team-guard 关闭）",
    )
    p_reb_user.add_argument("--national-team-min-score", type=float, default=0.40, help="国家队风险过滤：risk_off 阈值（默认 0.40）")
    p_reb_user.add_argument("--national-team-warn-score", type=float, default=0.55, help="国家队风险过滤：caution 阈值（默认 0.55）")
    p_reb_user.add_argument("--national-team-scale-low", type=float, default=0.35, help="risk_off 时最大仓位缩放比例（默认 0.35）")
    p_reb_user.add_argument("--national-team-scale-mid", type=float, default=0.70, help="caution 时最大仓位缩放比例（默认 0.70）")
    p_reb_user.add_argument("--national-team-max-positions-low", type=int, default=1, help="risk_off 时最多持仓数上限（默认 1）")
    p_reb_user.add_argument(
        "--fund-flow-check",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="主力资金复核开关（默认启用；--no-fund-flow-check 关闭）",
    )
    p_reb_user.add_argument(
        "--fund-flow-check-mode",
        choices=["auto", "meta_only"],
        default="auto",
        help="主力资金复核模式：auto=优先 signals.meta.fund_flow，再用 TuShare/Eastmoney；meta_only=只用 signals.meta.fund_flow",
    )
    p_reb_user.add_argument("--fund-flow-block-score", type=float, default=0.45, help="主力资金复核：score01 < 该值阻断 buy（默认 0.45）")
    p_reb_user.add_argument("--fund-flow-warn-score", type=float, default=0.55, help="主力资金复核：score01 < 该值警告（默认 0.55）")
    p_reb_user.add_argument("--returns-cache-dir", default=None, help="相关性/分散用的日线缓存目录（默认 data/cache；会按 asset 分子目录）")
    p_reb_user.add_argument(
        "--diversify",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="启用分散过滤（相关性去重/同主题限仓；默认启用；可用 --no-diversify 关闭）",
    )
    p_reb_user.add_argument("--diversify-window-weeks", type=int, default=104, help="相关性计算使用的周收益窗口（默认 104）")
    p_reb_user.add_argument("--diversify-min-overlap-weeks", type=int, default=26, help="相关性最小重叠周数（默认 26）")
    p_reb_user.add_argument("--diversify-max-corr", type=float, default=0.95, help="相关性阈值（|corr|>=该值视为同一类，默认 0.95）")
    p_reb_user.add_argument("--max-corr", type=float, default=None, help="别名：--diversify-max-corr（优先级更高）")
    p_reb_user.add_argument("--max-per-theme", type=int, default=0, help="同主题最多持有 N 个（0=不限制；默认 0）")
    p_reb_user.set_defaults(func=cmd_rebalance_user)

    p_nt = sub.add_parser("national-team", help="国家队/托底代理指标（ETF份额/资金流/尾盘护盘；研究用途）")
    p_nt.add_argument("--as-of", default=None, help="截止日期（YYYYMMDD 或 YYYY-MM-DD；默认=今天/最新）")
    p_nt.add_argument("--index-symbol", default="sh000300", help="盯盘指数（用于尾盘护盘特征；默认 sh000300）")
    p_nt.add_argument("--wide-etfs", default="", help="宽基ETF列表（逗号分隔；默认内置：300/50/500/创业板/科创50等）")
    p_nt.add_argument("--flow-lookback-days", type=int, default=120, help="ETF主力净流入回看窗口（天；默认 120）")
    p_nt.add_argument("--tail-window-minutes", type=int, default=30, help="尾盘窗口（分钟；默认 30）")
    p_nt.add_argument("--w-etf-flow", type=float, default=0.55, help="综合分权重：ETF资金流（默认 0.55）")
    p_nt.add_argument("--w-etf-shares", type=float, default=0.25, help="综合分权重：ETF份额Δ（默认 0.25）")
    p_nt.add_argument("--w-tail", type=float, default=0.20, help="综合分权重：尾盘护盘（默认 0.20）")
    p_nt.add_argument("--w-northbound", type=float, default=0.0, help="综合分权重：北向（默认 0；免费源常缺）")
    p_nt.add_argument("--cache-ttl-hours", type=float, default=6.0, help="缓存 TTL（小时；默认 6）")
    p_nt.add_argument("--out", default=None, help="输出 JSON 路径（可选；不传则打印到 stdout）")
    p_nt.set_defaults(func=cmd_national_team)

    p_ntb = sub.add_parser("national-team-backtest", help="回测：ETF主力净流入 proxy（研究用途；受数据源近120日限制）")
    p_ntb.add_argument("--out-dir", default=None, help="输出目录（默认 outputs/nt_backtest_YYYYMMDD；同日重复会自动加后缀）")
    p_ntb.add_argument("--index-symbol", default="sh000300", help="对齐的指数（默认 sh000300）")
    p_ntb.add_argument("--wide-etfs", default="", help="宽基ETF列表（逗号分隔；默认同 national-team 内置）")
    p_ntb.add_argument("--start-date", default=None, help="开始日期（YYYYMMDD 或 YYYY-MM-DD；可选）")
    p_ntb.add_argument("--end-date", default=None, help="结束日期（YYYYMMDD 或 YYYY-MM-DD；可选）")
    p_ntb.add_argument("--lookback-days", type=int, default=60, help="rolling z-score 回看窗口（默认 60）")
    p_ntb.add_argument("--cache-ttl-hours", type=float, default=24.0, help="缓存 TTL（小时；默认 24）")
    p_ntb.set_defaults(func=cmd_national_team_backtest)

    p_run = sub.add_parser(
        "run",
        help="日常跑批：scan-strategy(默认,因子库) -> (可选 legacy 对照/兜底) -> holdings-user -> rebalance-user -> report（研究用途）",
    )
    p_run.add_argument("--out-dir", default=None, help="输出目录（默认 outputs/run_YYYYMMDD；同日重复会自动加后缀）")
    p_run.add_argument(
        "--signals",
        action="append",
        default=[],
        help="直接使用已有 signals.json（可重复传多次；多份会自动 signals-merge；提供则跳过 scan-etf）",
    )
    p_run.add_argument("--signals-merge-conflict", choices=["risk_first", "priority", "vote"], default="risk_first", help="多份 signals 冲突裁决模式（默认 risk_first）")
    p_run.add_argument("--signals-merge-weights", default="", help="signals-merge 权重：bbb_weekly=1,trend_pullback_weekly=0.8（可选）")
    p_run.add_argument("--signals-merge-priority", default="", help="signals-merge 优先级：trend_pullback_weekly,bbb_weekly,...（conflict=priority 时生效）")
    p_run.add_argument("--signals-merge-top-k", type=int, default=0, help="signals-merge 只输出前 K 个（默认 0=全量）")
    p_run.add_argument("--holdings-path", default=str(Path("data") / "user_holdings.json"), help="持仓快照路径（默认 data/user_holdings.json）")
    p_run.add_argument(
        "--regime-index",
        default="sh000300",
        help="大盘指数（默认 sh000300；支持逗号分隔多指数；可用 ';' 显式指定 canary：sh000300,sz399006;sh000852；填 off 关闭）",
    )
    p_run.add_argument(
        "--regime-canary",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="多指数 regime 的 canary 降级开关（默认启用；--no-regime-canary 更灵敏）",
    )
    p_run.add_argument("--scan-freq", choices=["daily", "weekly"], default="weekly", help="扫描频率（默认 weekly；scan-strategy/scan-etf 共用）")
    p_run.add_argument("--scan-limit", type=int, default=200, help="扫描数量（默认 200；按代码排序）")
    p_run.add_argument("--scan-min-weeks", type=int, default=60, help="legacy scan-etf 最少周K根数（默认 60；scan-strategy 不使用）")
    p_run.add_argument(
        "--scan-mode",
        choices=["auto", "strategy", "legacy"],
        default="auto",
        help="扫描模式：auto=优先 scan-strategy(因子库) 不行再回退 legacy；strategy=只用因子库；legacy=只用旧 scan-etf",
    )
    p_run.add_argument(
        "--scan-strategy-config",
        default=str(Path("config") / "strategy_configs.yaml"),
        help="scan-strategy 的策略配置文件（默认 config/strategy_configs.yaml）",
    )
    p_run.add_argument("--scan-strategy", default="bbb_weekly", help="scan-strategy 的策略 key（默认 bbb_weekly）")
    p_run.add_argument("--scan-top-k", type=int, default=30, help="扫描输出 TopK（同时作用于 scan-strategy/scan-etf；默认 30）")
    p_run.add_argument(
        "--scan-shadow-legacy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="稳健切换：额外跑一份 legacy scan-etf 作为对照 + 生成 strategy-align 报告（默认启用；想快就 --no-scan-shadow-legacy）",
    )
    p_run.add_argument("--scan-align-top-k", type=int, default=30, help="strategy-align 对齐用的 TopK（默认 30）")
    p_run.add_argument(
        "--scan-left",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="额外跑一份“左侧低吸(高赔率)” scan-strategy 并写入 report（默认启用；想快就 --no-scan-left）",
    )
    p_run.add_argument("--scan-left-strategy", default="left_dip_rr", help="左侧 scan-strategy 的策略 key（默认 left_dip_rr）")
    p_run.add_argument("--scan-left-top-k", type=int, default=30, help="左侧候选输出 TopK（默认 30）")
    p_run.add_argument("--scan-stock", action=argparse.BooleanOptionalAction, default=False, help="额外跑一份 stock 的 scan-strategy 并写入 report（默认关闭）")
    p_run.add_argument(
        "--merge-stock-signals",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="scan-stock 时把 signals_stock 合并进 signals.json 供 rebalance-user 使用（默认启用；--no-merge-stock-signals 关闭）",
    )
    p_run.add_argument(
        "--core-satellite",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="ETF+stock 混合时启用核心+卫星（默认启用；--no-core-satellite 关闭；传给 rebalance-user）",
    )
    p_run.add_argument(
        "--national-team-guard",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="启用国家队 proxy 风险过滤（默认启用；--no-national-team-guard 关闭；传给 rebalance-user）",
    )
    p_run.add_argument("--national-team-min-score", type=float, default=0.40, help="国家队风险过滤：risk_off 阈值（默认 0.40）")
    p_run.add_argument("--national-team-warn-score", type=float, default=0.55, help="国家队风险过滤：caution 阈值（默认 0.55）")
    p_run.add_argument("--national-team-scale-low", type=float, default=0.35, help="risk_off 时最大仓位缩放比例（默认 0.35）")
    p_run.add_argument("--national-team-scale-mid", type=float, default=0.70, help="caution 时最大仓位缩放比例（默认 0.70）")
    p_run.add_argument("--national-team-max-positions-low", type=int, default=1, help="risk_off 时最多持仓数上限（默认 1）")
    p_run.add_argument(
        "--fund-flow-check",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="透传给 rebalance-user：主力资金复核开关（默认启用；--no-fund-flow-check 关闭）",
    )
    p_run.add_argument(
        "--fund-flow-check-mode",
        choices=["auto", "meta_only"],
        default="auto",
        help="透传给 rebalance-user：主力资金复核模式（auto/meta_only）",
    )
    p_run.add_argument("--fund-flow-block-score", type=float, default=0.45, help="透传给 rebalance-user：主力资金复核阻断阈值（默认 0.45）")
    p_run.add_argument("--fund-flow-warn-score", type=float, default=0.55, help="透传给 rebalance-user：主力资金复核警告阈值（默认 0.55）")
    p_run.add_argument("--scan-stock-universe", default="hs300", help="stock 股票池（默认 hs300；可选 hs300/index:000300/all）")
    p_run.add_argument("--scan-stock-limit", type=int, default=300, help="stock 扫描数量（默认 300；0=不限制）")
    p_run.add_argument("--scan-stock-top-k", type=int, default=30, help="stock 输出 TopK（默认 30）")
    p_run.add_argument("--scan-stock-strategy", default=None, help="stock scan-strategy 的策略 key（默认同 --scan-strategy）")
    p_run.add_argument("--scan-stock-source", choices=["akshare", "tushare", "auto"], default="auto", help="stock 数据源（默认 auto=TuShare失败回退AkShare）")
    p_run.add_argument("--deep-holdings", action=argparse.BooleanOptionalAction, default=False, help="可选：逐持仓跑 analyze 并聚合 report_holdings.md（默认关闭）")
    p_run.add_argument("--rebalance-mode", choices=["add", "rotate"], default="add", help="调仓模式（默认 add=只买不卖）")
    p_run.add_argument(
        "--rebalance-schedule",
        choices=["any_day", "fri_close_mon_open"],
        default=None,
        help="调仓执行窗：any_day=每天都允许输出 rebalance 单；fri_close_mon_open=只有周五收盘后才输出（执行=下周一开盘）。默认读 user_holdings.trade_rules.rebalance_schedule。",
    )
    p_run.add_argument("--cache-ttl-hours", type=float, default=6.0, help="K线缓存 TTL（小时；默认 6）")
    p_run.add_argument("--stock-adjust", default="qfq", help="股票复权方式（qfq/hfq；默认 qfq）")
    p_run.add_argument("--limit-up-pct", type=float, default=0.0, help="涨停阈值（比例；默认 0=不启用；例如 0.10=10%%）")
    p_run.add_argument("--limit-down-pct", type=float, default=0.0, help="跌停阈值（比例；默认 0=不启用；例如 0.10=10%%）")
    p_run.add_argument(
        "--halt-vol-zero",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="把 volume/amount=0 视为停牌/无成交（默认启用；可用 --no-halt-vol-zero 关闭）",
    )
    p_run.add_argument("--vol-target", type=float, default=0.0, help="透传给 rebalance-user：波动率目标（年化；0=关闭；只降不升）")
    p_run.add_argument("--vol-lookback-days", type=int, default=20, help="透传给 rebalance-user：vol-target 回看天数（默认 20）")
    p_run.add_argument("--max-exposure-pct", type=float, default=None, help="透传给 rebalance-user：最大总持仓比例（例如 0.6=60%%；默认随牛熊自动选）")
    p_run.add_argument("--min-trade-notional-yuan", type=float, default=None, help="透传给 rebalance-user：单笔买入最小成交额门槛（元；避免最低佣金磨损）")
    p_run.add_argument("--max-turnover-pct", type=float, default=0.0, help="透传给 rebalance-user：单次调仓 buy 侧最大换手（占 equity；0=不限制）")
    p_run.add_argument("--max-corr", type=float, default=None, help="透传给 rebalance-user：相关性阈值（别名 --diversify-max-corr）")
    p_run.add_argument("--max-per-theme", type=int, default=0, help="透传给 rebalance-user：同主题最多持有 N 个（默认 0=不限制）")
    p_run.add_argument("--max-positions", type=int, default=None, help="透传给 rebalance-user：最多持仓标的数（默认随牛熊自动选）")
    p_run.add_argument("--max-position-pct", type=float, default=None, help="透传给 rebalance-user：单标的最大仓位占比（例如 0.30=30%%）")
    p_run.set_defaults(func=cmd_run)

    p_chat = sub.add_parser("chat", help="自然语言入口：调度 run/analyze（武器库），并自动注入/写入偏好记忆")
    p_chat.add_argument("--text", default="", help="自然语言请求（可选；不传则进入交互/或从 stdin 读取）")
    p_chat.add_argument("--provider", choices=["openai", "gemini"], default="openai", help="LLM provider（默认 openai）")
    p_chat.add_argument("--planner", choices=["auto", "llm", "rule"], default="auto", help="计划生成：auto=有 LLM 就用，否则规则；默认 auto")
    p_chat.add_argument("--dry-run", action="store_true", help="只生成 plan，不执行任何子命令")
    p_chat.add_argument("--out-dir", default="", help="chat 输出目录（默认 outputs/chat_YYYYMMDD_HHMMSS）")
    p_chat.set_defaults(func=cmd_chat)

    p_rec = sub.add_parser("reconcile", help="对账闭环：成交->持仓/现金回写 + 审计台账（默认 dry-run；研究用途）")
    p_rec.add_argument("--fills", required=True, help="真实成交明细文件（csv/json/jsonl）")
    p_rec.add_argument("--fills-format", default=None, choices=["csv", "json", "jsonl"], help="强制指定 fills 格式（可选）")
    p_rec.add_argument("--encoding", default=None, help="CSV 编码（可选）")
    p_rec.add_argument("--out-dir", default=None, help="输出目录（默认 outputs/reconcile_YYYYMMDD；同日重复会自动加后缀）")
    p_rec.add_argument("--holdings-path", default=str(Path("data") / "user_holdings.json"), help="持仓快照路径（默认 data/user_holdings.json）")
    p_rec.add_argument("--orders", default=None, help="可选：run 的 orders_next_open.json（用于核对预期 vs 实际）")
    p_rec.add_argument("--ledger-path", default=str(Path("data") / "ledger_trades.jsonl"), help="审计台账路径（jsonl；默认 data/ledger_trades.jsonl）")
    p_rec.add_argument("--apply", action="store_true", help="真正写回 holdings + 追加 ledger（默认只 dry-run）")
    p_rec.set_defaults(func=cmd_reconcile)

    p_scan_stock = sub.add_parser("scan-stock", help="扫描全A个股，输出“当前买入信号 + 历史胜率/磨损”（研究用途）")
    p_scan_stock.add_argument("--freq", choices=["daily", "weekly"], default="weekly", help="扫描频率（默认 weekly）")
    p_scan_stock.add_argument("--window", type=int, default=500, help="每个标的取最近 N 根K线（默认 500；周线≈10年）")
    p_scan_stock.add_argument("--start-date", default="20100101", help="开始日期（默认 20100101；可选）")
    p_scan_stock.add_argument("--end-date", default=None, help="结束日期（可选）")
    p_scan_stock.add_argument("--adjust", default=None, help="复权方式（qfq/hfq/空，可选；默认 qfq）")
    p_scan_stock.add_argument(
        "--regime-index",
        default="sh000300",
        help="大盘指数代码（用于输出 market_regime；默认 sh000300=沪深300；支持逗号分隔多指数；可用 ';' 显式指定 canary：sh000300,sz399006;sh000852；填 off 关闭）",
    )
    p_scan_stock.add_argument(
        "--regime-canary",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="多指数 regime 的 canary 降级开关（默认启用；--no-regime-canary 更灵敏）",
    )
    p_scan_stock.add_argument(
        "--daily-filter",
        choices=["none", "ma20", "macd"],
        default="macd",
        help="日线辅助过滤（默认 macd）",
    )
    p_scan_stock.add_argument(
        "--base-filters",
        default="trend_template",
        help="基础环境过滤器，逗号分隔（默认 trend_template；填 none 关闭）",
    )
    p_scan_stock.add_argument("--tt-near-high", type=float, default=0.25, help="趋势模板：距离52周高点最大回撤比例（默认 0.25）")
    p_scan_stock.add_argument("--tt-above-low", type=float, default=0.30, help="趋势模板：高于52周低点最小涨幅比例（默认 0.30）")
    p_scan_stock.add_argument("--tt-slope-weeks", type=int, default=4, help="趋势模板：MA40 上行判断回看周数（默认 4）")
    p_scan_stock.add_argument("--horizons", default="4,8,12", help="胜率统计持有周数，逗号分隔（默认 4,8,12）")
    p_scan_stock.add_argument("--rank-horizon", type=int, default=8, help="榜单排序使用的 horizon（默认 8）")
    p_scan_stock.add_argument("--min-weeks", type=int, default=120, help="周K 少于该值直接跳过（默认 120）")
    p_scan_stock.add_argument("--min-trades", type=int, default=12, help="排序口径 trades 少于该值的直接过滤（默认 12）")
    p_scan_stock.add_argument("--min-price", type=float, default=0.0, help="过滤股价低于该值的标的（默认 0=不过滤）")
    p_scan_stock.add_argument("--max-price", type=float, default=0.0, help="过滤股价高于该值的标的（默认 0=不过滤）")
    p_scan_stock.add_argument(
        "--symbols",
        default=None,
        help="只扫描指定标的：逗号/空格分隔（支持 6 位代码或 sh/sz/bj 前缀，例如：600000,sz000001）",
    )
    p_scan_stock.add_argument(
        "--symbols-file",
        default=None,
        help="只扫描指定标的：从文件读取（每行一个；支持逗号/空格分隔；'#' 开头视为注释）",
    )
    p_scan_stock.add_argument(
        "--min-amount",
        type=float,
        default=0.0,
        help="过滤周线成交额(优先用 amount；否则 close*volume) 小于该值的标的（默认 0=不过滤）",
    )
    p_scan_stock.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="过滤 OpportunityScore(0~1) 低于该值的标的（默认 0=不过滤；不改变原信号口径）",
    )
    p_scan_stock.add_argument("--limit", type=int, default=0, help="只扫描前 N 个（默认 0=全量；按代码排序）")
    p_scan_stock.add_argument("--top-k", type=int, default=50, help="输出 Top K（默认 50）")
    p_scan_stock.add_argument("--workers", type=int, default=8, help="并发抓数/计算线程数（默认 8）")
    p_scan_stock.add_argument("--buy-cost", type=float, default=0.001, help="买入成本（默认 0.001=0.10%%）")
    p_scan_stock.add_argument("--sell-cost", type=float, default=0.002, help="卖出成本（默认 0.002=0.20%%，含印花税的保守估计）")
    p_scan_stock.add_argument("--allow-overlap", action="store_true", help="允许信号样本重叠（默认不允许，避免假高胜率）")
    p_scan_stock.add_argument("--include-st", action="store_true", help="包含 ST/*ST（默认排除）")
    p_scan_stock.add_argument("--exclude-bj", action="store_true", help="排除北交所（默认包含）")
    p_scan_stock.add_argument("--cache-dir", default=None, help="缓存目录（默认 data/cache/stock）")
    p_scan_stock.add_argument("--cache-ttl-hours", type=float, default=24.0, help="缓存有效期（小时，默认 24）")
    p_scan_stock.add_argument("--out-dir", default=None, help="输出目录（默认 outputs/stock_scan_<timestamp>）")
    p_scan_stock.add_argument("--verbose", action="store_true", help="打印扫描进度（可选）")
    p_scan_stock.set_defaults(func=cmd_scan_stock)

    # Phase3：策略迁移到“因子配置”的 scan（输出 signals.json；不替换 scan-etf/scan-stock 口径）
    p_ss = sub.add_parser("scan-strategy", help="按 strategy_configs.yaml 扫描并输出 signals.json（研究用途；默认不影响旧口径）")
    p_ss.add_argument("--asset", choices=["etf", "stock", "index"], required=True, help="资产类型：etf/stock/index")
    p_ss.add_argument("--freq", choices=["daily", "weekly"], default="weekly", help="扫描频率（默认 weekly）")
    p_ss.add_argument("--universe", default=None, help="股票池：hs300 / index:000300 / all（仅 stock 生效；默认 hs300）")
    p_ss.add_argument("--symbol", action="append", default=[], help="额外指定 symbol（可多次指定；index 默认用它；etf/stock 会 append 到 universe）")
    p_ss.add_argument("--include-all-funds", action="store_true", help="ETF：包含场内基金/LOF（默认只扫主流股票ETF）")
    p_ss.add_argument("--include-st", action="store_true", help="stock=all：包含 ST（默认不包含）")
    p_ss.add_argument("--include-bj", action=argparse.BooleanOptionalAction, default=True, help="stock=all：是否包含北交所（默认包含）")
    p_ss.add_argument("--strategy-config", default=str(Path("config") / "strategy_configs.yaml"), help="策略配置文件（默认 config/strategy_configs.yaml）")
    p_ss.add_argument("--strategy", required=True, help="策略 key（例：bbb_weekly / conservative）")
    p_ss.add_argument("--min-score", type=float, default=0.0, help="过滤 composite score(0~1) 低于该值的标的（默认 0=不过滤）")
    p_ss.add_argument("--limit", type=int, default=0, help="只扫描前 N 个（默认 0=全量）")
    p_ss.add_argument("--top-k", type=int, default=30, help="输出 Top K（默认 30）")
    p_ss.add_argument("--window", type=int, default=400, help="每个标的取最近 N 根K线（默认 400）")
    p_ss.add_argument("--out-dir", default=None, help="输出目录（默认 outputs/scan_strategy_<asset>_<strategy>_<ts>）")
    p_ss.add_argument("--workers", type=int, default=8, help="并发线程数（默认 8）")
    p_ss.add_argument("--whitelist", default=None, help="仅扫描白名单 symbol（逗号分隔或文件路径；文件=每行一个；可选）")
    p_ss.add_argument("--blacklist", default=None, help="排除黑名单 symbol（逗号分隔或文件路径；文件=每行一个；可选）")
    p_ss.add_argument("--cache-dir", default=None, help="缓存目录（默认 data/cache/<asset>）")
    p_ss.add_argument("--cache-ttl-hours", type=float, default=24.0, help="缓存 TTL（小时；默认 24）")
    p_ss.add_argument(
        "--source",
        choices=["akshare", "tushare", "auto"],
        default=None,
        help="数据源：akshare/tushare/auto（默认：etf=akshare；stock/index=auto；auto=TuShare失败回退AkShare）",
    )
    p_ss.add_argument("--regime-index", default="sh000300", help="市场 regime 指数（用于 allowed_regimes；默认 sh000300；off=关闭）")
    p_ss.add_argument("--regime-canary", action=argparse.BooleanOptionalAction, default=True, help="regime canary 降级开关（默认启用）")
    p_ss.set_defaults(func=cmd_scan_strategy)

    p_sa = sub.add_parser("strategy-align", help="新旧 signals.json 对齐报告（输出 outputs/strategy_alignment_*）")
    p_sa.add_argument("--base", required=True, help="基准 signals.json（schema_version=1）")
    p_sa.add_argument("--new", required=True, help="新 signals.json（schema_version=1）")
    p_sa.add_argument("--top-k", type=int, default=30, help="TopK overlap 统计的 K（默认 30）")
    p_sa.add_argument("--out-dir", default=None, help="输出目录（默认 outputs/strategy_alignment_<ts>）")
    p_sa.set_defaults(func=cmd_strategy_align)

    p_eval_bbb = sub.add_parser("eval-bbb", help="BBB 稳健性评估（walk-forward + 参数敏感性；研究用途）")
    p_eval_bbb.add_argument("--symbol", action="append", default=[], help="ETF 代码（可多次指定；例：--symbol sh510300）")
    p_eval_bbb.add_argument("--input", default=None, help="从 top_bbb.json 读取 symbol 列表")
    p_eval_bbb.add_argument("--limit", type=int, default=0, help="最多评估 N 个标的（默认 0=不限制）")
    p_eval_bbb.add_argument("--start-date", default=None, help="数据起始日期（YYYYMMDD，可选）")
    p_eval_bbb.add_argument("--end-date", default=None, help="数据结束日期（YYYYMMDD，可选）")
    p_eval_bbb.add_argument("--cache-dir", default=None, help="ETF 日线缓存目录（默认 data/cache/etf）")
    p_eval_bbb.add_argument("--cache-ttl-hours", type=float, default=24.0, help="缓存有效期（小时；0=不使用缓存；默认 24）")

    # 成本口径
    p_eval_bbb.add_argument("--capital-yuan", type=float, default=3000.0, help="单笔投入资金（元；用于固定磨损换算；默认 3000）")
    p_eval_bbb.add_argument("--roundtrip-cost-yuan", type=float, default=10.0, help="一进一出总磨损（元；默认 10）")
    p_eval_bbb.add_argument("--buy-cost", type=float, default=None, help="买入比例成本（覆盖 roundtrip 换算）")
    p_eval_bbb.add_argument("--sell-cost", type=float, default=None, help="卖出比例成本（覆盖 roundtrip 换算）")

    # BBB 参数
    p_eval_bbb.add_argument("--bbb-mode", choices=["strict", "pullback", "early"], default="pullback", help="BBB 模式（默认 pullback）")
    p_eval_bbb.add_argument("--bbb-entry-ma", type=int, default=50, help="BBB 入场参考均线周期（周线；默认 50）")
    p_eval_bbb.add_argument("--bbb-dist-ma-max", type=float, default=0.12, help="BBB 允许离入场均线的最大偏离（默认 0.12）")
    p_eval_bbb.add_argument("--bbb-max-above-20w", type=float, default=0.05, help="BBB 允许高于20W上轨的最大比例（默认 0.05）")
    p_eval_bbb.add_argument("--min-weeks", type=int, default=60, help="周K 最少根数（默认 60）")
    p_eval_bbb.add_argument("--horizon-weeks", type=int, default=8, help="统计持有周数（默认 8）")
    p_eval_bbb.add_argument("--score-mode", choices=["win_rate", "annualized"], default="annualized", help="选参/排序口径（默认 annualized）")
    p_eval_bbb.add_argument("--allow-overlap", action="store_true", help="允许样本重叠（默认不允许）")

    # walk-forward
    p_eval_bbb.add_argument("--train-weeks", type=int, default=156, help="训练窗长度（周，默认 156≈3年）")
    p_eval_bbb.add_argument("--test-weeks", type=int, default=26, help="验证窗长度（周，默认 26≈半年）")
    p_eval_bbb.add_argument("--step-weeks", type=int, default=26, help="滚动步长（周，默认 26）")
    p_eval_bbb.add_argument(
        "--include-mode-variants",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="参数扰动是否包含模式变体（strict/pullback/early；默认包含；可用 --no-include-mode-variants 关闭）",
    )

    p_eval_bbb.add_argument("--out-dir", default=None, help="输出目录（默认 outputs/eval_bbb_<timestamp>）")
    p_eval_bbb.add_argument("--verbose", action="store_true", help="打印进度（可选）")
    p_eval_bbb.set_defaults(func=cmd_eval_bbb)

    p_paper = sub.add_parser("paper-sim", help="组合级模拟盘/回测（paper sim；研究用途）")
    p_paper.add_argument(
        "--strategy",
        choices=["bbb_etf", "bbb_stock", "rot_stock_weekly"],
        default="bbb_etf",
        help="策略（默认 bbb_etf）",
    )
    p_paper.add_argument("--signals", default=None, help="signals.json（可选，用 items[].symbol 做 watchlist）")
    p_paper.add_argument("--symbol", action="append", default=[], help="标的代码（可重复传多次；可与 --signals 叠加）")
    p_paper.add_argument(
        "--universe-index",
        default=None,
        help="仅股票：用指数成分股做 watchlist（例：000300 或 sh000300+sh000905；可与 --symbol/--signals 叠加）",
    )
    p_paper.add_argument("--limit", type=int, default=0, help="最多使用前 N 个标的（默认 0=不限制）")
    p_paper.add_argument("--start-date", default=None, help="开始日期（YYYYMMDD 或 YYYY-MM-DD，可选）")
    p_paper.add_argument("--end-date", default=None, help="结束日期（YYYYMMDD 或 YYYY-MM-DD，可选）")
    p_paper.add_argument("--regime-index", default="sh000300", help="牛熊/震荡分段指数（默认 sh000300；off=关闭）")

    p_paper.add_argument("--capital-yuan", type=float, default=100000.0, help="初始资金（默认 100000）")
    p_paper.add_argument("--roundtrip-cost-yuan", type=float, default=10.0, help="来回固定磨损（元，默认 10）")
    p_paper.add_argument("--min-fee-yuan", type=float, default=0.0, help="最低佣金（每边，默认 0）")
    p_paper.add_argument("--buy-cost", type=float, default=0.0, help="买入比例成本（默认 0）")
    p_paper.add_argument("--sell-cost", type=float, default=0.0, help="卖出比例成本（默认 0）")
    p_paper.add_argument(
        "--slippage-mode",
        choices=["none", "fixed", "liquidity"],
        default="none",
        help="滑点/冲击成本模型（默认 none；fixed=固定bps；liquidity=按近20日均成交额估算）",
    )
    p_paper.add_argument("--slippage-bps", type=float, default=0.0, help="fixed/liquidity 模式基础 bps（默认 0）")
    p_paper.add_argument("--slippage-ref-amount-yuan", type=float, default=1e8, help="liquidity 模式参考成交额（元，默认 1e8）")
    p_paper.add_argument("--slippage-bps-min", type=float, default=0.0, help="liquidity 模式最小 bps（默认 0）")
    p_paper.add_argument("--slippage-bps-max", type=float, default=30.0, help="liquidity 模式最大 bps（默认 30）")
    p_paper.add_argument("--slippage-unknown-bps", type=float, default=10.0, help="liquidity 模式成交额缺失时的 bps（默认 10）")
    p_paper.add_argument("--slippage-vol-mult", type=float, default=0.0, help="波动放大系数（0=不按波动放大；默认 0）")
    p_paper.add_argument("--lot-size", type=int, default=100, help="最小交易单位（默认 100）")
    p_paper.add_argument("--max-positions", type=int, default=0, help="最多同时持仓数（默认 0=不限制）")
    p_paper.add_argument("--max-exposure-pct", type=float, default=0.0, help="最大仓位比例（默认 0=不限制；例如 0.6=60%%）")
    p_paper.add_argument(
        "--vol-target",
        type=float,
        default=0.0,
        help="波动率目标（年化；0=关闭；例如 0.15=15%%）。当前实现=用指数波动率对 max_exposure_pct 做“只降不升”的缩放（研究用途）。",
    )
    p_paper.add_argument("--vol-lookback-days", type=int, default=20, help="vol-target 的日线波动率回看天数（默认 20）")
    p_paper.add_argument(
        "--max-turnover-pct",
        type=float,
        default=0.0,
        help="单日“买入侧”最大换手（占 equity；0=不限制；例如 0.10=10%%）。说明：当前只限制 buy，不限制 sell（KISS）。",
    )
    p_paper.add_argument("--max-corr", type=float, default=0.0, help="相关性阈值（|corr|>=该值视为同类，入场会跳过；0=关闭；研究用途）")
    p_paper.add_argument("--max-per-theme", type=int, default=0, help="同主题最多持有 N 个（0=不限制；研究用途）")
    p_paper.add_argument("--limit-up-pct", type=float, default=0.0, help="涨停阈值（0=关闭；例 0.1=10%%；研究用途）")
    p_paper.add_argument("--limit-down-pct", type=float, default=0.0, help="跌停阈值（0=关闭；例 0.1=10%%；研究用途）")
    p_paper.add_argument(
        "--halt-vol-zero",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="volume/amount=0 视为停牌/无成交（默认启用；--no-halt-vol-zero 关闭）",
    )

    p_paper.add_argument("--cache-dir", default=None, help="数据缓存目录（默认 data/cache/<asset>）")
    p_paper.add_argument("--cache-ttl-hours", type=float, default=24.0, help="缓存有效期（小时，默认 24）")
    p_paper.add_argument("--adjust", default="qfq", help="仅股票：复权方式（qfq/hfq；默认 qfq）")

    # BBB 参数（短线参数先走 paper_sim 默认值，别过度设计）
    p_paper.add_argument("--bbb-entry-gap-max", type=float, default=0.015, help="BBB：开盘跳空过滤（默认 0.015=1.5%%）")
    p_paper.add_argument(
        "--bbb-entry-rank-mode",
        choices=["ma20_dist", "factor7"],
        default="ma20_dist",
        help="BBB：同日多信号抢额度/现金时的入场排序（默认 ma20_dist=离MA20(上一日)最近；factor7=7因子加权，不改硬规则）",
    )
    p_paper.add_argument(
        "--bbb-rs-index",
        default="sh000300+sh000905",
        help="BBB factor7 的 RS 基准指数（支持 '+' 合成；默认 sh000300+sh000905；auto=跟随 --regime-index；off=关闭）",
    )
    p_paper.add_argument(
        "--bbb-factor7-weights",
        default="",
        help="BBB 7因子权重：rs=0.35,trend=0.15,vol=0.15,drawdown=0.15,liquidity=0.10,boll=0.05,volume=0.05（留空=默认；会自动归一化）",
    )
    p_paper.add_argument("--bbb-entry-ma", type=int, default=20, help="BBB：entry_ma（默认 20）")
    p_paper.add_argument("--bbb-dist-ma-max", type=float, default=0.12, help="BBB：距离 entry_ma 最大偏离（默认 0.12）")
    p_paper.add_argument("--bbb-max-above-20w", type=float, default=0.05, help="BBB：不追高阈值（默认 0.05）")
    p_paper.add_argument("--bbb-min-weeks", type=int, default=60, help="BBB：周K最少根数（默认 60）")
    p_paper.add_argument("--bbb-min-hold-days", type=int, default=5, help="BBB：最少持有天数（默认 5）")
    p_paper.add_argument("--bbb-cooldown-days", type=int, default=0, help="BBB：冷却天数（默认 0）")

    # 组合级风控（通用；研究用途）
    p_paper.add_argument(
        "--portfolio-dd-stop",
        type=float,
        default=0.0,
        help="组合级最大回撤熔断阈值（0=关闭；例如 0.5=最大回撤 -50%% 触发清仓到现金；研究用途）",
    )
    p_paper.add_argument(
        "--portfolio-dd-cooldown-days",
        type=int,
        default=0,
        help="组合级熔断清仓后的冷却期（天；默认 0；研究用途）",
    )
    p_paper.add_argument(
        "--portfolio-dd-restart-ma-days",
        type=int,
        default=0,
        help="组合级熔断后的重启闸门：冷却期结束后，需指数连续 N 天收盘 > MA20 才允许重新开仓（默认 0=关闭；研究用途）",
    )

    # rot_stock_weekly 参数（研究用途）
    p_paper.add_argument("--rot-rebalance-weeks", type=int, default=1, help="轮动：调仓周期（周；默认 1=每周）")
    p_paper.add_argument("--rot-hold-n", type=int, default=6, help="轮动：持仓只数（默认 6；<=0 则回退到 --max-positions）")
    p_paper.add_argument("--rot-buffer-n", type=int, default=2, help="轮动：缓冲区（原持仓排名<=N+buffer 则尽量续留；默认 2）")
    p_paper.add_argument("--rot-rank-mode", choices=["factor7", "mom63", "mom126"], default="factor7", help="轮动：排序指标（默认 factor7）")
    p_paper.add_argument("--rot-gap-max", type=float, default=0.015, help="轮动：开盘跳空过滤（默认 0.015=1.5%%）")
    p_paper.add_argument("--rot-split-exec-days", type=int, default=1, help="轮动：买入分几天执行（1=一次性；2=两天 50/50；默认 1）")

    p_paper.add_argument(
        "--core",
        default="",
        help="核心beta持仓（用来填满闲置仓位，减少现金拖累；研究用途）。格式：sh510300=0.5,sh510500=0.5 或 510300,510500（等权）；留空=关闭",
    )
    p_paper.add_argument(
        "--core-min-pct",
        type=float,
        default=0.0,
        help="core 最低仓位占比（0~1）。当需要给 BBB 腾预算/现金时，最多只会卖到该下限；默认 0=不限制（研究用途）",
    )
    p_paper.add_argument(
        "--min-trade-notional-yuan",
        type=float,
        default=0.0,
        help="最小下单金额（元）。买入(含 core fill)低于该金额直接跳过，避免 5 元起步手续费把收益磨没；例如 2000；默认 0=关闭（研究用途）",
    )

    p_paper.add_argument("--out-dir", default=None, help="输出目录（默认 outputs/paper_sim_<timestamp>）")
    p_paper.set_defaults(func=cmd_paper_sim)

    p_race = sub.add_parser("race", help="经典策略赛马（按牛熊/震荡分段）")
    p_race.add_argument("--asset", default="etf", help="资产类型：etf/index/stock（默认 etf）")
    p_race.add_argument("--universe", default="", help="标的集合：etf=全ETF；etf_all=全场内基金；空=手动 --symbol/--input")
    p_race.add_argument("--symbol", action="append", default=[], help="标的代码（可重复传多次）")
    p_race.add_argument("--input", default=None, help="输入文件（top_*.json，读取 items[].symbol）")
    p_race.add_argument("--limit", type=int, default=0, help="最多处理 N 个 symbol（默认 0=不限制）")
    p_race.add_argument("--strategies", default="", help="策略列表（逗号分隔；默认跑内置全套）")
    p_race.add_argument("--include-buyhold", action="store_true", help="把 buyhold 也参与“最佳策略”评选（默认不参与）")
    p_race.add_argument("--start-date", default=None, help="起始日期 YYYYMMDD/YY-MM-DD（可选）")
    p_race.add_argument("--end-date", default=None, help="结束日期 YYYYMMDD/YY-MM-DD（可选）")
    p_race.add_argument("--regime-index", default="sh000300", help="牛熊判定指数（默认 sh000300；off=关闭）")
    p_race.add_argument("--top-n", type=int, default=10, help="leaderboards 每个分段输出 TopN（默认 10）")
    p_race.add_argument("--min-weeks-total", type=int, default=104, help="总历史周数门槛（默认 104≈2年；0=不过滤）")
    p_race.add_argument("--min-regime-weeks", type=int, default=26, help="单分段最少周数门槛（默认 26≈半年；0=不过滤）")
    p_race.add_argument("--min-trades", type=int, default=3, help="策略最少交易次数门槛（默认 3；0=不过滤）")
    p_race.add_argument("--min-amount-avg20", type=float, default=0.0, help="过滤近20日均成交额小于该值的标的（默认 0=不过滤）")
    p_race.add_argument("--cache-dir", default=None, help="数据缓存目录（默认 data/cache/<asset>）")
    p_race.add_argument("--cache-ttl-hours", type=float, default=24.0, help="缓存有效期（小时，默认 24）")
    p_race.add_argument("--workers", type=int, default=8, help="并发线程数（默认 8）")
    p_race.add_argument(
        "--analysis-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="启用派生结果缓存（加速重复赛马；默认启用；可用 --no-analysis-cache 关闭）",
    )
    p_race.add_argument("--analysis-cache-dir", default=None, help="派生结果缓存目录（默认 data/cache/analysis/race）")
    p_race.add_argument("--capital-yuan", type=float, default=3000.0, help="单笔资金（元，默认 3000）")
    p_race.add_argument("--roundtrip-cost-yuan", type=float, default=10.0, help="来回固定磨损（元，默认 10）")
    p_race.add_argument("--buy-cost", type=float, default=None, help="买入比例成本（覆盖 capital+磨损换算）")
    p_race.add_argument("--sell-cost", type=float, default=None, help="卖出比例成本（覆盖 capital+磨损换算）")
    p_race.add_argument("--out-dir", default=None, help="输出目录（默认 outputs/race_<timestamp>）")
    p_race.add_argument("--verbose", action="store_true", help="打印进度（可选）")
    p_race.set_defaults(func=cmd_race_strategies)

    p_replay = sub.add_parser("replay", help="一键复跑：从 run_config/run_meta/report 复现一次运行")
    p_replay.add_argument("--from", dest="src", required=True, help="run_config.json/run_meta.json/report.json 或包含这些文件的目录")
    p_replay.add_argument("--out-dir", default=None, help="输出目录（默认 outputs/replay_<cmd>_<timestamp>）")
    p_replay.add_argument("--print-only", action="store_true", help="只打印复跑命令，不执行")
    p_replay.set_defaults(func=cmd_replay)

    p_clean = sub.add_parser("clean-outputs", help="清理 outputs 历史产物（默认 dry-run）")
    p_clean.add_argument("--path", default="outputs", help="输出目录（默认 outputs）")
    p_clean.add_argument("--keep-days", type=float, default=7.0, help="保留最近 N 天（默认 7）")
    p_clean.add_argument("--keep-last", type=int, default=20, help="额外保留最近 N 个条目（默认 20）")
    p_clean.add_argument("--include-logs", action="store_true", help="连 outputs 下的 .log 文件也一起清（默认不动日志）")
    p_clean.add_argument("--apply", action="store_true", help="真正执行删除（默认只 dry-run）")
    p_clean.set_defaults(func=cmd_clean_outputs)

    p_dd = sub.add_parser("data-doctor", help="数据质量/偏差/可复现性体检（默认抽样检查）")
    p_dd.add_argument("--cache-dir", default=str(Path("data") / "cache"), help="数据缓存根目录（默认 data/cache）")
    p_dd.add_argument("--outputs-dir", default="outputs", help="outputs 目录（默认 outputs）")
    p_dd.add_argument(
        "--include-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="检查 data/cache（默认启用；可用 --no-include-cache 关闭）",
    )
    p_dd.add_argument(
        "--include-outputs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="检查 outputs（默认启用；可用 --no-include-outputs 关闭）",
    )
    p_dd.add_argument("--cache-recent-days", type=int, default=3, help="只检查最近 N 天修改过的 cache 文件（默认 3；0=不限）")
    p_dd.add_argument("--cache-max-files", type=int, default=200, help="最多检查多少个 cache CSV（默认 200；0=不查）")
    p_dd.add_argument("--outputs-max-dirs", type=int, default=30, help="最多检查 outputs 下多少个目录（默认 30；0=不查）")
    p_dd.add_argument(
        "--fail-on",
        choices=["never", "warn", "error"],
        default="never",
        help="返回码策略：never=永不失败；warn=有 warning 返回 1；error=有 error 返回 2（默认 never）",
    )
    p_dd.add_argument("--out", default=None, help="输出 JSON 路径（默认打印到 stdout）")
    p_dd.set_defaults(func=cmd_data_doctor)

    p_vp = sub.add_parser("verify-prices", help="校验涨幅口径/复权差异")
    p_vp.add_argument("--asset", choices=["etf", "index", "stock"], required=True, help="数据类型：etf / index / stock")
    p_vp.add_argument("--symbol", required=True, help="代码或名称（建议带前缀如 sh600426）")
    p_vp.add_argument(
        "--source",
        choices=["auto", "akshare", "tushare"],
        default="auto",
        help="数据源：auto(优先TuShare)/akshare/tushare",
    )
    p_vp.add_argument("--basis", default="raw", help="主口径：raw/qfq/hfq（默认 raw）")
    p_vp.add_argument("--compare", default=None, help="对比口径：raw/qfq/hfq（可选）")
    p_vp.add_argument("--threshold", type=float, default=0.015, help="对比阈值（默认 0.015=1.5%）")
    p_vp.add_argument("--cache-dir", default=None, help="缓存目录（默认 data/cache/<asset>）")
    p_vp.add_argument("--cache-ttl-hours", type=float, default=24.0, help="缓存 TTL 小时（默认 24）")
    p_vp.set_defaults(func=cmd_verify_prices)

    p_fr = sub.add_parser("factor-research", help="Phase1：因子研究最小闭环（IC/IR/衰减/成本/交易性）")
    p_fr.add_argument("--asset", choices=["etf", "stock", "index"], required=True, help="研究对象：etf/stock/index")
    p_fr.add_argument("--freq", choices=["daily", "weekly"], default="daily", help="研究频率（默认 daily）")
    p_fr.add_argument("--universe", default="", help="股票池：stock 默认 hs300；可选 hs300 / index:000300 / all")
    p_fr.add_argument("--symbol", default=None, help="仅 index：指数代码（默认 sh000300）")
    p_fr.add_argument("--limit", type=int, default=200, help="universe 最多取 N 个（默认 200）")
    p_fr.add_argument("--include-all-funds", action="store_true", help="仅 etf：把 LOF/场内基金也纳入（默认只扫股票ETF）")
    p_fr.add_argument("--include-st", action="store_true", help="仅 stock:all：是否包含 ST（默认不包含）")
    p_fr.add_argument("--include-bj", action="store_true", help="仅 stock:all：是否包含北交所（默认包含）")
    p_fr.add_argument("--start-date", default=None, help="起始日期 YYYYMMDD/YY-MM-DD（可选）")
    p_fr.add_argument("--as-of", default=None, help="截止日期 YYYYMMDD/YY-MM-DD（可选；不传则取共同最小 last_date）")
    p_fr.add_argument("--horizons", default="1,5,10,20", help="forward horizons（逗号分隔，默认 1,5,10,20）")
    p_fr.add_argument("--cache-dir", default=None, help="数据缓存目录（默认 data/cache/<asset>）")
    p_fr.add_argument("--cache-ttl-hours", type=float, default=24.0, help="缓存有效期（小时，默认 24）")
    p_fr.add_argument("--out-dir", default=None, help="输出目录（默认 outputs/factor_reports_<asset>_<freq>_<ts>）")
    p_fr.add_argument("--limit-up-pct", type=float, default=None, help="涨停阈值（默认 stock=0.095；ETF/指数=0）")
    p_fr.add_argument("--limit-down-pct", type=float, default=None, help="跌停阈值（默认 stock=0.095；ETF/指数=0）")
    p_fr.add_argument("--min-fee-yuan", type=float, default=5.0, help="最低佣金（每边，默认 5 元）")
    p_fr.add_argument("--slippage-bps", type=float, default=10.0, help="滑点（每边，bps，默认 10）")
    p_fr.add_argument("--notional-yuan", type=float, default=2000.0, help="单笔名义金额（用于摊薄最低佣金；默认 2000）")
    p_fr.add_argument("--include-tushare-micro", action="store_true", help="纳入 TuShare 个股 microstructure（moneyflow 大单/超大单 proxy；仅 stock 有意义）")
    p_fr.add_argument("--include-tushare-macro", action="store_true", help="输出 TuShare 宏观温度计（ERP proxy + HSGT north/south；会生成 factor_research_macro.json）")
    p_fr.add_argument("--context-index", default="sh000300", help="宏观温度计的 context 指数（默认 sh000300；支持 000300 / 000300.SH / sh000300）")
    p_fr.add_argument("--max-tushare-symbols", type=int, default=80, help="microstructure 最多拉多少个 symbol（默认 80；0=不限）")
    p_fr.set_defaults(func=cmd_factor_research)

    p_dw = sub.add_parser("dynamic-weights", help="Phase4：动态权重研究闭环（regime-aware；输出 OOS walk-forward 报告）")
    p_dw.add_argument("--asset", choices=["etf", "stock", "index"], required=True, help="研究对象：etf/stock/index")
    p_dw.add_argument("--freq", choices=["weekly"], default="weekly", help="研究频率（当前仅 weekly）")
    p_dw.add_argument("--universe", default="", help="股票池：stock 默认 hs300；可选 hs300 / index:000300 / all")
    p_dw.add_argument("--symbol", default=None, help="仅 index：指数代码（默认 sh000300）")
    p_dw.add_argument("--limit", type=int, default=200, help="universe 最多取 N 个（默认 200）")
    p_dw.add_argument("--include-all-funds", action="store_true", help="仅 etf：把 LOF/场内基金也纳入（默认只扫股票ETF）")
    p_dw.add_argument("--include-st", action="store_true", help="仅 stock:all：是否包含 ST（默认不包含）")
    p_dw.add_argument("--include-bj", action="store_true", help="仅 stock:all：是否包含北交所（默认包含）")
    p_dw.add_argument("--start-date", default=None, help="起始日期 YYYYMMDD/YY-MM-DD（可选）")
    p_dw.add_argument("--as-of", default=None, help="截止日期 YYYYMMDD/YY-MM-DD（可选；不传则取共同最小 last_date）")
    p_dw.add_argument("--horizons", default="1,5,10,20", help="forward horizons（逗号分隔，默认 1,5,10,20）")
    p_dw.add_argument("--cache-dir", default=None, help="数据缓存目录（默认 data/cache/<asset>）")
    p_dw.add_argument("--cache-ttl-hours", type=float, default=24.0, help="缓存有效期（小时，默认 24）")
    p_dw.add_argument("--out-dir", default=None, help="输出目录（默认 outputs/walk_forward_<asset>_<ts>）")
    p_dw.add_argument("--limit-up-pct", type=float, default=None, help="涨停阈值（默认 stock=0.095；ETF/指数=0）")
    p_dw.add_argument("--limit-down-pct", type=float, default=None, help="跌停阈值（默认 stock=0.095；ETF/指数=0）")
    p_dw.add_argument("--min-fee-yuan", type=float, default=5.0, help="最低佣金（每边，默认 5 元）")
    p_dw.add_argument("--slippage-bps", type=float, default=10.0, help="滑点（每边，bps，默认 10）")
    p_dw.add_argument("--notional-yuan", type=float, default=2000.0, help="单笔名义金额（用于摊薄最低佣金；默认 2000）")
    p_dw.add_argument("--walk-forward", action=argparse.BooleanOptionalAction, default=True, help="启用 walk-forward（默认启用；--no-walk-forward 关闭）")
    p_dw.add_argument("--train-window", type=int, default=252, help="walk-forward 训练窗口（默认 252）")
    p_dw.add_argument("--test-window", type=int, default=63, help="walk-forward 测试窗口（默认 63）")
    p_dw.add_argument("--step-window", type=int, default=63, help="walk-forward 步长（默认 63）")
    p_dw.add_argument("--min-cross-n", type=int, default=30, help="单日横截面最小样本数门槛（默认 30）")
    p_dw.add_argument("--top-quantile", type=float, default=0.8, help="top20 的 quantile 阈值（默认 0.8=Top20%%）")
    p_dw.add_argument("--context-index", default="sh000300", help="市场 regime 的 context 指数（默认 sh000300；支持 '+' 合成：sh000300+sh000905）")
    p_dw.add_argument("--regime-weights", default=str(Path("config") / "regime_weights.yaml"), help="regime 权重表（默认 config/regime_weights.yaml）")
    p_dw.add_argument("--baseline-regime", default="neutral", help="static baseline regime（默认 neutral）")
    p_dw.set_defaults(func=cmd_dynamic_weights)

    p_sql_init = sub.add_parser("sql-init", help="初始化本地 DuckDB 数据仓库（data/ + outputs/ SQL 化）")
    p_sql_init.add_argument("--db", default=str(Path("data") / "warehouse.duckdb"), help="DuckDB 文件路径（默认 data/warehouse.duckdb）")
    p_sql_init.set_defaults(func=cmd_sql_init)

    p_sql_sync = sub.add_parser("sql-sync", help="刷新 DuckDB 文件目录索引（wh.file_catalog）")
    p_sql_sync.add_argument("--db", default=str(Path("data") / "warehouse.duckdb"), help="DuckDB 文件路径（默认 data/warehouse.duckdb）")
    p_sql_sync.set_defaults(func=cmd_sql_sync)

    p_sql_q = sub.add_parser("sql-query", help="执行 SQL（默认 limit 50；可导出 CSV）")
    p_sql_q.add_argument("--db", default=str(Path("data") / "warehouse.duckdb"), help="DuckDB 文件路径（默认 data/warehouse.duckdb）")
    gq = p_sql_q.add_mutually_exclusive_group(required=True)
    gq.add_argument("--sql", default=None, help="SQL 语句（建议用引号包起来）")
    gq.add_argument("--file", default=None, help="SQL 文件路径")
    p_sql_q.add_argument("--limit", type=int, default=50, help="默认 50；-1=不限制")
    p_sql_q.add_argument("--out", default=None, help="导出 CSV 路径（可选）")
    p_sql_q.set_defaults(func=cmd_sql_query)

    p_mon = sub.add_parser("monitor", help="监控/回顾 outputs 产物（汇总 report.json；研究用途）")
    p_mon.add_argument("--outputs-dir", default="outputs", help="outputs 目录（默认 outputs）")
    p_mon.add_argument("--max-dirs", type=int, default=200, help="最多扫描 outputs 下多少个一级目录（默认 200；0=不限）")
    p_mon.add_argument("--include-cmds", default="", help="只包含哪些 cmd（逗号分隔；例 run,paper-sim；默认全量）")
    p_mon.add_argument("--exclude-cmds", default="", help="排除哪些 cmd（逗号分隔；默认空）")
    p_mon.add_argument("--out-dir", default=None, help="输出目录（默认 outputs/monitor_YYYYMMDD；同日重复自动加后缀）")
    p_mon.set_defaults(func=cmd_monitor)

    p_brief = sub.add_parser("daily-brief", help="生成每日量化简报（候选池/模拟持仓/执行草案；研究用途）")
    p_brief.add_argument("--run-dir", required=True, help="run 输出目录（例如 outputs/run_YYYYMMDD_close）")
    p_brief.add_argument("--out", default=None, help="输出 Markdown 路径（默认 outputs/agents/daily_brief.md）")
    p_brief.add_argument("--max-candidates", type=int, default=6, help="候选池数量（默认 6）")
    p_brief.add_argument("--max-portfolio", type=int, default=3, help="模拟持仓最多标的（默认 3）")
    p_brief.add_argument("--max-warnings", type=int, default=10, help="最多输出警告条数（默认 10）")
    p_brief.set_defaults(func=cmd_daily_brief)

    p_merge = sub.add_parser("signals-merge", help="合并多份 signals.json（多策略聚合）")
    p_merge.add_argument("--in", dest="inputs", action="append", required=True, help="signals.json 路径（可重复传多次）")
    p_merge.add_argument("--out", default=None, help="输出 JSON 路径（默认打印到 stdout）")
    p_merge.add_argument("--weights", default="", help="策略权重：bbb_weekly=1,trend_pullback_weekly=0.8（可选）")
    p_merge.add_argument("--priority", default="", help="优先级列表（conflict=priority 时生效）：trend_pullback_weekly,bbb_weekly,...（可选）")
    p_merge.add_argument("--conflict", choices=["risk_first", "priority", "vote"], default="risk_first", help="冲突裁决模式（默认 risk_first）")
    p_merge.add_argument("--top-k", type=int, default=0, help="只输出前 K 个（默认 0=全量）")
    p_merge.set_defaults(func=cmd_signals_merge)

    # skill：把“策略/研究/回测”的 SOP 变成可调用的命令（也可被 chat 自动触发）。
    p_skill = sub.add_parser("skill", help="运行 skills：strategy/research/backtest（研究用途）")
    skill_sub = p_skill.add_subparsers(dest="skill_cmd", required=True)

    p_skill_strategy = skill_sub.add_parser("strategy", help="基于 outputs/run_* 生成策略执行清单（LLM 可选）")
    p_skill_strategy.add_argument("--run-dir", default="", help="输入 run 目录（默认自动选最新的 outputs/run_* 或 chat_run_*）")
    p_skill_strategy.add_argument("--out", default=str(Path("outputs") / "agents" / "strategy_action.md"), help="输出 markdown 路径")
    p_skill_strategy.add_argument("--provider", choices=["openai", "gemini"], default="openai", help="LLM provider（默认 openai）")
    p_skill_strategy.add_argument("--no-llm", action="store_true", help="禁用 LLM（降级为规则模板输出）")
    p_skill_strategy.set_defaults(func=cmd_skill)

    p_skill_research = skill_sub.add_parser("research", help="舆情/新闻线索：抓取新闻 + 输出 research.md（LLM 可选）")
    p_skill_research.add_argument("--run-dir", default="", help="用于自动提取查询词的 run 目录（可选）")
    p_skill_research.add_argument("--queries", default="", help="逗号分隔查询词/代码（可选；为空则从 run_dir/持仓自动提取）")
    p_skill_research.add_argument("--pages", type=int, default=2, help="抓取页数（默认 2）")
    p_skill_research.add_argument("--page-size", type=int, default=10, help="每页条数（默认 10；上限 50）")
    p_skill_research.add_argument("--out-dir", default=str(Path("outputs") / "agents"), help="输出目录（默认 outputs/agents）")
    p_skill_research.add_argument("--provider", choices=["openai", "gemini"], default="openai", help="LLM provider（默认 openai）")
    p_skill_research.add_argument("--no-llm", action="store_true", help="禁用 LLM（只输出线索汇总/规则摘要）")
    p_skill_research.set_defaults(func=cmd_skill)

    p_skill_backtest = skill_sub.add_parser("backtest", help="回测：出场规则对比（t 日信号 -> t+1 开盘成交）")
    p_skill_backtest.add_argument("--asset", default="etf", choices=["etf", "stock", "index", "crypto"], help="资产类型（默认 etf）")
    p_skill_backtest.add_argument("--symbols", required=True, help="逗号分隔，如 sh518880,sh159937")
    p_skill_backtest.add_argument("--source", default="akshare", help="akshare/tushare/auto（默认 akshare）")
    p_skill_backtest.add_argument("--start", default="", help="YYYY-MM-DD（可选）")
    p_skill_backtest.add_argument("--end", default="", help="YYYY-MM-DD（可选）")
    p_skill_backtest.add_argument("--cache-ttl-hours", type=float, default=24.0, help="缓存 TTL 小时（默认 24）")
    p_skill_backtest.add_argument("--fee-bps", type=float, default=10.0, help="单边手续费 bps（默认 10）")
    p_skill_backtest.add_argument("--slippage-bps", type=float, default=5.0, help="单边滑点 bps（默认 5）")
    p_skill_backtest.add_argument(
        "--out",
        default=str(Path("outputs") / "agents" / "backtest_report.md"),
        help="输出 markdown 路径（默认 outputs/agents/backtest_report.md）",
    )
    p_skill_backtest.set_defaults(func=cmd_skill)

    # memory：对话/偏好/复盘的持久化（data/ 里，默认 gitignore）
    p_mem = sub.add_parser("memory", help="记忆库：偏好/复盘/对话持久化（data/memory/）")
    mem_sub = p_mem.add_subparsers(dest="memory_cmd", required=True)

    p_mem_status = mem_sub.add_parser("status", help="显示记忆库状态（路径/文件数）")
    p_mem_status.add_argument("--json", action="store_true", help="输出 JSON（便于脚本调用）")
    p_mem_status.set_defaults(func=cmd_memory)

    p_mem_rem = mem_sub.add_parser("remember", help="写入记忆：--text 追加；--set k=v 更新结构化偏好")
    p_mem_rem.add_argument("--text", default="", help="要记住的内容（会自动加时间戳）")
    p_mem_rem.add_argument("--daily", action="store_true", help="写入每日记忆（默认写入长期 MEMORY.md）")
    p_mem_rem.add_argument("--title", default="", help="每日记忆小标题（仅 --daily 生效，可选）")
    p_mem_rem.add_argument(
        "--set",
        action="append",
        default=[],
        help="结构化偏好：k=v（k 支持点号路径；v 尝试按 JSON 解析，可重复传多次）",
    )
    p_mem_rem.set_defaults(func=cmd_memory)

    p_mem_search = mem_sub.add_parser("search", help="搜索记忆（keyword/vector/hybrid）")
    p_mem_search.add_argument("query", help="查询文本")
    p_mem_search.add_argument("--mode", choices=["keyword", "vector", "hybrid"], default="keyword", help="检索模式（默认 keyword）")
    p_mem_search.add_argument("--max-results", type=int, default=20, help="最多返回条数（默认 20）")
    p_mem_search.add_argument("--min-score", type=float, default=0.15, help="向量/混合检索最小分数阈值（默认 0.15）")
    p_mem_search.add_argument("--reindex", action="store_true", help="强制重建向量索引（vector/hybrid 生效）")
    p_mem_search.add_argument("--context-lines", type=int, default=2, help="上下文行数（keyword 生效，默认 2）")
    p_mem_search.set_defaults(func=cmd_memory)

    p_mem_index = mem_sub.add_parser("index", help="构建/更新向量索引（需要 EMBEDDINGS_* 配置）")
    p_mem_index.add_argument("--force", action="store_true", help="强制全量重建（默认增量）")
    p_mem_index.add_argument("--verbose", action="store_true", help="输出索引统计信息")
    p_mem_index.set_defaults(func=cmd_memory)

    p_mem_export = mem_sub.add_parser("export-prompt", help="导出一段可直接喂给 LLM 的记忆上下文")
    p_mem_export.add_argument("--max-chars", type=int, default=6000, help="最多字符数（默认 6000）")
    p_mem_export.add_argument("--daily-days", type=int, default=2, help="包含最近 N 天每日记忆（默认 2）")
    p_mem_export.add_argument("--no-long-term", action="store_true", help="不包含 MEMORY.md")
    p_mem_export.add_argument("--no-profile", action="store_true", help="不包含 user_profile.json")
    p_mem_export.set_defaults(func=cmd_memory)

    p_mem_sync = mem_sub.add_parser("sync", help="从 data/user_holdings.json 同步 trade_rules → user_profile.json")
    p_mem_sync.add_argument("--holdings-path", default=str(Path("data") / "user_holdings.json"), help="持仓快照路径")
    p_mem_sync.set_defaults(func=cmd_memory)

    p_mem_arch = mem_sub.add_parser("archive", help="归档 daily 记忆（rollup 到 archive/；默认 dry-run）")
    p_mem_arch.add_argument("--keep-days", type=int, default=7, help="保留最近 N 天 daily 原文（默认 7）")
    p_mem_arch.add_argument("--group", choices=["month", "week"], default="month", help="归档分组（默认 month）")
    p_mem_arch.add_argument("--apply", action="store_true", help="真执行（默认不动文件，仅输出 plan JSON）")
    p_mem_arch.set_defaults(func=cmd_memory)

    return parser
def main(argv: list[str] | None = None) -> int:
    if os.name == "nt":
        os.environ.setdefault("PYTHONUTF8", "1")
    # 艹，沙箱/容器里写不了 ~/.cache 的话就全给我塞回仓库里，别让 matplotlib/别的库瞎拉屎。
    try:
        root = Path(__file__).resolve().parents[1]
        cache_root = root / "data" / "cache"
        os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "_xdg"))
        os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "_mpl"))
        (cache_root / "_xdg").mkdir(parents=True, exist_ok=True)
        (cache_root / "_mpl").mkdir(parents=True, exist_ok=True)
    except (AttributeError):  # noqa: BLE001
        pass
    parser = build_parser()
    args = parser.parse_args(argv)
    # 让每个子命令都能拿到“实际 argv”（用于 run_meta 可复现）
    try:
        setattr(args, "_argv", list(argv) if argv is not None else list(sys.argv[1:]))
    except (AttributeError):  # noqa: BLE001
        pass
    return int(args.func(args))
