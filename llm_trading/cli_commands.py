from __future__ import annotations

import argparse


# 说明：
# - cli.py 只负责 argparse；真正的命令实现在 llm_trading/commands/*。
# - 这里用“延迟导入”的轻量 wrapper，避免 `python -m llm_trading --help` 也把一堆重依赖/大文件导入。


def cmd_fetch(args: argparse.Namespace) -> int:
    from .commands.fetch import cmd_fetch as _cmd

    return _cmd(args)


def cmd_analyze(args: argparse.Namespace) -> int:
    from .commands.analyze import cmd_analyze as _cmd

    return _cmd(args)


def cmd_scan_etf(args: argparse.Namespace) -> int:
    from .commands.scan import cmd_scan_etf as _cmd

    return _cmd(args)


def cmd_plan_etf(args: argparse.Namespace) -> int:
    from .commands.portfolio import cmd_plan_etf as _cmd

    return _cmd(args)


def cmd_holdings_etf(args: argparse.Namespace) -> int:
    from .commands.portfolio import cmd_holdings_etf as _cmd

    return _cmd(args)


def cmd_holdings_user(args: argparse.Namespace) -> int:
    from .commands.portfolio import cmd_holdings_user as _cmd

    return _cmd(args)


def cmd_rebalance_user(args: argparse.Namespace) -> int:
    from .commands.portfolio import cmd_rebalance_user as _cmd

    return _cmd(args)


def cmd_scan_stock(args: argparse.Namespace) -> int:
    from .commands.scan import cmd_scan_stock as _cmd

    return _cmd(args)

def cmd_eval_bbb(args: argparse.Namespace) -> int:
    from .commands.eval import cmd_eval_bbb as _cmd

    return _cmd(args)


def cmd_paper_sim(args: argparse.Namespace) -> int:
    from .commands.paper_sim import cmd_paper_sim as _cmd

    return _cmd(args)


def cmd_run(args: argparse.Namespace) -> int:
    from .commands.run import cmd_run as _cmd

    return _cmd(args)


def cmd_national_team(args: argparse.Namespace) -> int:
    from .commands.national_team import cmd_national_team as _cmd

    return _cmd(args)


def cmd_national_team_backtest(args: argparse.Namespace) -> int:
    from .commands.national_team import cmd_national_team_backtest as _cmd

    return _cmd(args)


def cmd_reconcile(args: argparse.Namespace) -> int:
    from .commands.reconcile import cmd_reconcile as _cmd

    return _cmd(args)


def cmd_race_strategies(args: argparse.Namespace) -> int:
    from .commands.race import cmd_race_strategies as _cmd

    return _cmd(args)


def cmd_replay(args: argparse.Namespace) -> int:
    from .commands.replay import cmd_replay as _cmd

    return _cmd(args)


def cmd_clean_outputs(args: argparse.Namespace) -> int:
    from .commands.ops import cmd_clean_outputs as _cmd

    return _cmd(args)


def cmd_data_doctor(args: argparse.Namespace) -> int:
    from .commands.ops import cmd_data_doctor as _cmd

    return _cmd(args)


def cmd_verify_prices(args: argparse.Namespace) -> int:
    from .commands.verify_prices import cmd_verify_prices as _cmd

    return _cmd(args)


def cmd_factor_research(args: argparse.Namespace) -> int:
    from .commands.factor_research import cmd_factor_research as _cmd

    return _cmd(args)

def cmd_dynamic_weights(args: argparse.Namespace) -> int:
    from .commands.dynamic_weights import cmd_dynamic_weights as _cmd

    return _cmd(args)


def cmd_sql_init(args: argparse.Namespace) -> int:
    from .commands.sql import cmd_sql_init as _cmd

    return _cmd(args)


def cmd_sql_sync(args: argparse.Namespace) -> int:
    from .commands.sql import cmd_sql_sync as _cmd

    return _cmd(args)


def cmd_sql_query(args: argparse.Namespace) -> int:
    from .commands.sql import cmd_sql_query as _cmd

    return _cmd(args)


def cmd_monitor(args: argparse.Namespace) -> int:
    from .commands.ops import cmd_monitor as _cmd

    return _cmd(args)


def cmd_signals_merge(args: argparse.Namespace) -> int:
    from .commands.signals_merge import cmd_signals_merge as _cmd

    return _cmd(args)


def cmd_scan_strategy(args: argparse.Namespace) -> int:
    from .commands.strategy import cmd_scan_strategy as _cmd

    return _cmd(args)


def cmd_strategy_align(args: argparse.Namespace) -> int:
    from .commands.strategy import cmd_strategy_align as _cmd

    return _cmd(args)


def cmd_memory(args: argparse.Namespace) -> int:
    from .commands.memory import cmd_memory as _cmd

    return _cmd(args)

def cmd_skill(args: argparse.Namespace) -> int:
    from .commands.skill import cmd_skill as _cmd

    return _cmd(args)


def cmd_chat(args: argparse.Namespace) -> int:
    from .commands.chat import cmd_chat as _cmd

    return _cmd(args)


def cmd_daily_brief(args: argparse.Namespace) -> int:
    from .commands.brief import cmd_daily_brief as _cmd

    return _cmd(args)


def cmd_commodity_chain(args: argparse.Namespace) -> int:
    from .commands.commodity_chain import cmd_commodity_chain as _cmd

    return _cmd(args)
