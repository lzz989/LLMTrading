from __future__ import annotations

import argparse
from pathlib import Path


def cmd_sql_init(args: argparse.Namespace) -> int:
    """
    初始化本地 DuckDB 数据仓库（把 data/ + outputs/ 里的结构化数据“SQL 化”）。
    """
    from ..warehouse import WarehouseError, sql_init

    db = str(getattr(args, "db", "") or "").strip() or None
    try:
        p = sql_init(db_path=db)
    except WarehouseError as exc:
        raise SystemExit(str(exc)) from exc

    print(str(p.resolve()))
    return 0


def cmd_sql_sync(args: argparse.Namespace) -> int:
    """
    刷新 DuckDB 的文件目录索引（wh.file_catalog）。
    """
    from ..warehouse import WarehouseError, sql_sync

    db = str(getattr(args, "db", "") or "").strip() or None
    try:
        p = sql_sync(db_path=db)
    except WarehouseError as exc:
        raise SystemExit(str(exc)) from exc

    print(str(p.resolve()))
    return 0


def cmd_sql_query(args: argparse.Namespace) -> int:
    """
    执行一段 SQL（默认 limit 50，防止你一条语句把终端喷成瀑布）。
    """
    from ..warehouse import WarehouseError, sql_query

    db = str(getattr(args, "db", "") or "").strip() or None
    out = str(getattr(args, "out", "") or "").strip() or None
    limit = int(getattr(args, "limit", 50) or 50)

    sql_raw = str(getattr(args, "sql", "") or "").strip()
    sql_file = str(getattr(args, "file", "") or "").strip()
    if (not sql_raw) and sql_file:
        p = Path(sql_file)
        if not p.exists():
            raise SystemExit(f"找不到 SQL 文件：{p}")
        sql_raw = p.read_text(encoding="utf-8")
    if not sql_raw:
        raise SystemExit("缺少 --sql 或 --file")

    try:
        sql_query(sql=sql_raw, db_path=db, limit=limit, out=out)
    except WarehouseError as exc:
        raise SystemExit(str(exc)) from exc

    return 0

