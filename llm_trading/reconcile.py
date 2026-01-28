from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Literal

from .akshare_source import resolve_symbol

Side = Literal["buy", "sell"]


@dataclass(frozen=True)
class TradeFill:
    """
    真实成交（fill）标准化结构：用于“对账闭环/审计台账”。

    说明：
    - 这不是“下单回执”，是“最终成交”（一个订单可能拆成多笔 fill）。
    - trade_id 最好是券商/接口给的唯一成交编号；否则只能退化用 hash，当你反复导出时可能会重复/冲突。
    """

    trade_id: str
    dt: datetime | None
    asset: str
    symbol: str
    side: Side
    price: float
    shares: int
    fee: float
    tax: float
    order_id: str | None = None
    raw: dict[str, Any] | None = None

    @property
    def amount(self) -> float:
        return float(self.price) * float(self.shares)


def _first_match(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    normalized = {c.strip(): c for c in columns}
    lower_map = {c.strip().lower(): c for c in columns}
    for cand in candidates:
        if cand in normalized:
            return normalized[cand]
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def _guess_asset_for_symbol(symbol: str) -> str:
    """
    粗暴推断（研究/个人工具够用了）：
    - 6/0/3 开头基本是股票；5/1/15/16/159 之类大概率是 ETF/基金。
    - 兜底：etf（别把默认搞反）
    """
    s = str(symbol or "").strip().lower()
    if s.startswith(("sh", "sz", "bj")):
        s = s[2:]
    if not s.isdigit():
        return "etf"
    if len(s) != 6:
        return "etf"
    if s.startswith(("6", "0", "3")):
        return "stock"
    return "etf"


def _norm_side(x: Any) -> Side | None:
    s = str(x or "").strip().lower()
    if not s:
        return None
    if s in {"b", "buy", "long", "open_long"}:
        return "buy"
    if s in {"s", "sell", "short", "close_long"}:
        return "sell"
    # 中文/券商常见
    if "买" in s:
        return "buy"
    if "卖" in s:
        return "sell"
    return None


def _parse_dt(x: Any) -> datetime | None:
    if x is None:
        return None
    if isinstance(x, datetime):
        return x
    s = str(x).strip()
    if not s:
        return None
    # 兼容 "2026-01-15 09:31:02" / "2026/01/15 09:31" / "20260115 093102" 等
    try:
        return datetime.fromisoformat(s.replace("/", "-"))
    except (AttributeError):  # noqa: BLE001
        pass
    try:
        import pandas as pd

        dt = pd.to_datetime(s, errors="coerce")
        if getattr(dt, "to_pydatetime", None) is not None:
            dt2 = dt.to_pydatetime()
            if isinstance(dt2, datetime):
                return dt2
        return None
    except (TypeError, ValueError, AttributeError):  # noqa: BLE001
        return None


def _stable_trade_id(*, dt: datetime | None, asset: str, symbol: str, side: Side, price: float, shares: int, fee: float, tax: float, order_id: str | None) -> str:
    """
    当上游不给 trade_id 时的退化方案：用关键字段 hash 一个“尽量稳定”的 id。
    """
    s = "|".join(
        [
            (dt.isoformat() if isinstance(dt, datetime) else ""),
            str(asset),
            str(symbol),
            str(side),
            f"{float(price):.10f}",
            str(int(shares)),
            f"{float(fee):.10f}",
            f"{float(tax):.10f}",
            str(order_id or ""),
        ]
    )
    return "hash_" + hashlib.sha1(s.encode("utf-8")).hexdigest()  # noqa: S324


def load_trade_fills(path: str | Path, *, fmt: str | None = None, encoding: str | None = None) -> list[TradeFill]:
    """
    读取成交明细（CSV/JSON/JSONL）并标准化为 TradeFill 列表。

    支持的最低字段：
    - symbol / code / 证券代码
    - side / direction / 买卖
    - price
    - shares / qty
    - datetime（可选，但强烈建议有）
    - trade_id（可选，但强烈建议有）
    - fee/tax（可选，缺失按 0；但对现金/盈亏会有偏差）
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"fills 文件不存在：{p}")

    fmt2 = (str(fmt or "").strip().lower() or None)
    if fmt2 is None:
        suf = p.suffix.lower()
        if suf in {".csv"}:
            fmt2 = "csv"
        elif suf in {".jsonl"}:
            fmt2 = "jsonl"
        else:
            fmt2 = "json"

    if fmt2 == "csv":
        return _load_trade_fills_csv(p, encoding=encoding)
    if fmt2 == "jsonl":
        return _load_trade_fills_jsonl(p)
    if fmt2 == "json":
        return _load_trade_fills_json(p)
    raise ValueError(f"不支持的 fills 格式：{fmt2}（只支持 csv/json/jsonl）")


def _load_trade_fills_json(path: Path) -> list[TradeFill]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    items: list[Any]
    if isinstance(raw, list):
        items = raw
    elif isinstance(raw, dict) and isinstance(raw.get("items"), list):
        items = list(raw.get("items") or [])
    else:
        raise ValueError("fills.json 根节点必须是 list，或 {items:[...]}。")
    fills: list[TradeFill] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        fills.append(_parse_fill_dict(it))
    return _post_process_fills(fills)


def _load_trade_fills_jsonl(path: Path) -> list[TradeFill]:
    fills: list[TradeFill] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = str(line).strip()
        if not s:
            continue
        obj = json.loads(s)
        if not isinstance(obj, dict):
            continue
        fills.append(_parse_fill_dict(obj))
    return _post_process_fills(fills)


def _load_trade_fills_csv(path: Path, *, encoding: str | None = None) -> list[TradeFill]:
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise RuntimeError("没装 pandas？先跑：pip install -r requirements.txt") from exc

    df = pd.read_csv(path, encoding=encoding)
    if df.empty:
        return []

    cols = list(df.columns)
    trade_id_col = _first_match(cols, ["trade_id", "成交编号", "成交序号", "成交ID", "成交编号"])
    order_id_col = _first_match(cols, ["order_id", "委托编号", "订单编号", "委托序号", "委托ID"])
    dt_col = _first_match(cols, ["datetime", "time", "timestamp", "成交时间", "成交日期", "日期时间", "时间"])
    symbol_col = _first_match(cols, ["symbol", "code", "证券代码", "代码", "ticker"])
    side_col = _first_match(cols, ["side", "direction", "买卖", "买卖方向", "业务名称", "操作"])
    price_col = _first_match(cols, ["price", "成交价", "成交价格", "price_avg", "成交均价"])
    shares_col = _first_match(cols, ["shares", "qty", "volume", "成交数量", "数量", "成交股数", "成交份额"])
    fee_col = _first_match(cols, ["fee", "手续费", "交易费", "佣金"])
    tax_col = _first_match(cols, ["tax", "印花税", "过户费", "规费"])
    asset_col = _first_match(cols, ["asset", "品种", "资产", "类型"])

    if not symbol_col:
        raise ValueError("CSV 找不到 symbol/证券代码 列。")
    if not side_col:
        raise ValueError("CSV 找不到 side/买卖方向 列。")
    if not price_col:
        raise ValueError("CSV 找不到 price/成交价 列。")
    if not shares_col:
        raise ValueError("CSV 找不到 shares/成交数量 列。")

    fills: list[TradeFill] = []
    for _, row in df.iterrows():
        it = {c: row.get(c) for c in cols}
        # 用列映射把字段塞回 dict，复用 _parse_fill_dict（少写一套）
        if trade_id_col:
            it["trade_id"] = row.get(trade_id_col)
        if order_id_col:
            it["order_id"] = row.get(order_id_col)
        if dt_col:
            it["datetime"] = row.get(dt_col)
        it["symbol"] = row.get(symbol_col)
        it["side"] = row.get(side_col)
        it["price"] = row.get(price_col)
        it["shares"] = row.get(shares_col)
        if fee_col:
            it["fee"] = row.get(fee_col)
        if tax_col:
            it["tax"] = row.get(tax_col)
        if asset_col:
            it["asset"] = row.get(asset_col)
        fills.append(_parse_fill_dict(it))

    return _post_process_fills(fills)


def _parse_fill_dict(it: dict[str, Any]) -> TradeFill:
    trade_id = str(it.get("trade_id") or it.get("id") or it.get("成交编号") or it.get("成交序号") or "").strip()
    order_id = str(it.get("order_id") or it.get("委托编号") or it.get("订单编号") or "").strip() or None

    dt = _parse_dt(it.get("datetime") or it.get("time") or it.get("timestamp") or it.get("成交时间") or it.get("成交日期"))

    sym_raw = str(it.get("symbol") or it.get("code") or it.get("证券代码") or it.get("ticker") or "").strip()
    if not sym_raw:
        raise ValueError("fills 里有一条记录 symbol 为空。")

    asset = str(it.get("asset") or "").strip().lower()
    if asset not in {"etf", "stock"}:
        asset = _guess_asset_for_symbol(sym_raw)

    side = _norm_side(it.get("side") or it.get("direction") or it.get("买卖") or it.get("买卖方向") or it.get("业务名称") or it.get("操作"))
    if side is None:
        raise ValueError(f"无法解析买卖方向：symbol={sym_raw} side={it.get('side') or it.get('买卖方向')}")

    try:
        price = float(it.get("price") if it.get("price") is not None else it.get("成交价"))
    except (TypeError, ValueError, OverflowError, AttributeError) as exc:  # noqa: BLE001
        raise ValueError(f"price 非法：symbol={sym_raw} price={it.get('price') or it.get('成交价')}") from exc
    if price <= 0:
        raise ValueError(f"price 非法（<=0）：symbol={sym_raw} price={price}")

    try:
        shares = int(float(it.get("shares") if it.get("shares") is not None else it.get("qty") if it.get("qty") is not None else it.get("成交数量") or it.get("数量") or 0))
    except (TypeError, ValueError, OverflowError, AttributeError) as exc:  # noqa: BLE001
        raise ValueError(f"shares 非法：symbol={sym_raw} shares={it.get('shares') or it.get('成交数量') or it.get('数量')}") from exc
    if shares <= 0:
        raise ValueError(f"shares 非法（<=0）：symbol={sym_raw} shares={shares}")

    fee = 0.0
    tax = 0.0
    for key, target in [("fee", "fee"), ("手续费", "fee"), ("佣金", "fee"), ("交易费", "fee")]:
        if key in it and it.get(key) is not None:
            try:
                fee = float(it.get(key) or 0.0)
            except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                fee = 0.0
            break
    for key, target in [("tax", "tax"), ("印花税", "tax"), ("过户费", "tax"), ("规费", "tax")]:
        if key in it and it.get(key) is not None:
            try:
                tax = float(it.get(key) or 0.0)
            except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                tax = 0.0
            break

    # 统一 symbol 前缀（sh/sz/bj）
    sym = resolve_symbol(asset, sym_raw)

    if not trade_id:
        trade_id = _stable_trade_id(dt=dt, asset=asset, symbol=sym, side=side, price=price, shares=shares, fee=fee, tax=tax, order_id=order_id)

    raw_keep = {k: it.get(k) for k in list(it.keys())[:80]}
    return TradeFill(
        trade_id=trade_id,
        dt=dt,
        asset=asset,
        symbol=sym,
        side=side,
        price=float(price),
        shares=int(shares),
        fee=float(fee or 0.0),
        tax=float(tax or 0.0),
        order_id=order_id,
        raw=raw_keep,
    )


def _post_process_fills(fills: list[TradeFill]) -> list[TradeFill]:
    # 去掉明显垃圾（shares/price<=0 在 parse 里已经挡了）
    out = [f for f in fills if isinstance(f, TradeFill)]
    # 按时间排序（没有 dt 的放最后；同一时间保持输入顺序）
    out.sort(key=lambda x: (0 if x.dt is not None else 1, x.dt or datetime.max, x.trade_id))
    return out


def _load_ledger_trade_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        s = str(line).strip()
        if not s:
            continue
        try:
            obj = json.loads(s)
        except (AttributeError):  # noqa: BLE001
            continue
        if not isinstance(obj, dict):
            continue
        tid = str(obj.get("trade_id") or "").strip()
        if tid:
            ids.add(tid)
    return ids


def reconcile_user_holdings(
    *,
    holdings_snapshot: dict[str, Any],
    fills: list[TradeFill],
    existing_ledger_trade_ids: set[str] | None = None,
    orders_next_open: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """
    对账核心：把 fills 合并进 user_holdings 快照，并生成审计台账 append 记录。

    返回：
    - result：对账摘要（warnings/changes/cash_before/after...）
    - next_snapshot：更新后的 holdings_snapshot（未落盘，由调用方决定是否 --apply）
    - ledger_appends：需要追加到 ledger_trades.jsonl 的行（dict）
    """
    # 这玩意是“状态机”，但别偷偷改调用方传进来的 dict（不然你调试会疯）。
    import copy

    holdings_snapshot = copy.deepcopy(holdings_snapshot)
    if not isinstance(holdings_snapshot, dict):
        raise ValueError("holdings_snapshot 必须是 dict")

    cash_obj = holdings_snapshot.get("cash")
    if not isinstance(cash_obj, dict):
        cash_obj = {}
        holdings_snapshot["cash"] = cash_obj
    cash_before_raw = cash_obj.get("amount")
    try:
        cash_before = float(cash_before_raw) if cash_before_raw is not None else None
    except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
        cash_before = None

    positions = holdings_snapshot.get("positions")
    if not isinstance(positions, list):
        positions = []
        holdings_snapshot["positions"] = positions

    # 建一个 symbol->position 的映射（直接引用原 dict，方便原地更新）
    pos_by_sym: dict[str, dict[str, Any]] = {}
    for p in positions:
        if not isinstance(p, dict):
            continue
        sym = str(p.get("symbol") or "").strip()
        if not sym:
            continue
        pos_by_sym[sym] = p

    existing_ids = set(existing_ledger_trade_ids or set())
    warnings: list[str] = []
    changes: list[dict[str, Any]] = []
    ledger_appends: list[dict[str, Any]] = []

    if not fills:
        res = {
            "ok": True,
            "fills_total": 0,
            "fills_new": 0,
            "cash_before": cash_before,
            "cash_after": cash_before,
            "changes": [],
            "warnings": ["fills 为空：没啥可对的。"],
        }
        return res, dict(holdings_snapshot), []

    # cash 为空也允许跑，但会打 warning（apply 时你最好补齐，否则持仓/现金对不上）
    cash = float(cash_before or 0.0)
    if cash_before is None:
        warnings.append("cash.amount 为空：现金会按 0 作为起点做对账（结果仅供参考）。")

    def _get_pos(sym: str) -> dict[str, Any]:
        if sym in pos_by_sym:
            return pos_by_sym[sym]
        # 新标的：先给个空壳
        p2: dict[str, Any] = {"symbol": sym, "shares": 0, "cost_basis": 0.0}
        pos_by_sym[sym] = p2
        positions.append(p2)
        return p2

    def _get_shares_cost(p: dict[str, Any]) -> tuple[int, float]:
        try:
            sh = int(p.get("shares") or 0)
        except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
            sh = 0
        raw_cost = p.get("cost_basis")
        if raw_cost is None:
            raw_cost = p.get("cost")
        try:
            cb = float(raw_cost or 0.0)
        except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
            cb = 0.0
        return sh, cb

    max_dt: datetime | None = None
    n_total = 0
    n_new = 0
    n_hash_id = 0
    for f in fills:
        n_total += 1
        if f.trade_id in existing_ids:
            continue
        existing_ids.add(f.trade_id)
        n_new += 1
        if str(f.trade_id).startswith("hash_"):
            n_hash_id += 1

        if f.dt is not None:
            max_dt = f.dt if max_dt is None else max(max_dt, f.dt)

        p = _get_pos(f.symbol)
        sh0, cb0 = _get_shares_cost(p)

        cash0 = float(cash)
        if f.side == "buy":
            # 买入：把费用并进成本（更贴近实际持仓成本口径）
            total_cost0 = float(cb0) * float(sh0)
            total_cost1 = total_cost0 + float(f.amount) + float(f.fee) + float(f.tax)
            sh1 = int(sh0) + int(f.shares)
            cb1 = float(total_cost1) / float(sh1) if sh1 > 0 else 0.0
            cash -= float(f.amount) + float(f.fee) + float(f.tax)
            realized = None
        else:
            # 卖出：不支持做空；如果出现，先报错别瞎对
            if sh0 < int(f.shares):
                raise ValueError(f"卖出超过持仓：{f.symbol} have={sh0} sell={f.shares} trade_id={f.trade_id}")
            sh1 = int(sh0) - int(f.shares)
            cb1 = float(cb0) if sh1 > 0 else 0.0
            cash += float(f.amount) - float(f.fee) - float(f.tax)
            realized = (float(f.price) - float(cb0)) * float(f.shares) - float(f.fee) - float(f.tax)

        # 写回 position
        p["shares"] = int(sh1)
        p["cost_basis"] = float(cb1)
        if f.asset in {"etf", "stock"}:
            p.setdefault("asset", f.asset)

        # 记录变化（按 trade 粒度，方便审计）
        ch = {
            "trade_id": f.trade_id,
            "datetime": (f.dt.isoformat() if f.dt is not None else None),
            "asset": f.asset,
            "symbol": f.symbol,
            "side": f.side,
            "price": float(f.price),
            "shares": int(f.shares),
            "fee": float(f.fee),
            "tax": float(f.tax),
            "amount": float(f.amount),
            "cash_before": float(cash0),
            "cash_after": float(cash),
            "pos_shares_before": int(sh0),
            "pos_shares_after": int(sh1),
            "pos_cost_basis_before": float(cb0),
            "pos_cost_basis_after": float(cb1),
            "realized_pnl": (float(realized) if realized is not None else None),
        }
        changes.append(ch)
        ledger_appends.append(
            {
                **ch,
                "order_id": f.order_id,
                "source": {"type": "fills", "raw": (f.raw or None)},
            }
        )

    # 清理 positions：shares=0 的从 positions 挪到 closed_positions（可选；不强制）
    closed = holdings_snapshot.get("closed_positions")
    if not isinstance(closed, list):
        closed = []
        holdings_snapshot["closed_positions"] = closed
    keep_positions: list[dict[str, Any]] = []
    for p in positions:
        if not isinstance(p, dict):
            continue
        sym = str(p.get("symbol") or "").strip()
        if not sym:
            continue
        sh = 0
        try:
            sh = int(p.get("shares") or 0)
        except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
            sh = 0
        if sh > 0:
            keep_positions.append(p)
            continue
        # shares==0：如果原来就没仓位/只是临时壳，就别塞进 closed_positions
        # 但如果确实发生过卖出导致清仓，留个审计记录
        touched = any((c.get("symbol") == sym and c.get("side") == "sell") for c in changes if isinstance(c, dict))
        if touched:
            note = "reconcile: 清仓"
            if max_dt is not None:
                note += f" at {max_dt.date().isoformat()}"
            closed.append({"symbol": sym, "note": note})
    holdings_snapshot["positions"] = keep_positions

    # cash 写回
    cash_obj["amount"] = float(cash)
    if max_dt is not None:
        holdings_snapshot["last_updated"] = max_dt.date().isoformat()

    if n_hash_id > 0:
        warnings.append(f"有 {n_hash_id} 条成交缺 trade_id：已退化用 hash_ 生成（反复导出可能重复；建议上游补齐唯一成交编号）")
    if float(cash) < -1e-6:
        warnings.append(f"现金为负：cash_after={cash:.2f}（融资/逆回购/未计入资金流水？自己核对）")

    # planned_allocations：能对上就标记（不强求，别把主流程搞复杂）
    try:
        planned = holdings_snapshot.get("planned_allocations")
        if isinstance(planned, list) and planned:
            by_sym_buy: dict[str, dict[str, Any]] = {}
            # 只统计“本次新增”的买入（ledger_appends 就是新增的）
            for rec in ledger_appends:
                if not isinstance(rec, dict) or str(rec.get("side") or "") != "buy":
                    continue
                sym = str(rec.get("symbol") or "")
                if not sym:
                    continue
                agg = by_sym_buy.get(sym)
                if agg is None:
                    by_sym_buy[sym] = {"shares": 0, "amount": 0.0}
                    agg = by_sym_buy[sym]
                agg["shares"] = int(agg.get("shares") or 0) + int(rec.get("shares") or 0)
                agg["amount"] = float(agg.get("amount") or 0.0) + float(rec.get("amount") or 0.0)

            for it in planned:
                if not isinstance(it, dict):
                    continue
                sym = str(it.get("symbol") or "").strip()
                if not sym:
                    continue
                st = str(it.get("status") or "").strip().lower()
                if st not in {"planned", "partial"}:
                    continue
                agg = by_sym_buy.get(sym)
                if not agg:
                    continue
                filled_sh = int(agg.get("shares") or 0)
                filled_amt = float(agg.get("amount") or 0.0)
                if filled_sh <= 0 or filled_amt <= 0:
                    continue
                avg = float(filled_amt) / float(filled_sh)
                it["filled_avg_price"] = float(avg)
                it["filled_shares"] = int(filled_sh)
                want = it.get("planned_shares")
                try:
                    want2 = int(want) if want is not None else None
                except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                    want2 = None
                if want2 is not None and want2 > 0 and filled_sh + 1e-9 < want2:
                    it["status"] = "partial"
                else:
                    it["status"] = "filled"
    except (TypeError, ValueError, OverflowError, KeyError, IndexError, AttributeError):  # noqa: BLE001
        pass

    # orders vs fills 的简易对比（你手动点确认也经常会改手数/撤单，这里只给提示，不做强校验）
    if orders_next_open:
        try:
            exp: dict[tuple[str, str], int] = {}
            for o in orders_next_open:
                if not isinstance(o, dict):
                    continue
                side = str(o.get("side") or "").strip().lower()
                sym = str(o.get("symbol") or "").strip()
                if side not in {"buy", "sell"} or not sym:
                    continue
                try:
                    sh = int(o.get("shares") or 0)
                except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                    sh = 0
                if sh <= 0:
                    continue
                exp[(side, sym)] = exp.get((side, sym), 0) + int(sh)

            got: dict[tuple[str, str], int] = {}
            for rec in ledger_appends:
                if not isinstance(rec, dict):
                    continue
                side = str(rec.get("side") or "").strip().lower()
                sym = str(rec.get("symbol") or "").strip()
                if side not in {"buy", "sell"} or not sym:
                    continue
                got[(side, sym)] = got.get((side, sym), 0) + int(rec.get("shares") or 0)

            for k, sh_exp in sorted(exp.items()):
                sh_got = int(got.get(k, 0))
                if sh_got <= 0:
                    warnings.append(f"orders 未成交？{k[0]} {k[1]} expected={sh_exp} got=0")
                elif sh_got != int(sh_exp):
                    warnings.append(f"orders 成交数量不一致：{k[0]} {k[1]} expected={sh_exp} got={sh_got}")

            # 计划里没有，但实际发生了（手动临时单/改手数/新开仓）
            for k, sh_got in sorted(got.items()):
                if k not in exp:
                    warnings.append(f"实际成交不在 orders 计划里：{k[0]} {k[1]} got={sh_got}")
        except (AttributeError):  # noqa: BLE001
            pass

    res = {
        "ok": True,
        "fills_total": int(n_total),
        "fills_new": int(n_new),
        "cash_before": cash_before,
        "cash_after": float(cash),
        "changes": changes,
        "warnings": warnings,
    }
    return res, dict(holdings_snapshot), ledger_appends


def load_orders_next_open(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"orders_next_open.json 不存在：{p}")
    obj = json.loads(p.read_text(encoding="utf-8"))
    orders = (obj.get("orders") if isinstance(obj, dict) else None) or []
    return [o for o in orders if isinstance(o, dict)]


def load_user_holdings_snapshot(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"user_holdings.json 不存在：{p}")
    obj = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError("user_holdings.json 根节点必须是 object")
    return obj


def load_ledger_trade_ids(path: str | Path) -> set[str]:
    return _load_ledger_trade_ids(Path(path))
