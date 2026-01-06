from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any


def _ensure_matplotlib_cache_dir():
    """
    这环境对 /home/root_zzl/.cache 没写权限（沙箱限制），matplotlib 默认写字体缓存会报 Permission denied。
    解决：把 MPLCONFIGDIR 指到仓库内的 .matplotlib 目录。
    """
    import os
    from pathlib import Path

    if os.environ.get("MPLCONFIGDIR"):
        return
    try:
        root = Path(__file__).resolve().parents[1]
        cache_dir = root / ".matplotlib"
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(cache_dir)
    except Exception:  # noqa: BLE001
        return


def _require_matplotlib():
    _ensure_matplotlib_cache_dir()
    try:
        import matplotlib  # noqa: F401
    except ModuleNotFoundError as exc:
        raise RuntimeError("没装 matplotlib？先跑：pip install -r requirements.txt（或 Python3.12 用 requirements-py312.txt）") from exc


def _get_project_root():
    from pathlib import Path

    return Path(__file__).resolve().parents[1]


def _ensure_cjk_font_file(*, filename: str, url: str) -> str | None:
    """
    没中文字体就自动下一个（落到仓库内 .matplotlib/fonts/），否则全是方框，谁看得下去？
    """
    import urllib.error
    import urllib.request
    from pathlib import Path

    root = _get_project_root()
    fonts_dir = root / ".matplotlib" / "fonts"
    fonts_dir.mkdir(parents=True, exist_ok=True)

    target = fonts_dir / filename
    if target.exists() and target.stat().st_size > 1024 * 1024:
        return str(target)

    tmp = Path(str(target) + ".tmp")
    try:
        with urllib.request.urlopen(url, timeout=120) as resp:
            with tmp.open("wb") as f:
                while True:
                    chunk = resp.read(1024 * 64)
                    if not chunk:
                        break
                    f.write(chunk)
        if tmp.stat().st_size < 1024 * 1024:
            tmp.unlink(missing_ok=True)
            return None
        tmp.replace(target)
        return str(target)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):  # noqa: PERF203
        try:
            tmp.unlink(missing_ok=True)
        except Exception:  # noqa: BLE001
            pass
        return None
    except Exception:  # noqa: BLE001
        try:
            tmp.unlink(missing_ok=True)
        except Exception:  # noqa: BLE001
            pass
        return None


def setup_chinese_font(preferred: str | None = None, *, font_path: str | None = None) -> str | None:
    _require_matplotlib()
    import matplotlib
    from matplotlib import font_manager

    fallback = ["DejaVu Sans", "DejaVu Sans Mono"]

    if font_path:
        fp = str(font_path)
        try:
            font_manager.fontManager.addfont(fp)
            name = font_manager.FontProperties(fname=fp).get_name()
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"加载字体失败：{fp}") from exc

        matplotlib.rcParams["font.sans-serif"] = [name, *fallback]
        matplotlib.rcParams["font.family"] = "sans-serif"
        matplotlib.rcParams["axes.unicode_minus"] = False
        return name

    if preferred:
        matplotlib.rcParams["font.sans-serif"] = [preferred, *fallback]
        matplotlib.rcParams["font.family"] = "sans-serif"
        matplotlib.rcParams["axes.unicode_minus"] = False
        return preferred

    candidates = [
        "SimHei",
        "Microsoft YaHei",
        "PingFang SC",
        "Heiti SC",
        "WenQuanYi Zen Hei",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "Arial Unicode MS",
    ]

    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            matplotlib.rcParams["font.sans-serif"] = [name, *fallback]
            matplotlib.rcParams["font.family"] = "sans-serif"
            matplotlib.rcParams["axes.unicode_minus"] = False
            return name

    # 这破环境大概率没装中文字体：自动拉一个 Noto CJK（体积不小，但总比全是方框强）
    noto_path = _ensure_cjk_font_file(
        filename="NotoSansCJKsc-Regular.otf",
        url="https://github.com/notofonts/noto-cjk/raw/refs/heads/main/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf",
    )
    if noto_path:
        try:
            font_manager.fontManager.addfont(noto_path)
            name = font_manager.FontProperties(fname=noto_path).get_name()
        except Exception:  # noqa: BLE001
            name = None
        if name:
            matplotlib.rcParams["font.sans-serif"] = [name]
            matplotlib.rcParams["font.family"] = "sans-serif"
            matplotlib.rcParams["axes.unicode_minus"] = False
            return name

    matplotlib.rcParams["font.sans-serif"] = fallback
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["axes.unicode_minus"] = False
    return None


def _parse_date(s: str) -> datetime | None:
    try:
        return datetime.strptime(s, "%Y-%m-%d")
    except Exception:  # noqa: BLE001
        return None


def plot_wyckoff_chart(
    df,
    *,
    analysis: dict[str, Any] | None,
    out_path: str,
    title: str | None = None,
    font_path: str | None = None,
    show_ad_line: bool = True,
):
    _require_matplotlib()
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    setup_chinese_font(font_path=font_path)

    use_ad = bool(show_ad_line and "ad_line" in df.columns)
    if use_ad:
        fig, (ax, ax_ad) = plt.subplots(
            nrows=2, ncols=1, sharex=True, figsize=(16, 9), gridspec_kw={"height_ratios": [3, 1]}
        )
        ax.tick_params(labelbottom=False)
    else:
        fig, ax = plt.subplots(figsize=(16, 9))
        ax_ad = None

    ax.plot(df["date"], df["close"], color="black", linewidth=1.2, label="收盘价")
    if "ma50" in df.columns:
        ax.plot(df["date"], df["ma50"], color="blue", linestyle="--", linewidth=1, label="MA50")
    if "ma200" in df.columns:
        ax.plot(df["date"], df["ma200"], color="red", linestyle="--", linewidth=1, label="MA200")

    y_min = float(df["close"].min())
    y_max = float(df["close"].max())

    if analysis:
        for zone in analysis.get("zones", []) or []:
            start_dt = _parse_date(str(zone.get("start_date", "")))
            end_dt = _parse_date(str(zone.get("end_date", "")))
            low = zone.get("low")
            high = zone.get("high")
            if not (start_dt and end_dt and isinstance(low, (int, float)) and isinstance(high, (int, float))):
                continue
            if high <= low:
                continue
            if end_dt <= start_dt:
                continue

            zone_type = str(zone.get("type", ""))
            face = "#a6e3a1" if zone_type == "Accumulation" else "#f2a7a7"
            edge = "#2b8a3e" if zone_type == "Accumulation" else "#c92a2a"

            x0 = mdates.date2num(start_dt)
            x1 = mdates.date2num(end_dt)
            rect = Rectangle(
                (x0, float(low)),
                x1 - x0,
                float(high) - float(low),
                facecolor=face,
                edgecolor=edge,
                linewidth=1.0,
                alpha=0.25,
            )
            ax.add_patch(rect)
            label = str(zone.get("label", "")).strip()
            if label:
                ax.text(
                    start_dt + (end_dt - start_dt) / 2,
                    float(high),
                    label,
                    color=edge,
                    fontsize=11,
                    ha="center",
                    va="bottom",
                )

        phases = analysis.get("phases", []) or []
        for phase in phases:
            start_dt = _parse_date(str(phase.get("start_date", "")))
            end_dt = _parse_date(str(phase.get("end_date", "")))
            if not (start_dt and end_dt) or end_dt <= start_dt:
                continue
            ax.axvline(start_dt, color="black", linestyle=(0, (6, 6)), linewidth=2, alpha=0.7)
            mid_dt = start_dt + (end_dt - start_dt) / 2
            label = str(phase.get("label") or phase.get("name") or "").strip()
            if label:
                ax.text(
                    mid_dt,
                    y_max + (y_max - y_min) * 0.02,
                    label,
                    color="darkred",
                    fontsize=14,
                    fontweight="bold",
                    ha="center",
                    va="bottom",
                )

        events = analysis.get("events", []) or []
        for i, ev in enumerate(events):
            dt = _parse_date(str(ev.get("date", "")))
            price = ev.get("price")
            text = str(ev.get("text", "")).strip()
            if not (dt and isinstance(price, (int, float)) and text):
                continue

            dx_days = (i % 5 - 2) * 8
            dy = (0.04 + 0.02 * (i % 5)) * (y_max - y_min)
            xytext = (dt + timedelta(days=dx_days), float(price) + dy)

            ax.annotate(
                text,
                xy=(dt, float(price)),
                xytext=xytext,
                textcoords="data",
                arrowprops={"arrowstyle": "->", "color": "#444", "lw": 1},
                fontsize=10,
                ha="left" if dx_days >= 0 else "right",
                va="bottom",
                bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "#999", "alpha": 0.8},
            )

    ax.set_title(title or "LLM辅助威科夫标注图", fontsize=16)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    locator = mdates.AutoDateLocator(minticks=5, maxticks=10)
    formatter = mdates.ConciseDateFormatter(locator)
    if ax_ad is not None:
        ax_ad.xaxis.set_major_locator(locator)
        ax_ad.xaxis.set_major_formatter(formatter)
    else:
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

    ax.set_ylim(y_min - (y_max - y_min) * 0.05, y_max + (y_max - y_min) * 0.12)

    if ax_ad is not None:
        ax_ad.plot(df["date"], df["ad_line"], color="#666", linewidth=1.2, label="A/D（累积/派发线）")
        ax_ad.grid(True, alpha=0.2)
        ax_ad.legend(loc="best")

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_chanlun_chart(
    df,
    *,
    structure: dict[str, Any],
    out_path: str,
    title: str | None = None,
    font_path: str | None = None,
):
    _require_matplotlib()
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    setup_chinese_font(font_path=font_path)

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.plot(df["date"], df["close"], color="black", linewidth=1.2, label="收盘价")
    if "ma50" in df.columns:
        ax.plot(df["date"], df["ma50"], color="blue", linestyle="--", linewidth=1, label="MA50")
    if "ma200" in df.columns:
        ax.plot(df["date"], df["ma200"], color="red", linestyle="--", linewidth=1, label="MA200")

    y_min = float(df["close"].min())
    y_max = float(df["close"].max())

    # 中枢
    for idx, c in enumerate(structure.get("centers", []) or []):
        start_dt = _parse_date(str(c.get("start_date", "")))
        end_dt = _parse_date(str(c.get("end_date", "")))
        low = c.get("low")
        high = c.get("high")
        if not (start_dt and end_dt and isinstance(low, (int, float)) and isinstance(high, (int, float))):
            continue
        if end_dt <= start_dt or high <= low:
            continue

        x0 = mdates.date2num(start_dt)
        x1 = mdates.date2num(end_dt)
        rect = Rectangle(
            (x0, float(low)),
            x1 - x0,
            float(high) - float(low),
            facecolor="#89b4fa",
            edgecolor="#1e66f5",
            linewidth=1.0,
            alpha=0.18,
        )
        ax.add_patch(rect)

        label = f"中枢#{idx + 1}\\n[{float(low):.2f}-{float(high):.2f}]"
        ax.text(
            start_dt + (end_dt - start_dt) / 2,
            float(high),
            label,
            color="#1e66f5",
            fontsize=10,
            ha="center",
            va="bottom",
        )

    # 笔
    for st in structure.get("strokes", []) or []:
        s = st.get("start") or {}
        e = st.get("end") or {}
        sd = _parse_date(str(s.get("date", "")))
        ed = _parse_date(str(e.get("date", "")))
        sp = s.get("price")
        ep = e.get("price")
        direction = str(st.get("direction", ""))
        if not (sd and ed and isinstance(sp, (int, float)) and isinstance(ep, (int, float))):
            continue

        color = "#2f9e44" if direction == "up" else "#e03131"
        ax.plot([sd, ed], [float(sp), float(ep)], color=color, linewidth=2.0, alpha=0.85)
        ax.scatter([sd, ed], [float(sp), float(ep)], color=color, s=18, zorder=5)

    ax.set_title(title or "缠论结构（笔/中枢）", fontsize=16)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    locator = mdates.AutoDateLocator(minticks=5, maxticks=10)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

    ax.set_ylim(y_min - (y_max - y_min) * 0.05, y_max + (y_max - y_min) * 0.12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_ichimoku_chart(
    df,
    *,
    out_path: str,
    title: str | None = None,
    font_path: str | None = None,
    prefix: str = "ichimoku_",
):
    _require_matplotlib()
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    setup_chinese_font(font_path=font_path)

    fig, ax = plt.subplots(figsize=(16, 9))

    ax.plot(df["date"], df["close"], color="black", linewidth=1.2, label="收盘价")
    if "ma50" in df.columns:
        ax.plot(df["date"], df["ma50"], color="blue", linestyle="--", linewidth=1, label="MA50")
    if "ma200" in df.columns:
        ax.plot(df["date"], df["ma200"], color="red", linestyle="--", linewidth=1, label="MA200")

    tenkan = prefix + "tenkan"
    kijun = prefix + "kijun"
    span_a = prefix + "span_a"
    span_b = prefix + "span_b"
    chikou = prefix + "chikou"

    if tenkan in df.columns:
        ax.plot(df["date"], df[tenkan], color="#fab005", linewidth=1.2, label="转换线(9)")
    if kijun in df.columns:
        ax.plot(df["date"], df[kijun], color="#7950f2", linewidth=1.2, label="基准线(26)")

    if span_a in df.columns and span_b in df.columns:
        sa = df[span_a].astype(float)
        sb = df[span_b].astype(float)
        x = df["date"]
        ax.fill_between(x, sa, sb, where=(sa >= sb), color="#a6e3a1", alpha=0.22, interpolate=True, label="云(多)")
        ax.fill_between(x, sa, sb, where=(sa < sb), color="#f2a7a7", alpha=0.22, interpolate=True, label="云(空)")

    if chikou in df.columns:
        ax.plot(df["date"], df[chikou], color="#666", linewidth=1.0, alpha=0.7, label="迟行线(26)")

    y_min = float(df["close"].min())
    y_max = float(df["close"].max())

    ax.set_title(title or "Ichimoku（一目均衡表）", fontsize=16)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", ncol=2)

    locator = mdates.AutoDateLocator(minticks=5, maxticks=10)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.set_ylim(y_min - (y_max - y_min) * 0.05, y_max + (y_max - y_min) * 0.12)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_turtle_chart(
    df,
    *,
    out_path: str,
    title: str | None = None,
    font_path: str | None = None,
    entry_upper_col: str = "donchian_entry_upper",
    entry_lower_col: str = "donchian_entry_lower",
    exit_upper_col: str = "donchian_exit_upper",
    exit_lower_col: str = "donchian_exit_lower",
):
    _require_matplotlib()
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    setup_chinese_font(font_path=font_path)

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.plot(df["date"], df["close"], color="black", linewidth=1.2, label="收盘价")

    if "ma50" in df.columns:
        ax.plot(df["date"], df["ma50"], color="blue", linestyle="--", linewidth=1, label="MA50")
    if "ma200" in df.columns:
        ax.plot(df["date"], df["ma200"], color="red", linestyle="--", linewidth=1, label="MA200")

    if entry_upper_col in df.columns:
        ax.plot(df["date"], df[entry_upper_col], color="#2f9e44", linestyle="--", linewidth=1.2, label="Donchian上轨(入场)")
    if entry_lower_col in df.columns:
        ax.plot(df["date"], df[entry_lower_col], color="#e03131", linestyle="--", linewidth=1.2, label="Donchian下轨(入场)")
    if exit_upper_col in df.columns:
        ax.plot(df["date"], df[exit_upper_col], color="#2f9e44", linestyle=":", linewidth=1.2, label="Donchian上轨(出场)")
    if exit_lower_col in df.columns:
        ax.plot(df["date"], df[exit_lower_col], color="#e03131", linestyle=":", linewidth=1.2, label="Donchian下轨(出场)")

    y_min = float(df["close"].min())
    y_max = float(df["close"].max())
    ax.set_ylim(y_min - (y_max - y_min) * 0.05, y_max + (y_max - y_min) * 0.12)

    ax.set_title(title or "Turtle / Donchian 趋势突破", fontsize=16)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", ncol=2)

    locator = mdates.AutoDateLocator(minticks=5, maxticks=10)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_momentum_chart(
    df,
    *,
    out_path: str,
    title: str | None = None,
    font_path: str | None = None,
    rsi_col: str = "rsi",
    macd_col: str = "macd",
    macd_signal_col: str = "macd_signal",
    macd_hist_col: str = "macd_hist",
    adx_col: str = "adx",
    di_plus_col: str = "di_plus",
    di_minus_col: str = "di_minus",
):
    _require_matplotlib()
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    setup_chinese_font(font_path=font_path)

    fig, (ax_price, ax_macd, ax_rsi, ax_adx) = plt.subplots(
        nrows=4,
        ncols=1,
        sharex=True,
        figsize=(16, 11),
        gridspec_kw={"height_ratios": [3, 1, 1, 1]},
    )
    ax_price.tick_params(labelbottom=False)
    ax_macd.tick_params(labelbottom=False)
    ax_rsi.tick_params(labelbottom=False)

    ax_price.plot(df["date"], df["close"], color="black", linewidth=1.2, label="收盘价")
    if "ma50" in df.columns:
        ax_price.plot(df["date"], df["ma50"], color="blue", linestyle="--", linewidth=1, label="MA50")
    if "ma200" in df.columns:
        ax_price.plot(df["date"], df["ma200"], color="red", linestyle="--", linewidth=1, label="MA200")
    ax_price.set_title(title or "Momentum（RSI/MACD/ADX）", fontsize=16)
    ax_price.grid(True, alpha=0.25)
    ax_price.legend(loc="best")

    # MACD
    ax_macd.axhline(0, color="#888", linewidth=1.0, alpha=0.7)
    if macd_hist_col in df.columns:
        hist = df[macd_hist_col].astype(float)
        colors = ["#2f9e44" if v >= 0 else "#e03131" for v in hist.fillna(0.0)]
        ax_macd.bar(df["date"], hist, color=colors, width=2, alpha=0.35, label="MACD柱")
    if macd_col in df.columns:
        ax_macd.plot(df["date"], df[macd_col], color="#1c7ed6", linewidth=1.2, label="MACD")
    if macd_signal_col in df.columns:
        ax_macd.plot(df["date"], df[macd_signal_col], color="#f59f00", linewidth=1.2, label="Signal")
    ax_macd.grid(True, alpha=0.2)
    ax_macd.legend(loc="best", ncol=3)

    # RSI
    if rsi_col in df.columns:
        ax_rsi.plot(df["date"], df[rsi_col], color="#6741d9", linewidth=1.2, label="RSI")
    ax_rsi.axhline(70, color="#e03131", linestyle="--", linewidth=1.0, alpha=0.8)
    ax_rsi.axhline(30, color="#2f9e44", linestyle="--", linewidth=1.0, alpha=0.8)
    ax_rsi.set_ylim(0, 100)
    ax_rsi.grid(True, alpha=0.2)
    ax_rsi.legend(loc="best")

    # ADX/DI
    if adx_col in df.columns:
        ax_adx.plot(df["date"], df[adx_col], color="#343a40", linewidth=1.2, label="ADX")
    if di_plus_col in df.columns:
        ax_adx.plot(df["date"], df[di_plus_col], color="#2f9e44", linewidth=1.0, alpha=0.9, label="+DI")
    if di_minus_col in df.columns:
        ax_adx.plot(df["date"], df[di_minus_col], color="#e03131", linewidth=1.0, alpha=0.9, label="-DI")
    ax_adx.axhline(20, color="#666", linestyle="--", linewidth=1.0, alpha=0.6)
    ax_adx.grid(True, alpha=0.2)
    ax_adx.legend(loc="best", ncol=3)

    locator = mdates.AutoDateLocator(minticks=5, maxticks=10)
    ax_adx.xaxis.set_major_locator(locator)
    ax_adx.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_dow_chart(
    df,
    *,
    swings: list[dict[str, Any]] | None,
    out_path: str,
    title: str | None = None,
    font_path: str | None = None,
):
    _require_matplotlib()
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    setup_chinese_font(font_path=font_path)

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.plot(df["date"], df["close"], color="black", linewidth=1.2, label="收盘价")
    if "ma50" in df.columns:
        ax.plot(df["date"], df["ma50"], color="blue", linestyle="--", linewidth=1, label="MA50")
    if "ma200" in df.columns:
        ax.plot(df["date"], df["ma200"], color="red", linestyle="--", linewidth=1, label="MA200")

    if swings:
        highs = [s for s in swings if s.get("kind") == "high"]
        lows = [s for s in swings if s.get("kind") == "low"]
        high_labeled = False
        low_labeled = False
        for s in highs:
            dt = _parse_date(str(s.get("date", "")))
            price = s.get("price")
            if dt and isinstance(price, (int, float)):
                ax.scatter(
                    [dt],
                    [float(price)],
                    color="#e03131",
                    s=40,
                    marker="^",
                    zorder=6,
                    label="Swing High" if not high_labeled else "_nolegend_",
                )
                high_labeled = True
        for s in lows:
            dt = _parse_date(str(s.get("date", "")))
            price = s.get("price")
            if dt and isinstance(price, (int, float)):
                ax.scatter(
                    [dt],
                    [float(price)],
                    color="#2f9e44",
                    s=40,
                    marker="v",
                    zorder=6,
                    label="Swing Low" if not low_labeled else "_nolegend_",
                )
                low_labeled = True

        pts = []
        for s in swings:
            dt = _parse_date(str(s.get("date", "")))
            price = s.get("price")
            if dt and isinstance(price, (int, float)):
                pts.append((dt, float(price)))
        if len(pts) >= 2:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color="#868e96", linewidth=1.4, alpha=0.8)

        last_high = highs[-1] if highs else None
        last_low = lows[-1] if lows else None
        if last_high and isinstance(last_high.get("price"), (int, float)):
            ax.axhline(float(last_high["price"]), color="#e03131", linewidth=1.0, alpha=0.35, linestyle="--")
        if last_low and isinstance(last_low.get("price"), (int, float)):
            ax.axhline(float(last_low["price"]), color="#2f9e44", linewidth=1.0, alpha=0.35, linestyle="--")

    ax.set_title(title or "Dow（趋势结构：HH/HL/LH/LL）", fontsize=16)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", ncol=2)

    locator = mdates.AutoDateLocator(minticks=5, maxticks=10)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

    y_min = float(df["close"].min())
    y_max = float(df["close"].max())
    ax.set_ylim(y_min - (y_max - y_min) * 0.05, y_max + (y_max - y_min) * 0.12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_vsa_chart(
    df,
    *,
    vsa_report: dict[str, Any] | None,
    out_path: str,
    title: str | None = None,
    font_path: str | None = None,
    rel_volume_col: str = "vsa_rel_volume",
):
    _require_matplotlib()
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    setup_chinese_font(font_path=font_path)

    use_vol = "volume" in df.columns
    if use_vol:
        fig, (ax, ax_vol) = plt.subplots(
            nrows=2, ncols=1, sharex=True, figsize=(16, 9), gridspec_kw={"height_ratios": [3, 1]}
        )
        ax.tick_params(labelbottom=False)
    else:
        fig, ax = plt.subplots(figsize=(16, 9))
        ax_vol = None

    ax.plot(df["date"], df["close"], color="black", linewidth=1.2, label="收盘价")
    if "ma50" in df.columns:
        ax.plot(df["date"], df["ma50"], color="blue", linestyle="--", linewidth=1, label="MA50")
    if "ma200" in df.columns:
        ax.plot(df["date"], df["ma200"], color="red", linestyle="--", linewidth=1, label="MA200")

    events = (vsa_report or {}).get("events", []) or []
    y_min = float(df["close"].min())
    y_max = float(df["close"].max())
    for i, ev in enumerate(events[-25:]):
        dt = _parse_date(str(ev.get("date", "")))
        price = ev.get("price")
        text = str(ev.get("label", "")).strip()
        if not (dt and isinstance(price, (int, float)) and text):
            continue
        dy = (0.04 + 0.015 * (i % 5)) * (y_max - y_min)
        ax.annotate(
            text,
            xy=(dt, float(price)),
            xytext=(dt, float(price) + dy),
            textcoords="data",
            arrowprops={"arrowstyle": "->", "color": "#444", "lw": 1},
            fontsize=10,
            ha="center",
            va="bottom",
            bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "#999", "alpha": 0.8},
        )

    ax.set_title(title or "VSA（量价行为特征）", fontsize=16)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    ax.set_ylim(y_min - (y_max - y_min) * 0.05, y_max + (y_max - y_min) * 0.12)

    locator = mdates.AutoDateLocator(minticks=5, maxticks=10)
    formatter = mdates.ConciseDateFormatter(locator)
    if ax_vol is not None:
        ax_vol.xaxis.set_major_locator(locator)
        ax_vol.xaxis.set_major_formatter(formatter)
    else:
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

    if ax_vol is not None:
        vol = df["volume"].fillna(0).astype(float)
        ax_vol.bar(df["date"], vol, color="#868e96", alpha=0.35, label="成交量")
        if rel_volume_col in df.columns:
            ax2 = ax_vol.twinx()
            ax2.plot(df["date"], df[rel_volume_col].astype(float), color="#1c7ed6", linewidth=1.2, label="相对成交量")
            ax2.axhline(1.0, color="#999", linewidth=1.0, alpha=0.6, linestyle="--")
            ax2.set_ylim(0, max(2.0, float(df[rel_volume_col].fillna(0.0).max()) * 1.15))
            ax2.legend(loc="upper right")
        ax_vol.grid(True, alpha=0.2)
        ax_vol.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
