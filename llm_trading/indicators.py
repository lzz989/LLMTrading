from __future__ import annotations


def add_moving_averages(df, *, ma_fast: int = 50, ma_slow: int = 200):
    df = df.copy()
    # 别用 min_periods=1 去“硬算” MA：新标的会冒出假的 MA50/MA200，误导你以为均线有效。
    df[f"ma{ma_fast}"] = df["close"].rolling(window=ma_fast, min_periods=ma_fast).mean()
    df[f"ma{ma_slow}"] = df["close"].rolling(window=ma_slow, min_periods=ma_slow).mean()
    return df


def add_accumulation_distribution_line(df, *, out_col: str = "ad_line"):
    """
    A/D（累积/派发线，常用量价强弱指标）
    公式：
      CLV = (2*Close - High - Low) / (High - Low)
      MFV = CLV * Volume
      AD  = cumsum(MFV)
    """
    need = {"high", "low", "close", "volume"}
    if not need.issubset(set(df.columns)):
        return df

    df2 = df.copy()
    high = df2["high"].astype(float)
    low = df2["low"].astype(float)
    close = df2["close"].astype(float)
    volume = df2["volume"].fillna(0).astype(float)

    denom = high - low
    safe = denom.where(denom != 0, 1.0)
    clv = (2 * close - high - low) / safe
    clv = clv.where(denom != 0, 0.0).fillna(0.0)

    mfv = (clv * volume).fillna(0.0)
    df2[out_col] = mfv.cumsum()
    return df2


def add_ichimoku(
    df,
    *,
    tenkan: int = 9,
    kijun: int = 26,
    span_b: int = 52,
    displacement: int = 26,
    prefix: str = "ichimoku_",
):
    need = {"high", "low", "close"}
    if not need.issubset(set(df.columns)):
        return df

    df2 = df.copy()
    high = df2["high"].astype(float)
    low = df2["low"].astype(float)
    close = df2["close"].astype(float)

    tenkan_line = (high.rolling(tenkan, min_periods=tenkan).max() + low.rolling(tenkan, min_periods=tenkan).min()) / 2.0
    kijun_line = (high.rolling(kijun, min_periods=kijun).max() + low.rolling(kijun, min_periods=kijun).min()) / 2.0

    # 乖乖：SpanA/SpanB 为了画云图会“前移”，但你拿最后一根去算状态就会全是 NaN（前移导致末尾缺值）。
    # 所以这里同时保留 raw（当前时点）和 shift（画图用），别再把前移值当现价云层了。
    span_a_raw = (tenkan_line + kijun_line) / 2.0
    span_b_raw = (high.rolling(span_b, min_periods=span_b).max() + low.rolling(span_b, min_periods=span_b).min()) / 2.0

    span_a_line = span_a_raw.shift(displacement)
    span_b_line = span_b_raw.shift(displacement)
    chikou_line = close.shift(-displacement)

    df2[prefix + "tenkan"] = tenkan_line
    df2[prefix + "kijun"] = kijun_line
    df2[prefix + "span_a_raw"] = span_a_raw
    df2[prefix + "span_b_raw"] = span_b_raw
    df2[prefix + "span_a"] = span_a_line
    df2[prefix + "span_b"] = span_b_line
    df2[prefix + "chikou"] = chikou_line
    return df2


def add_atr(df, *, period: int = 14, out_col: str = "atr"):
    need = {"high", "low", "close"}
    if not need.issubset(set(df.columns)):
        return df

    df2 = df.copy()
    high = df2["high"].astype(float)
    low = df2["low"].astype(float)
    close = df2["close"].astype(float)
    prev_close = close.shift(1)

    tr = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = tr.combine(tr2, max).combine(tr3, max).fillna(tr)

    alpha = 1.0 / float(period) if period > 0 else 1.0
    df2[out_col] = true_range.ewm(alpha=alpha, adjust=False, min_periods=1).mean()
    return df2


def add_donchian_channels(
    df,
    *,
    window: int = 20,
    upper_col: str = "donchian_upper",
    lower_col: str = "donchian_lower",
    shift: int = 1,
):
    need = {"high", "low"}
    if not need.issubset(set(df.columns)):
        return df

    df2 = df.copy()
    high = df2["high"].astype(float)
    low = df2["low"].astype(float)
    # 唐奇安通道：window 不足就别算，算出来也是“假的上轨/下轨”。
    df2[upper_col] = high.rolling(window, min_periods=window).max().shift(shift)
    df2[lower_col] = low.rolling(window, min_periods=window).min().shift(shift)
    return df2


def add_rsi(df, *, period: int = 14, out_col: str = "rsi"):
    if "close" not in df.columns:
        return df

    df2 = df.copy()
    close = df2["close"].astype(float)
    delta = close.diff()
    gain = delta.clip(lower=0.0).fillna(0.0)
    loss = (-delta.clip(upper=0.0)).fillna(0.0)

    alpha = 1.0 / float(period) if period > 0 else 1.0
    avg_gain = gain.ewm(alpha=alpha, adjust=False, min_periods=1).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False, min_periods=1).mean()

    rs = avg_gain / avg_loss.replace({0.0: float("nan")})
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.fillna(0.0)
    df2[out_col] = rsi.clip(lower=0.0, upper=100.0)
    return df2


def add_macd(
    df,
    *,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    macd_col: str = "macd",
    signal_col: str = "macd_signal",
    hist_col: str = "macd_hist",
):
    if "close" not in df.columns:
        return df

    df2 = df.copy()
    close = df2["close"].astype(float)
    ema_fast = close.ewm(span=fast, adjust=False, min_periods=1).mean()
    ema_slow = close.ewm(span=slow, adjust=False, min_periods=1).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=1).mean()
    hist = macd_line - signal_line

    df2[macd_col] = macd_line
    df2[signal_col] = signal_line
    df2[hist_col] = hist
    return df2


def add_adx(
    df,
    *,
    period: int = 14,
    adx_col: str = "adx",
    di_plus_col: str = "di_plus",
    di_minus_col: str = "di_minus",
):
    need = {"high", "low", "close"}
    if not need.issubset(set(df.columns)):
        return df

    df2 = df.copy()
    high = df2["high"].astype(float)
    low = df2["low"].astype(float)
    close = df2["close"].astype(float)
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)

    up_move = high - prev_high
    down_move = prev_low - low

    plus_dm = up_move.where((up_move > down_move) & (up_move > 0.0), 0.0).fillna(0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0.0), 0.0).fillna(0.0)

    tr = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = tr.combine(tr2, max).combine(tr3, max).fillna(tr)

    alpha = 1.0 / float(period) if period > 0 else 1.0
    tr_sm = true_range.ewm(alpha=alpha, adjust=False, min_periods=1).mean()
    plus_sm = plus_dm.ewm(alpha=alpha, adjust=False, min_periods=1).mean()
    minus_sm = minus_dm.ewm(alpha=alpha, adjust=False, min_periods=1).mean()

    di_plus = 100.0 * (plus_sm / tr_sm.replace({0.0: float("nan")}))
    di_minus = 100.0 * (minus_sm / tr_sm.replace({0.0: float("nan")}))
    di_plus = di_plus.fillna(0.0)
    di_minus = di_minus.fillna(0.0)

    dx = 100.0 * ((di_plus - di_minus).abs() / (di_plus + di_minus).replace({0.0: float("nan")}))
    dx = dx.fillna(0.0)
    adx = dx.ewm(alpha=alpha, adjust=False, min_periods=1).mean()

    df2[di_plus_col] = di_plus
    df2[di_minus_col] = di_minus
    df2[adx_col] = adx
    return df2
