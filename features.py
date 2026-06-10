#!/usr/bin/env python3
"""
BTC 10分钟方向预测 — 特征全集实现
对应 btc_feature_universe.pdf，覆盖维度 1~5、9、12

输入: btcusdt_1m_24h.csv  (open_time, open, high, low, close, volume, ...)
输出: btcusdt_features.csv / btcusdt_features.parquet

维度说明（本文件实现的）:
  Dim1  价格动量与趋势
  Dim2  波动率与regime
  Dim3  成交量微观结构（无 taker 数据时跳过 3.3/3.4）
  Dim4  K线形态微观结构
  Dim5  支撑阻力与价格位置
  Dim9  时间/日历特征
  Dim12 统计/信息论特征
"""

import math
import warnings
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, entropy as scipy_entropy
from scipy.signal import find_peaks

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════════

def safe_div(a, b, fill=np.nan):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = np.where(b != 0, a / b, fill)
    return result


def rolling_linreg(series: pd.Series, window: int):
    """返回 (slope, r2) 两个 Series"""
    n = len(series)
    slopes = np.full(n, np.nan)
    r2s = np.full(n, np.nan)
    arr = series.values.astype(float)
    x = np.arange(window, dtype=float)
    xm = x.mean()
    ss_x = ((x - xm) ** 2).sum()
    for i in range(window - 1, n):
        y = arr[i - window + 1: i + 1]
        if np.isnan(y).any():
            continue
        ym = y.mean()
        ss_xy = ((x - xm) * (y - ym)).sum()
        ss_y = ((y - ym) ** 2).sum()
        slope = ss_xy / ss_x
        r2 = (ss_xy ** 2 / (ss_x * ss_y)) if ss_y > 0 else 0.0
        slopes[i] = slope
        r2s[i] = r2
    return pd.Series(slopes, index=series.index), pd.Series(r2s, index=series.index)


def rolling_adx(high, low, close, window=14):
    """手工实现 ADX"""
    h, l, c = high.values, low.values, close.values
    n = len(c)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    for i in range(1, n):
        up = h[i] - h[i - 1]
        dn = l[i - 1] - l[i]
        plus_dm[i] = up if up > dn and up > 0 else 0
        minus_dm[i] = dn if dn > up and dn > 0 else 0
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))

    def ema_wilder(arr, w):
        out = np.zeros(len(arr))
        out[w - 1] = arr[:w].sum()
        for i in range(w, len(arr)):
            out[i] = out[i - 1] - out[i - 1] / w + arr[i]
        return out

    atr_w = ema_wilder(tr, window)
    pdm_w = ema_wilder(plus_dm, window)
    mdm_w = ema_wilder(minus_dm, window)
    with np.errstate(divide='ignore', invalid='ignore'):
        pdi = np.where(atr_w > 0, 100 * pdm_w / atr_w, 0)
        mdi = np.where(atr_w > 0, 100 * mdm_w / atr_w, 0)
        dx = np.where((pdi + mdi) > 0, 100 * np.abs(pdi - mdi) / (pdi + mdi), 0)
    adx = ema_wilder(dx, window)
    return pd.Series(adx, index=close.index)


def hurst_exponent(series_arr, lags_range=range(2, 20)):
    """R/S 法估计 Hurst 指数，series_arr 为 1D numpy array"""
    if len(series_arr) < 30 or np.std(series_arr) == 0:
        return np.nan
    rs_vals, lag_vals = [], []
    for lag in lags_range:
        if lag >= len(series_arr):
            break
        sub = series_arr[:lag]
        mean = sub.mean()
        dev = np.cumsum(sub - mean)
        r = dev.max() - dev.min()
        s = sub.std()
        if s > 0:
            rs_vals.append(r / s)
            lag_vals.append(lag)
    if len(rs_vals) < 2:
        return np.nan
    log_rs = np.log(rs_vals)
    log_lag = np.log(lag_vals)
    h, _ = np.polyfit(log_lag, log_rs, 1)
    return h


def rolling_hurst(series: pd.Series, window: int = 60):
    arr = series.values.astype(float)
    n = len(arr)
    out = np.full(n, np.nan)
    for i in range(window - 1, n):
        out[i] = hurst_exponent(arr[i - window + 1: i + 1])
    return pd.Series(out, index=series.index)


# ═══════════════════════════════════════════════════════════════════
# 维度1: 价格动量与趋势
# ═══════════════════════════════════════════════════════════════════

def add_dim1(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]
    high = df["high"]
    low = df["low"]

    # 1.1 多周期 return
    for n in [1, 3, 5, 10, 20, 60, 120]:
        df[f"ret_{n}m"] = close.pct_change(n)

    # 1.2 动量加速度
    df["ret_diff"] = df["ret_5m"] - df["ret_10m"]
    df["ret_ratio"] = safe_div(df["ret_3m"].values, df["ret_10m"].values)

    # 1.3 趋势强度
    for n in [14, 20]:
        df[f"adx_{n}"] = rolling_adx(high, low, close, n)

    for n in [5, 10, 20]:
        df[f"trend_consistency_{n}"] = (
            (close.diff() > 0).rolling(n).mean()
        )
        slope, r2 = rolling_linreg(close, n)
        df[f"linear_reg_slope_{n}"] = slope / close  # 归一化
        df[f"linear_reg_r2_{n}"] = r2

    # 1.4 均线系统
    for n in [5, 10, 20, 60]:
        df[f"ma_{n}"] = close.rolling(n).mean()

    df["ma_diff_fast_slow"] = safe_div(
        (df["ma_5"] - df["ma_20"]).values, df["ma_20"].values
    )
    for n in [5, 10, 20]:
        df[f"ma_slope_{n}"] = df[f"ma_{n}"].diff()
        df[f"price_vs_ma_{n}"] = safe_div(
            (close - df[f"ma_{n}"]).values, df[f"ma_{n}"].values
        )

    # ma_alignment: MA5>MA10>MA20=1, MA5<MA10<MA20=-1, else 0
    df["ma_alignment"] = np.select(
        [
            (df["ma_5"] > df["ma_10"]) & (df["ma_10"] > df["ma_20"]),
            (df["ma_5"] < df["ma_10"]) & (df["ma_10"] < df["ma_20"]),
        ],
        [1, -1],
        default=0,
    )

    # 1.5 MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["macd_line"] = ema12 - ema26
    df["signal_line"] = df["macd_line"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd_line"] - df["signal_line"]
    df["macd_hist_diff"] = df["macd_hist"].diff()

    # 金叉死叉距离
    cross = np.sign(df["macd_hist"])
    cross_events = (cross != cross.shift(1)) & cross.notna()
    cross_idx = np.where(cross_events)[0]
    dist = np.full(len(df), np.nan)
    for pos in cross_idx:
        end = cross_idx[cross_idx > pos][0] if cross_idx[cross_idx > pos].size else len(df)
        dist[pos:end] = np.arange(0, end - pos)
    df["macd_cross_distance"] = dist

    return df


# ═══════════════════════════════════════════════════════════════════
# 维度2: 波动率与 regime
# ═══════════════════════════════════════════════════════════════════

def add_dim2(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]
    high = df["high"]
    low = df["low"]
    ret = close.pct_change()

    # 2.1 已实现波动率
    for n in [5, 10, 20, 60]:
        df[f"rvol_{n}"] = ret.rolling(n).std()

    df["rvol_ratio"] = safe_div(df["rvol_5"].values, df["rvol_20"].values)
    df["rvol_percentile"] = df["rvol_20"].rolling(240).rank(pct=True)

    # 2.2 ATR
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    for n in [5, 14, 20]:
        df[f"atr_{n}"] = tr.ewm(span=n, adjust=False).mean()
        df[f"natr_{n}"] = safe_div(df[f"atr_{n}"].values, close.values)

    df["atr_ratio"] = safe_div(df["atr_5"].values, df["atr_20"].values)

    # 2.3 波动率变化率
    df["rvol_acceleration"] = df["rvol_5"].diff()
    slope_atr, _ = rolling_linreg(df["atr_5"], 3)
    df["atr_slope"] = slope_atr

    # 2.4 布林带
    bb_n = 20
    bb_std = 2
    bb_mid = close.rolling(bb_n).mean()
    bb_std_val = close.rolling(bb_n).std()
    bb_upper = bb_mid + bb_std * bb_std_val
    bb_lower = bb_mid - bb_std * bb_std_val
    bb_range = bb_upper - bb_lower

    df["bb_position"] = safe_div((close - bb_lower).values, bb_range.values)
    df["bb_width"] = safe_div(bb_range.values, bb_mid.values)
    df["bb_width_change"] = df["bb_width"].diff()
    df["bb_squeeze"] = df["bb_width"].rolling(60).rank(pct=True)

    return df


# ═══════════════════════════════════════════════════════════════════
# 维度3: 成交量微观结构
# ═══════════════════════════════════════════════════════════════════

def add_dim3(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]
    vol = df["volume"]
    ret = close.pct_change()

    # 3.1 成交量基础
    for n in [3, 5, 10, 20]:
        df[f"vol_{n}"] = vol.rolling(n).mean()

    df["vol_ratio"] = safe_div(df["vol_3"].values, df["vol_20"].values)
    df["vol_spike"] = safe_div(vol.values, df["vol_20"].values)

    slope_vol, _ = rolling_linreg(vol, 10)
    df["vol_trend"] = slope_vol

    # 3.2 量价关系
    for n in [5, 10, 20]:
        df[f"vol_price_corr_{n}"] = (
            vol.rolling(n).corr(ret.abs())
        )
        vwr = (vol * ret).rolling(n).sum() / vol.rolling(n).sum()
        df[f"volume_weighted_return_{n}"] = vwr

        up_mask = ret > 0
        dn_mask = ret < 0
        up_vol = (vol * up_mask).rolling(n).sum()
        dn_vol = (vol * dn_mask).rolling(n).sum()
        df[f"up_vol_ratio_{n}"] = safe_div(up_vol.values, dn_vol.values)

    # 3.3 Taker (CryptoCompare 无 taker 字段，用代理：上涨 bar 视为 buy)
    # 若有真实 taker 数据可替换
    proxy_taker_buy = vol * (ret > 0).astype(float)
    proxy_taker_sell = vol * (ret <= 0).astype(float)
    df["taker_buy_ratio"] = safe_div(proxy_taker_buy.values, vol.values)
    for n in [3, 10]:
        df[f"taker_buy_ratio_ma_{n}"] = df["taker_buy_ratio"].rolling(n).mean()
    df["taker_buy_ratio_change"] = (
        df["taker_buy_ratio_ma_3"] - df["taker_buy_ratio_ma_10"]
    )
    for n in [5, 10, 20]:
        net_flow = (proxy_taker_buy - proxy_taker_sell).rolling(n).sum()
        df[f"net_taker_flow_{n}"] = safe_div(net_flow.values, df[f"vol_{n}"].values)

    imbal = (proxy_taker_buy - proxy_taker_sell).abs()
    df["taker_imbalance_intensity"] = safe_div(imbal.values, vol.values)

    return df


# ═══════════════════════════════════════════════════════════════════
# 维度4: K线形态微观结构
# ═══════════════════════════════════════════════════════════════════

def add_dim4(df: pd.DataFrame) -> pd.DataFrame:
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    total_range = h - l

    # 4.1 K线内部结构
    df["body_ratio"] = safe_div((c - o).abs().values, total_range.values)
    df["upper_shadow_ratio"] = safe_div(
        (h - pd.concat([o, c], axis=1).max(axis=1)).values,
        total_range.values,
    )
    df["lower_shadow_ratio"] = safe_div(
        (pd.concat([o, c], axis=1).min(axis=1) - l).values,
        total_range.values,
    )
    df["shadow_imbalance"] = df["upper_shadow_ratio"] - df["lower_shadow_ratio"]

    # 4.2 K线形态序列
    for n in [5, 10, 20]:
        slope_br, _ = rolling_linreg(df["body_ratio"], n)
        df[f"body_ratio_trend_{n}"] = slope_br
        df[f"upper_shadow_ma_{n}"] = df["upper_shadow_ratio"].rolling(n).mean()

        # 连续同向最大长度
        direction = np.sign(c - o)
        def max_consec(w):
            mx = 0
            cur = 0
            ref = w.iloc[0]
            for v in w:
                if v == ref and v != 0:
                    cur += 1
                    mx = max(mx, cur)
                else:
                    cur = 1
                    ref = v
            return mx
        df[f"direction_consistency_{n}"] = (
            direction.rolling(n).apply(max_consec, raw=False)
        )

        df[f"doji_count_{n}"] = (
            (df["body_ratio"] < 0.2).rolling(n).sum()
        )

    # 4.3 K线能量特征
    ret_each = c.pct_change()
    for n in [5, 10, 20]:
        net_ret = c.pct_change(n)
        sum_abs = ret_each.abs().rolling(n).sum()
        df[f"candle_efficiency_{n}"] = safe_div(net_ret.values, sum_abs.values)

        slope_range, _ = rolling_linreg(total_range, n)
        slope_body, _ = rolling_linreg((c - o).abs(), n)
        df[f"range_body_divergence_{n}"] = (
            (slope_range > 0).astype(int) - (slope_body > 0).astype(int)
        )

    return df


# ═══════════════════════════════════════════════════════════════════
# 维度5: 支撑阻力与价格位置
# ═══════════════════════════════════════════════════════════════════

def add_dim5(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]
    high = df["high"]
    low = df["low"]
    atr14 = df["atr_14"]

    # 5.1 动态支撑阻力
    for n in [20, 60, 120, 288]:
        hh = high.rolling(n).max()
        ll = low.rolling(n).min()
        rng = hh - ll
        df[f"dist_to_high_{n}"] = safe_div((close - hh).values, atr14.values)
        df[f"dist_to_low_{n}"] = safe_div((close - ll).values, atr14.values)
        df[f"range_position_{n}"] = safe_div((close - ll).values, rng.values)

    # 5.2 关键价格水平
    # 距最近整数关口（万位）
    round_levels = np.array([
        round(p / 1000) * 1000 for p in close.values
    ], dtype=float)
    df["dist_to_round_number"] = safe_div(
        (close.values - round_levels), atr14.values
    )

    # 日内 UTC 0:00 开盘价
    ts = pd.to_datetime(df["time_str"])
    daily_open = (
        df.assign(_date=ts.dt.date)
        .groupby("_date")["open"]
        .transform("first")
    )
    df["dist_to_daily_open"] = safe_div(
        (close - daily_open).values, atr14.values
    )

    # 前一交易日收盘价
    daily_close = (
        df.assign(_date=ts.dt.date)
        .groupby("_date")["close"]
        .transform("last")
    ).shift(1)  # 前一天
    df["dist_to_prev_daily_close"] = safe_div(
        (close - daily_close).values, atr14.values
    )

    # VWAP（当日累计）
    typical = (high + low + close) / 3
    cum_tp_vol = (typical * df["volume"]).groupby(ts.dt.date).cumsum()
    cum_vol = df["volume"].groupby(ts.dt.date).cumsum()
    vwap = cum_tp_vol / cum_vol
    df["vwap"] = vwap
    df["dist_to_vwap"] = safe_div((close - vwap).values, atr14.values)

    # 5.3 突破
    for n in [20, 60]:
        hh = high.rolling(n).max().shift(1)
        ll = low.rolling(n).min().shift(1)
        df[f"breakout_{n}"] = np.select(
            [close > hh, close < ll], [1, -1], default=0
        )
        # 突破后经过的 K 线数
        bo = df[f"breakout_{n}"].values
        age = np.zeros(len(bo))
        cnt = 0
        for i in range(len(bo)):
            if bo[i] != 0:
                cnt += 1
            else:
                cnt = 0
            age[i] = cnt
        df[f"breakout_age_{n}"] = age

        # 假突破：突破后 3 根内回落
        def false_bo(w):
            count = 0
            for i in range(1, len(w)):
                if w.iloc[i - 1] != 0 and w.iloc[i] == 0:
                    count += 1
            return count
        df[f"false_breakout_count_{n}"] = (
            df[f"breakout_{n}"].rolling(20).apply(false_bo, raw=False)
        )

    return df


# ═══════════════════════════════════════════════════════════════════
# 维度9: 时间/日历特征
# ═══════════════════════════════════════════════════════════════════

def add_dim9(df: pd.DataFrame) -> pd.DataFrame:
    ts = pd.to_datetime(df["time_str"])
    utc_ts = ts.dt.tz_localize(None)  # 视为 UTC

    hour = utc_ts.dt.hour + utc_ts.dt.minute / 60.0
    dow = utc_ts.dt.dayofweek.astype(float)

    # 9.1 周期性编码
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)

    # Session: 亚盘 1-9 UTC, 欧盘 7-16 UTC, 美盘 13-21 UTC（重叠算欧美）
    h_int = utc_ts.dt.hour
    session_id = np.select(
        [
            (h_int >= 13) & (h_int < 21),   # 美盘
            (h_int >= 7) & (h_int < 16),    # 欧盘
            (h_int >= 1) & (h_int < 9),     # 亚盘
        ],
        [2, 1, 0],
        default=3,
    )
    df["session_id"] = session_id

    # Session 边界（近似）
    session_start_hour = np.select(
        [session_id == 2, session_id == 1, session_id == 0],
        [13, 7, 1],
        default=0,
    ).astype(float)
    session_duration = np.select(
        [session_id == 2, session_id == 1, session_id == 0],
        [8, 9, 8],
        default=6,
    ).astype(float)
    minutes_in_session = (h_int.values - session_start_hour) * 60 + utc_ts.dt.minute.values
    df["session_progress"] = np.clip(minutes_in_session / (session_duration * 60), 0, 1)
    df["minute_of_session"] = np.clip(minutes_in_session.astype(int), 0, None)

    # 9.2 CME 窗口 (CME 开盘 14:30 UTC, 收盘 21:00 UTC)
    total_min = h_int.values * 60 + utc_ts.dt.minute.values
    cme_open_min = 14 * 60 + 30
    cme_close_min = 21 * 60
    df["is_cme_open"] = (np.abs(total_min - cme_open_min) <= 30).astype(int)
    df["is_cme_close"] = (np.abs(total_min - cme_close_min) <= 30).astype(int)
    df["is_weekend"] = (utc_ts.dt.dayofweek >= 5).astype(int)

    # Funding 每 8 小时一次（UTC 0, 8, 16 点）
    funding_mins = np.array([0, 8 * 60, 16 * 60])
    dist_to_funding = np.min(
        np.abs(total_min[:, None] - funding_mins[None, :]), axis=1
    )
    df["is_funding_window"] = (dist_to_funding <= 10).astype(int)
    df["time_to_next_funding"] = np.min(
        np.mod(funding_mins[None, :] - total_min[:, None], 480), axis=1
    )

    return df


# ═══════════════════════════════════════════════════════════════════
# 维度12: 统计/信息论特征
# ═══════════════════════════════════════════════════════════════════

def add_dim12(df: pd.DataFrame) -> pd.DataFrame:
    ret = df["close"].pct_change()

    # 12.1 分布特征
    for n in [20, 60, 120]:
        df[f"return_skewness_{n}"] = ret.rolling(n).apply(
            lambda x: skew(x, nan_policy="omit") if len(x) >= 4 else np.nan,
            raw=True,
        )
        df[f"return_kurtosis_{n}"] = ret.rolling(n).apply(
            lambda x: kurtosis(x, nan_policy="omit") if len(x) >= 4 else np.nan,
            raw=True,
        )

        def entropy_fn(x):
            x = x[~np.isnan(x)]
            if len(x) < 4:
                return np.nan
            counts, _ = np.histogram(x, bins=10)
            counts = counts[counts > 0].astype(float)
            probs = counts / counts.sum()
            return -(probs * np.log(probs + 1e-12)).sum()

        df[f"return_entropy_{n}"] = ret.rolling(n).apply(entropy_fn, raw=True)

    # 12.2 自相关特征
    for n in [30, 60, 120]:
        df[f"return_autocorr_lag1_{n}"] = ret.rolling(n).apply(
            lambda x: pd.Series(x).autocorr(lag=1) if len(x) >= 3 else np.nan,
            raw=True,
        )
        df[f"return_autocorr_lag2_{n}"] = ret.rolling(n).apply(
            lambda x: pd.Series(x).autocorr(lag=2) if len(x) >= 4 else np.nan,
            raw=True,
        )

    df["hurst_60"] = rolling_hurst(ret.fillna(0), window=60)
    df["hurst_120"] = rolling_hurst(ret.fillna(0), window=120)

    # 12.3 regime 分类（基于 rvol 分位数）
    rvol20 = df["rvol_20"]
    p33 = rvol20.rolling(240).quantile(0.33)
    p66 = rvol20.rolling(240).quantile(0.66)
    df["vol_regime"] = np.select(
        [rvol20 <= p33, rvol20 <= p66],
        [0, 1],
        default=2,
    )  # 0=低vol, 1=中vol, 2=高vol

    # trend_regime: 基于 linear_reg_r2_20 和 adx_14
    df["trend_regime"] = np.select(
        [
            (df["linear_reg_r2_20"] > 0.7) & (df["adx_14"] > 25),
            (df["linear_reg_r2_20"] < 0.3) | (df["adx_14"] < 15),
        ],
        [1, -1],
        default=0,
    )  # 1=趋势, -1=震荡, 0=中间

    return df


# ═══════════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════════

def build_features(csv_path: str) -> pd.DataFrame:
    print(f"读取数据: {csv_path}")
    df = pd.read_csv(csv_path)
    df = df.rename(columns={
        "volumefrom": "volume",
        "volumeto": "quote_volume",
    })
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    print(f"原始行数: {len(df)}")

    print("计算维度1: 价格动量与趋势...")
    df = add_dim1(df)
    print("计算维度2: 波动率与regime...")
    df = add_dim2(df)
    print("计算维度3: 成交量微观结构...")
    df = add_dim3(df)
    print("计算维度4: K线形态微观结构...")
    df = add_dim4(df)
    print("计算维度5: 支撑阻力与价格位置...")
    df = add_dim5(df)
    print("计算维度9: 时间/日历特征...")
    df = add_dim9(df)
    print("计算维度12: 统计/信息论特征...")
    df = add_dim12(df)

    # 删除辅助列
    for col in [c for c in df.columns if c.startswith("ma_") and c in ["ma_5","ma_10","ma_20","ma_60"]]:
        pass  # 保留 MA 列供参考

    print(f"\n特征构造完成，总列数: {len(df.columns)}")
    print(f"有效行数（去掉预热期 NaN）: {df.dropna(thresh=len(df.columns)//2).shape[0]}")
    return df


# def main():
#     df = build_features("/content/drive/MyDrive/binace_date_std/BTCUSDT_2026_05_1m.csv")

#     # 输出特征列表
#     feat_cols = [c for c in df.columns if c not in
#                  {"time", "time_str", "open", "high", "low", "close",
#                   "volume", "quote_volume"}]
#     print(f"\n特征总数: {len(feat_cols)}")
#     for i, c in enumerate(feat_cols, 1):
#         print(f"  {i:>3}. {c}")

#     # 保存
#     out_csv = "btcusdt_features_30d.csv"
#     df.to_csv(out_csv, index=False)
#     print(f"\n已保存 → {out_csv}")

#     # 打印最后一行非 NaN 特征的数值（最新一根 K 线的特征）
#     last_valid = df[feat_cols].dropna(how="all").iloc[-1]
#     print("\n=== 最新K线特征值（样例）===")
#     for col in feat_cols[:30]:
#         v = last_valid.get(col, np.nan)
#         print(f"  {col:<40} {v:.6f}" if not pd.isna(v) else f"  {col:<40} NaN")


# if __name__ == "__main__":
#     main()
