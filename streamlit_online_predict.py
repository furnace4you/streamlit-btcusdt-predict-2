#!/usr/bin/env python3
"""Streamlit app for real-time Binance BTCUSDT prediction and 30-minute verification."""

import datetime as dt
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import requests
import streamlit as st

try:
    import xgboost as xgb
except ImportError as exc:  # pragma: no cover
    raise ImportError("xgboost 未安装，请先执行 `pip install xgboost`") from exc

BASE_URL = "https://api3.binance.com/api/v3/klines"
PRICE_URL = "https://api.binance.com/api/v3/ticker/price"
LABEL_MAP = {-1: 0, 0: 1, 1: 2}
INVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}


def fetch_klines(symbol: str, interval: str = "1m", limit: int = 1000) -> list[list[Any]]:
    params = {"symbol": symbol.upper(), "interval": interval, "limit": limit}
    response = requests.get(BASE_URL, params=params, timeout=20)
    response.raise_for_status()
    return response.json()


def fetch_current_price(symbol: str) -> float:
    params = {"symbol": symbol.upper()}
    response = requests.get(PRICE_URL, params=params, timeout=20)
    response.raise_for_status()
    data = response.json()
    return float(data["price"])


def klines_to_df(rows: list[list[Any]]) -> pd.DataFrame:
    columns = [
        "time", "open", "high", "low", "close", "volumefrom",
        "close_time", "volumeto", "trades",
        "buy_base", "buy_quote", "ignore"
    ]
    if not rows:
        return pd.DataFrame(columns=columns)

    df = pd.DataFrame(rows, columns=columns)
    df = df.rename(columns={"volumefrom": "volume", "volumeto": "quote_volume"})
    for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df = df.drop(columns=["close_time", "trades", "buy_base", "buy_quote"])
    df["time_str"] = pd.to_datetime(df["time"], unit="ms").dt.strftime("%Y/%m/%d %H:%M:%S")
    return df


def add_online_features(df: pd.DataFrame) -> pd.DataFrame:
    from features import add_dim1, add_dim2, add_dim3, add_dim4, add_dim5, add_dim9, add_dim12

    work = df.copy()
    work = work.sort_values("time", kind="mergesort").reset_index(drop=True)
    work = add_dim1(work)
    work = add_dim2(work)
    work = add_dim3(work)
    work = add_dim4(work)
    work = add_dim5(work)
    work = add_dim9(work)
    work = add_dim12(work)
    return work


def build_feature_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    work = add_online_features(df)
    excluded = {"time", "time_str", "future_close", "label", "next_high", "next_low", "future_high_31", "future_low_31"}
    feature_cols = [c for c in work.columns if c not in excluded and pd.api.types.is_numeric_dtype(work[c])]
    if not feature_cols:
        raise ValueError("未能生成在线特征列")
    X = work[feature_cols].astype(np.float32)
    X = X.replace([np.inf, -np.inf], np.nan)
    return work, feature_cols


def strict_label_from_history(history: pd.DataFrame, idx: int) -> int:
    if idx + 31 >= len(history):
        return 0
    next_low = history.loc[idx + 1, "low"]
    future_high_31 = history.loc[idx + 31, "high"]
    next_high = history.loc[idx + 1, "high"]
    future_low_31 = history.loc[idx + 31, "low"]

    if next_low > future_high_31:
        return -1
    if next_high < future_low_31:
        return 1
    return 0


def predict_one_row(model: xgb.XGBClassifier, history: pd.DataFrame, idx: int, threshold: float = 0.7) -> dict[str, Any]:
    work, feature_cols = build_feature_frame(history)
    row = work.iloc[[idx]][feature_cols].astype(np.float32)
    proba = model.predict_proba(row)[0]
    pred_index = int(np.argmax(proba))
    model_pred_label = INVERSE_LABEL_MAP[pred_index]
    confidence = float(proba[pred_index])

    pred_label = model_pred_label
    if pred_label in (-1, 1) and confidence < threshold:
        pred_label = 0

    return {
        "idx": idx,
        "time": int(history.loc[idx, "time"]),
        "time_str": history.loc[idx, "time_str"],
        "open": float(history.loc[idx, "open"]),
        "high": float(history.loc[idx, "high"]),
        "low": float(history.loc[idx, "low"]),
        "close": float(history.loc[idx, "close"]),
        "price": float(history.loc[idx, "close"]),
        "pred_label": pred_label,
        "model_pred_label": model_pred_label,
        "confidence": confidence,
        "probabilities": {str(k): float(v) for k, v in zip([-1, 0, 1], proba)},
    }


STATE_FILE = Path(__file__).resolve().parents[0] / "streamlit_online_predict_state.json"

@st.cache_resource
def load_model(model_path: str) -> xgb.XGBClassifier:
    model = xgb.XGBClassifier()
    model.load_model(str(model_path))
    return model


def save_state() -> None:
    if not hasattr(st, "session_state"):
        return
    history_records = None
    if st.session_state.history is not None:
        history_records = st.session_state.history.to_dict(orient="records")

    data = {
        "history_initialized": st.session_state.history_initialized,
        "history": history_records,
        "predictions": st.session_state.predictions,
        "pending": st.session_state.pending,
        "verified": st.session_state.verified,
        "latest_prediction": st.session_state.latest_prediction,
        "last_fetch_time": st.session_state.last_fetch_time,
        "model_path": st.session_state.model_path,
        "symbol": st.session_state.get("symbol"),
        "interval": st.session_state.get("interval"),
        "limit": st.session_state.get("limit"),
        "threshold": st.session_state.get("threshold"),
    }
    with STATE_FILE.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_saved_state() -> Dict[str, Any] | None:
    if not STATE_FILE.exists():
        return None
    try:
        with STATE_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("history") is not None:
            data["history"] = pd.DataFrame(data["history"])
        return data
    except Exception:
        return None


def init_session_state() -> None:
    if "history" not in st.session_state:
        st.session_state.history = None
    if "predictions" not in st.session_state:
        st.session_state.predictions = []
    if "pending" not in st.session_state:
        st.session_state.pending = []
    if "verified" not in st.session_state:
        st.session_state.verified = []
    if "latest_prediction" not in st.session_state:
        st.session_state.latest_prediction = None
    if "last_fetch_time" not in st.session_state:
        st.session_state.last_fetch_time = None
    if "model_path" not in st.session_state:
        st.session_state.model_path = None
    if "model" not in st.session_state:
        st.session_state.model = None
    if "history_initialized" not in st.session_state:
        st.session_state.history_initialized = False


def append_verified_item(item: Dict[str, Any]) -> None:
    st.session_state.verified.insert(0, item)


def update_pending_and_verify(history: pd.DataFrame) -> None:
    idx = len(history) - 1
    to_verify = []
    for item in st.session_state.pending:
        if item["verified"]:
            continue
        if item["idx"] + 31 <= idx:
            actual = strict_label_from_history(history, item["idx"])
            correct = int(item["pred_label"] == actual)
            item["verified"] = True
            item["actual_label"] = actual
            item["correct"] = correct
            item["verify_time"] = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            to_verify.append(item)

    for item in to_verify:
        st.session_state.pending.remove(item)
        append_verified_item(item)


def fetch_history(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    rows = fetch_klines(symbol, interval=interval, limit=limit)
    return klines_to_df(rows)


def append_new_rows(history: pd.DataFrame, latest: pd.DataFrame) -> pd.DataFrame:
    if history is None or history.empty:
        return latest
    history_last = int(history.loc[len(history) - 1, "time"])
    new_rows = latest[latest["time"] > history_last].copy()
    if new_rows.empty:
        return history
    return pd.concat([history, new_rows], ignore_index=True)


def format_prediction(result: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame.from_records([{
        "时间": result["time_str"],
        "开": result["open"],
        "高": result["high"],
        "低": result["low"],
        "收": result["close"],
        "价格": result["price"],
        "模型预测": result["model_pred_label"],
        "阈值决策": result["pred_label"],
        "置信度": result["confidence"],
    }])


def main() -> None:
    st.set_page_config(page_title="BTCUSDT 实时预测", layout="wide")
    st.title("Binance BTCUSDT 实时预测与 30 分钟后验证")

    with st.sidebar:
        st.header("参数设置")
        symbol = st.text_input("交易对", "BTCUSDT")
        interval = st.selectbox("K 线周期", ["1m", "3m", "5m", "15m", "30m"], index=0)
        limit = st.number_input("冷启动 K 线数量", min_value=100, max_value=2000, value=1000, step=100)
        threshold = st.slider("信号置信度阈值", min_value=0.0, max_value=1.0, value=0.7, step=0.01)
        model_path = st.text_input("XGBoost 模型路径", str(Path(__file__).resolve().parents[0] / "xgb_strict_label_model.json"))
        refresh_interval = st.number_input("自动刷新间隔（秒）", min_value=10, max_value=600, value=60, step=10)
        auto_refresh = st.checkbox("启用自动刷新", value=True)
        reset_state = st.button("重新加载历史数据")

    init_session_state()
    if reset_state:
        st.session_state.history = None
        st.session_state.predictions = []
        st.session_state.pending = []
        st.session_state.verified = []
        st.session_state.latest_prediction = None
        st.session_state.last_fetch_time = None
        st.session_state.history_initialized = False
        st.session_state.model_path = None
        st.session_state.model = None
        if STATE_FILE.exists():
            try:
                STATE_FILE.unlink()
            except Exception:
                pass

    saved_state = load_saved_state()
    if saved_state is not None and st.session_state.model_path is None:
        st.session_state.history_initialized = saved_state.get("history_initialized", False)
        st.session_state.history = saved_state.get("history")
        st.session_state.predictions = saved_state.get("predictions", [])
        st.session_state.pending = saved_state.get("pending", [])
        st.session_state.verified = saved_state.get("verified", [])
        st.session_state.latest_prediction = saved_state.get("latest_prediction")
        st.session_state.last_fetch_time = saved_state.get("last_fetch_time")
        st.session_state.model_path = saved_state.get("model_path")
        if saved_state.get("symbol") is not None:
            st.session_state.symbol = saved_state.get("symbol")
        if saved_state.get("interval") is not None:
            st.session_state.interval = saved_state.get("interval")
        if saved_state.get("limit") is not None:
            st.session_state.limit = saved_state.get("limit")
        if saved_state.get("threshold") is not None:
            st.session_state.threshold = saved_state.get("threshold")

    model_path_obj = Path(model_path)
    if not model_path_obj.exists():
        st.error(f"模型文件不存在: {model_path_obj}")
        return

    if st.session_state.model is None or st.session_state.model_path != str(model_path_obj):
        st.session_state.model = load_model(str(model_path_obj))
        st.session_state.model_path = str(model_path_obj)

    st.session_state.symbol = symbol
    st.session_state.interval = interval
    st.session_state.limit = limit
    st.session_state.threshold = threshold

    if auto_refresh:
        st.markdown(f"<meta http-equiv='refresh' content='{refresh_interval}'>", unsafe_allow_html=True)

    force_refresh = st.button("刷新")
    if force_refresh:
        st.rerun()

    try:
        if not st.session_state.history_initialized:
            st.info(f"冷启动：拉取最近 {limit} 条 {interval} K 线...")
            st.session_state.history = fetch_history(symbol, interval, limit)
            st.session_state.last_fetch_time = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.history_initialized = True
            if not st.session_state.history.empty:
                idx = len(st.session_state.history) - 1
                result = predict_one_row(st.session_state.model, st.session_state.history, idx, threshold=threshold)
                try:
                    result["price"] = fetch_current_price(symbol)
                except Exception:
                    result["price"] = float(st.session_state.history.loc[idx, "close"])
                st.session_state.latest_prediction = result
                st.session_state.predictions.insert(0, result)
                if result["model_pred_label"] != 0:
                    st.session_state.pending.append({
                        **result,
                        "verified": False,
                        "actual_label": None,
                        "correct": None,
                        "verify_time": None,
                    })
                save_state()
        else:
            latest = fetch_history(symbol, interval, 5)
            updated = append_new_rows(st.session_state.history, latest)
            if len(updated) > len(st.session_state.history):
                st.session_state.history = updated
                st.session_state.last_fetch_time = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
                idx = len(st.session_state.history) - 1
                result = predict_one_row(st.session_state.model, st.session_state.history, idx, threshold=threshold)
                try:
                    result["price"] = fetch_current_price(symbol)
                except Exception:
                    result["price"] = float(st.session_state.history.loc[idx, "close"])
                st.session_state.latest_prediction = result
                st.session_state.predictions.insert(0, result)
                if result["model_pred_label"] != 0:
                    st.session_state.pending.append({
                        **result,
                        "verified": False,
                        "actual_label": None,
                        "correct": None,
                        "verify_time": None,
                    })
            else:
                st.info("暂无新 K 线，等待下一次刷新")

        if st.session_state.history is not None and not st.session_state.history.empty:
            update_pending_and_verify(st.session_state.history)
            save_state()

    except Exception as exc:
        st.error(f"预测过程出错: {exc}")
        return

    with st.expander("当前状态", expanded=True):
        st.write(f"符号: {symbol}")
        st.write(f"周期: {interval}")
        st.write(f"冷启动 K 线数量: {limit}")
        st.write(f"最后刷新时间: {st.session_state.last_fetch_time}")
        st.write(f"历史 K 线条数: {len(st.session_state.history) if st.session_state.history is not None else 0}")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("最新预测")
        if st.session_state.latest_prediction:
            st.table(format_prediction(st.session_state.latest_prediction))
            st.write("概率分布:", st.session_state.latest_prediction["probabilities"])
        else:
            st.info("等待第一条预测结果")

    with col2:
        st.subheader("预测历史（滚动）")
        if st.session_state.predictions:
            st.table(pd.DataFrame(st.session_state.predictions)[[
                "time_str", "model_pred_label", "pred_label", "confidence", "open", "high", "low", "close"
            ]].rename(columns={
                "time_str": "时间",
                "model_pred_label": "模型预测",
                "pred_label": "阈值决策",
                "confidence": "置信度",
                "open": "开",
                "high": "高",
                "low": "低",
                "close": "收",
            }))
        else:
            st.info("尚无预测历史")

    st.subheader("未验证预测")
    if st.session_state.pending:
        st.table(pd.DataFrame(st.session_state.pending)[[
            "time_str", "model_pred_label", "pred_label", "confidence", "open", "high", "low", "close"
        ]].rename(columns={
            "time_str": "时间",
            "model_pred_label": "模型预测",
            "pred_label": "阈值决策",
            "confidence": "置信度",
            "open": "开",
            "high": "高",
            "low": "低",
            "close": "收",
        }))
    else:
        st.info("当前无待验证预测")

    st.subheader("已验证结果")
    if st.session_state.verified:
        st.table(pd.DataFrame(st.session_state.verified)[[
            "time_str", "model_pred_label", "pred_label", "actual_label", "correct", "confidence", "verify_time"
        ]].rename(columns={
            "time_str": "时间",
            "model_pred_label": "模型预测",
            "pred_label": "阈值决策",
            "actual_label": "实际标签",
            "correct": "是否正确",
            "confidence": "置信度",
            "verify_time": "验证时间",
        }))
    else:
        st.info("尚无验证结果")

    st.caption("本页面会基于最新 K 线进行预测，并在 30 分钟后自动验证模型预测结果。")


if __name__ == "__main__":
    main()
