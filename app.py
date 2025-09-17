# TSLA Strategy Dashboard (Colab-stable)
# - Live data via yfinance (falls back to synthetic)
# - Robust normalization (guarantees 'Date' + 'Close' before use)
# - Streamlit native charts (reliable in Colab)

import io
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="TSLA Strategy Dashboard", layout="wide")

# ---------- helpers ----------
def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    d = series.diff()
    up = d.clip(lower=0).rolling(period).mean()
    dn = (-d.clip(upper=0)).rolling(period).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    m = ema_fast - ema_slow
    s = m.ewm(span=signal, adjust=False).mean()
    h = m - s
    return m, s, h

def normalize_df(df_in: pd.DataFrame) -> pd.DataFrame | None:
    """Return df with at least Date & Close columns. Handles MultiIndex from yfinance."""
    df = df_in.copy()

    # 1) Flatten MultiIndex columns (e.g., ('Close','TSLA') -> 'Close_TSLA')
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            "_".join([str(x) for x in tup if str(x)]).strip()
            for tup in df.columns
        ]

    # 2) Helper: find first column whose normalized name starts with any prefix
    def find_col(prefixes):
        for c in df.columns:
            key = str(c).replace(" ", "").replace("_", "").lower()
            for p in prefixes:
                if key.startswith(p):
                    return c
        return None

    # 3) Make sure Date is a *column*
    date_c = find_col(["date", "datetime", "timestamp"])
    if date_c is None and getattr(df.index, "name", None) is not None:
        df = df.reset_index()
        date_c = find_col(["date", "datetime", "timestamp", "index"])

    # 4) Find price/volume columns (accepts Close_TSLA, AdjClose_TSLA, etc.)
    close_c = find_col(["close", "adjclose"])
    open_c  = find_col(["open"])
    high_c  = find_col(["high"])
    low_c   = find_col(["low"])
    vol_c   = find_col(["volume", "vol"])

    # 5) Rename to standard names
    rename_map = {}
    if date_c:  rename_map[date_c]  = "Date"
    if close_c: rename_map[close_c] = "Close"
    if open_c:  rename_map[open_c]  = "Open"
    if high_c:  rename_map[high_c]  = "High"
    if low_c:   rename_map[low_c]   = "Low"
    if vol_c:   rename_map[vol_c]   = "Volume"
    df = df.rename(columns=rename_map)

    # 6) Bail out if still missing
    if "Date" not in df.columns or "Close" not in df.columns:
        return None

    # 7) Coerce types and clean
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for c in ["Open","High","Low","Close","Volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["Date","Close"]).sort_values("Date").reset_index(drop=True)
    return df


# ---------- sidebar (position only for now; we can add ladders later) ----------
st.sidebar.header("Position")
shares_held = st.sidebar.number_input("Shares held", 1, step=1, value=220)
cost_basis  = st.sidebar.number_input("Cost basis (USD)", min_value=0.0, step=0.1, value=400.0, format="%.2f")
use_live    = st.sidebar.checkbox("Use live TSLA data (yfinance)", value=True)
uploaded    = st.sidebar.file_uploader("Or upload CSV (Date,Open,High,Low,Close,Volume)", type=["csv"])

# ---------- data load ----------
df_raw = None
if uploaded is not None:
    try:
        df_raw = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()

if df_raw is None and use_live:
    try:
        import yfinance as yf
        raw = yf.download("TSLA", period="2y", interval="1d", auto_adjust=False, progress=False)
        if not raw.empty:
            df_raw = raw.reset_index()
    except Exception as e:
        df_raw = None

# Fallback synthetic so the app always runs
if df_raw is None:
    dates = pd.bdate_range(end=pd.Timestamp.today(), periods=250)
    price = 400.0
    rng = np.random.default_rng(42)
    path = price * (1 + rng.normal(0, 0.015, size=len(dates))).cumprod() / (1 + 0* rng.normal())
    df_raw = pd.DataFrame({"Date": dates, "Close": path})
    df_raw["Open"] = df_raw["Close"].shift(1).fillna(df_raw["Close"])
    df_raw["High"] = df_raw[["Open","Close"]].max(axis=1) * 1.01
    df_raw["Low"]  = df_raw[["Open","Close"]].min(axis=1) * 0.99
    df_raw["Volume"] = 1e8

# ---------- normalize (prevents KeyError) ----------
st.caption("Columns BEFORE normalize: " + ", ".join(map(str, df_raw.columns)))
df = normalize_df(df_raw)
if df is None:
    st.error("Data is missing 'Date'/'Close' even after normalization. "
             f"Columns now: {list(df_raw.columns)}. "
             "Please upload a CSV with Date, Close (and optionally OHLC/Volume).")
    st.dataframe(df_raw.head(), use_container_width=True)
    st.stop()

# Drop today's partial bar if present
if len(df) and df["Date"].iloc[-1].date() >= pd.Timestamp.now(tz="America/New_York").date():
    if len(df) > 1:
        df = df.iloc[:-1].reset_index(drop=True)

# ---------- indicators ----------
df["MA20"]  = df["Close"].rolling(20).mean()
df["MA50"]  = df["Close"].rolling(50).mean()
df["MA200"] = df["Close"].rolling(200).mean()
df["RSI14"] = rsi(df["Close"], 14)
df["MACD"], df["MACD_signal"], df["MACD_hist"] = macd(df["Close"])

# ---------- metrics ----------
last_close = float(pd.to_numeric(df["Close"], errors="coerce").dropna().iloc[-1])
st.title("TSLA Strategy Dashboard")
c1,c2,c3,c4 = st.columns(4)
c1.metric("Last Close (from data)", f"${last_close:,.2f}")
c2.metric("Shares Held", f"{int(shares_held)}")
c3.metric("Cost Basis", f"${float(cost_basis):,.2f}")
c4.metric("Unrealized P/L", f"${(last_close - float(cost_basis))*float(shares_held):,.2f}")

st.markdown("---")

# ---------- charts (Streamlit-native) ----------
st.subheader("Price + Moving Averages")
st.line_chart(df.set_index("Date")[["Close","MA20","MA50","MA200"]])

st.subheader("RSI(14)")
st.line_chart(df.set_index("Date")[["RSI14"]])

st.subheader("MACD")
st.line_chart(df.set_index("Date")[["MACD","MACD_signal"]])
st.bar_chart(df.set_index("Date")[["MACD_hist"]])

st.caption("If you still see errors, check the 'Columns BEFORE normalize' line above and share it.")
