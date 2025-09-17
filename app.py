# TSLA Strategy Dashboard (Streamlit / Cloud-friendly)
# - Live data via yfinance (falls back to synthetic)
# - Robust normalization (guarantees 'Date' + 'Close')
# - Fresh daily close override (and optional last-minute price)
# - Streamlit-native charts

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="TSLA Strategy Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================
# Helpers
# =========================
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

    # 3) Ensure Date is a column (yfinance often uses index)
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

    # 6) Bail if still missing
    if "Date" not in df.columns or "Close" not in df.columns:
        return None

    # 7) Types & clean
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for c in ["Open","High","Low","Close","Volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Date","Close"]).sort_values("Date").reset_index(drop=True)

    return df

# =========================
# Sidebar controls
# =========================
st.sidebar.header("Position")
shares_held = st.sidebar.number_input("Shares held", 1, step=1, value=220)
cost_basis  = st.sidebar.number_input("Cost basis (USD)", min_value=0.0, step=0.1, value=400.0, format="%.2f")

st.sidebar.header("Data")
use_live       = st.sidebar.checkbox("Use live TSLA data (yfinance)", value=True)
include_today  = st.sidebar.checkbox("Include today's bar (if available)", value=True)
live_override  = st.sidebar.checkbox("Use last-minute price in metric", value=False)

uploaded = st.sidebar.file_uploader("Or upload CSV (Date,Open,High,Low,Close,Volume)", type=["csv"])
if st.sidebar.button("Refresh data"):
    st.experimental_rerun()

# =========================
# Data load
# =========================
df_raw = None

# 1) CSV
if uploaded is not None:
    try:
        df_raw = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()

# 2) Live via yfinance
if df_raw is None and use_live:
    try:
        import yfinance as yf
        raw = yf.download(
            "TSLA",
            period="5y",
            interval="1d",
            auto_adjust=True,     # adjusted prices (handles splits/dividends)
            actions=False,
            progress=False,
            threads=True,
        )
        if not raw.empty:
            raw = raw[raw["Volume"] > 0]        # drop empty/holiday bars
            df_raw = raw.reset_index()          # Date index -> column
    except Exception:
        df_raw = None

# 3) Fallback synthetic so the app always runs
if df_raw is None:
    dates = pd.bdate_range(end=pd.Timestamp.today(), periods=250)
    price = 400.0
    rng = np.random.default_rng(42)
    path = price * (1 + rng.normal(0, 0.015, size=len(dates))).cumprod()
    df_raw = pd.DataFrame({"Date": dates, "Close": path})
    df_raw["Open"] = df_raw["Close"].shift(1).fillna(df_raw["Close"])
    df_raw["High"] = df_raw[["Open","Close"]].max(axis=1) * 1.01
    df_raw["Low"]  = df_raw[["Open","Close"]].min(axis=1) * 0.99
    df_raw["Volume"] = 1e8

# Normalize for robust Date/Close columns
st.caption("Columns BEFORE normalize: " + ", ".join(map(str, df_raw.columns)))
df = normalize_df(df_raw)
if df is None:
    st.error(
        "Data is missing 'Date'/'Close' even after normalization. "
        f"Columns now: {list(df_raw.columns)}. "
        "Please upload a CSV with Date, Close (and optionally OHLC/Volume)."
    )
    st.dataframe(df_raw.head(), use_container_width=True)
    st.stop()

# Keep/skip today's bar
ny_now = pd.Timestamp.now(tz="America/New_York")
if not include_today and len(df) and df["Date"].iloc[-1].date() == ny_now.date():
    if len(df) > 1:
        df = df.iloc[:-1].reset_index(drop=True)

# =========================
# Indicators
# =========================
df["MA20"]  = df["Close"].rolling(20).mean()
df["MA50"]  = df["Close"].rolling(50).mean()
df["MA200"] = df["Close"].rolling(200).mean()
df["RSI14"] = rsi(df["Close"], 14)
df["MACD"], df["MACD_signal"], df["MACD_hist"] = macd(df["Close"])

# =========================
# Fresh daily close (and optional minute) for the metric
# =========================
display_close = None
try:
    import yfinance as yf
    latest = yf.Ticker("TSLA").history(
        period="10d",
        interval="1d",
        auto_adjust=True,
        actions=False,
    ).dropna(subset=["Close"])
    if not latest.empty:
        display_close = float(latest["Close"].iloc[-1])
except Exception:
    pass

if live_override:
    try:
        m = yf.Ticker("TSLA").history(period="1d", interval="1m", auto_adjust=True)
        if not m.empty:
            display_close = float(m["Close"].dropna().iloc[-1])
    except Exception:
        pass

if display_close is None:
    display_close = float(pd.to_numeric(df["Close"], errors="coerce").dropna().iloc[-1])

# =========================
# Layout: metrics
# =========================
st.title("TSLA Strategy Dashboard")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Last Close (from data)", f"${display_close:,.2f}")
c2.metric("Shares Held", f"{int(shares_held)}")
c3.metric("Cost Basis", f"${float(cost_basis):,.2f}")
c4.metric("Unrealized P/L", f"${(display_close - float(cost_basis))*float(shares_held):,.2f}")

st.markdown("---")

# =========================
# Charts (native)
# =========================
st.subheader("Price + Moving Averages")
st.line_chart(df.set_index("Date")[["Close","MA20","MA50","MA200"]])

st.subheader("RSI(14)")
st.line_chart(df.set_index("Date")[["RSI14"]])

st.subheader("MACD")
st.line_chart(df.set_index("Date")[["MACD","MACD_signal"]])
st.bar_chart(df.set_index("Date")[["MACD_hist"]])

st.caption("Tip: Toggle 'Include today’s bar' or 'Use last-minute price' in the sidebar if the close looks stale. This tool is for education only — not financial advice.")

