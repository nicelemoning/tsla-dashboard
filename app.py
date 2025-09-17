# TSLA Strategy Dashboard (robust price + diagnostics)
import numpy as np
import pandas as pd
import streamlit as st
import requests

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
    """Return df with at least Date & Close columns. Handles yfinance MultiIndex."""
    df = df_in.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            "_".join([str(x) for x in tup if str(x)]).strip()
            for tup in df.columns
        ]

    def find_col(prefixes):
        for c in df.columns:
            key = str(c).replace(" ", "").replace("_", "").lower()
            for p in prefixes:
                if key.startswith(p):
                    return c
        return None

    date_c = find_col(["date", "datetime", "timestamp"])
    if date_c is None and getattr(df.index, "name", None) is not None:
        df = df.reset_index()
        date_c = find_col(["date", "datetime", "timestamp", "index"])

    close_c = find_col(["close", "adjclose"])
    open_c  = find_col(["open"])
    high_c  = find_col(["high"])
    low_c   = find_col(["low"])
    vol_c   = find_col(["volume", "vol"])

    rename_map = {}
    if date_c:  rename_map[date_c]  = "Date"
    if close_c: rename_map[close_c] = "Close"
    if open_c:  rename_map[open_c]  = "Open"
    if high_c:  rename_map[high_c]  = "High"
    if low_c:   rename_map[low_c]   = "Low"
    if vol_c:   rename_map[vol_c]   = "Volume"
    df = df.rename(columns=rename_map)

    if "Date" not in df.columns or "Close" not in df.columns:
        return None

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for c in ["Open","High","Low","Close","Volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["Date","Close"]).sort_values("Date").reset_index(drop=True)
    return df

def fetch_fresh_price() -> tuple[float | None, str]:
    """Return freshest TSLA price (USD), plus a label for the source used."""
    # 1) Yahoo Quote API (very fresh, usually current trading price)
    try:
        r = requests.get(
            "https://query1.finance.yahoo.com/v7/finance/quote?symbols=TSLA",
            timeout=5,
        )
        q = r.json()["quoteResponse"]["result"][0]
        p = q.get("regularMarketPrice")
        if p:
            return float(p), "yahoo-quote (regularMarketPrice)"
    except Exception:
        pass

    # 2) yfinance daily close (adjusted)
    try:
        import yfinance as yf
        h = yf.Ticker("TSLA").history(period="10d", interval="1d", auto_adjust=True, actions=False)
        if not h.empty:
            return float(h["Close"].dropna().iloc[-1]), "yfinance daily"
    except Exception:
        pass

    # 3) yfinance 1-minute (fallback live)
    try:
        import yfinance as yf
        m = yf.Ticker("TSLA").history(period="1d", interval="1m", auto_adjust=True)
        if not m.empty:
            return float(m["Close"].dropna().iloc[-1]), "yfinance 1m"
    except Exception:
        pass

    return None, "fallback df"

# =========================
# Sidebar controls
# =========================
st.sidebar.header("Position")
shares_held = st.sidebar.number_input("Shares held", 1, step=1, value=220)
cost_basis  = st.sidebar.number_input("Cost basis (USD)", min_value=0.0, step=0.1, value=400.0, format="%.2f")

st.sidebar.header("Data")
use_live       = st.sidebar.checkbox("Use live TSLA data (yfinance)", value=True)
include_today  = st.sidebar.checkbox("Include today's bar (if available)", value=True)
show_debug     = st.sidebar.checkbox("Show diagnostics", value=True)
uploaded = st.sidebar.file_uploader("Or upload CSV (Date,Open,High,Low,Close,Volume)", type=["csv"])
if st.sidebar.button("Refresh data"):
    st.experimental_rerun()

# =========================
# Data load
# =========================
df_raw, data_source = None, ""

# 1) CSV
if uploaded is not None:
    try:
        df_raw = pd.read_csv(uploaded)
        data_source = "uploaded CSV"
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()

# 2) yfinance
if df_raw is None and use_live:
    try:
        import yfinance as yf
        raw = yf.download(
            "TSLA",
            period="5y",
            interval="1d",
            auto_adjust=True,      # adjusted prices
            actions=False,
            progress=False,
            threads=True,
        )
        if not raw.empty:
            raw = raw[raw["Volume"] > 0]         # drop empty/holiday bars
            df_raw = raw.reset_index()           # Date index -> column
            data_source = "yfinance daily (auto_adjust)"
    except Exception:
        df_raw = None

# 3) Synthetic fallback
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
    data_source = "synthetic fallback (API unavailable)"

# Normalize
df = normalize_df(df_raw)
if df is None:
    st.error(
        "Data is missing 'Date'/'Close' even after normalization.\n"
        f"Columns: {list(df_raw.columns)}"
    )
    st.dataframe(df_raw.head(), use_container_width=True)
    st.stop()

# Keep/skip today's bar
ny_now = pd.Timestamp.now(tz="America/New_York")
if not include_today and len(df) and df["Date"].iloc[-1].date() == ny_now.date():
    if len(df) > 1:
        df = df.iloc[:-1].reset_index(drop=True)

if show_debug:
    st.caption(
        f"Data source: {data_source} | last row date: {df['Date'].iloc[-1].date()} "
        f"| rows: {len(df)}"
    )

# =========================
# Indicators
# =========================
df["MA20"]  = df["Close"].rolling(20).mean()
df["MA50"]  = df["Close"].rolling(50).mean()
df["MA200"] = df["Close"].rolling(200).mean()
df["RSI14"] = rsi(df["Close"], 14)
df["MACD"], df["MACD_signal"], df["MACD_hist"] = macd(df["Close"])

# =========================
# Fresh price for the metric
# =========================
fresh_price, price_source = fetch_fresh_price()
if fresh_price is None:
    fresh_price = float(pd.to_numeric(df["Close"], errors="coerce").dropna().iloc[-1])
    price_source = "df last row"

# =========================
# Layout: metrics
# =========================
st.title("TSLA Strategy Dashboard")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Last Price / Close", f"${fresh_price:,.2f}")
c2.metric("Shares Held", f"{int(shares_held)}")
c3.metric("Cost Basis", f"${float(cost_basis):,.2f}")
c4.metric("Unrealized P/L", f"${(fresh_price - float(cost_basis))*float(shares_held):,.2f}")

if show_debug:
    st.caption(f"Price source: {price_source}")

st.markdown("---")

# =========================
# Charts
# =========================
st.subheader("Price + Moving Averages")
st.line_chart(df.set_index("Date")[["Close","MA20","MA50","MA200"]])

st.subheader("RSI(14)")
st.line_chart(df.set_index("Date")[["RSI14"]])

st.subheader("MACD")
st.line_chart(df.set_index("Date")[["MACD","MACD_signal"]])
st.bar_chart(df.set_index("Date")[["MACD_hist"]])

st.caption("If the number still looks off, check the diagnostics above. This tool is for education only — not financial advice.")
