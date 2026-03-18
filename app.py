# app.py — Daily Transactions & Volume with Industry Pivot + NA-safe DoD + PF vs DM Pie
# Expected headers (from your SQL export):
#   PY_DM_TAG, MERCHANT_CAT, TRANSACTION_DATE, PERIOD, MONTH, MCC, INDUSTRY,
#   MERCHANT_NAME, TRXN_CODE, AMOUNT, TRXN_COUNT, CARD_ACCEPTOR_ID,
#   USD_PHP_TAG, DB_CR_TAG, DB_CR_AMOUNT, CONVERTED_AMOUNT

import csv
import io
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# -------------------------------------------------
# Page config & title
# -------------------------------------------------
st.set_page_config(page_title="Daily Volume by Industry", page_icon="📊", layout="wide")
st.title("📊 Daily Transactions & Volume")

# -------------------------------------------------
# 1) Load CSV robustly
# -------------------------------------------------
uploaded = st.file_uploader("Upload your CSV", type=["csv"])
if not uploaded:
    st.info("Please upload your aggregated CSV to proceed.")
    st.stop()

try:
    # Read as text first to avoid silent coercion and row loss
    df = pd.read_csv(
        uploaded,
        dtype=str,
        low_memory=False,
        encoding="utf-8-sig",      # handles BOM from Excel
        keep_default_na=False,     # avoids auto NA conversion of strings like "N/A"
        quoting=csv.QUOTE_MINIMAL,
    )
except Exception as e:
    st.error(f"Could not read CSV: {e}")
    st.stop()

# Normalize headers
df.columns = [c.strip() for c in df.columns]

with st.expander("🔎 Debug: columns & sample"):
    st.write("Columns:", list(df.columns))
    st.dataframe(df.head(10), use_container_width=True)

# Basic schema sanity
must_have = ["TRANSACTION_DATE", "TRXN_COUNT", "INDUSTRY", "PY_DM_TAG"]
missing = [c for c in must_have if c not in df.columns]
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

# -------------------------------------------------
# 2) Text normalization & numeric parsing
# -------------------------------------------------
def normalize_text(series: pd.Series) -> pd.Series:
    """Trim + collapse internal whitespace. Keeps case."""
    return series.astype(str).str.replace(r"\s+", " ", regex=True).str.strip()

def norm_one(x) -> str:
    """Normalize a single scalar safely to lowercase token."""
    if pd.isna(x):
        return ""
    return normalize_text(pd.Series([x])).iloc[0].lower()

for col in ["PY_DM_TAG", "MCC", "INDUSTRY", "MERCHANT_CAT", "MERCHANT_NAME", "TRXN_CODE", "USD_PHP_TAG", "CARD_ACCEPTOR_ID"]:
    if col in df.columns:
        df[col] = normalize_text(df[col])

def to_float_safe(s: pd.Series) -> pd.Series:
    """Parse floats that may contain commas, $, and parentheses for negatives."""
    s = s.astype(str).str.strip()
    s = s.str.replace(r"[,$]", "", regex=True)                 # remove thousands separators and $
    s = s.str.replace(r"\(([\d\.]+)\)", r"-\1", regex=True)    # (123.45) -> -123.45
    return pd.to_numeric(s, errors="coerce")

def to_int_safe(s: pd.Series) -> pd.Series:
    """Parse ints that may contain commas/underscores; returns pandas Int64 (nullable)."""
    s = s.astype(str).str.strip()
    s = s.str.replace(r"[,_]", "", regex=True)
    return pd.to_numeric(s, errors="coerce").astype("Int64")

# Apply parsing
df["TRXN_COUNT"] = to_int_safe(df["TRXN_COUNT"])
for col in ["AMOUNT", "DB_CR_AMOUNT", "CONVERTED_AMOUNT"]:
    if col in df.columns:
        df[col] = to_float_safe(df[col])

# -------------------------------------------------
# 3) Date parsing with heuristics (pick best)
# -------------------------------------------------
raw_date = df["TRANSACTION_DATE"].astype(str).str.strip()

parsed0 = pd.to_datetime(raw_date, errors="coerce"); bad0 = parsed0.isna().sum()
parsed1 = pd.to_datetime(raw_date, errors="coerce", dayfirst=True); bad1 = parsed1.isna().sum()

def try_fmt(fmt: str) -> pd.Series:
    try:
        return pd.to_datetime(raw_date, format=fmt, errors="coerce")
    except Exception:
        return pd.Series(pd.NaT, index=raw_date.index)

parsed2 = try_fmt("%d/%m/%Y"); bad2 = parsed2.isna().sum()
parsed3 = try_fmt("%m/%d/%Y"); bad3 = parsed3.isna().sum()

candidates = [(parsed0, bad0, "default"),
              (parsed1, bad1, "dayfirst"),
              (parsed2, bad2, "fmt:%d/%m/%Y"),
              (parsed3, bad3, "fmt:%m/%d/%Y")]
best_parsed, best_bad, best_label = sorted(candidates, key=lambda x: x[1])[0]

df["TRANSACTION_DATE_PARSED"] = best_parsed
df["day"] = df["TRANSACTION_DATE_PARSED"].dt.normalize()  # daily bucket
unparsed_rows = int(df["TRANSACTION_DATE_PARSED"].isna().sum())

with st.expander("🧪 Parse quality"):
    st.write({
        "rows_read": int(len(df)),
        "date_parse_mode_selected": best_label,
        "unparsed_date_rows": unparsed_rows,
        "TRXN_COUNT_total_raw": int(pd.to_numeric(df["TRXN_COUNT"].fillna(0), errors="coerce").sum()),
        "PY_DM_TAG values": df["PY_DM_TAG"].dropna().unique().tolist()[:10],
    })

# -------------------------------------------------
# 4) Volume resolution (prefer CONVERTED_AMOUNT → DB_CR_AMOUNT → AMOUNT signed)
# -------------------------------------------------
def ensure_signed_amount(frame: pd.DataFrame, raw_col: str) -> pd.Series:
    """Signs AMOUNT via DB_CR_TAG: 'Credit'/CR => negative, else positive."""
    if "DB_CR_TAG" not in frame.columns:
        return to_float_safe(frame[raw_col]).fillna(0.0)
    tag = frame["DB_CR_TAG"].astype(str).str.upper().str.strip()
    factor = np.where(tag.str.startswith("CR"), -1, 1)  # 'Credit' or 'CR' => -1
    return to_float_safe(frame[raw_col]).fillna(0.0) * factor

if "CONVERTED_AMOUNT" in df.columns:
    amt_col = "CONVERTED_AMOUNT"
    df["_volume"] = to_float_safe(df["CONVERTED_AMOUNT"]).fillna(0.0)
elif "DB_CR_AMOUNT" in df.columns:
    amt_col = "DB_CR_AMOUNT"
    df["_volume"] = to_float_safe(df["DB_CR_AMOUNT"]).fillna(0.0)
elif "AMOUNT" in df.columns:
    amt_col = "AMOUNT"
    df["_volume"] = ensure_signed_amount(df, "AMOUNT")
else:
    amt_col = None
    df["_volume"] = 0.0

# -------------------------------------------------
# 5) PY_DM_TAG filter (single-select)
# -------------------------------------------------
# Canonicalize common labels but allow any distinct values from data
tag_options = ["All"] + sorted(df["PY_DM_TAG"].dropna().astype(str).unique().tolist())
selected_tag = st.selectbox("Filter: PY_DM_TAG", options=tag_options, index=0)

df_filtered = df.copy()
if selected_tag != "All":
    tgt_norm = norm_one(selected_tag)
    df_filtered = df_filtered[df_filtered["PY_DM_TAG"].map(norm_one) == tgt_norm].copy()

# -------------------------------------------------
# 6) Daily totals + NA-safe DoD
# -------------------------------------------------
df_valid = df_filtered[df_filtered["day"].notna()].copy()

daily_totals = (
    df_valid
    .groupby("day", as_index=False)
    .agg(
        daily_trxn_count=("TRXN_COUNT", lambda s: pd.to_numeric(s, errors="coerce").fillna(0).sum()),
        daily_volume=("_volume", "sum"),
    )
    .sort_values("day")
)

# --- NA-safe DoD computation (no boolean on <NA>) ---
prev_count = daily_totals["daily_trxn_count"].shift(1).astype("float")
prev_vol   = daily_totals["daily_volume"].shift(1).astype("float")

daily_totals["DoD_Δ_count"] = daily_totals["daily_trxn_count"].astype("float") - prev_count
daily_totals["DoD_Δ_vol"]   = daily_totals["daily_volume"].astype("float")      - prev_vol

# Denominator is NaN if previous is 0 or NaN → result NaN (undefined) instead of raising
den_count = prev_count.replace(0.0, np.nan)
den_vol   = prev_vol.replace(0.0, np.nan)

daily_totals["DoD_%_count"] = daily_totals["DoD_Δ_count"] / den_count
daily_totals["DoD_%_vol"]   = daily_totals["DoD_Δ_vol"]   / den_vol

# -------------------------------------------------
# 7) Daily INDUSTRY pivot (volume) & matrix
# -------------------------------------------------
industry_pivot = (
    df_valid
    .pivot_table(index="day", columns="INDUSTRY", values="_volume", aggfunc="sum", fill_value=0.0)
    .sort_index()
)

matrix = daily_totals.set_index("day").join(industry_pivot, how="left").reset_index()

# Column order: day, totals, DoD columns, then industries by latest-day volume
if not industry_pivot.empty:
    latest_day_idx = industry_pivot.index.max()
    industry_order = industry_pivot.loc[latest_day_idx].sort_values(ascending=False).index.tolist()
else:
    industry_order = []

ordered_cols = [
    "day",
    "daily_trxn_count", "daily_volume",
    "DoD_Δ_count", "DoD_%_count",
    "DoD_Δ_vol", "DoD_%_vol",
] + industry_order

matrix = matrix.reindex(columns=[c for c in ordered_cols if c in matrix.columns])

# -------------------------------------------------
# 8) KPI row (latest day) with simple formatting
# -------------------------------------------------
def fmt_int(x):
    return "—" if pd.isna(x) else f"{int(x):,}"

def fmt_float0(x):
    return "—" if pd.isna(x) else f"{x:,.0f}"

def fmt_pct(x):
    return "—" if pd.isna(x) else f"{x*100:,.1f}%"

if not matrix.empty:
    latest_row = matrix.iloc[-1]
    latest_day = pd.to_datetime(latest_row["day"]).date()
    kpi_count = latest_row.get("daily_trxn_count", np.nan)
    kpi_vol   = latest_row.get("daily_volume", np.nan)
    kpi_dod_vol = latest_row.get("DoD_%_vol", np.nan)
else:
    latest_day, kpi_count, kpi_vol, kpi_dod_vol = None, np.nan, np.nan, np.nan

c1, c2, c3, c4 = st.columns(4)
c1.metric("Latest day", latest_day.strftime("%Y-%m-%d") if latest_day else "—")
c2.metric("Total TXN (latest day)", fmt_int(kpi_count))
c3.metric(f"Total Volume (latest day, {amt_col or 'N/A'})", fmt_float0(kpi_vol))
c4.metric("DoD % (Volume)", fmt_pct(kpi_dod_vol))

# -------------------------------------------------
# 9) Charts (totals + top‑N industries)
# -------------------------------------------------
left, right = st.columns(2)
with left:
    st.subheader("Daily Transaction Count (Total)")
    fig_cnt = px.line(matrix, x="day", y="daily_trxn_count", markers=True, title="Daily TRXN_COUNT")
    fig_cnt.update_layout(xaxis_title="Day", yaxis_title="TRXN_COUNT")
    st.plotly_chart(fig_cnt, use_container_width=True)

with right:
    st.subheader(f"Daily Volume (Total, {amt_col or 'N/A'})")
    fig_vol = px.line(matrix, x="day", y="daily_volume", markers=True, title=f"Daily Volume ({amt_col or 'no amount column'})")
    fig_vol.update_layout(xaxis_title="Day", yaxis_title="Volume")
    st.plotly_chart(fig_vol, use_container_width=True)

if industry_order:
    st.subheader("Daily Volume — Top Industries (by latest day)")
    top_n = st.slider("Top N industries to plot", min_value=3, max_value=min(20, len(industry_order)), value=min(10, len(industry_order)))
    top_cols = industry_order[:top_n]
    melt_df = matrix.melt(id_vars=["day"], value_vars=top_cols, var_name="INDUSTRY", value_name="volume")
    fig_ind = px.line(melt_df, x="day", y="volume", color="INDUSTRY", markers=False, title=f"Daily Volume — Top {top_n} Industries")
    fig_ind.update_layout(xaxis_title="Day", yaxis_title="Volume")
    st.plotly_chart(fig_ind, use_container_width=True)

# -------------------------------------------------
# 10) PF vs DM pie (Latest day or all days; Volume or Count)
# -------------------------------------------------
st.markdown("---")
st.subheader("Payment Facilitator vs Direct Merchant — Share")

pie_basis = st.selectbox("Pie basis", ["Latest day", "Selected range (all days read)"], index=0)
pie_metric = st.selectbox("Pie metric", ["Volume", "Transaction Count"], index=0)

# Build a working frame that ignores the PY_DM_TAG filter (so you always see both slices)
df_pie = df[df["day"].notna()].copy()

if pie_basis == "Latest day":
    latest_day_global = df_pie["day"].max()
    df_pie = df_pie[df_pie["day"] == latest_day_global]

val_col = "_volume" if pie_metric == "Volume" else "TRXN_COUNT"

pie = (
    df_pie
    .groupby("PY_DM_TAG", as_index=False)
    .agg(value=(val_col, lambda s: pd.to_numeric(s, errors="coerce").fillna(0).sum()))
    .sort_values("value", ascending=False)
)

fig_pie = px.pie(pie, names="PY_DM_TAG", values="value",
                 title=f"PF vs DM — {pie_metric} ({pie_basis})")
st.plotly_chart(fig_pie, use_container_width=True)

# -------------------------------------------------
# 11) Table + Export
# -------------------------------------------------
st.subheader("Daily Matrix (Totals & DoD on the left; industries to the right)")
st.dataframe(matrix, use_container_width=True)

buf = io.StringIO()
matrix.to_csv(buf, index=False)
st.download_button(
    "⬇️ Download daily matrix (CSV)",
    buf.getvalue(),
    file_name="daily_matrix_with_DoD_by_industry.csv",
    mime="text/csv"
)
