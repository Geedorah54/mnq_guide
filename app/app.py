import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh

# Utils
from utils.data_fetch import get_mnq_data, get_vix_data, get_options_chain
from utils.indicators import add_emas, add_atr, classify_vix, classify_gamma
from utils.options import get_call_put_walls, calc_gamma_exposure
from utils.scoring import (
    compute_trend_score,
    classify_bias,
    compute_expected_move,
    classify_market_mode,
)

# ====================================================================================
# CONFIG
# ====================================================================================
st.set_page_config(page_title="MNQ Trading Dashboard", layout="wide")
st_autorefresh(interval=15000, key="mnq_dashboard_refresh")

# Journal storage
JOURNAL_DIR = "journal"
JOURNAL_PATH = os.path.join(JOURNAL_DIR, "trades.csv")
os.makedirs(JOURNAL_DIR, exist_ok=True)

# Load CSS if exists
try:
    with open("app/assets/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except:
    pass

def header(text):
    st.markdown(f"<h2 style='padding-top:0.5rem;'>{text}</h2>", unsafe_allow_html=True)

# ====================================================================================
# SIDEBAR NAV
# ====================================================================================
st.sidebar.title("MNQ Strategy Panel")
view = st.sidebar.radio(
    "View",
    ["Dashboard", "Options Levels", "Charts & Stats", "Checklist", "Journal"],
)

period = st.sidebar.selectbox("MNQ Lookback", ["1d", "5d"], index=0)
interval = "1m" if period == "1d" else "5m"

# ====================================================================================
# DATA FETCH
# ====================================================================================
mnq = get_mnq_data(period, interval)
vix = get_vix_data(period, interval)

def flatten_cols(df):
    df.columns = [
        "_".join(col).strip() if isinstance(col, tuple) else col
        for col in df.columns
    ]
    for col in df.columns:
        if col.startswith("Close"):
            df.rename(columns={col: "Close"}, inplace=True)
        if col.startswith("High"):
            df.rename(columns={col: "High"}, inplace=True)
        if col.startswith("Low"):
            df.rename(columns={col: "Low"}, inplace=True)
    return df

if not mnq.empty:
    mnq = flatten_cols(mnq)

if not vix.empty:
    vix = flatten_cols(vix)

# ====================================================================================
# SANITIZE VIX
# ====================================================================================
if not vix.empty:
    raw_vix = vix["Close"].iloc[-1]
    try:
        vix_now = float(raw_vix.iloc[0]) if isinstance(raw_vix, pd.Series) else float(raw_vix)
    except:
        vix_now = np.nan
else:
    vix_now = np.nan

vix_regime, vix_comment = classify_vix(vix_now)

# ====================================================================================
# EMAS + ATR
# ====================================================================================
if not mnq.empty:
    mnq = add_emas(mnq)
    mnq = add_atr(mnq, period=14)

for col in ["EMA_8", "EMA_21", "ATR_14"]:
    if col not in mnq.columns:
        mnq[col] = np.nan

# ====================================================================================
# OPTIONS CHAIN (Call/Put Walls + Gamma)
# ====================================================================================
calls, puts, exp = get_options_chain("QQQ")

if calls is not None and puts is not None:
    call_wall, put_wall = get_call_put_walls(calls, puts)

    try:
        raw_call = call_wall["strike"]
        call_strike = float(raw_call.iloc[0]) if isinstance(raw_call, pd.Series) else float(raw_call)
    except:
        call_strike = None

    try:
        raw_put = put_wall["strike"]
        put_strike = float(raw_put.iloc[0]) if isinstance(raw_put, pd.Series) else float(raw_put)
    except:
        put_strike = None

    try:
        raw_gex = calc_gamma_exposure(calls, puts)
        gex = float(raw_gex.iloc[0]) if isinstance(raw_gex, pd.Series) else float(raw_gex)
    except:
        gex = np.nan

else:
    call_strike = None
    put_strike = None
    gex = np.nan

gamma_regime, gamma_comment = classify_gamma(gex)

# ====================================================================================
# MNQ PRICE
# ====================================================================================
if not mnq.empty:
    raw_price = mnq["Close"].iloc[-1]
    try:
        price = float(raw_price.iloc[0]) if isinstance(raw_price, pd.Series) else float(raw_price)
    except:
        price = 0.0
else:
    price = 0.0

# ====================================================================================
# TREND SCORE + EXPECTED MOVE + MARKET MODE
# ====================================================================================
trend_score, trend_dir = compute_trend_score(
    df=mnq,
    vix_value=vix_now,
    gex=gex,
    call_wall_strike=call_strike,
    put_wall_strike=put_strike,
)

bias_label, bias_style = classify_bias(trend_score, trend_dir, vix_now)

exp_move_abs, exp_move_pct = compute_expected_move(price, vix_now)
market_mode, market_mode_comment = classify_market_mode(
    trend_score, gamma_regime, vix_regime
)

# ====================================================================================
# JOURNAL HELPERS
# ====================================================================================
def load_journal():
    if os.path.exists(JOURNAL_PATH):
        try:
            df = pd.read_csv(JOURNAL_PATH, parse_dates=["timestamp"])
        except Exception:
            df = pd.read_csv(JOURNAL_PATH)
        return df
    cols = [
        "timestamp",
        "symbol",
        "direction",
        "entry_price",
        "exit_price",
        "size",
        "pnl_points",
        "pnl_dollars",
        "notes",
    ]
    return pd.DataFrame(columns=cols)

def save_journal(df):
    df.to_csv(JOURNAL_PATH, index=False)

def calc_journal_stats(df):
    if df.empty:
        return {}
    stats = {}
    stats["total_trades"] = len(df)
    stats["wins"] = (df["pnl_dollars"] > 0).sum()
    stats["losses"] = (df["pnl_dollars"] < 0).sum()
    stats["win_rate"] = stats["wins"] / stats["total_trades"] * 100
    stats["total_pnl"] = df["pnl_dollars"].sum()
    stats["avg_pnl"] = df["pnl_dollars"].mean()
    return stats

# ====================================================================================
# PLOTLY CHART HELPERS
# ====================================================================================
def plot_mnq_chart(df, price, exp_move_abs):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df["Close"], mode="lines",
        name="Close", line=dict(color="#3b82f6", width=2)
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=df["EMA_8"], mode="lines",
        name="EMA 8", line=dict(color="#10b981", width=1.5, dash="dot")
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=df["EMA_21"], mode="lines",
        name="EMA 21", line=dict(color="#fbbf24", width=1.5, dash="dot")
    ))

    # Expected-move bands (today)
    if exp_move_abs is not None and not np.isnan(exp_move_abs) and price > 0:
        upper = price + exp_move_abs
        lower = price - exp_move_abs

        fig.add_hline(
            y=upper,
            line=dict(color="#f97316", width=1, dash="dash"),
            annotation_text=f"+EM ({upper:.1f})",
            annotation_position="top left",
        )
        fig.add_hline(
            y=lower,
            line=dict(color="#f97316", width=1, dash="dash"),
            annotation_text=f"-EM ({lower:.1f})",
            annotation_position="bottom left",
        )

    fig.update_layout(
        template="plotly_dark",
        height=500,
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
    )
    return fig

def plot_vix_chart(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df["Close"], mode="lines",
        name="VIX", line=dict(color="#ef4444", width=2)
    ))
    fig.update_layout(
        template="plotly_dark",
        height=350,
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
    )
    return fig

def plot_atr_chart(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df["ATR_14"], mode="lines",
        name="ATR 14", line=dict(color="#a855f7", width=2)
    ))
    fig.update_layout(
        template="plotly_dark",
        height=300,
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
    )
    return fig

# ====================================================================================
# VIEW: DASHBOARD
# ====================================================================================
if view == "Dashboard":

    header("📈 MNQ Options Flow Trading Dashboard")

    st.markdown("### Market Snapshot")
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown("#### MNQ Price")
        st.markdown(f"<h2 style='color:#10b981'>{price:.2f}</h2>", unsafe_allow_html=True)

    with c2:
        st.markdown("#### VIX")
        st.markdown(f"<h2 style='color:#3b82f6'>{vix_now:.2f}</h2>", unsafe_allow_html=True)
        st.caption(vix_comment)

    with c3:
        st.markdown("#### EMA Trend")
        color = "#10b981" if trend_dir == "Uptrend" else "#ef4444"
        st.markdown(f"<h2 style='color:{color}'>{trend_dir}</h2>", unsafe_allow_html=True)

    with c4:
        st.markdown("#### Gamma Regime")
        st.markdown(f"<h2 style='color:#eab308'>{gamma_regime}</h2>", unsafe_allow_html=True)
        st.caption(gamma_comment)

    st.markdown("### Session Intelligence")
    c5, c6 = st.columns(2)

    with c5:
        st.markdown("#### Expected Daily Move")
        if not np.isnan(exp_move_abs):
            st.markdown(
                f"<h3>± {exp_move_abs:.1f} pts "
                f"(<span style='color:#f97316'>{exp_move_pct:.2f}%</span>)</h3>",
                unsafe_allow_html=True,
            )
        else:
            st.write("N/A")

    with c6:
        st.markdown("#### Market Mode")
        st.markdown(f"<h3>{market_mode}</h3>", unsafe_allow_html=True)
        st.caption(market_mode_comment)

    st.markdown(
        f'<div class="big-badge bias-{bias_style}">{bias_label}</div>',
        unsafe_allow_html=True,
    )
    st.write(f"**Trend Score:** {trend_score}/100")
    st.progress(trend_score / 100)

    header("Price Action + Expected Move Bands")
    st.plotly_chart(plot_mnq_chart(mnq, price, exp_move_abs), use_container_width=True)

# ====================================================================================
# VIEW: OPTIONS LEVELS
# (keep your existing OI charts + advanced dealer stuff here)
# ====================================================================================
# ====================================================================================
# VIEW: OPTIONS LEVELS (FULL OPTIONS SUITE)
# ====================================================================================
elif view == "Options Levels":

    header("🎯 Options Chain Key Levels")

    if calls is None or puts is None:
        st.error("Options data unavailable.")
    else:
        colA, colB = st.columns(2)
        with colA:
            st.subheader("Call Wall (Resistance)")
            st.dataframe(call_wall)
        with colB:
            st.subheader("Put Wall (Support)")
            st.dataframe(put_wall)

        st.write(f"Expiration: `{exp}`")

        # ============================================================
        # CALL OI HEATMAP
        # ============================================================
        st.subheader("📊 Calls Open Interest Heatmap")

        fig_call_heat = go.Figure(
            data=[
                go.Bar(
                    x=calls["strike"],
                    y=calls["openInterest"],
                    marker=dict(
                        color=calls["openInterest"],
                        colorscale="Viridis",
                        showscale=True,
                        colorbar=dict(title="Call OI"),
                    ),
                    name="Call OI",
                )
            ]
        )
        fig_call_heat.update_layout(
            template="plotly_dark",
            height=300,
            margin=dict(l=10, r=10, t=40, b=10),
            xaxis_title="Strike",
            yaxis_title="Open Interest",
        )
        st.plotly_chart(fig_call_heat, use_container_width=True)

        # ============================================================
        # PUT OI HEATMAP
        # ============================================================
        st.subheader("📊 Puts Open Interest Heatmap")

        fig_put_heat = go.Figure(
            data=[
                go.Bar(
                    x=puts["strike"],
                    y=puts["openInterest"],
                    marker=dict(
                        color=puts["openInterest"],
                        colorscale="Inferno",
                        showscale=True,
                        colorbar=dict(title="Put OI"),
                    ),
                    name="Put OI",
                )
            ]
        )
        fig_put_heat.update_layout(
            template="plotly_dark",
            height=300,
            margin=dict(l=10, r=10, t=40, b=10),
            xaxis_title="Strike",
            yaxis_title="Open Interest",
        )
        st.plotly_chart(fig_put_heat, use_container_width=True)

        # ============================================================
        # TOTAL OI HEATMAP
        # ============================================================
        st.subheader("📊 Total Open Interest (Calls + Puts)")

        oi_calls = calls[["strike", "openInterest"]].rename(columns={"openInterest": "call_oi"})
        oi_puts = puts[["strike", "openInterest"]].rename(columns={"openInterest": "put_oi"})
        oi = pd.merge(oi_calls, oi_puts, on="strike", how="outer").fillna(0.0)
        oi["total_oi"] = oi["call_oi"] + oi["put_oi"]

        fig_total_heat = go.Figure(
            data=[
                go.Bar(
                    x=oi["strike"],
                    y=oi["total_oi"],
                    marker=dict(
                        color=oi["total_oi"],
                        colorscale="Plasma",
                        showscale=True,
                        colorbar=dict(title="Total OI"),
                    ),
                    name="Total OI",
                )
            ]
        )
        fig_total_heat.update_layout(
            template="plotly_dark",
            height=300,
            margin=dict(l=10, r=10, t=40, b=10),
            xaxis_title="Strike",
            yaxis_title="Total OI",
        )
        st.plotly_chart(fig_total_heat, use_container_width=True)

        # ============================================================
        # ADVANCED DEALER POSITIONING — DELTA WEIGHTED OI
        # ============================================================
        st.header("📘 Dealer Positioning: Delta-Weighted OI")

        # Ensure delta/gamma exist
        def compute_missing_greeks(df, is_call):
            if "delta" not in df.columns or df["delta"].isna().all():
                df["delta"] = np.clip(
                    np.where(
                        is_call,
                        0.5 * np.exp(-(abs(df["strike"] - price) / 25)),
                        -0.5 * np.exp(-(abs(df["strike"] - price) / 25)),
                    ),
                    -1,
                    1,
                )
            if "gamma" not in df.columns or df["gamma"].isna().all():
                df["gamma"] = np.exp(-(abs(df["strike"] - price) / 15)) * 0.01
            return df

        calls = compute_missing_greeks(calls, is_call=True)
        puts = compute_missing_greeks(puts, is_call=False)

        calls["delta_oi"] = calls["delta"].abs() * calls["openInterest"]
        puts["delta_oi"] = puts["delta"].abs() * puts["openInterest"]

        fig_delta = go.Figure()
        fig_delta.add_trace(go.Bar(
            x=calls["strike"], y=calls["delta_oi"],
            name="Call Δ·OI", marker_color="#60a5fa"
        ))
        fig_delta.add_trace(go.Bar(
            x=puts["strike"], y=puts["delta_oi"],
            name="Put Δ·OI", marker_color="#f87171"
        ))
        fig_delta.update_layout(
            template="plotly_dark",
            barmode="group",
            height=350,
            title="Dealer Hedging Pressure by Strike (Δ·OI)",
        )
        st.plotly_chart(fig_delta, use_container_width=True)

        # ============================================================
        # GAMMA EXPOSURE (GEX)
        # ============================================================
        st.header("📗 Gamma Exposure (GEX) by Strike")

        calls["gex"] = calls["gamma"] * calls["openInterest"]
        puts["gex"] = -puts["gamma"] * puts["openInterest"]

        gex_df = pd.DataFrame({
            "strike": calls["strike"],
            "GEX": calls["gex"] + puts["gex"],
        }).sort_values("strike")

        fig_gex = go.Figure()
        fig_gex.add_trace(go.Bar(
            x=gex_df["strike"], y=gex_df["GEX"],
            marker_color=np.where(gex_df["GEX"] >= 0, "#34d399", "#ef4444"),
        ))
        fig_gex.update_layout(
            template="plotly_dark",
            height=350,
            title="Gamma Exposure by Strike",
            xaxis_title="Strike",
            yaxis_title="Gamma Exposure",
        )
        st.plotly_chart(fig_gex, use_container_width=True)

        # ============================================================
        # GAMMA FLIP
        # ============================================================
        st.subheader("⚡ Gamma Flip Level")

        gamma_flip = None
        for i in range(1, len(gex_df)):
            if gex_df["GEX"].iloc[i] * gex_df["GEX"].iloc[i - 1] < 0:
                gamma_flip = gex_df["strike"].iloc[i]
                break

        if gamma_flip:
            st.success(f"Gamma Flip Detected at Strike: **{gamma_flip}**")
        else:
            st.info("No Gamma Flip detected today.")

        # ============================================================
        # PCR (Put/Call Ratio)
        # ============================================================
        st.header("📙 Put/Call Ratio (PCR) by Strike")

        pcr_df = pd.DataFrame({
            "strike": calls["strike"],
            "call_oi": calls["openInterest"],
            "put_oi": puts["openInterest"],
        })
        pcr_df["PCR"] = pcr_df["put_oi"] / pcr_df["call_oi"].replace(0, np.nan)

        fig_pcr = go.Figure()
        fig_pcr.add_trace(go.Bar(
            x=pcr_df["strike"], y=pcr_df["PCR"],
            marker_color="#fbbf24",
        ))
        fig_pcr.update_layout(
            template="plotly_dark",
            height=300,
            title="Put/Call Ratio by Strike",
            xaxis_title="Strike",
            yaxis_title="PCR",
        )
        st.plotly_chart(fig_pcr, use_container_width=True)


# ====================================================================================
# VIEW: CHARTS & STATS
# ====================================================================================
elif view == "Charts & Stats":

    header("📉 MNQ Price with EMAs (Interactive)")
    st.plotly_chart(plot_mnq_chart(mnq, price, exp_move_abs), use_container_width=True)

    header("📈 VIX (Interactive)")
    if not vix.empty:
        st.plotly_chart(plot_vix_chart(vix), use_container_width=True)

    header("📊 Volatility (ATR 14)")
    st.plotly_chart(plot_atr_chart(mnq), use_container_width=True)

# ====================================================================================
# VIEW: CHECKLIST
# ====================================================================================
elif view == "Checklist":

    header("📝 A+ Setup Checklist")

    high_vol = st.checkbox("High Volume Session (NY Open)")
    clean_pa = st.checkbox("Clean Price Action (Low Chop)")
    clear_sr = st.checkbox("Clear Support/Resistance")
    no_macro = st.checkbox("No Major Macro Events")
    intuition = st.checkbox("Gut Agrees")

    st.markdown("---")
    st.write(f"Trend Score: **{trend_score}/100** — Bias: **{bias_label}**")
    st.write(f"Market Mode: **{market_mode}**")

    if all([high_vol, clean_pa, clear_sr, no_macro, intuition]) and trend_score >= 70:
        st.success("A+ Setup Confirmed ✔✔✔")
    elif trend_score < 50:
        st.warning("Market conditions unfavorable.")
    else:
        st.info("Mixed signals — be selective.")

# ====================================================================================
# VIEW: JOURNAL
# ====================================================================================
elif view == "Journal":
    header("📓 Trade Journal & P/L Stats")

    journal_df = load_journal()

    # --- Stats row ---
    stats = calc_journal_stats(journal_df)
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("Total Trades", stats.get("total_trades", 0))
    with c2:
        st.metric("Win Rate", f"{stats.get('win_rate', 0):.1f}%" if stats else "0.0%")
    with c3:
        st.metric("Total P/L ($)", f"{stats.get('total_pnl', 0):.2f}" if stats else "0.00")
    with c4:
        st.metric("Avg P/L ($)", f"{stats.get('avg_pnl', 0):.2f}" if stats else "0.00")

    st.markdown("---")
    st.subheader("Log New Trade")

    with st.form("trade_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            symbol = st.text_input("Symbol", value="MNQ")
            direction = st.selectbox("Direction", ["Long", "Short"])
        with col2:
            entry_price = st.number_input("Entry Price", min_value=0.0, step=0.25)
            exit_price = st.number_input("Exit Price", min_value=0.0, step=0.25)
        with col3:
            size = st.number_input("Size (contracts)", min_value=0.0, step=0.1)
        notes = st.text_area("Notes", placeholder="Reason for entry, context, emotions, etc.")
        submitted = st.form_submit_button("Save Trade")

    if submitted:
        if entry_price > 0 and exit_price > 0 and size > 0:
            points = (exit_price - entry_price) * (1 if direction == "Long" else -1)
            # MNQ = $2 per point
            pnl_dollars = points * 2.0 * size

            new_row = pd.DataFrame(
                [{
                    "timestamp": pd.Timestamp.utcnow(),
                    "symbol": symbol,
                    "direction": direction,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "size": size,
                    "pnl_points": points,
                    "pnl_dollars": pnl_dollars,
                    "notes": notes,
                }]
            )

            journal_df = pd.concat([journal_df, new_row], ignore_index=True)
            save_journal(journal_df)
            st.success(f"Trade saved. P/L: {pnl_dollars:.2f} USD ({points:.1f} pts)")
        else:
            st.error("Please fill in entry, exit, and size.")

    st.markdown("---")
    st.subheader("Recent Trades")

    if journal_df.empty:
        st.info("No trades logged yet.")
    else:
        st.dataframe(journal_df.sort_values("timestamp", ascending=False).head(25))
