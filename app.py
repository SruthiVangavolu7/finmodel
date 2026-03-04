"""Try Trading Streamlit app with a styled educational simulator layout."""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"

# Requested color system
COLOR_BG = "#F2F4F3"
COLOR_CARD_DARK = "#22333B"
COLOR_LIGHT = "#F2F4F3"
COLOR_ACCENT = "#A9927D"
COLOR_TEXT_SECONDARY = "#5E503F"

LONG_THRESHOLD = 0.55
SHORT_THRESHOLD = 0.45
LOOKBACK_BARS = 30


@st.cache_data(show_spinner=False)
def load_csv(file_name: str) -> pd.DataFrame:
    """Load a CSV from the results directory."""
    path = RESULTS_DIR / file_name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def inject_css() -> None:
    """Inject custom CSS theme and lightweight motion effects."""
    st.markdown(
        f"""
<style>
.main {{ background-color: {COLOR_BG}; }}
.stApp {{ background-color: {COLOR_BG}; }}
.block-container {{ max-width: 980px; padding-top: 1.6rem; padding-bottom: 2.2rem; }}

html, body, [class*="css"] {{
  font-family: "Iowan Old Style", "Times New Roman", "Georgia", serif;
  color: #0A0908;
}}

h1, h2, h3, h4 {{
  font-family: "Iowan Old Style", "Times New Roman", "Georgia", serif;
  letter-spacing: 0.2px;
}}

.hero-shell {{
  background: {COLOR_LIGHT};
  color: #0A0908;
  border-radius: 14px;
  padding: 2.2rem 1.8rem;
  text-align: center;
  margin-bottom: 1.2rem;
  animation: riseIn 0.6s ease-out;
}}

.hero-sub {{
  color: {COLOR_TEXT_SECONDARY};
  max-width: 680px;
  margin: 0.8rem auto 0.4rem auto;
  font-size: 1.1rem;
}}

.type-heading {{
  display: inline-block;
  overflow: hidden;
  white-space: nowrap;
  animation: typing 2.4s steps(30, end), blink .8s step-end infinite;
  font-size: clamp(2rem, 5vw, 3.5rem);
  line-height: 1.05;
  margin-bottom: 0.6rem;
}}

.reveal-text {{
  display: inline-block;
  color: {COLOR_TEXT_SECONDARY};
  clip-path: inset(0 100% 0 0);
  animation: revealClip 1.5s ease forwards;
  animation-delay: 0.2s;
}}

.pill-btn {{
  display: inline-block;
  background: {COLOR_ACCENT};
  color: #0A0908;
  border-radius: 999px;
  padding: 0.45rem 1rem;
  font-weight: 600;
  margin-top: 0.8rem;
  text-decoration: none;
}}

.surface {{
  background: {COLOR_LIGHT};
  color: #0A0908;
  border-radius: 14px;
  padding: 1.2rem;
  margin-bottom: 1rem;
  animation: riseIn 0.6s ease-out;
}}

.section-label {{
  display: inline-block;
  background: {COLOR_ACCENT};
  color: #0A0908;
  border-radius: 999px;
  padding: 0.22rem 0.62rem;
  font-size: 0.78rem;
  margin-bottom: 0.55rem;
  letter-spacing: 0.2px;
}}

.step-card {{
  background: {COLOR_CARD_DARK};
  border-radius: 12px;
  padding: 1rem;
  min-height: 160px;
  color: {COLOR_LIGHT};
  text-align: center;
  display: flex;
  flex-direction: column;
  justify-content: center;
  transition: transform 220ms ease, box-shadow 220ms ease;
}}
.step-card:hover {{
  transform: translateY(-4px);
  box-shadow: 0 14px 24px rgba(0, 0, 0, 0.25);
}}
.step-num {{ color: {COLOR_ACCENT}; font-weight: 700; font-size: 1.2rem; margin-bottom: 0.4rem; }}

.kpi-card {{
  background: {COLOR_CARD_DARK};
  color: {COLOR_LIGHT};
  border-radius: 10px;
  padding: 0.85rem 0.95rem;
  margin-top: 0.6rem;
}}
.kpi-label {{ color: #d5d8d8; font-size: 0.85rem; }}
.kpi-value {{ color: {COLOR_LIGHT}; font-size: 1.3rem; font-weight: 700; }}

div[data-testid="stRadio"] > label,
div[data-testid="stRadio"] span {{
  color: #000000 !important;
}}

/* Force widget labels/values to black for Stock, Date index, and decision controls */
div[data-testid="stWidgetLabel"] p,
div[data-testid="stWidgetLabel"] span,
div[data-testid="stSelectbox"] *,
div[data-testid="stSlider"] *,
div[data-testid="stRadio"] *,
div[data-baseweb="select"] *,
div[data-baseweb="slider"] * {{
  color: #000000 !important;
}}

/* Force listed section headers to black */
h2, h3, h4, h5, label, .pill-btn {{
  color: #000000 !important;
}}

.out-good {{ color: #1f9d55; font-weight: 700; }}
.out-bad {{ color: #d94f4f; font-weight: 700; }}
.out-flat {{ color: {COLOR_TEXT_SECONDARY}; font-weight: 700; }}

@keyframes riseIn {{
  from {{ opacity: 0; transform: translateY(16px); }}
  to {{ opacity: 1; transform: translateY(0); }}
}}
@keyframes typing {{ from {{ width: 0 }} to {{ width: 100% }} }}
@keyframes blink {{ 50% {{ border-color: transparent; }} }}
@keyframes revealClip {{ to {{ clip-path: inset(0 0 0 0); }} }}
</style>
        """,
        unsafe_allow_html=True,
    )


def metric_lookup(metrics: pd.DataFrame, metric_name: str) -> float | None:
    """Return a numeric metric value by name."""
    if metrics.empty:
        return None
    row = metrics[metrics["metric"] == metric_name]
    if row.empty:
        return None
    try:
        return float(row["value"].iloc[0])
    except (TypeError, ValueError):
        return None


def model_action(probability: float) -> str:
    """Map probability to strategy action buckets."""
    if probability > LONG_THRESHOLD:
        return "Buy"
    if probability < SHORT_THRESHOLD:
        return "Sell"
    return "Stay Flat"


def render_hero(metrics: pd.DataFrame) -> None:
    """Render hero section and top KPIs."""
    st.markdown(
        """
<div class="hero-shell">
  <div class="type-heading">Trading Market<br/>Simulator</div>
  <div class="hero-sub reveal-text">
    Practice reading market patterns, make a trade decision, and compare your choice to what the model expected.
  </div>
  <a class="pill-btn" href="#try-trading">Try Trading Now</a>
</div>
        """,
        unsafe_allow_html=True,
    )

    auc = metric_lookup(metrics, "auc_roc")
    ic = metric_lookup(metrics, "ic")
    sharpe = metric_lookup(metrics, "annualized_sharpe")
    total_ret = metric_lookup(metrics, "total_return")

    c1, c2, c3, c4 = st.columns(4)
    cards = [
        ("AUC", f"{auc:.3f}" if auc is not None else "N/A"),
        ("Mean IC", f"{ic:.3f}" if ic is not None else "N/A"),
        ("Sharpe", f"{sharpe:.2f}" if sharpe is not None else "N/A"),
        ("Total Return", f"{(total_ret * 100):.1f}%" if total_ret is not None else "N/A"),
    ]
    for col, (label, value) in zip((c1, c2, c3, c4), cards):
        with col:
            st.markdown(
                f"<div class='kpi-card'><div class='kpi-label'>{label}</div><div class='kpi-value'>{value}</div></div>",
                unsafe_allow_html=True,
            )


def render_steps() -> None:
    """Render the three-step simulator explanation row."""
    st.markdown("<h2 style='text-align:center; margin-top:1.4rem; color:#000000;'>How it Works</h2>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            """
<div class="step-card">
  <div class="step-num">1</div>
  <div>Pick a stock and a day based on the model patterns and confidence.</div>
</div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
<div class="step-card">
  <div class="step-num">2</div>
  <div>Choose whether to buy, sell, or stay flat.</div>
</div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            """
<div class="step-card">
  <div class="step-num">3</div>
  <div>We reveal what happened and compare your choice vs the model signal.</div>
</div>
            """,
            unsafe_allow_html=True,
        )


def render_learning(daily_pnl: pd.DataFrame, monthly_ic: pd.DataFrame) -> None:
    """Render educational learn section with model interpretation guidance."""
    st.markdown(
        """
<div class='surface'>
  <h2 style='text-align:center; margin-bottom:0.2rem;'>Learn</h2>
</div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1, 2])
    with left:
        st.markdown(
            """
<div class='surface'>
  <p style='color:#5E503F;'>
  Focus on probability first, then thresholds, then consistency over many months.
  </p>
</div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            """
<div class='surface'>
  <p style='margin:0;'>
  - Start with predicted probability: closer to 1.0 means stronger "up" confidence, closer to 0.0 means stronger "down" confidence.<br/>
  - Use threshold logic: above 0.55 suggests Buy, below 0.45 suggests Sell, in between suggests no trade.<br/>
  - Check AUC for direction skill, IC for ranking quality, and Sharpe/Drawdown for risk-adjusted behavior.<br/>
  - Judge consistency over time using monthly IC and equity trend, not a single trade.
  </p>
</div>
            """,
            unsafe_allow_html=True,
        )

    c1, c2 = st.columns(2)
    with c1:
        if not monthly_ic.empty:
            mic = monthly_ic.copy()
            mic["date"] = pd.to_datetime(mic["date"])
            st.markdown("#### Monthly IC")
            st.bar_chart(mic.sort_values("date").set_index("date")["monthly_ic"])
    with c2:
        if not daily_pnl.empty:
            pnl = daily_pnl.copy()
            pnl["date"] = pd.to_datetime(pnl["date"])
            st.markdown("#### Equity Path")
            st.line_chart(pnl.sort_values("date").set_index("date")[["strategy_equity", "benchmark_equity"]])


def render_simulator(predictions: pd.DataFrame) -> None:
    """Render interactive scenario picker and decision outcome."""
    if predictions.empty:
        st.error("No prediction data available. Run python main.py first.")
        return

    df = predictions.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    st.markdown("<a id='try-trading'></a>", unsafe_allow_html=True)
    st.markdown("<div class='surface'><h2 style='text-align:center; margin-bottom:0; color:#000000;'>Try Trading</h2></div>", unsafe_allow_html=True)

    c1, c2 = st.columns([1.6, 2.4])
    tickers = sorted(df["ticker"].dropna().unique().tolist())

    with c1:
        ticker = st.selectbox("Stock", tickers)

    ticker_df = df[df["ticker"] == ticker].copy().sort_values("date").reset_index(drop=True)
    with c2:
        max_idx = max(len(ticker_df) - 1, 0)
        chosen_idx = st.slider("Date index", min_value=0, max_value=max_idx, value=min(20, max_idx), step=1)
    row = ticker_df.iloc[chosen_idx]
    window = ticker_df.iloc[max(0, chosen_idx - LOOKBACK_BARS) : chosen_idx + 1].copy()

    st.markdown(
        f"""
<div class='surface'>
  <span class='section-label'>Scenario</span>
  <p><b>{ticker}</b> on <b>{row['date'].date()}</b></p>
  <p style='color:{COLOR_TEXT_SECONDARY}; margin-bottom:0;'>
    Predicted probability of next-day up move: <b>{row['pred_prob']:.1%}</b>
  </p>
</div>
        """,
        unsafe_allow_html=True,
    )

    ch1, ch2 = st.columns(2)
    with ch1:
        st.markdown("##### Recent Price Pattern")
        st.line_chart(window.set_index("date")["Close"])
    with ch2:
        st.markdown("##### Confidence Pattern")
        st.line_chart(window.set_index("date")["pred_prob"])

    action = st.radio("Your decision", ["Buy", "Sell", "Stay Flat"], horizontal=True)

    forward_ret = float(row["forward_return"])
    actual_up = int(row["label"]) == 1

    if action == "Buy":
        pnl = forward_ret
    elif action == "Sell":
        pnl = -forward_ret
    else:
        pnl = 0.0

    actual_text = "Up" if actual_up else "Down"
    model_pick = model_action(float(row["pred_prob"]))

    if action == "Stay Flat":
        cls = "out-flat"
        verdict = "You stayed flat, so your P/L is 0 for this scenario."
    elif pnl > 0:
        cls = "out-good"
        verdict = "Good read. Your direction matched the next move."
    else:
        cls = "out-bad"
        verdict = "This trade direction was wrong for the next day."

    st.markdown(
        f"""
<div class='surface'>
  <span class='section-label'>Outcome</span>
  <p>Next-day move: <b>{actual_text}</b> ({forward_ret:.2%})</p>
  <p>Your action: <b>{action}</b> | Model action: <b>{model_pick}</b></p>
  <p class='{cls}'>{verdict}</p>
</div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    """Entrypoint for the redesigned Try Trading app."""
    st.set_page_config(page_title="Try Trading", page_icon="📈", layout="wide")
    inject_css()

    if not RESULTS_DIR.exists():
        st.error("No results directory found. Run python main.py first.")
        return

    predictions = load_csv("predictions_oof.csv")
    daily_pnl = load_csv("daily_pnl.csv")
    metrics = load_csv("metrics.csv")
    monthly_ic = load_csv("monthly_ic.csv")

    render_hero(metrics)
    render_steps()
    render_learning(daily_pnl, monthly_ic)
    render_simulator(predictions)


if __name__ == "__main__":
    main()
