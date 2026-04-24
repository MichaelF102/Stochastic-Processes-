import streamlit as st
import QuantLib as ql
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import norm, lognorm

# Page Config
st.set_page_config(page_title="Black-Scholes-Merton Process ", layout="wide")

# Custom Glassmorphism CSS
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: #ffffff;
    }
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 25px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 20px;
    }
    h1, h2, h3 {
        color: #00d4ff !important;
    }
    .stMetric {
        background: rgba(255, 255, 255, 0.05);
        padding: 15px;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.header("🕹️ Control Panel")

# Parameters
S0 = st.sidebar.number_input("Initial Price (S₀)", 100.0, step=1.0)
r = st.sidebar.slider("Risk-free Rate (r)", 0.0, 0.2, 0.05, step=0.01)
q = st.sidebar.slider("Dividend Yield (q)", 0.0, 0.2, 0.02, step=0.01)
sigma = st.sidebar.slider("Volatility (σ)", 0.01, 1.0, 0.2, step=0.01)
T = st.sidebar.slider("Time Horizon (T in years)", 0.1, 5.0, 1.0, step=0.1)

st.sidebar.markdown("---")
st.sidebar.subheader("Simulation Settings")
steps = st.sidebar.slider("Steps", 50, 1000, 252)
n_paths = st.sidebar.number_input("Number of Paths", 1, 10000, 100)
seed = st.sidebar.number_input("Random Seed", 0, 10000, 42)
method = st.sidebar.selectbox("Simulation Method", ["NumPy (Fast)", "QuantLib"])

if n_paths > 2000:
    st.sidebar.warning("⚠️ High path count may affect responsiveness.")

# Global Setup
today = ql.Date.todaysDate()
ql.Settings.instance().evaluationDate = today

# --- Logic Functions ---

def simulate_bsm_np(S0, r, q, sigma, T, steps, n_paths, seed):
    np.random.seed(seed)
    dt = T / steps
    # Vectorized simulation for BSM
    # Drift = r - q - 0.5 * sigma^2
    Z = np.random.standard_normal((n_paths, steps))
    log_returns = (r - q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    
    price_paths = np.zeros((n_paths, steps + 1))
    price_paths[:, 0] = S0
    price_paths[:, 1:] = S0 * np.exp(np.cumsum(log_returns, axis=1))
    
    time = np.linspace(0, T, steps + 1)
    return time, price_paths

def simulate_bsm_ql(S0, r, q, sigma, T, steps, n_paths, seed):
    day_count = ql.Actual365Fixed()
    calendar = ql.NullCalendar()
    
    s0_handle = ql.QuoteHandle(ql.SimpleQuote(S0))
    r_handle = ql.YieldTermStructureHandle(ql.FlatForward(today, r, day_count))
    q_handle = ql.YieldTermStructureHandle(ql.FlatForward(today, q, day_count))
    v_handle = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, calendar, sigma, day_count))
    
    # Use BlackScholesMertonProcess
    process = ql.BlackScholesMertonProcess(s0_handle, q_handle, r_handle, v_handle)
    
    rsg = ql.GaussianRandomSequenceGenerator(
        ql.UniformRandomSequenceGenerator(steps, ql.UniformRandomGenerator(seed))
    )
    
    path_generator = ql.GaussianPathGenerator(process, T, steps, rsg, False)
    
    paths = np.zeros((n_paths, steps + 1))
    for i in range(n_paths):
        sample_path = path_generator.next().value()
        paths[i, :] = [sample_path[j] for j in range(len(sample_path))]
        
    time = np.linspace(0, T, steps + 1)
    return time, paths

# --- Main App ---
st.title("📈 Black-Scholes-Merton Process")
st.write("Professional-grade stochastic simulation for assets with continuous dividend yields.")

tab_theory, tab_sim, tab_pricing, tab_insights = st.tabs([
    "📚 Theory & Math", "⚡ Simulation", "🎯 Option Pricing", "🧠 Insights"
])

# Run Simulation
if method == "NumPy (Fast)":
    time, data = simulate_bsm_np(S0, r, q, sigma, T, steps, n_paths, seed)
else:
    time, data = simulate_bsm_ql(S0, r, q, sigma, T, steps, n_paths, seed)

final_prices = data[:, -1]

with tab_theory:
    st.markdown("""
    ## 🔍 The Merton Extension
    
    The Black-Scholes-Merton (BSM) model extends the original framework to account for assets that pay a **continuous dividend yield** ($q$).
    
    ### ❗ The Cost of Carry
    In the BSM model, the net growth rate of the asset under the risk-neutral measure is the **cost of carry** ($b = r - q$):
    - **Risk-free Rate ($r$):** The return earned by holding cash.
    - **Dividend Yield ($q$):** The return leaked from the stock price and paid to the holder.
    
    > **Crucial Difference:** While the stock price grows at rate $r-q$, the total return (price + dividends) still grows at rate $r$ under the risk-neutral measure.
    
    ### 📐 Mathematical SDE
    Under the risk-neutral measure ($\mathbb{Q}$):
    """)
    st.latex(r"dS_t = (r - q) S_t \, dt + \sigma S_t \, dW_t")
    
    st.markdown("### 📌 Analytical Solution")
    st.latex(r"S_t = S_0 \exp\left((r - q - \tfrac{1}{2}\sigma^2)t + \sigma W_t\right)")
    
    st.info("Notice how the dividend yield $q$ acts as a 'negative drift' on the spot price evolution.")

with tab_sim:
    col_paths, col_stats = st.columns([2, 1])
    
    with col_paths:
        st.subheader("BSM Price Path Simulation")
        fig_paths = go.Figure()
        
        n_display = min(n_paths, 50)
        for i in range(n_display):
            fig_paths.add_trace(go.Scatter(
                x=time, y=data[i], mode='lines', 
                line=dict(width=1), opacity=0.4,
                name=f"Path {i+1}", hoverinfo='name+y'
            ))
            
        p5 = np.percentile(data, 5, axis=0)
        p50 = np.percentile(data, 50, axis=0)
        p95 = np.percentile(data, 95, axis=0)
        
        fig_paths.add_trace(go.Scatter(x=time, y=p5, line=dict(color='rgba(255,255,255,0)', dash='dash'), showlegend=False, hoverinfo='skip'))
        fig_paths.add_trace(go.Scatter(x=time, y=p95, fill='tonexty', fillcolor='rgba(0, 212, 255, 0.1)', line=dict(color='rgba(255,255,255,0)'), name="90% Confidence Interval"))
        fig_paths.add_trace(go.Scatter(x=time, y=p50, line=dict(color='#00d4ff', width=3), name="Median Path (P50)"))
        
        fig_paths.update_layout(
            template="plotly_dark", height=500,
            xaxis_title="Time (T)", yaxis_title="Price ($)",
            hovermode="closest"
        )
        st.plotly_chart(fig_paths, use_container_width=True)

    with col_stats:
        st.subheader("Simulation Metrics")
        st.metric("Final Mean", f"${np.mean(final_prices):.2f}")
        st.metric("Final Std Dev", f"${np.std(final_prices):.2f}")
        st.metric("Total Drift (r-q)", f"{(r-q)*100:.1f}%")

    st.markdown("---")
    
    col_returns, col_dist = st.columns(2)
    
    with col_returns:
        st.subheader("📊 Log-Returns Analysis")
        log_rets = np.log(data[:, 1:] / data[:, :-1]).flatten()
        
        fig_rets = go.Figure()
        fig_rets.add_trace(go.Histogram(x=log_rets, histnorm='probability density', marker_color='#00d4ff', opacity=0.6, name="Simulated"))
        
        mu_ret = (r - q - 0.5 * sigma**2) * (T/steps)
        std_ret = sigma * np.sqrt(T/steps)
        x_range = np.linspace(min(log_rets), max(log_rets), 100)
        y_pdf = norm.pdf(x_range, mu_ret, std_ret)
        fig_rets.add_trace(go.Scatter(x=x_range, y=y_pdf, line=dict(color='#ff007f', width=2), name="Normal PDF (Theory)"))
        
        fig_rets.update_layout(template="plotly_dark", title="Distribution of Log-Returns")
        st.plotly_chart(fig_rets, use_container_width=True)

    with col_dist:
        st.subheader("🎯 Terminal Price Distribution")
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(x=final_prices, histnorm='probability density', marker_color='#00d4ff', opacity=0.6, name="Simulated"))
        
        mean_log = np.log(S0) + (r - q - 0.5 * sigma**2) * T
        sd_log = sigma * np.sqrt(T)
        
        x_range_p = np.linspace(min(final_prices), max(final_prices), 100)
        y_pdf_p = lognorm.pdf(x_range_p, s=sd_log, scale=np.exp(mean_log))
        fig_dist.add_trace(go.Scatter(x=x_range_p, y=y_pdf_p, line=dict(color='#ff007f', width=2), name="Log-Normal PDF (Theory)"))
        
        fig_dist.update_layout(template="plotly_dark", title="Final Price Distribution")
        st.plotly_chart(fig_dist, use_container_width=True)

with tab_pricing:
    st.subheader("Merton Option Pricing Workbench")
    
    col_p1, col_p2 = st.columns([1, 2])
    
    with col_p1:
        st.markdown("### Parameters")
        K = st.number_input("Strike Price (K)", value=S0, step=1.0)
        opt_type = st.radio("Option Type", ["Call", "Put"])
        
        # QuantLib Setup
        day_count = ql.Actual365Fixed()
        calendar = ql.NullCalendar()
        exercise_date = today + ql.Period(int(T*365), ql.Days)
        
        s0_handle = ql.QuoteHandle(ql.SimpleQuote(S0))
        r_handle = ql.YieldTermStructureHandle(ql.FlatForward(today, r, day_count))
        q_handle = ql.YieldTermStructureHandle(ql.FlatForward(today, q, day_count))
        v_handle = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, calendar, sigma, day_count))
        process = ql.BlackScholesMertonProcess(s0_handle, q_handle, r_handle, v_handle)
        
        type_enum = ql.Option.Call if opt_type == "Call" else ql.Option.Put
        payoff = ql.PlainVanillaPayoff(type_enum, K)
        exercise = ql.EuropeanExercise(exercise_date)
        option = ql.EuropeanOption(payoff, exercise)
        option.setPricingEngine(ql.AnalyticEuropeanEngine(process))
        
        bsm_price = option.NPV()
        delta = option.delta()
        vega = option.vega() / 100.0
        
        # Monte Carlo Price
        if opt_type == "Call":
            payoffs = np.maximum(final_prices - K, 0)
        else:
            payoffs = np.maximum(K - final_prices, 0)
        mc_price = np.mean(payoffs) * np.exp(-r * T) # Discount at r
        
        st.markdown("---")
        st.metric("Analytical BSM Price", f"${bsm_price:.4f}")
        st.metric("Monte Carlo Price", f"${mc_price:.4f}")

    with col_p2:
        st.markdown("### Greeks & Performance")
        st.write(f"**Delta (Δ):** {delta:.4f}")
        st.write(f"**Vega (ν):** {vega:.4f}")
        
        # Convergence Chart
        cum_mc = (np.cumsum(payoffs) / (np.arange(n_paths) + 1)) * np.exp(-r * T)
        fig_conv = go.Figure()
        fig_conv.add_trace(go.Scatter(y=cum_mc, name="MC Estimate", line=dict(color='#00d4ff')))
        fig_conv.add_hline(y=bsm_price, line_dash="dash", line_color="#ff007f", annotation_text="BSM Analytical")
        fig_conv.update_layout(template="plotly_dark", height=300)
        st.plotly_chart(fig_conv, use_container_width=True)

with tab_insights:
    st.markdown("""
    ## 🧠 The Power of the Dividend Extension
    
    The Black-Scholes-Merton model is more versatile than the standard Black-Scholes model because $q$ can represent more than just stock dividends.
    
    ### 🌏 Real-World Applications
    1.  **Equity Indices:** Indices like the S&P 500 have a continuous dividend yield.
    2.  **Foreign Exchange (Garman-Kohlhagen):** In FX, $q$ represents the **foreign risk-free rate** ($r_f$). The BSM model becomes the Garman-Kohlhagen model.
    3.  **Commodities:** $q$ can represent the **convenience yield** minus the storage cost.
    
    ### 📉 Impact of Dividends on Options
    - **Calls:** Dividends make calls **less valuable** (the price drops after dividend payments).
    - **Puts:** Dividends make puts **more valuable**.
    - **Early Exercise:** For American options, a high dividend yield is the primary reason one might exercise a call option early.
    
    """)