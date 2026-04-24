import streamlit as st
import QuantLib as ql
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import norm, lognorm

# Page Config
st.set_page_config(page_title="Garman-Kohlhagen Process", layout="wide")

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
S0 = st.sidebar.number_input("Exchange Rate (Domestic/Foreign)", 1.20, step=0.01)
rd = st.sidebar.slider("Domestic Interest Rate (rd)", 0.0, 0.2, 0.05, step=0.01)
rf = st.sidebar.slider("Foreign Interest Rate (rf)", 0.0, 0.2, 0.03, step=0.01)
sigma = st.sidebar.slider("FX Volatility (σ)", 0.01, 1.0, 0.15, step=0.01)
T = st.sidebar.slider("Time Horizon (T in years)", 0.1, 5.0, 1.0, step=0.1)

st.sidebar.markdown("---")
st.sidebar.subheader("Simulation Settings")
steps = st.sidebar.slider("Steps", 50, 1000, 252)
n_paths = st.sidebar.number_input("Number of Paths", 1, 10000, 100)
seed = st.sidebar.number_input("Random Seed", 0, 10000, 42)
method = st.sidebar.selectbox("Simulation Method", ["NumPy (Fast)", "QuantLib"])

# Global Setup
today = ql.Date.todaysDate()
ql.Settings.instance().evaluationDate = today

# --- Logic Functions ---

def simulate_gk_np(S0, rd, rf, sigma, T, steps, n_paths, seed):
    np.random.seed(seed)
    dt = T / steps
    # Drift = rd - rf - 0.5 * sigma^2
    Z = np.random.standard_normal((n_paths, steps))
    log_returns = (rd - rf - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    
    price_paths = np.zeros((n_paths, steps + 1))
    price_paths[:, 0] = S0
    price_paths[:, 1:] = S0 * np.exp(np.cumsum(log_returns, axis=1))
    
    time = np.linspace(0, T, steps + 1)
    return time, price_paths

def simulate_gk_ql(S0, rd, rf, sigma, T, steps, n_paths, seed):
    day_count = ql.Actual365Fixed()
    calendar = ql.NullCalendar()
    
    s0_handle = ql.QuoteHandle(ql.SimpleQuote(S0))
    rd_handle = ql.YieldTermStructureHandle(ql.FlatForward(today, rd, day_count))
    rf_handle = ql.YieldTermStructureHandle(ql.FlatForward(today, rf, day_count))
    v_handle = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, calendar, sigma, day_count))
    
    # QuantLib's GarmanKohlagenProcess
    process = ql.GarmanKohlagenProcess(s0_handle, rf_handle, rd_handle, v_handle)
    
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
st.title("💱 Garman-Kohlhagen Process")
st.write("The industry standard for pricing Foreign Exchange (FX) options.")

tab_theory, tab_sim, tab_pricing, tab_insights = st.tabs([
    "📚 Theory & Math", "⚡ Simulation", "🎯 Pricing", "🧠 FX Insights"
])

# Run Simulation
if method == "NumPy (Fast)":
    time, data = simulate_gk_np(S0, rd, rf, sigma, T, steps, n_paths, seed)
else:
    time, data = simulate_gk_ql(S0, rd, rf, sigma, T, steps, n_paths, seed)

final_prices = data[:, -1]

with tab_theory:
    st.markdown("""
    ## 🔍 What is the Garman-Kohlhagen Model?
    
    The **Garman-Kohlhagen (1983)** model is an extension of the Black-Scholes model specifically designed for the **Foreign Exchange (FX)** market. 
    
    In the FX market, an exchange rate (e.g., EUR/USD) is a relative price between two currencies. Both currencies have their own risk-free interest rates.
    
    ### ❗ The Two-Rate Logic
    - **Domestic Rate ($r_d$):** The interest rate earned on the currency in which the option is priced.
    - **Foreign Rate ($r_f$):** The interest rate earned on the currency being modeled. Holding the foreign currency is like holding a dividend-paying stock where the "dividend" is the foreign interest rate.
    
    ### 📐 Mathematical SDE
    The evolution of the exchange rate $S_t$ under the domestic risk-neutral measure is:
    """)
    st.latex(r"dS_t = (r_d - r_f) S_t \, dt + \sigma S_t \, dW_t")
    
    st.markdown("### 📌 Analytical Solution")
    st.latex(r"S_t = S_0 \exp\left((r_d - r_f - \tfrac{1}{2}\sigma^2)t + \sigma W_t\right)")
    
    st.info("The drift term $(r_d - r_f)$ reflects the **Interest Rate Parity** condition.")

with tab_sim:
    col_paths, col_stats = st.columns([2, 1])
    
    with col_paths:
        st.subheader("FX Exchange Rate Simulation")
        fig_paths = go.Figure()
        
        n_display = min(n_paths, 50)
        for i in range(n_display):
            fig_paths.add_trace(go.Scatter(
                x=time, y=data[i], mode='lines', 
                line=dict(width=1), opacity=0.3,
                name=f"Path {i+1}", hoverinfo='name+y'
            ))
            
        p50 = np.percentile(data, 50, axis=0)
        fig_paths.add_trace(go.Scatter(x=time, y=p50, line=dict(color='#00d4ff', width=3), name="Median Path"))
        
        fig_paths.update_layout(template="plotly_dark", height=500, xaxis_title="Time", yaxis_title="Exchange Rate", hovermode="closest")
        st.plotly_chart(fig_paths, use_container_width=True)

    with col_stats:
        st.subheader("Metrics")
        st.metric("Final Mean", f"{np.mean(final_prices):.4f}")
        st.metric("Expected Forward Rate", f"{S0 * np.exp((rd-rf)*T):.4f}")
        st.metric("Implied Drift", f"{(rd-rf)*100:.2f}%")

    st.markdown("---")
    st.subheader("Terminal Value Distribution")
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(x=final_prices, histnorm='probability density', marker_color='#00d4ff', opacity=0.6))
    
    mu_l = np.log(S0) + (rd - rf - 0.5 * sigma**2) * T
    sd_l = sigma * np.sqrt(T)
    xr = np.linspace(min(final_prices), max(final_prices), 100)
    fig_hist.add_trace(go.Scatter(x=xr, y=lognorm.pdf(xr, s=sd_l, scale=np.exp(mu_l)), line=dict(color='#ff007f', width=2), name="Log-Normal PDF"))
    
    fig_hist.update_layout(template="plotly_dark", xaxis_title="Exchange Rate")
    st.plotly_chart(fig_hist, use_container_width=True)

with tab_pricing:
    st.subheader("FX Option Workbench")
    
    col_k, col_res = st.columns([1, 2])
    
    with col_k:
        K = st.number_input("Strike Price (K)", value=S0, step=0.01)
        opt_type = st.radio("Option Type", ["Call", "Put"])
        
        # Analytical Garman-Kohlhagen
        d1 = (np.log(S0/K) + (rd - rf + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if opt_type == "Call":
            price = S0 * np.exp(-rf * T) * norm.cdf(d1) - K * np.exp(-rd * T) * norm.cdf(d2)
            payoffs = np.maximum(final_prices - K, 0)
        else:
            price = K * np.exp(-rd * T) * norm.cdf(-d2) - S0 * np.exp(-rf * T) * norm.cdf(-d1)
            payoffs = np.maximum(K - final_prices, 0)
            
        mc_price = np.mean(payoffs) * np.exp(-rd * T)
        
        st.metric("GK Analytical Price", f"{price:.6f}")
        st.metric("Monte Carlo Price", f"{mc_price:.6f}")

    with col_res:
        st.markdown("### MC Convergence")
        cum_mc = (np.cumsum(payoffs) / (np.arange(n_paths) + 1)) * np.exp(-rd * T)
        fig_conv = go.Figure()
        fig_conv.add_trace(go.Scatter(y=cum_mc, name="MC Estimate", line=dict(color='#00d4ff')))
        fig_conv.add_hline(y=price, line_dash="dash", line_color="#ff007f", annotation_text="GK Analytical")
        fig_conv.update_layout(template="plotly_dark", height=300)
        st.plotly_chart(fig_conv, use_container_width=True)

with tab_insights:
    st.markdown("""
    ## 🧠 FX Market Realities
    
    ### 1. Interest Rate Parity
    The drift of an exchange rate is determined by the difference between domestic and foreign interest rates. If $r_d > r_f$, the foreign currency is expected to appreciate relative to the domestic currency to prevent arbitrage.
    
    ### 2. FX Volatility Skew
    Unlike equity markets, where the skew is almost always negative, the FX market often exhibits a **Volatility Smile**. This is because large moves in either direction (appreciation or depreciation) are possible, reflecting the relative nature of currency pricing.
    
    ### 3. Risk Reversals
    In FX trading, traders use "Risk Reversals" to measure the skewness of the market's expectations. This compares the implied volatility of out-of-the-money calls vs. puts.
    
    ### 🛠️ Pro Quant Tip
    "Always remember which currency is 'domestic' and which is 'foreign'. A common error in GK pricing is swapping $r_d$ and $r_f$, which completely flips the forward rate and the option value."
    """)

st.markdown("---")
st.caption("Quantitative Finance Portfolio | Garman-Kohlhagen (FX Options) Implementation")