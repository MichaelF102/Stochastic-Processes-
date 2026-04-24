import streamlit as st
import QuantLib as ql
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import norm, lognorm

# Page Config
st.set_page_config(page_title="Black-76 Process", layout="wide")

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
F0 = st.sidebar.number_input("Futures Price (F₀)", 100.0, step=1.0)
r = st.sidebar.slider("Risk-free Rate (r)", 0.0, 0.2, 0.05, step=0.01)
sigma = st.sidebar.slider("Volatility (σ)", 0.01, 1.0, 0.2, step=0.01)
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

def simulate_black_np(F0, sigma, T, steps, n_paths, seed):
    np.random.seed(seed)
    dt = T / steps
    # SDE for Futures: dF = sigma * F * dW
    # Solution: F(t) = F(0) * exp(-0.5*sigma^2*t + sigma*W(t))
    Z = np.random.standard_normal((n_paths, steps))
    log_returns = (-0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    
    price_paths = np.zeros((n_paths, steps + 1))
    price_paths[:, 0] = F0
    price_paths[:, 1:] = F0 * np.exp(np.cumsum(log_returns, axis=1))
    
    time = np.linspace(0, T, steps + 1)
    return time, price_paths

def simulate_black_ql(F0, r, sigma, T, steps, n_paths, seed):
    day_count = ql.Actual365Fixed()
    calendar = ql.NullCalendar()
    
    f0_handle = ql.QuoteHandle(ql.SimpleQuote(F0))
    # In Black process, the risk-free rate is used for discounting, 
    # but the drift of the underlying is 0 (q = r).
    r_handle = ql.YieldTermStructureHandle(ql.FlatForward(today, r, day_count))
    v_handle = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, calendar, sigma, day_count))
    
    # QuantLib's BlackProcess
    process = ql.BlackProcess(f0_handle, r_handle, v_handle)
    
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
st.title("🌑 The Black Process (Black-76)")
st.write("The industry standard for pricing options on futures, forwards, and swaptions.")

tab_theory, tab_sim, tab_pricing, tab_insights = st.tabs([
    "📚 Theory & Math", "⚡ Simulation", "🎯 Pricing", "🧠 Use Cases"
])

# Run Simulation
if method == "NumPy (Fast)":
    time, data = simulate_black_np(F0, sigma, T, steps, n_paths, seed)
else:
    time, data = simulate_black_ql(F0, r, sigma, T, steps, n_paths, seed)

final_prices = data[:, -1]

with tab_theory:
    st.markdown("""
    ## 🔍 What is the Black Process?
    
    The **Black Process** (also known as the **Black-76 model**) is a mathematical model for the pricing of options on forward contracts, futures contracts, and swaptions. 
    
    Developed by Fischer Black in 1976, it is a variation of the Black-Scholes model where the underlying is a **futures price** rather than a stock price.
    
    ### ❗ The Driftless Underlyer
    In a risk-neutral world, the current price of a futures contract $F_0$ is the expected future price. Consequently, the futures price process has **zero drift**.
    
    ### 📐 Mathematical SDE
    The evolution of the futures price $F_t$ is described by:
    """)
    st.latex(r"dF_t = \sigma F_t \, dW_t")
    
    st.markdown("### 📌 Analytical Solution")
    st.latex(r"F_t = F_0 \exp\left(-\tfrac{1}{2}\sigma^2 t + \sigma W_t\right)")
    
    st.info("Note the absence of the interest rate ($r$) in the SDE. The interest rate only enters the model during the discounting of the final payoff.")

with tab_sim:
    col_paths, col_stats = st.columns([2, 1])
    
    with col_paths:
        st.subheader("Futures Price Simulation (Driftless)")
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
        
        fig_paths.update_layout(template="plotly_dark", height=500, xaxis_title="Time", yaxis_title="Futures Price", hovermode="closest")
        st.plotly_chart(fig_paths, use_container_width=True)

    with col_stats:
        st.subheader("Simulation Metrics")
        st.metric("Expected Price (F₀)", f"${F0:.2f}")
        st.metric("Simulated Mean", f"${np.mean(final_prices):.2f}")
        st.metric("Drift Observed", f"{((np.mean(final_prices)/F0)-1)*100:.2f}%")
        st.caption("Theoretical drift for futures is 0.00%.")

    st.markdown("---")
    
    col_hist, col_log = st.columns(2)
    with col_hist:
        st.subheader("🎯 Terminal Distribution")
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(x=final_prices, histnorm='probability density', marker_color='#00d4ff', opacity=0.6))
        
        # Theoretical Log-Normal
        ml = np.log(F0) - 0.5 * sigma**2 * T
        sl = sigma * np.sqrt(T)
        xp = np.linspace(min(final_prices), max(final_prices), 100)
        fig_hist.add_trace(go.Scatter(x=xp, y=lognorm.pdf(xp, s=sl, scale=np.exp(ml)), line=dict(color='#ff007f', width=2), name="Log-Normal PDF"))
        fig_hist.update_layout(template="plotly_dark", xaxis_title="Price")
        st.plotly_chart(fig_hist, use_container_width=True)
        
    with col_log:
        st.subheader("📊 Log-Returns")
        log_rets = np.log(data[:, 1:] / data[:, :-1]).flatten()
        fig_log = go.Figure()
        fig_log.add_trace(go.Histogram(x=log_rets, histnorm='probability density', marker_color='#00d4ff', opacity=0.6))
        
        mu_r = -0.5 * sigma**2 * (T/steps)
        sd_r = sigma * np.sqrt(T/steps)
        xr = np.linspace(min(log_rets), max(log_rets), 100)
        fig_log.add_trace(go.Scatter(x=xr, y=norm.pdf(xr, mu_r, sd_r), line=dict(color='#ff007f', width=2), name="Normal PDF"))
        fig_log.update_layout(template="plotly_dark", xaxis_title="Log-Return")
        st.plotly_chart(fig_log, use_container_width=True)

with tab_pricing:
    st.subheader("Black-76 Pricing Workbench")
    
    col_k, col_res = st.columns([1, 2])
    
    with col_k:
        K = st.number_input("Strike Price (K)", value=F0)
        opt_type = st.radio("Option Type", ["Call", "Put"])
        
        # Analytical Black-76
        d1 = (np.log(F0/K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if opt_type == "Call":
            price = np.exp(-r * T) * (F0 * norm.cdf(d1) - K * norm.cdf(d2))
            payoffs = np.maximum(final_prices - K, 0)
        else:
            price = np.exp(-r * T) * (K * norm.cdf(-d2) - F0 * norm.cdf(-d1))
            payoffs = np.maximum(K - final_prices, 0)
            
        mc_price = np.mean(payoffs) * np.exp(-r * T)
        
        st.metric("Black-76 Analytical Price", f"${price:.4f}")
        st.metric("Monte Carlo Price", f"${mc_price:.4f}")

    with col_res:
        st.markdown("### MC Convergence")
        cum_mc = (np.cumsum(payoffs) / (np.arange(n_paths) + 1)) * np.exp(-r * T)
        fig_conv = go.Figure()
        fig_conv.add_trace(go.Scatter(y=cum_mc, name="MC Estimate", line=dict(color='#00d4ff')))
        fig_conv.add_hline(y=price, line_dash="dash", line_color="#ff007f", annotation_text="Black-76 Analytical")
        fig_conv.update_layout(template="plotly_dark", height=300)
        st.plotly_chart(fig_conv, use_container_width=True)

with tab_insights:
    st.markdown("""
    ## 🧠 Why Use the Black Model?
    
    The Black model is pervasive in fixed income and commodities trading because it elegantly bypasses the need to model the spot price of the underlying asset.
    
    ### 1. Options on Futures
    Unlike stocks, futures contracts cost nothing to enter (initial margin is a performance bond, not a cost). Therefore, the underlying does not grow at the risk-free rate; it is already "fairly priced" at $F_0$.
    
    ### 2. Swaptions (Interest Rate Options)
    The Black model is the market standard for pricing European Swaptions. Here, the underlying is the **Forward Swap Rate**. Since the swap rate is a martingale under the annuity measure, it behaves exactly like a futures price with zero drift.
    
    ### 3. Caps and Floors
    A cap is a portfolio of **Caplets**. Each caplet is priced using the Black model where the underlying is the **Forward Libor/OIS Rate**.
    
    ### 🛠️ Pro Quant Tip
    "When you see a volatility quoted in the interest rate markets (e.g., '20% vol on a 10Y swaption'), it is almost always **Black Volatility**, implying the use of this specific model."
    """)

st.markdown("---")
st.caption("Quantitative Finance Portfolio | Black-76 (Options on Futures) Implementation")