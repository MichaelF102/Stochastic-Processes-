import streamlit as st
import QuantLib as ql
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import norm, lognorm
import math

# Page Config
st.set_page_config(page_title="Merton Jump-Diffusion Process", layout="wide")

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

# Continuous Params
S0 = st.sidebar.number_input("Initial Price (S₀)", 100.0, step=1.0)
r = st.sidebar.slider("Risk-free Rate (r)", 0.0, 0.2, 0.05, step=0.01)
sigma = st.sidebar.slider("Volatility (σ)", 0.01, 1.0, 0.2, step=0.01)
T = st.sidebar.slider("Time Horizon (T)", 0.1, 5.0, 1.0, step=0.1)

# Jump Params
st.sidebar.markdown("---")
st.sidebar.subheader("Jump Parameters")
jump_lambda = st.sidebar.slider("Jump Intensity (λ)", 0.0, 10.0, 1.0, step=0.5)
jump_mu = st.sidebar.slider("Mean Log-Jump (μJ)", -0.5, 0.5, -0.1, step=0.05)
jump_delta = st.sidebar.slider("Jump Volatility (δ)", 0.01, 0.5, 0.1, step=0.01)

# Settings
st.sidebar.markdown("---")
steps = st.sidebar.slider("Steps", 100, 1000, 500)
n_paths = st.sidebar.number_input("Number of Paths", 1, 5000, 100)
seed = st.sidebar.number_input("Random Seed", 0, 10000, 42)
method = st.sidebar.selectbox("Simulation Method", ["NumPy (Fast)", "QuantLib"])

# Global Setup
today = ql.Date.todaysDate()
ql.Settings.instance().evaluationDate = today

# --- Logic Functions ---

def simulate_merton_np(S0, r, sigma, jump_lambda, jump_mu, jump_delta, T, steps, n_paths, seed):
    np.random.seed(seed)
    dt = T / steps
    time = np.linspace(0, T, steps + 1)
    
    # Expected relative jump size: kappa = E[exp(J)-1]
    kappa = np.exp(jump_mu + 0.5 * jump_delta**2) - 1
    
    # Risk-neutral drift: r - lambda * kappa - 0.5 * sigma^2
    drift = (r - jump_lambda * kappa - 0.5 * sigma**2) * dt
    
    paths = np.zeros((n_paths, steps + 1))
    paths[:, 0] = S0
    
    for j in range(steps):
        # Diffusion part
        Z = np.random.standard_normal(n_paths)
        log_increments = drift + sigma * np.sqrt(dt) * Z
        
        # Jump part
        n_jumps = np.random.poisson(jump_lambda * dt, n_paths)
        jumps = np.zeros(n_paths)
        # Sum of n_jumps normal variables: N(n*mu, n*delta^2)
        mask = n_jumps > 0
        if np.any(mask):
            jumps[mask] = np.random.normal(
                n_jumps[mask] * jump_mu, 
                np.sqrt(n_jumps[mask]) * jump_delta
            )
            
        paths[:, j+1] = paths[:, j] * np.exp(log_increments + jumps)
        
    return time, paths

def simulate_merton_ql(S0, r, sigma, jump_lambda, jump_mu, jump_delta, T, steps, n_paths, seed):
    day_count = ql.Actual365Fixed()
    s0_handle = ql.QuoteHandle(ql.SimpleQuote(S0))
    r_handle = ql.YieldTermStructureHandle(ql.FlatForward(today, r, day_count))
    v_handle = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, ql.NullCalendar(), sigma, day_count))
    
    # Merton 76 Process
    process = ql.Merton76Process(
        s0_handle, r_handle, v_handle,
        ql.QuoteHandle(ql.SimpleQuote(jump_lambda)),
        ql.QuoteHandle(ql.SimpleQuote(jump_mu)),
        ql.QuoteHandle(ql.SimpleQuote(jump_delta))
    )
    
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
st.title("📉 Merton Jump-Diffusion Process")
st.write("Modeling asset prices with continuous diffusion and discrete, random jumps.")

tab_theory, tab_sim, tab_pricing, tab_insights = st.tabs([
    "📚 Theory", "⚡ Simulation", "🎯 Pricing", "🧠 Insights"
])

# Run Simulation
if method == "NumPy (Fast)":
    time, data = simulate_merton_np(S0, r, sigma, jump_lambda, jump_mu, jump_delta, T, steps, n_paths, seed)
else:
    time, data = simulate_merton_ql(S0, r, sigma, jump_lambda, jump_mu, jump_delta, T, steps, n_paths, seed)

final_prices = data[:, -1]

with tab_theory:
    st.markdown("""
    ## 🔍 Beyond Black-Scholes: The Merton Model
    
    The **Merton Jump-Diffusion (1976)** model extends Black-Scholes by allowing the asset price to "jump" at random intervals. This is critical for modeling markets prone to sudden shocks (e.g., earnings, geopolitical news).
    
    ### 📐 The SDE
    Under the risk-neutral measure ($\mathbb{Q}$):
    """)
    st.latex(r"dS_t = (r - \lambda \kappa) S_t \, dt + \sigma S_t \, dW_t + S_t (J - 1) \, dN_t")
    
    st.markdown("""
    Where:
    - **$dN_t$**: A Poisson process with intensity $\lambda$ (jump frequency).
    - **$J$**: The jump size, where $\ln(J) \sim \mathcal{N}(\mu_J, \delta^2)$.
    - **$\kappa$**: The expected relative jump size, $\mathbb{E}[J - 1] = \exp(\mu_J + \tfrac{1}{2}\delta^2) - 1$.
    
    ### ⚖️ The Martingale Condition
    To ensure the model is arbitrage-free, the drift is adjusted by $-\lambda \kappa$. This compensates for the "drift" introduced by the jump process itself, ensuring that $\mathbb{E}[e^{-rt}S_t] = S_0$.
    """)
    
    st.info("The addition of jumps makes the distribution of returns **Leptokurtic** (fat-tailed), more closely matching real-world market behavior.")

with tab_sim:
    col_paths, col_stats = st.columns([2, 1])
    
    with col_paths:
        st.subheader("Jump-Diffusion Paths")
        fig_paths = go.Figure()
        
        n_display = min(n_paths, 30)
        for i in range(n_display):
            fig_paths.add_trace(go.Scatter(x=time, y=data[i], mode='lines', line=dict(width=1), opacity=0.4, showlegend=False))
            
        fig_paths.update_layout(template="plotly_dark", height=500, xaxis_title="Time", yaxis_title="Price", hovermode="closest")
        st.plotly_chart(fig_paths, use_container_width=True)

    with col_stats:
        st.subheader("Metrics")
        st.metric("Final Mean", f"${np.mean(final_prices):.2f}")
        st.metric("Theoretical Expected", f"${S0 * np.exp(r * T):.2f}")
        
        # Count paths with significant moves
        diffs = np.abs(np.diff(data, axis=1))
        jumps_found = np.sum(diffs > (sigma * np.sqrt(T/steps) * 4))
        st.metric("Spikes Detected", jumps_found)
        st.caption("Movements exceeding 4σ threshold.")

    st.markdown("---")
    
    col_hist, col_log = st.columns(2)
    with col_hist:
        st.subheader("Final Price Distribution")
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(x=final_prices, histnorm='probability density', marker_color='#00d4ff', opacity=0.6, name="Merton"))
        
        # Compare with pure GBM (Black-Scholes)
        xr = np.linspace(min(final_prices), max(final_prices), 100)
        mu_log = np.log(S0) + (r - 0.5 * sigma**2) * T
        sd_log = sigma * np.sqrt(T)
        fig_hist.add_trace(go.Scatter(x=xr, y=lognorm.pdf(xr, s=sd_log, scale=np.exp(mu_log)), line=dict(color='#ff007f', width=2), name="Pure GBM (BS)"))
        
        fig_hist.update_layout(template="plotly_dark", xaxis_title="Price")
        st.plotly_chart(fig_hist, use_container_width=True)
        
    with col_log:
        st.subheader("Log-Returns (Fat Tails)")
        log_rets = np.log(data[:, 1:] / data[:, :-1]).flatten()
        fig_log = go.Figure()
        fig_log.add_trace(go.Histogram(x=log_rets, histnorm='probability density', marker_color='#00d4ff', opacity=0.6))
        
        # Normal Overlay
        mu_r = (r - 0.5 * sigma**2) * (T/steps)
        sd_r = sigma * np.sqrt(T/steps)
        x_r = np.linspace(min(log_rets), max(log_rets), 100)
        fig_log.add_trace(go.Scatter(x=x_r, y=norm.pdf(x_r, mu_r, sd_r), line=dict(color='#ff007f', width=2), name="Normal Dist"))
        
        fig_log.update_layout(template="plotly_dark", xaxis_title="Log-Return")
        st.plotly_chart(fig_log, use_container_width=True)

with tab_pricing:
    st.subheader("Merton Analytical Pricing")
    
    col_k, col_res = st.columns([1, 2])
    
    with col_k:
        K = st.number_input("Strike (K)", value=S0)
        opt_type = st.radio("Option Type", ["Call", "Put"])
        
        # Merton Analytical Formula: Infinite sum of BS weighted by Poisson probabilities
        # We sum up to 50 jumps for accuracy
        m_price = 0
        for n in range(50):
            weight = (math.exp(-jump_lambda * T) * (jump_lambda * T)**n) / math.factorial(n)
            
            # Adjusted params for n jumps
            sigma_n = math.sqrt(sigma**2 + n * jump_delta**2 / T)
            r_n = r - jump_lambda * (math.exp(jump_mu + 0.5 * jump_delta**2) - 1) + n * (jump_mu + 0.5 * jump_delta**2) / T
            
            d1 = (math.log(S0/K) + (r_n + 0.5 * sigma_n**2) * T) / (sigma_n * math.sqrt(T))
            d2 = d1 - sigma_n * math.sqrt(T)
            
            if opt_type == "Call":
                bs_n = S0 * norm.cdf(d1) - K * math.exp(-r_n * T) * norm.cdf(d2)
            else:
                bs_n = K * math.exp(-r_n * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
                
            m_price += weight * bs_n
            
        if opt_type == "Call":
            payoffs = np.maximum(final_prices - K, 0)
        else:
            payoffs = np.maximum(K - final_prices, 0)
            
        mc_price = np.mean(payoffs) * np.exp(-r * T)
        
        st.metric("Merton Analytical Price", f"${m_price:.4f}")
        st.metric("Monte Carlo Price", f"${mc_price:.4f}")

    with col_res:
        st.markdown("### MC Convergence")
        cum_mc = (np.cumsum(payoffs) / (np.arange(n_paths) + 1)) * np.exp(-r * T)
        fig_conv = go.Figure()
        fig_conv.add_trace(go.Scatter(y=cum_mc, name="MC Estimate", line=dict(color='#00d4ff')))
        fig_conv.add_hline(y=m_price, line_dash="dash", line_color="#ff007f", annotation_text="Merton Analytical")
        fig_conv.update_layout(template="plotly_dark", height=350)
        st.plotly_chart(fig_conv, use_container_width=True)

with tab_insights:
    st.markdown("""
    ## 🧠 Why Merton Matters
    
    The Merton model was a groundbreaking step because it admitted that price movements are not always continuous.
    
    ### 📉 The Volatility Smile
    One of the greatest failures of Black-Scholes is that it cannot explain the **volatility smile**. By adding jumps, the Merton model creates fat tails, which naturally leads to higher implied volatilities for out-of-the-money options—effectively creating the smile.
    
    ### 💥 Modeling Crashes
    Standard GBM assumes that a 20% market drop is a "once in a billion years" event. The Merton model, by adjusting $\lambda$ and $\mu_J$, allows such crashes to be rare but **statistically possible**.
    
    ### 🛠️ Pro Quant Tip
    "When you see a large gap in a stock price overnight due to an earnings release, you are seeing a **jump** in action. Black-Scholes can't handle it, but Merton can. This is why Jump-Diffusion is still used to price short-dated equity options where jump risk is most prominent."
    """)

st.markdown("---")
st.caption("Quantitative Finance Portfolio | Merton Jump-Diffusion Implementation")