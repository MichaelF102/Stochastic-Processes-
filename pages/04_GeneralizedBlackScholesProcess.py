import streamlit as st
import QuantLib as ql
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import norm, lognorm

# Page Config
st.set_page_config(page_title="Generalized Black-Scholes Process", layout="wide")

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

# Model Presets
st.sidebar.subheader("Model Presets")
preset = st.sidebar.selectbox("Select Model Logic", [
    "Custom (b)", 
    "Black-Scholes (b = r)", 
    "Black-Scholes-Merton (b = r - q)", 
    "Black-76 (b = 0)", 
    "Garman-Kohlhagen (b = rd - rf)"
])

# Parameters
S0 = st.sidebar.number_input("Initial Price (S₀)", 100.0, step=1.0)
r = st.sidebar.slider("Risk-free Rate (r)", 0.0, 0.2, 0.05, step=0.01)

# Logic for b (Cost of Carry)
if preset == "Custom (b)":
    b = st.sidebar.slider("Cost of Carry (b)", -0.2, 0.2, 0.05, step=0.01)
elif preset == "Black-Scholes (b = r)":
    b = r
elif preset == "Black-Scholes-Merton (b = r - q)":
    q = st.sidebar.slider("Dividend Yield (q)", 0.0, 0.2, 0.02, step=0.01)
    b = r - q
elif preset == "Black-76 (b = 0)":
    b = 0.0
elif preset == "Garman-Kohlhagen (b = rd - rf)":
    rf = st.sidebar.slider("Foreign Rate (rf)", 0.0, 0.2, 0.03, step=0.01)
    b = r - rf

sigma = st.sidebar.slider("Volatility (σ)", 0.01, 1.0, 0.2, step=0.01)
T = st.sidebar.slider("Time Horizon (T)", 0.1, 5.0, 1.0, step=0.1)

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

def simulate_gbs_np(S0, b, sigma, T, steps, n_paths, seed):
    np.random.seed(seed)
    dt = T / steps
    # SDE: dS = b*S*dt + sigma*S*dW
    Z = np.random.standard_normal((n_paths, steps))
    log_returns = (b - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    
    price_paths = np.zeros((n_paths, steps + 1))
    price_paths[:, 0] = S0
    price_paths[:, 1:] = S0 * np.exp(np.cumsum(log_returns, axis=1))
    
    time = np.linspace(0, T, steps + 1)
    return time, price_paths

def simulate_gbs_ql(S0, r, b, sigma, T, steps, n_paths, seed):
    day_count = ql.Actual365Fixed()
    calendar = ql.NullCalendar()
    
    s0_handle = ql.QuoteHandle(ql.SimpleQuote(S0))
    r_handle = ql.YieldTermStructureHandle(ql.FlatForward(today, r, day_count))
    
    # Calculate implied dividend yield q = r - b
    q = r - b
    q_handle = ql.YieldTermStructureHandle(ql.FlatForward(today, q, day_count))
    v_handle = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, calendar, sigma, day_count))
    
    # QuantLib uses BSM process as the generalized form
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
st.title("📊 Generalized Black-Scholes Process")
st.write("A unified framework for pricing derivatives across stocks, indices, currencies, and futures.")

tab_theory, tab_sim, tab_pricing, tab_insights = st.tabs([
    "📚 Theory", "⚡ Simulation", "🎯 Pricing", "🧠 Use Cases"
])

# Run Simulation
if method == "NumPy (Fast)":
    time, data = simulate_gbs_np(S0, b, sigma, T, steps, n_paths, seed)
else:
    time, data = simulate_gbs_ql(S0, r, b, sigma, T, steps, n_paths, seed)

final_prices = data[:, -1]

with tab_theory:
    st.markdown("""
    ## 🔍 The Unified Framework
    
    The **Generalized Black-Scholes (GBS)** model is a "meta-model" that captures several specific pricing models by varying the **cost of carry ($b$)**.
    
    ### 📐 The SDE
    The asset price evolution is governed by:
    """)
    st.latex(r"dS_t = b S_t \, dt + \sigma S_t \, dW_t")
    
    st.markdown("""
    ### ⚖️ Cost of Carry ($b$) Mappings
    Depending on the asset class, $b$ is defined differently:
    
    | Model | Asset Type | Cost of Carry ($b$) |
    | :--- | :--- | :--- |
    | **Black-Scholes** | Non-dividend stock | $b = r$ |
    | **Black-Scholes-Merton** | Stock with div yield ($q$) | $b = r - q$ |
    | **Black-76** | Futures / Options on Futures | $b = 0$ |
    | **Garman-Kohlhagen** | Foreign Exchange (FX) | $b = r_d - r_f$ |
    """)
    
    st.info(f"Current Model: **{preset}** | Cost of Carry ($b$): **{b:.4f}**")

with tab_sim:
    col_paths, col_stats = st.columns([2, 1])
    
    with col_paths:
        st.subheader("Price Path Simulation")
        fig_paths = go.Figure()
        
        n_display = min(n_paths, 50)
        for i in range(n_display):
            fig_paths.add_trace(go.Scatter(
                x=time, y=data[i], mode='lines', 
                line=dict(width=1), opacity=0.3,
                name=f"Path {i+1}", hoverinfo='name+y'
            ))
            
        p5 = np.percentile(data, 5, axis=0)
        p50 = np.percentile(data, 50, axis=0)
        p95 = np.percentile(data, 95, axis=0)
        
        fig_paths.add_trace(go.Scatter(x=time, y=p5, line=dict(color='rgba(255,255,255,0)'), showlegend=False, hoverinfo='skip'))
        fig_paths.add_trace(go.Scatter(x=time, y=p95, fill='tonexty', fillcolor='rgba(0, 212, 255, 0.1)', line=dict(color='rgba(255,255,255,0)'), name="90% CI"))
        fig_paths.add_trace(go.Scatter(x=time, y=p50, line=dict(color='#00d4ff', width=3), name="Median"))
        
        fig_paths.update_layout(template="plotly_dark", height=500, xaxis_title="Years", yaxis_title="Price", hovermode="closest")
        st.plotly_chart(fig_paths, use_container_width=True)

    with col_stats:
        st.subheader("Metrics")
        st.metric("Expected Price (E[ST])", f"${S0 * np.exp(b * T):.2f}")
        st.metric("Simulated Mean", f"${np.mean(final_prices):.2f}")
        st.metric("Drift (b)", f"{b*100:.1f}%")

    st.markdown("---")
    
    col_rets, col_hist = st.columns(2)
    with col_rets:
        log_rets = np.log(data[:, 1:] / data[:, :-1]).flatten()
        fig_rets = go.Figure()
        fig_rets.add_trace(go.Histogram(x=log_rets, histnorm='probability density', marker_color='#00d4ff', opacity=0.6))
        
        # Theoretical Normal
        mu_r = (b - 0.5 * sigma**2) * (T/steps)
        sd_r = sigma * np.sqrt(T/steps)
        xr = np.linspace(min(log_rets), max(log_rets), 100)
        fig_rets.add_trace(go.Scatter(x=xr, y=norm.pdf(xr, mu_r, sd_r), line=dict(color='#ff007f', width=2), name="Normal PDF"))
        fig_rets.update_layout(template="plotly_dark", title="Log-Returns Distribution")
        st.plotly_chart(fig_rets, use_container_width=True)

    with col_hist:
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(x=final_prices, histnorm='probability density', marker_color='#00d4ff', opacity=0.6))
        
        # Theoretical Log-Normal
        ml = np.log(S0) + (b - 0.5 * sigma**2) * T
        sl = sigma * np.sqrt(T)
        xp = np.linspace(min(final_prices), max(final_prices), 100)
        fig_dist.add_trace(go.Scatter(x=xp, y=lognorm.pdf(xp, s=sl, scale=np.exp(ml)), line=dict(color='#ff007f', width=2), name="Log-Normal PDF"))
        fig_dist.update_layout(template="plotly_dark", title="Final Price Distribution")
        st.plotly_chart(fig_dist, use_container_width=True)

with tab_pricing:
    st.subheader("GBS Analytical Pricing")
    
    col_k, col_res = st.columns([1, 2])
    
    with col_k:
        K = st.number_input("Strike (K)", value=S0)
        opt_type = st.radio("Type", ["Call", "Put"])
        
        # Analytical Logic (using cost of carry b)
        d1 = (np.log(S0/K) + (b + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        
        if opt_type == "Call":
            price = S0 * np.exp((b-r)*T) * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
            payoffs = np.maximum(final_prices - K, 0)
        else:
            price = K * np.exp(-r*T) * norm.cdf(-d2) - S0 * np.exp((b-r)*T) * norm.cdf(-d1)
            payoffs = np.maximum(K - final_prices, 0)
            
        mc_price = np.mean(payoffs) * np.exp(-r * T)
        
        st.metric("GBS Analytical Price", f"${price:.4f}")
        st.metric("Monte Carlo Price", f"${mc_price:.4f}")

    with col_res:
        st.markdown("### Model Convergence")
        cum_mc = (np.cumsum(payoffs) / (np.arange(n_paths) + 1)) * np.exp(-r * T)
        fig_conv = go.Figure()
        fig_conv.add_trace(go.Scatter(y=cum_mc, name="MC Estimate", line=dict(color='#00d4ff')))
        fig_conv.add_hline(y=price, line_dash="dash", line_color="#ff007f", annotation_text="GBS Analytical")
        fig_conv.update_layout(template="plotly_dark", height=350)
        st.plotly_chart(fig_conv, use_container_width=True)

with tab_insights:
    st.markdown("""
    ## 🧠 The "Swiss Army Knife" of Option Pricing
    
    The **Generalized Black-Scholes** model is essential because it allows a single code implementation to handle multiple asset classes by simply adjusting the relationship between $r$ and $b$.
    
    ### ⚡ Key takeaways for each sub-model:
    
    - **Black-Scholes ($b=r$):** The standard for equity options where the stock pays no dividends.
    - **BSM ($b=r-q$):** Used for equity indices and stocks with continuous dividend streams.
    - **Black-76 ($b=0$):** Used for options on futures (e.g., Eurodollar futures, Oil futures). Since futures have no cost of carry (initial margin is not the price), $b=0$.
    - **Garman-Kohlhagen ($b=rd-rf$):** Used for FX options. Holding foreign currency provides a yield equal to the foreign interest rate ($rf$).
    """)
