import streamlit as st
import QuantLib as ql
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Page Config
st.set_page_config(page_title="Heston SLV (Stochastic Local Volatility)", layout="wide")

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
</style>
""", unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.header("🕹️ Control Panel")

# Heston Parameters
S0 = st.sidebar.number_input("Initial Price (S₀)", 100.0)
v0 = st.sidebar.slider("Initial Variance (v₀)", 0.01, 0.5, 0.04)
kappa = st.sidebar.slider("Mean Reversion (κ)", 0.1, 5.0, 1.5)
theta = st.sidebar.slider("Long-term Var (θ)", 0.01, 0.5, 0.04)
sigma_v = st.sidebar.slider("Vol of Vol (σv)", 0.1, 1.0, 0.4)
rho = st.sidebar.slider("Correlation (ρ)", -1.0, 1.0, -0.6)

# Local Vol Component
st.sidebar.markdown("---")
st.sidebar.subheader("Local Vol Leverage")
# Simplified leverage function: L(S) = (S/S0)^beta
# beta < 0 creates a local vol skew (downward)
beta = st.sidebar.slider("Leverage Function Skew (β)", -2.0, 2.0, -0.5)

# Settings
st.sidebar.markdown("---")
steps = st.sidebar.slider("Steps", 100, 500, 252)
n_paths = st.sidebar.number_input("Paths", 1, 500, 100)
seed = st.sidebar.number_input("Seed", 0, 1000, 42)

# --- Logic Functions ---

def simulate_slv_np(S0, v0, kappa, theta, sigma_v, rho, beta, T, steps, n_paths, seed):
    np.random.seed(seed)
    dt = T / steps
    time = np.linspace(0, T, steps + 1)
    
    # Correlated shocks
    Z1 = np.random.standard_normal((n_paths, steps))
    Z_temp = np.random.standard_normal((n_paths, steps))
    Z2 = rho * Z1 + np.sqrt(1 - rho**2) * Z_temp
    
    prices = np.zeros((n_paths, steps + 1))
    prices_base = np.zeros((n_paths, steps + 1)) # Pure Heston for comparison
    variances = np.zeros((n_paths, steps + 1))
    
    prices[:, 0] = S0
    prices_base[:, 0] = S0
    variances[:, 0] = v0
    
    for j in range(steps):
        v = np.maximum(variances[:, j], 0)
        
        # Leverage Function: L(S) = (S/S0)^beta
        # This is a simplified model of the actual market leverage function
        L = (prices[:, j] / S0)**beta
        
        # Heston SLV: dS = r*S*dt + sqrt(v)*L(S)*S*dW1
        # (Assuming r=0 for path clarity)
        prices[:, j+1] = prices[:, j] * np.exp(-0.5 * v * (L**2) * dt + np.sqrt(v) * L * np.sqrt(dt) * Z1[:, j])
        
        # Pure Heston: dS = r*S*dt + sqrt(v)*S*dW1
        prices_base[:, j+1] = prices_base[:, j] * np.exp(-0.5 * v * dt + np.sqrt(v) * np.sqrt(dt) * Z1[:, j])
        
        # Variance: dv = kappa(theta - v)dt + sigma_v*sqrt(v)*dW2
        variances[:, j+1] = variances[:, j] + kappa * (theta - v) * dt + sigma_v * np.sqrt(v * dt) * Z2[:, j]
        
    return time, prices, prices_base, variances

# --- Main App ---
st.title("📊 Heston Stochastic Local Volatility (SLV)")
st.write("The 'Gold Standard' for pricing path-dependent options (Barriers, Autocallables).")

tab_theory, tab_sim, tab_insights = st.tabs(["📚 Theory", "⚡ Simulation", "🧠 Insights"])

# Run Simulation
time, prices_slv, prices_heston, var_data = simulate_slv_np(S0, v0, kappa, theta, sigma_v, rho, beta, 1.0, steps, n_paths, seed)

with tab_theory:
    st.markdown("""
    ## 🔍 Why do we need SLV?
    
    The **Heston SLV** model is a hybrid model that combines the strengths of the **Heston Stochastic Volatility** model and the **Dupire Local Volatility** model.
    
    ### 🏗️ The Problem
    1.  **Local Volatility (Dupire)**: Can perfectly match the current market prices of all vanilla options (the "vol surface"). However, it has unrealistic dynamics—it assumes volatility is a deterministic function of the spot price.
    2.  **Stochastic Volatility (Heston)**: Has realistic "forward volatility" dynamics and captures the volatility smile. However, it cannot perfectly match every single market price simultaneously.
    
    ### 🤝 The Solution: Heston SLV
    The SLV model introduces a **leverage function** $L(S, t)$ that modifies the stochastic volatility process to ensure perfect calibration to the market surface.
    
    ### 📐 The SDEs
    """)
    st.latex(r"dS_t = r S_t dt + \sqrt{v_t} L(S_t, t) S_t dW_{1,t}")
    st.latex(r"dv_t = \kappa (\theta - v_t) dt + \sigma_v \sqrt{v_t} dW_{2,t}")
    st.markdown("""
    Where $L(S, t)$ is the leverage function, often calibrated using a Fokker-Planck equation or particle methods to satisfy the **Marginal Condition**.
    """)

with tab_sim:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Heston SLV Paths (with Leverage)")
        fig_slv = go.Figure()
        for i in range(min(n_paths, 10)):
            fig_slv.add_trace(go.Scatter(x=time, y=prices_slv[i], mode='lines', line=dict(width=1.5), showlegend=False))
        fig_slv.update_layout(template="plotly_dark", height=450, xaxis_title="Time", yaxis_title="Price")
        st.plotly_chart(fig_slv, use_container_width=True)
        
    with col2:
        st.subheader("Pure Heston Paths (No Leverage)")
        fig_h = go.Figure()
        for i in range(min(n_paths, 10)):
            fig_h.add_trace(go.Scatter(x=time, y=prices_heston[i], mode='lines', line=dict(width=1.5, color='gray'), opacity=0.5, showlegend=False))
        fig_h.update_layout(template="plotly_dark", height=450, xaxis_title="Time", yaxis_title="Price")
        st.plotly_chart(fig_h, use_container_width=True)

    st.markdown("### How the Leverage Function $(\\beta)$ Changes Paths")
    st.info(f"With β = {beta}, the volatility is scaled by (S/S0)^{beta}. " + 
             ("If prices fall, volatility accelerates (Skew)." if beta < 0 else "If prices rise, volatility accelerates." if beta > 0 else "Volatility is independent of spot (Pure Heston)."))

with tab_insights:
    st.markdown("""
    ## 🧠 High-Level Insights
    
    ### 1. Barrier Option Pricing
    Standard Heston models often misprice **Barrier Options** because they don't match the market's implied volatility for all strikes. SLV is the industry standard for these "Exotic" products because it ensures the model "sees" the same vol surface as the market.
    
    ### 2. Forward Volatility Dynamics
    Unlike pure Local Vol, Heston SLV produces a realistic **Volatility of Volatility**. This is crucial for products where the value depends on future volatility levels (e.g., Forward Starting Options).
    
    ### 3. Vanna-Volga Effects
    SLV models naturally capture the **Vanna** (sensitivity of delta to vol) and **Volga** (sensitivity of vega to vol) effects that are prominent in FX markets.
    
    ### 🛠️ Pro Quant Tip
    "Think of Heston SLV as a Heston model that has been 'bent' by the leverage function to fit the market. If your leverage function $L(S, t)$ is close to 1.0, your Heston model was already doing a good job. If it's highly skewed, your Heston parameters need recalibration."
    """)

st.markdown("---")
st.caption("Quantitative Finance Portfolio | Heston Stochastic Local Volatility (SLV) Implementation")