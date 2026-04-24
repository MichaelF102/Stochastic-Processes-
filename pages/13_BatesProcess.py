import streamlit as st
import QuantLib as ql
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Page Config
st.set_page_config(page_title="Bates Stochastic Volatility with Jumps", layout="wide")

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

# Heston Part
S0 = st.sidebar.number_input("Initial Price (S₀)", 100.0)
v0 = st.sidebar.slider("Initial Variance (v₀)", 0.01, 0.5, 0.04)
kappa = st.sidebar.slider("Mean Reversion (κ)", 0.1, 5.0, 2.0)
theta = st.sidebar.slider("Long-term Var (θ)", 0.01, 0.5, 0.04)
sigma_v = st.sidebar.slider("Vol of Vol (σv)", 0.1, 1.0, 0.3)
rho = st.sidebar.slider("Correlation (ρ)", -1.0, 1.0, -0.7)

# Jump Part
st.sidebar.markdown("---")
st.sidebar.subheader("Jump Parameters")
jump_lambda = st.sidebar.slider("Jump Intensity (λ)", 0.0, 5.0, 1.0)
jump_mu = st.sidebar.slider("Mean Jump Size (μJ)", -0.5, 0.5, -0.1)
jump_delta = st.sidebar.slider("Jump Vol (δJ)", 0.01, 0.5, 0.1)

# Settings
st.sidebar.markdown("---")
steps = st.sidebar.slider("Steps", 100, 500, 252)
n_paths = st.sidebar.number_input("Paths", 1, 500, 100)
seed = st.sidebar.number_input("Seed", 0, 1000, 42)

# --- Logic Functions ---

def simulate_bates_np(S0, v0, r, kappa, theta, sigma_v, rho, j_lambda, j_mu, j_delta, T, steps, n_paths, seed):
    np.random.seed(seed)
    dt = T / steps
    time = np.linspace(0, T, steps + 1)
    
    # Pre-calculate shocks
    Z1 = np.random.standard_normal((n_paths, steps))
    Z_temp = np.random.standard_normal((n_paths, steps))
    Z2 = rho * Z1 + np.sqrt(1 - rho**2) * Z_temp
    
    # Jump setup
    # Expected relative jump size: k = exp(mu + 0.5*delta^2) - 1
    k_jump = np.exp(j_mu + 0.5 * j_delta**2) - 1
    
    prices = np.zeros((n_paths, steps + 1))
    variances = np.zeros((n_paths, steps + 1))
    prices[:, 0] = S0
    variances[:, 0] = v0
    
    for j in range(steps):
        v = np.maximum(variances[:, j], 0)
        
        # Continuous part
        prices[:, j+1] = prices[:, j] * np.exp((r - j_lambda * k_jump - 0.5 * v) * dt + np.sqrt(v * dt) * Z1[:, j])
        
        # Jump part
        n_jumps = np.random.poisson(j_lambda * dt, n_paths)
        if np.any(n_jumps > 0):
            mask = n_jumps > 0
            # Jumps are log-normally distributed
            jumps = np.random.normal(n_jumps[mask] * j_mu, np.sqrt(n_jumps[mask]) * j_delta)
            prices[mask, j+1] *= np.exp(jumps)
            
        # Variance part
        variances[:, j+1] = variances[:, j] + kappa * (theta - v) * dt + sigma_v * np.sqrt(v * dt) * Z2[:, j]
        
    return time, prices, variances

# --- Main App ---
st.title("📉 The Bates Process")
st.write("A 'Super-Model' combining Stochastic Volatility (Heston) with Jump-Diffusion (Merton).")

tab_theory, tab_sim, tab_insights = st.tabs(["📚 Theory", "⚡ Simulation", "🧠 Insights"])

# Run Simulation
time, price_data, var_data = simulate_bates_np(S0, v0, 0.05, kappa, theta, sigma_v, rho, jump_lambda, jump_mu, jump_delta, 1.0, steps, n_paths, seed)

with tab_theory:
    st.markdown("""
    ## 🔍 Why the Bates Model?
    
    The **Bates Model (1996)** is one of the most comprehensive equity models. It solves the specific failures of both the Heston and Merton models.
    
    ### 🏗️ The Problem
    1.  **Heston Model**: Excellent at capturing the long-term volatility skew, but fails to match the very steep skew observed in short-dated options.
    2.  **Merton Model**: Excellent at capturing short-term skew via jumps, but the volatility is constant between jumps, which is unrealistic for long horizons.
    
    ### 🤝 The Solution: Bates
    By combining them, the model uses **Jumps** to explain sudden market crashes (short-term skew) and **Stochastic Volatility** to explain the varying levels of market "nervousness" (long-term smile).
    
    ### 📐 The SDEs
    """)
    st.latex(r"dS_t = (r - \lambda \kappa) S_t dt + \sqrt{v_t} S_t dW_{1,t} + S_t (J - 1) dN_t")
    st.latex(r"dv_t = \kappa (\theta - v_t) dt + \sigma_v \sqrt{v_t} dW_{2,t}")
    st.markdown("""
    Where $dN_t$ is a Poisson process and $J$ is the random jump size.
    """)

with tab_sim:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Price Paths (with Jumps)")
        fig_p = go.Figure()
        for i in range(min(n_paths, 15)):
            fig_p.add_trace(go.Scatter(x=time, y=price_data[i], mode='lines', line=dict(width=1.5), showlegend=False))
        fig_p.update_layout(template="plotly_dark", height=450, xaxis_title="Time", yaxis_title="Price")
        st.plotly_chart(fig_p, use_container_width=True)
        
    with col2:
        st.subheader("Variance Paths (Stochastic Vol)")
        fig_v = go.Figure()
        for i in range(min(n_paths, 15)):
            fig_v.add_trace(go.Scatter(x=time, y=var_data[i], mode='lines', line=dict(width=1.5, color='#00d4ff'), opacity=0.6, showlegend=False))
        fig_v.update_layout(template="plotly_dark", height=450, xaxis_title="Time", yaxis_title="Variance")
        st.plotly_chart(fig_v, use_container_width=True)

with tab_insights:
    st.markdown("""
    ## 🧠 High-Level Insights
    
    ### 1. Calibration to the "Whole" Surface
    The Bates model is the standard choice for quants who need a single model that can price options across **all maturities** (from 1 week to 10 years). The jumps handle the 1-week skew, and the Heston component handles the 10-year smile.
    
    ### 2. Risk Management (VaR)
    Because it includes both jumps and stochastic volatility, the Bates model provides much more realistic **Value at Risk (VaR)** estimates during crisis periods than standard Gaussian models.
    
    ### 3. Hedging Complexity
    Trading under the Bates model is complex because you are hedging two different risks: the continuous volatility risk (Vega) and the discrete jump risk (Gamma/Jump-to-Default).
    
    ### 🛠️ Pro Quant Tip
    "When you see the Bates model, look at $\lambda$. If $\lambda$ is high, the model expects frequent shocks. If $\sigma_v$ is high, it expects the 'volatility of volatility' to be the main driver. Calibration involves finding the right balance between these two 'engines of uncertainty'."
    """)

st.markdown("---")
st.caption("Quantitative Finance Portfolio | Bates Stochastic Volatility with Jumps Implementation")