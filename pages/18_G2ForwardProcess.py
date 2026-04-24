import streamlit as st
import QuantLib as ql
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Page Config
st.set_page_config(page_title="G2++ Forward Measure", layout="wide")

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

r0 = st.sidebar.number_input("Initial Rate (r₀)", 0.03, step=0.005, format="%.3f")

st.sidebar.subheader("Factor 1 (x)")
a = st.sidebar.slider("Reversion (a)", 0.01, 1.0, 0.1)
sigma = st.sidebar.slider("Volatility (σ)", 0.001, 0.1, 0.01, format="%.3f")

st.sidebar.subheader("Factor 2 (y)")
b = st.sidebar.slider("Reversion (b)", 0.01, 1.0, 0.5)
eta = st.sidebar.slider("Volatility (η)", 0.001, 0.1, 0.008, format="%.3f")

st.sidebar.subheader("Measure Maturity")
T_measure = st.sidebar.slider("Forward Maturity (T)", 1.0, 30.0, 10.0)
rho = st.sidebar.slider("Correlation (ρ)", -1.0, 1.0, -0.7)

# Settings
st.sidebar.markdown("---")
T_horizon = st.sidebar.slider("Horizon (Years)", 1.0, 30.0, 5.0)
steps = st.sidebar.slider("Steps", 100, 500, 252)
n_paths = st.sidebar.number_input("Paths", 1, 500, 50)
seed = st.sidebar.number_input("Seed", 0, 1000, 42)

# Global QuantLib Setup
today = ql.Date.todaysDate()
ql.Settings.instance().evaluationDate = today
day_count = ql.Actual365Fixed()
ts = ql.YieldTermStructureHandle(ql.FlatForward(today, r0, day_count))

# --- Logic Functions ---

def simulate_g2_forward_ql(a, sigma, b, eta, rho, T_horizon, T_measure, steps, n_paths, seed):
    # G2 Forward Process (Constructor takes 5 parameters: a, sigma, b, eta, rho)
    process = ql.G2ForwardProcess(a, sigma, b, eta, rho)
    process.setForwardMeasureTime(float(T_measure))
    
    rsg = ql.GaussianMultiPathGenerator(process, ql.TimeGrid(T_horizon, steps), 
                                        ql.GaussianRandomSequenceGenerator(
                                            ql.UniformRandomSequenceGenerator(2 * steps, ql.UniformRandomGenerator(seed))
                                        ))
    
    x_paths = np.zeros((n_paths, steps + 1))
    y_paths = np.zeros((n_paths, steps + 1))
    r_paths = np.zeros((n_paths, steps + 1))
    
    for i in range(n_paths):
        multi_path = rsg.next().value()
        x_paths[i, :] = [multi_path[0][j] for j in range(len(multi_path[0]))]
        y_paths[i, :] = [multi_path[1][j] for j in range(len(multi_path[1]))]
        r_paths[i, :] = x_paths[i, :] + y_paths[i, :] + r0 # Simplified phi(t)
        
    time = np.linspace(0, T_horizon, steps + 1)
    return time, x_paths, y_paths, r_paths

# --- Main App ---
st.title("📉 G2++ Forward Process")
st.write("Two-factor interest rate dynamics under the T-Forward Measure.")

tab_theory, tab_sim, tab_analytics, tab_insights = st.tabs([
    "📚 Theory", "⚡ Simulation", "🎯 Analytics", "🧠 Insights"
])

# Run Simulation
time, x_data, y_data, r_data = simulate_g2_forward_ql(a, sigma, b, eta, rho, float(T_horizon), float(T_measure), steps, n_paths, seed)

with tab_theory:
    st.markdown("""
    ## 🔍 Multi-Factor Measure Theory
    
    The **G2 Forward Process** represents the evolution of the G2++ factors under the **$T$-Forward Measure** ($\mathbb{Q}^T$). In this measure, the numeraire is the Zero-Coupon Bond $P(t, T)$.
    
    ### 🏗️ Why it's more complex?
    When you have two factors, a change of measure introduces a **drift adjustment** to both factors. These adjustments depend on:
    1.  The volatility of the factor itself.
    2.  The sensitivity of the bond numeraire to that factor ($B_x$ and $B_y$).
    3.  The correlation between the factors ($\rho$).
    
    ### 📐 The Forward SDEs
    The factors $x_t$ and $y_t$ gain a deterministic drift component:
    """)
    st.latex(r"dx_t = [-a x_t - \text{adj}_x(t, T)] dt + \sigma dW_{1,t}^T")
    st.latex(r"dy_t = [-b y_t - \text{adj}_y(t, T)] dt + \eta dW_{2,t}^T")
    st.markdown("""
    Where the adjustments ensure that all asset prices divided by $P(t, T)$ are martingales.
    """)

with tab_sim:
    st.subheader(f"Short Rate Paths (T={T_measure}Y Measure)")
    fig_r = go.Figure()
    for i in range(min(n_paths, 15)):
        fig_r.add_trace(go.Scatter(x=time, y=r_data[i], mode='lines', line=dict(width=1.5), showlegend=False))
    fig_r.update_layout(template="plotly_dark", height=450, xaxis_title="Years", yaxis_title="Short Rate")
    st.plotly_chart(fig_r, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Factor x (Forward Drift)")
        fig_x = go.Figure()
        for i in range(min(n_paths, 10)):
            fig_x.add_trace(go.Scatter(x=time, y=x_data[i], mode='lines', line=dict(width=1, color='#00d4ff'), opacity=0.5, showlegend=False))
        fig_x.update_layout(template="plotly_dark", height=300)
        st.plotly_chart(fig_x, use_container_width=True)
    with col2:
        st.subheader("Factor y (Forward Drift)")
        fig_y = go.Figure()
        for i in range(min(n_paths, 10)):
            fig_y.add_trace(go.Scatter(x=time, y=y_data[i], mode='lines', line=dict(width=1, color='#ff007f'), opacity=0.5, showlegend=False))
        fig_y.update_layout(template="plotly_dark", height=300)
        st.plotly_chart(fig_y, use_container_width=True)

with tab_analytics:
    st.subheader("Joint Drift Adjustment Analysis")
    
    def get_b(t, T, k):
        return (1.0 - np.exp(-k * (T - t))) / k
        
    t_plot = np.linspace(0, min(T_horizon, T_measure), 100)
    
    # Simplified drift adjustments for visualization
    # Adj_x = sigma^2 * Bx + rho * sigma * eta * By
    adj_x = [sigma**2 * get_b(t, T_measure, a) + rho * sigma * eta * get_b(t, T_measure, b) for t in t_plot]
    adj_y = [eta**2 * get_b(t, T_measure, b) + rho * sigma * eta * get_b(t, T_measure, a) for t in t_plot]
    
    fig_adj = go.Figure()
    fig_adj.add_trace(go.Scatter(x=t_plot, y=adj_x, name="Drift Adj x", line=dict(color='#00d4ff')))
    fig_adj.add_trace(go.Scatter(x=t_plot, y=adj_y, name="Drift Adj y", line=dict(color='#ff007f')))
    fig_adj.update_layout(template="plotly_dark", title="Piecewise Factor Drift Adjustments", xaxis_title="Time (t)", yaxis_title="Adjustment Value")
    st.plotly_chart(fig_adj, use_container_width=True)
    
    st.info("Note how the correlation ρ impacts the drift adjustment of BOTH factors. This is the hallmark of multi-factor measure theory.")

with tab_insights:
    st.markdown("""
    ## 🧠 Advanced Quantitative Insights
    
    ### 1. The "Measure" of the Curve
    Working in the G2 forward measure is the standard approach for pricing **European Swaptions** using Monte Carlo or analytical approximations. By choosing $T$ as the swaption expiry, the underlying swap rate becomes much easier to manage.
    
    ### 2. Multi-Factor Convexity
    In a two-factor model, convexity adjustments are not just a function of individual factor volatilities; they are coupled by the **correlation** $\rho$. This explains why the spread between different forward rates can be non-zero even in a flat curve environment.
    
    ### 3. Path-Dependent Exotics
    For path-dependent products like **Bermudan Swaptions** or **Range Accruals**, the G2 Forward process provides a consistent way to simulate the joint evolution of multiple points on the yield curve while maintaining no-arbitrage consistency.
    
    ### 🛠️ Pro Quant Tip
    "When you change the forward maturity $T$, observe the 'Drift Adj y'. Because Factor 2 (y) usually has higher mean reversion ($b > a$), its drift adjustment decays faster as $t$ approaches $T$. This creates the complex curve dynamics that banks use to hedge their largest interest rate exposures."
    """)

st.markdown("---")
st.caption("Quantitative Finance Portfolio | G2++ Forward Measure Implementation")