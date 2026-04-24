import streamlit as st
import QuantLib as ql
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Page Config
st.set_page_config(page_title="Hull-White Forward Measure", layout="wide")

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

# HW Parameters
r0 = st.sidebar.number_input("Initial Short Rate (r₀)", 0.03, step=0.005, format="%.3f")
a = st.sidebar.slider("Mean Reversion Speed (a)", 0.01, 1.0, 0.1)
sigma = st.sidebar.slider("Short Rate Volatility (σ)", 0.001, 0.1, 0.01, format="%.3f")

# Forward Measure Parameter
st.sidebar.markdown("---")
st.sidebar.subheader("Measure Settings")
T_measure = st.sidebar.slider("Forward Measure Maturity (T)", 1.0, 30.0, 10.0)

# Settings
st.sidebar.markdown("---")
T_horizon = st.sidebar.slider("Simulation Horizon", 1.0, 30.0, 5.0)
steps = st.sidebar.slider("Steps", 100, 1000, 252)
n_paths = st.sidebar.number_input("Paths", 1, 500, 100)
seed = st.sidebar.number_input("Seed", 0, 1000, 42)

# Global QuantLib Setup
today = ql.Date.todaysDate()
ql.Settings.instance().evaluationDate = today
day_count = ql.Actual365Fixed()
ts = ql.YieldTermStructureHandle(ql.FlatForward(today, r0, day_count))

# --- Logic Functions ---

def simulate_hw_forward_ql(r0, a, sigma, T_horizon, T_measure, steps, n_paths, seed):
    # The Pythonic way to simulate the HW process under an arbitrary T-forward measure.
    # We initialize the Forward Process and explicitly set the measure maturity.
    process = ql.HullWhiteForwardProcess(ts, a, sigma)
    process.setForwardMeasureTime(float(T_measure))
    
    rsg = ql.GaussianRandomSequenceGenerator(
        ql.UniformRandomSequenceGenerator(steps, ql.UniformRandomGenerator(seed))
    )
    
    path_generator = ql.GaussianPathGenerator(process, float(T_horizon), steps, rsg, False)
    
    paths = np.zeros((n_paths, steps + 1))
    for i in range(n_paths):
        sample_path = path_generator.next().value()
        paths[i, :] = [sample_path[j] for j in range(len(sample_path))]
        
    time = np.linspace(0, T_horizon, steps + 1)
    return time, paths

# --- Main App ---
st.title("📈 Hull-White Forward Process")
st.write("Modeling interest rate dynamics under the T-Forward Measure.")

tab_theory, tab_sim, tab_analytics, tab_insights = st.tabs([
    "📚 Theory", "⚡ Simulation", "🎯 Analytics", "🧠 Insights"
])

# Run Simulation
time, rate_paths = simulate_hw_forward_ql(r0, a, sigma, T_horizon, T_measure, steps, n_paths, seed)

with tab_theory:
    st.markdown("""
    ## 🔍 What is the Forward Measure?
    
    In the standard Hull-White model, we work in the **Risk-Neutral Measure** ($\mathbb{Q}$), where the numeraire is the cash account $B_t$. However, for pricing interest rate derivatives, it is often easier to work in the **$T$-Forward Measure** ($\mathbb{Q}^T$).
    
    ### 🏗️ Change of Numeraire
    In the $T$-forward measure, the numeraire is the **Zero-Coupon Bond** $P(t, T)$ maturing at time $T$.
    
    ### 📐 The Forward SDE
    The dynamics of the short rate $r_t$ under the $T$-forward measure are:
    """)
    st.latex(r"dr_t = [\theta(t) - a r_t - \sigma^2 B(t, T)] dt + \sigma dW_t^T")
    st.markdown("""
    Where:
    - **$B(t, T)$**: The sensitivity of the bond price to the short rate, $B(t, T) = \frac{1}{a}(1 - e^{-a(T-t)})$.
    - **$\sigma^2 B(t, T)$**: The **Drift Adjustment** (convexity correction) required when changing from the risk-neutral to the forward measure.
    
    ### 🎯 Why use it?
    In this measure, any asset price $X_t$ divided by the bond price $P(t, T)$ is a martingale. This makes the pricing of options on bonds or forward rates much simpler.
    """)

with tab_sim:
    st.subheader(f"Short Rate Paths under the {T_measure}Y-Forward Measure")
    
    fig_hw = go.Figure()
    
    # Average path
    mean_path = np.mean(rate_paths, axis=0)
    fig_hw.add_trace(go.Scatter(x=time, y=mean_path, name="Empirical Mean Path", line=dict(color='#ff007f', width=3)))
    
    n_display = min(n_paths, 25)
    for i in range(n_display):
        fig_hw.add_trace(go.Scatter(x=time, y=rate_paths[i], mode='lines', line=dict(width=1), opacity=0.3, showlegend=False))
        
    fig_hw.update_layout(template="plotly_dark", height=500, xaxis_title="Years", yaxis_title="Short Rate (r)", hovermode="x")
    st.plotly_chart(fig_hw, use_container_width=True)
    
    st.info(f"The paths are simulated under a measure where the {T_measure}Y bond is the numeraire. Note the drift adjustment relative to the standard HW process.")

with tab_analytics:
    st.subheader("Measure Analytics")
    
    col_a1, col_a2 = st.columns(2)
    
    with col_a1:
        st.markdown(f"### Drift Adjustment Term")
        # Plot sigma^2 * B(t, T) over the simulation horizon
        def get_b(t, T, a):
            return (1.0 - np.exp(-a * (T - t))) / a
            
        t_plot = np.linspace(0, min(T_horizon, T_measure), 100)
        drift_adj = [sigma**2 * get_b(t, T_measure, a) for t in t_plot]
        
        fig_adj = go.Figure()
        fig_adj.add_trace(go.Scatter(x=t_plot, y=drift_adj, name="Drift Adj", line=dict(color='#00d4ff')))
        fig_adj.update_layout(template="plotly_dark", xaxis_title="Current Time (t)", yaxis_title="Convexity Correction", 
                              title=f"Adjustment relative to {T_measure}Y Maturity")
        st.plotly_chart(fig_adj, use_container_width=True)

    with col_a2:
        st.markdown("### Forward Rate Martingale Test")
        # Under the T-forward measure, the forward rate F(t, T-dt, T) should be a martingale
        st.info("In the $T$-forward measure, the forward interest rate $F(t, T^*, T)$ for $t \le T^* \le T$ is a martingale (has zero drift).")
        st.metric("Measure Numeraire", f"{T_measure}Y Zero Bond")
        st.metric("Volatility Parameter (σ)", f"{sigma*100:.2f}%")

with tab_insights:
    st.markdown("""
    ## 🧠 High-Level Insights
    
    ### 1. Convexity Adjustments
    The difference between the expected short rate in the risk-neutral measure and the forward measure is the **Convexity Adjustment**. This is why "Forward Rates" are not equal to "Expected Future Spot Rates" in a stochastic world.
    
    ### 2. Pricing Options on Bonds
    If you are pricing an option that matures at time $T$, working in the $T$-forward measure is the standard "shortcut." It allows you to ignore the stochastic nature of the discount factor until the very last step.
    
    ### 3. Connection to Libor Market Model (LMM)
    The $T$-forward measure is the building block of the **Libor Market Model**. In LMM, each forward rate is modeled in its own natural forward measure, where it is a simple driftless Brownian motion.
    
    ### 🛠️ Pro Quant Tip
    "When you change the measure maturity $T$, you are changing the 'units' of your economy. If $T$ is very far in the future, the drift adjustment becomes larger, reflecting the higher sensitivity of long-term bonds to interest rate volatility."
    """)