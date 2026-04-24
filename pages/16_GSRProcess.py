import streamlit as st
import QuantLib as ql
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Page Config
st.set_page_config(page_title="GSR (Gaussian Short Rate) Process", layout="wide")

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

# Yield curve base
r0 = st.sidebar.number_input("Initial Yield (r₀)", 0.03, step=0.005, format="%.3f")

# Piecewise Volatilities
st.sidebar.subheader("Piecewise Volatilities")
vol1 = st.sidebar.slider("Vol 0-2Y", 0.001, 0.05, 0.01, format="%.3f")
vol2 = st.sidebar.slider("Vol 2Y-10Y", 0.001, 0.05, 0.008, format="%.3f")
vol3 = st.sidebar.slider("Vol 10Y+", 0.001, 0.05, 0.012, format="%.3f")

# Piecewise Reversions
st.sidebar.subheader("Piecewise Reversions")
rev1 = st.sidebar.slider("Rev 0-5Y", 0.0, 1.0, 0.1)
rev2 = st.sidebar.slider("Rev 5Y+", 0.0, 1.0, 0.05)

# Forward Measure Maturity
st.sidebar.markdown("---")
T_measure = st.sidebar.slider("Measure Maturity (T)", 1.0, 30.0, 10.0)

# Settings
st.sidebar.markdown("---")
T_horizon = st.sidebar.slider("Simulation Horizon", 1, 30, 10)
steps = st.sidebar.slider("Steps", 100, 1000, 252)
n_paths = st.sidebar.number_input("Paths", 1, 500, 100)
seed = st.sidebar.number_input("Seed", 0, 1000, 42)

# Global QuantLib Setup
today = ql.Date.todaysDate()
ql.Settings.instance().evaluationDate = today
day_count = ql.Actual365Fixed()
ts = ql.YieldTermStructureHandle(ql.FlatForward(today, r0, day_count))

# --- Logic Functions ---

def simulate_gsr_ql(vol_list, rev_list, T_horizon, T_measure, steps, n_paths, seed):
    # GSR requires piecewise dates
    vol_dates = [today + ql.Period(2, ql.Years), today + ql.Period(10, ql.Years)]
    rev_dates = [today + ql.Period(5, ql.Years)]
    
    # QuantLib uses QuoteHandles for piecewise parameters
    vols = [ql.QuoteHandle(ql.SimpleQuote(v)) for v in vol_list]
    revs = [ql.QuoteHandle(ql.SimpleQuote(r)) for r in rev_list]
    
    # GSR Process with explicit vector types for SWIG
    process = ql.GsrProcess(
        ql.QuoteHandleVector(vols), 
        ql.DateVector(vol_dates), 
        ql.QuoteHandleVector(revs), 
        ql.DateVector(rev_dates), 
        ts
    )
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
st.title("📉 Gaussian Short Rate (GSR) Process")
st.write("A professional one-factor model with piecewise constant parameters.")

tab_theory, tab_sim, tab_analytics, tab_insights = st.tabs([
    "📚 Theory", "⚡ Simulation", "🎯 Analytics", "🧠 Insights"
])

# Run Simulation
time, rate_paths = simulate_gsr_ql([vol1, vol2, vol3], [rev1, rev2], T_horizon, T_measure, steps, n_paths, seed)

with tab_theory:
    st.markdown("""
    ## 🔍 What is the GSR Process?
    
    The **Gaussian Short Rate (GSR)** model is a generalization of the Hull-White model. While standard Hull-White assumes constant mean reversion and volatility, GSR allows these parameters to be **piecewise constant** functions of time.
    
    ### 🏗️ Why it's superior for Calibration
    In professional markets, the volatility of interest rates is not uniform across all time horizons. By allowing $\sigma(t)$ and $a(t)$ to change at specific dates, the GSR model can be **perfectly calibrated** to an entire strip of swaptions or caps.
    
    ### 📐 The SDE (in T-forward measure)
    """)
    st.latex(r"dr_t = [\theta(t) - a(t) r_t - \text{convexity}(t, T)] dt + \sigma(t) dW_t^T")
    st.markdown("""
    The model remains "Gaussian," meaning the distribution of the short rate at any future time is still normal, but the variance is now an integral of the piecewise volatility function.
    """)

with tab_sim:
    st.subheader(f"GSR Paths under {T_measure}Y-Forward Measure")
    
    fig_gsr = go.Figure()
    # Expected path
    mean_path = np.mean(rate_paths, axis=0)
    fig_gsr.add_trace(go.Scatter(x=time, y=mean_path, name="Mean Path", line=dict(color='#ff007f', width=3)))
    
    n_display = min(n_paths, 20)
    for i in range(n_display):
        fig_gsr.add_trace(go.Scatter(x=time, y=rate_paths[i], mode='lines', line=dict(width=1), opacity=0.3, showlegend=False))
        
    fig_gsr.update_layout(template="plotly_dark", height=500, xaxis_title="Years", yaxis_title="Short Rate (r)")
    st.plotly_chart(fig_gsr, use_container_width=True)

with tab_analytics:
    st.subheader("Piecewise Parameter Structure")
    
    col_v, col_r = st.columns(2)
    
    with col_v:
        st.markdown("### Piecewise Volatility $\sigma(t)$")
        t_v = [0, 2, 2, 10, 10, 30]
        v_vals = [vol1, vol1, vol2, vol2, vol3, vol3]
        fig_v = go.Figure()
        fig_v.add_trace(go.Scatter(x=t_v, y=v_vals, line=dict(color='#00d4ff', shape='hv'), fill='tozeroy'))
        fig_v.update_layout(template="plotly_dark", xaxis_title="Time (Years)", yaxis_title="Vol")
        st.plotly_chart(fig_v, use_container_width=True)

    with col_r:
        st.markdown("### Piecewise Reversion $a(t)$")
        t_r = [0, 5, 5, 30]
        r_vals = [rev1, rev1, rev2, rev2]
        fig_r = go.Figure()
        fig_r.add_trace(go.Scatter(x=t_r, y=r_vals, line=dict(color='#ff007f', shape='hv'), fill='tozeroy'))
        fig_r.update_layout(template="plotly_dark", xaxis_title="Time (Years)", yaxis_title="Reversion")
        st.plotly_chart(fig_r, use_container_width=True)

with tab_insights:
    st.markdown("""
    ## 🧠 High-Level Insights
    
    ### 1. Calibration to Swaption Surface
    GSR is the primary "workhorse" for modeling interest rates when you need to match the market prices of **European Swaptions** for different maturities. The piecewise steps in volatility are calibrated to these market prices.
    
    ### 2. Time-Dependent Dynamics
    If a central bank is expected to change its policy regime in 2 years, a GSR model can account for this by shifting the mean reversion or volatility at the 2Y mark.
    
    ### 3. Comparison with Hull-White
    GSR is essentially a "non-homogeneous" Hull-White model. It retains all the mathematical tractability of HW (Gaussian distributions, analytical bond prices) while adding the flexibility needed for real-world market fitting.
    
    ### 🛠️ Pro Quant Tip
    "When setting up a GSR model, ensure your piecewise dates match the market pillars (e.g., 1Y, 2Y, 5Y, 10Y swaption expiry). If the volatility jumps too sharply between periods, it can create unrealistic 'spikes' in your forward volatility term structure."
    """)

st.markdown("---")
st.caption("Quantitative Finance Portfolio | GSR Process Implementation")