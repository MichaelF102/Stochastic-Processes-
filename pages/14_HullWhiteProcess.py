import streamlit as st
import QuantLib as ql
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Page Config
st.set_page_config(page_title="Hull-White Short Rate Model", layout="wide")

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

# Settings
st.sidebar.markdown("---")
T_horizon = st.sidebar.slider("Time Horizon (Years)", 1, 30, 10)
steps = st.sidebar.slider("Steps", 100, 1000, 252)
n_paths = st.sidebar.number_input("Paths", 1, 500, 100)
seed = st.sidebar.number_input("Seed", 0, 1000, 42)

# Global QuantLib Setup
today = ql.Date.todaysDate()
ql.Settings.instance().evaluationDate = today
day_count = ql.Actual365Fixed()
yield_curve = ql.FlatForward(today, r0, day_count)
ts = ql.YieldTermStructureHandle(yield_curve)

# --- Logic Functions ---

def simulate_hw_ql(r0, a, sigma, T, steps, n_paths, seed):
    process = ql.HullWhiteProcess(ts, a, sigma)
    
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
st.title("📈 Hull-White Short Rate Model")
st.write("Professional-grade interest rate modeling and analytics.")

tab_theory, tab_sim, tab_analytics, tab_insights = st.tabs([
    "📚 Theory", "⚡ Simulation", "🎯 Analytics", "🧠 Insights"
])

# Run Simulation
time, rate_paths = simulate_hw_ql(r0, a, sigma, float(T_horizon), steps, n_paths, seed)

with tab_theory:
    st.markdown("""
    The **Hull-White Model (1990)** is a "short-rate" model used to describe the stochastic evolution of instantaneous interest rates. It is an extension of the **Vasicek model** that allows the model to fit the current yield curve perfectly.

    ### 1. The Stochastic Differential Equation (SDE)
    The evolution of the short rate $r_t$ is given by:
    """)
    st.latex(r"dr_t = [\theta(t) - a r_t] dt + \sigma dW_t")
    st.markdown("""
    Where:
    - **$a$**: The speed of mean reversion.
    - **$\sigma$**: The volatility of the short rate.
    - **$\theta(t)$**: A time-dependent drift calibrated to the initial yield curve.

    ### 2. Analytical Zero-Coupon Bond Price
    One of the greatest advantages of the Hull-White model is that it is an **Affine Term Structure Model**. The price of a Zero-Coupon Bond $P(t, T)$ can be written as:
    """)
    st.latex(r"P(t, T) = A(t, T) e^{-B(t, T) r_t}")
    st.markdown("""
    Where $B(t, T) = \frac{1}{a}(1 - e^{-a(T-t)})$, and $A(t, T)$ is a more complex function related to the initial yield curve.
    """)

with tab_sim:
    st.subheader("Short Rate Path Simulation")
    fig_hw = go.Figure()
    
    # Expected path (Mean Reversion)
    mean_path = np.mean(rate_paths, axis=0)
    fig_hw.add_trace(go.Scatter(x=time, y=mean_path, name="Mean Path", line=dict(color='#ff007f', width=3)))
    
    n_display = min(n_paths, 25)
    for i in range(n_display):
        fig_hw.add_trace(go.Scatter(x=time, y=rate_paths[i], mode='lines', line=dict(width=1), opacity=0.3, showlegend=False))
        
    fig_hw.update_layout(template="plotly_dark", height=500, xaxis_title="Years", yaxis_title="Short Rate (r)", hovermode="x")
    st.plotly_chart(fig_hw, use_container_width=True)

with tab_analytics:
    st.subheader("Yield Curve & ZCB Analytics")
    
    col_a1, col_a2 = st.columns(2)
    
    # Generate maturities for analytics
    maturities = np.linspace(0.1, 30, 100)
    
    with col_a1:
        st.markdown("### Zero-Coupon Bond Prices $P(0, T)$")
        # Prices today based on the initial flat curve r0
        zcb_prices = [ql.FlatForward(today, r0, day_count).discount(m) for m in maturities]
        
        fig_zcb = go.Figure()
        fig_zcb.add_trace(go.Scatter(x=maturities, y=zcb_prices, name="ZCB Price", line=dict(color='#00d4ff')))
        fig_zcb.update_layout(template="plotly_dark", xaxis_title="Maturity (Years)", yaxis_title="Price ($)")
        st.plotly_chart(fig_zcb, use_container_width=True)

    with col_a2:
        st.markdown("### Bond Volatility Term Structure")
        # In HW, the volatility of a ZCB with maturity T is sigma * B(0, T)
        def get_b_const(t, T, a):
            return (1.0 - np.exp(-a * (T - t))) / a
            
        bond_vols = [sigma * get_b_const(0, m, a) for m in maturities]
        
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Scatter(x=maturities, y=bond_vols, name="ZCB Volatility", line=dict(color='#ff007f')))
        fig_vol.update_layout(template="plotly_dark", xaxis_title="Maturity (Years)", yaxis_title="Absolute Volatility")
        st.plotly_chart(fig_vol, use_container_width=True)

    st.markdown("---")
    st.subheader("European Swaption Pricing")
    
    # Setup a simple European Swaption
    sw_maturity = st.selectbox("Swaption Maturity (Years)", [1, 2, 5, 10], index=1)
    sw_length = st.selectbox("Swap Length (Years)", [1, 5, 10, 20], index=2)
    
    # QuantLib Pricing
    settle_date = today + ql.Period(sw_maturity, ql.Years)
    exercise = ql.EuropeanExercise(settle_date)
    
    # Fixed rate bond for the swap (simplified as a coupon bond for demonstration)
    # In HW, swaptions have analytical solutions
    hw_model = ql.HullWhite(ts, a, sigma)
    hw_engine = ql.JamshidianSwaptionEngine(hw_model)
    
    # We'll display the Jamshidian analytical capacity
    st.info(f"The Hull-White model allows for analytical pricing of European Swaptions via **Jamshidian's Decomposition**. This transforms a swaption into a portfolio of options on zero-coupon bonds.")
    
    # Display analytical ZCB option price (Caplet-like)
    st.markdown(f"**Sample Analytics for {sw_maturity}Y maturity:**")
    st.metric("ZCB Price P(0, {sw_maturity}Y)", f"{ts.discount(sw_maturity):.4f}")
    st.metric("Short Rate Volatility (Input σ)", f"{sigma*100:.2f}%")

with tab_insights:
    st.markdown("""
    ## 🧠 Quantitative Insights
    
    ### 1. Fitting the Term Structure
    The primary reason major banks use Hull-White is its ability to **perfectly fit the initial yield curve**. By making the drift $\theta(t)$ time-dependent, the model ensures that the price of a zero-coupon bond today matches the market exactly.
    
    ### 2. Mean Reversion vs. Long-Term Vol
    The parameter $a$ is critical. A higher $a$ means the short rate returns to its mean faster, which **dampens** the volatility of long-term rates. This is why the "Bond Volatility" curve levels off at higher maturities.
    
    ### 3. Bermudan Swaptions
    While European swaptions have analytical formulas in HW, **Bermudan swaptions** (options to enter a swap on multiple dates) are priced using a **Hull-White Tree** or **Lattice**. The HW model's simple structure makes these trees very efficient to build.
    
    ### 🛠️ Pro Quant Tip
    "The Hull-White model is a 'one-factor' model. This means all points on the yield curve are perfectly correlated. For complex structures like **Spread Options** or **Yield Curve Steepeners**, you would need a multi-factor model like **G2++**."
    """)

st.markdown("---")
st.caption("Quantitative Finance Portfolio | Hull-White Short Rate Implementation & Analytics")