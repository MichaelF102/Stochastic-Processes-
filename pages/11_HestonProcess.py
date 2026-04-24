import streamlit as st
import QuantLib as ql
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import norm

# Page Config
st.set_page_config(page_title="Heston Stochastic Volatility", layout="wide")

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

# Asset Parameters
S0 = st.sidebar.number_input("Initial Price (S₀)", 100.0, step=1.0)
r = st.sidebar.slider("Risk-free Rate (r)", 0.0, 0.2, 0.05, step=0.01)

# Variance Parameters
v0 = st.sidebar.slider("Initial Variance (v₀)", 0.01, 1.0, 0.04, step=0.01)
kappa = st.sidebar.slider("Mean Reversion Speed (κ)", 0.1, 10.0, 2.0, step=0.1)
theta = st.sidebar.slider("Long-term Variance (θ)", 0.01, 1.0, 0.04, step=0.01)
sigma_v = st.sidebar.slider("Vol-of-Vol (σv)", 0.01, 1.0, 0.3, step=0.01)
rho = st.sidebar.slider("Correlation (ρ)", -1.0, 1.0, -0.7, step=0.05)

# Simulation Settings
st.sidebar.markdown("---")
T = st.sidebar.slider("Time Horizon (T)", 0.1, 5.0, 1.0, step=0.1)
steps = st.sidebar.slider("Steps", 100, 1000, 252)
n_paths = st.sidebar.number_input("Number of Paths", 1, 1000, 100)
seed = st.sidebar.number_input("Random Seed", 0, 10000, 42)
method = st.sidebar.selectbox("Simulation Method", ["NumPy (Euler)", "QuantLib"])

# Global Setup
today = ql.Date.todaysDate()
ql.Settings.instance().evaluationDate = today

# --- Logic Functions ---

def simulate_heston_np(S0, v0, r, kappa, theta, sigma_v, rho, T, steps, n_paths, seed):
    np.random.seed(seed)
    dt = T / steps
    
    # Pre-calculate correlated shocks
    Z1 = np.random.standard_normal((n_paths, steps))
    Z_temp = np.random.standard_normal((n_paths, steps))
    Z2 = rho * Z1 + np.sqrt(1 - rho**2) * Z_temp
    
    prices = np.zeros((n_paths, steps + 1))
    variances = np.zeros((n_paths, steps + 1))
    
    prices[:, 0] = S0
    variances[:, 0] = v0
    
    for j in range(steps):
        v_prev = np.maximum(variances[:, j], 0) # Full truncation
        
        # S(t+dt) = S(t) * exp((r - 0.5*v)*dt + sqrt(v*dt)*Z1)
        prices[:, j+1] = prices[:, j] * np.exp((r - 0.5 * v_prev) * dt + np.sqrt(v_prev * dt) * Z1[:, j])
        
        # v(t+dt) = v(t) + kappa*(theta - v)*dt + sigma_v*sqrt(v*dt)*Z2
        variances[:, j+1] = variances[:, j] + kappa * (theta - v_prev) * dt + sigma_v * np.sqrt(v_prev * dt) * Z2[:, j]
        variances[:, j+1] = np.maximum(variances[:, j+1], 0)
        
    time = np.linspace(0, T, steps + 1)
    return time, prices, variances

def simulate_heston_ql(S0, v0, r, kappa, theta, sigma_v, rho, T, steps, n_paths, seed):
    day_count = ql.Actual365Fixed()
    calendar = ql.NullCalendar()
    
    s0_handle = ql.QuoteHandle(ql.SimpleQuote(S0))
    r_handle = ql.YieldTermStructureHandle(ql.FlatForward(today, r, day_count))
    q_handle = ql.YieldTermStructureHandle(ql.FlatForward(today, 0.0, day_count))
    
    process = ql.HestonProcess(r_handle, q_handle, s0_handle, v0, kappa, theta, sigma_v, rho)
    
    # Heston requires 2D shocks
    rsg = ql.GaussianMultiPathGenerator(process, ql.TimeGrid(T, steps), 
                                        ql.GaussianRandomSequenceGenerator(
                                            ql.UniformRandomSequenceGenerator(2 * steps, ql.UniformRandomGenerator(seed))
                                        ))
    
    price_paths = np.zeros((n_paths, steps + 1))
    var_paths = np.zeros((n_paths, steps + 1))
    
    for i in range(n_paths):
        multi_path = rsg.next().value()
        price_paths[i, :] = [multi_path[0][j] for j in range(len(multi_path[0]))]
        var_paths[i, :] = [multi_path[1][j] for j in range(len(multi_path[1]))]
        
    time = np.linspace(0, T, steps + 1)
    return time, price_paths, var_paths

# --- Main App ---
st.title("📉 The Heston Model")
st.write("A stochastic volatility model that captures the 'Leverage Effect' and the Volatility Smile.")

tab_theory, tab_sim, tab_pricing, tab_insights = st.tabs([
    "📚 Theory", "⚡ Simulation", "🎯 Pricing", "🧠 Insights"
])

# Run Simulation
if method == "NumPy (Euler)":
    time, price_data, var_data = simulate_heston_np(S0, v0, r, kappa, theta, sigma_v, rho, T, steps, n_paths, seed)
else:
    time, price_data, var_data = simulate_heston_ql(S0, v0, r, kappa, theta, sigma_v, rho, T, steps, n_paths, seed)

with tab_theory:
    st.markdown("""
    ## 🔍 Beyond Constant Volatility
    
    The **Heston Model (1993)** is one of the most popular stochastic volatility models. It recognizes that volatility is not constant but evolves randomly over time, exhibiting **mean reversion** and **correlation** with asset returns.
    
    ### 📐 The SDEs
    The model consists of two coupled stochastic differential equations:
    """)
    st.latex(r"dS_t = r S_t dt + \sqrt{v_t} S_t dW_{1,t}")
    st.latex(r"dv_t = \kappa (\theta - v_t) dt + \sigma_v \sqrt{v_t} dW_{2,t}")
    st.markdown("""
    Where:
    - **$v_t$**: The variance of the asset returns.
    - **$\kappa$**: The speed of mean reversion of the variance.
    - **$\theta$**: The long-term average variance.
    - **$\sigma_v$**: The volatility of volatility (vol-of-vol).
    - **$\rho$**: The correlation between $dW_{1,t}$ and $dW_{2,t}$.
    """)
    
    # Feller Condition
    feller = 2 * kappa * theta - sigma_v**2
    if feller > 0:
        st.success(f"✅ **Feller Condition Satisfied:** $2\kappa\theta > \sigma_v^2$ ({feller:.4f} > 0). Variance will stay strictly positive.")
    else:
        st.warning(f"⚠️ **Feller Condition Violated:** $2\kappa\theta \le \sigma_v^2$ ({feller:.4f} ≤ 0). Variance may hit zero.")

with tab_sim:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Asset Price Paths (S)")
        fig_p = go.Figure()
        n_display = min(n_paths, 20)
        for i in range(n_display):
            fig_p.add_trace(go.Scatter(x=time, y=price_data[i], mode='lines', line=dict(width=1), opacity=0.4, showlegend=False))
        fig_p.update_layout(template="plotly_dark", xaxis_title="Time", yaxis_title="Price", height=400)
        st.plotly_chart(fig_p, use_container_width=True)

    with col2:
        st.subheader("Variance Paths (v)")
        fig_v = go.Figure()
        fig_v.add_trace(go.Scatter(x=time, y=[theta]*(steps+1), name="Long-term Mean θ", line=dict(color='#ff007f', dash='dash')))
        for i in range(n_display):
            fig_v.add_trace(go.Scatter(x=time, y=var_data[i], mode='lines', line=dict(width=1), opacity=0.4, showlegend=False))
        fig_v.update_layout(template="plotly_dark", xaxis_title="Time", yaxis_title="Variance", height=400)
        st.plotly_chart(fig_v, use_container_width=True)

    st.markdown("---")
    st.subheader("Leverage Effect Visualization")
    # Show scatter of returns vs change in vol
    rets = np.diff(np.log(price_data), axis=1).flatten()
    v_diffs = np.diff(var_data, axis=1).flatten()
    
    fig_corr = go.Figure()
    fig_corr.add_trace(go.Scatter(x=rets, y=v_diffs, mode='markers', marker=dict(size=2, opacity=0.2, color='#00d4ff')))
    fig_corr.update_layout(template="plotly_dark", title=f"Returns vs ΔVariance (Observed ρ ≈ {np.corrcoef(rets, v_diffs)[0,1]:.2f})", 
                           xaxis_title="Log-Returns", yaxis_title="ΔVariance")
    st.plotly_chart(fig_corr, use_container_width=True)

with tab_pricing:
    st.subheader("Heston Pricing Workbench")
    
    col_p1, col_p2 = st.columns([1, 2])
    
    with col_p1:
        K = st.number_input("Strike Price (K)", value=S0)
        opt_type = st.radio("Option Type", ["Call", "Put"])
        
        # QuantLib Heston Analytical Pricing
        day_count = ql.Actual365Fixed()
        exercise_date = today + ql.Period(int(T*365), ql.Days)
        
        s0_handle = ql.QuoteHandle(ql.SimpleQuote(S0))
        r_handle = ql.YieldTermStructureHandle(ql.FlatForward(today, r, day_count))
        q_handle = ql.YieldTermStructureHandle(ql.FlatForward(today, 0.0, day_count))
        
        process = ql.HestonProcess(r_handle, q_handle, s0_handle, v0, kappa, theta, sigma_v, rho)
        model = ql.HestonModel(process)
        engine = ql.AnalyticHestonEngine(model)
        
        type_enum = ql.Option.Call if opt_type == "Call" else ql.Option.Put
        payoff = ql.PlainVanillaPayoff(type_enum, K)
        exercise = ql.EuropeanExercise(exercise_date)
        option = ql.EuropeanOption(payoff, exercise)
        option.setPricingEngine(engine)
        
        h_price = option.NPV()
        
        # Compare with BS (using current sqrt(v0))
        bs_process = ql.BlackScholesProcess(s0_handle, r_handle, ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, ql.NullCalendar(), np.sqrt(v0), day_count)))
        bs_option = ql.EuropeanOption(payoff, exercise)
        bs_option.setPricingEngine(ql.AnalyticEuropeanEngine(bs_process))
        bs_price = bs_option.NPV()
        
        st.metric("Heston Analytical Price", f"${h_price:.4f}")
        st.metric("Black-Scholes Price (Fixed σ)", f"${bs_price:.4f}", delta=f"{h_price-bs_price:.4f}")
        st.caption(f"Difference indicates the value of volatility convexity/skew.")

    with col_p2:
        st.markdown("### Monte Carlo Check")
        if opt_type == "Call":
            payoffs = np.maximum(price_data[:, -1] - K, 0)
        else:
            payoffs = np.maximum(K - price_data[:, -1], 0)
        mc_price = np.mean(payoffs) * np.exp(-r * T)
        
        cum_mc = (np.cumsum(payoffs) / (np.arange(n_paths) + 1)) * np.exp(-r * T)
        fig_conv = go.Figure()
        fig_conv.add_trace(go.Scatter(y=cum_mc, name="MC Estimate", line=dict(color='#00d4ff')))
        fig_conv.add_hline(y=h_price, line_dash="dash", line_color="#ff007f", annotation_text="Heston Analytical")
        fig_conv.update_layout(template="plotly_dark", height=300)
        st.plotly_chart(fig_conv, use_container_width=True)

with tab_insights:
    st.markdown("""
    ## 🧠 Quantitative Insights
    
    ### 1. The Volatility Smile
    The Heston model can calibrate to the market **volatility smile**. By adjusting $\sigma_v$ (which controls the kurtosis/smile) and $\rho$ (which controls the skew), quants can match the prices of options across different strikes.
    
    ### 2. The Leverage Effect ($\rho$)
    In equity markets, $\rho$ is typically negative (around -0.6 to -0.8). This means that when stock prices fall, volatility tends to rise. This creates a **downward-sloping skew** in implied volatility.
    
    ### 3. Mean Reversion of Volatility
    The parameter $\kappa$ determines how quickly volatility returns to its long-term average $\theta$. This is consistent with empirical observations that high-volatility regimes are usually followed by calmer periods.
    
    ### 4. The Feller Condition
    The condition $2\kappa\theta > \sigma_v^2$ ensures that the variance process $v_t$ never hits zero. In many market-calibrated sets, this condition is actually violated, requiring robust numerical schemes (like Full Truncation) to handle near-zero variance.
    
    ### 🛠️ Pro Quant Tip
    "When you see a Heston model being used, look at $\rho$. If $\rho=0$, the distribution is symmetric but fat-tailed (smile). If $\rho < 0$, it's skewed (equity-like). If $\rho > 0$, it's skewed the other way (sometimes seen in certain commodities)."
    """)

st.markdown("---")
st.caption("Quantitative Finance Portfolio | Heston Stochastic Volatility Implementation")