import streamlit as st
import QuantLib as ql
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import norm, gamma

# Page Config
st.set_page_config(page_title="Variance Gamma Process", layout="wide")

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
S0 = st.sidebar.number_input("Initial Price (S₀)", 100.0, step=1.0)
r = st.sidebar.slider("Risk-free Rate (r)", 0.0, 0.2, 0.05, step=0.01)
sigma = st.sidebar.slider("Volatility (σ)", 0.01, 1.0, 0.2, step=0.01)
nu = st.sidebar.slider("Gamma Variance (ν)", 0.01, 2.0, 0.2, step=0.05)
theta = st.sidebar.slider("Skewness (θ)", -1.0, 1.0, -0.1, step=0.05)
T = st.sidebar.slider("Time Horizon (T)", 0.1, 5.0, 1.0, step=0.1)

st.sidebar.markdown("---")
st.sidebar.subheader("Simulation Settings")
steps = st.sidebar.slider("Steps", 50, 1000, 252)
n_paths = st.sidebar.number_input("Number of Paths", 1, 5000, 100)
seed = st.sidebar.number_input("Random Seed", 0, 10000, 42)
method = st.sidebar.selectbox("Simulation Method", ["NumPy (Fast)", "QuantLib"])

# Global Setup
today = ql.Date.todaysDate()
ql.Settings.instance().evaluationDate = today

# --- Logic Functions ---

def simulate_vg_np(S0, r, sigma, nu, theta, T, steps, n_paths, seed):
    np.random.seed(seed)
    dt = T / steps
    time = np.linspace(0, T, steps + 1)
    
    # Drift correction for risk-neutrality:
    # omega = (1/nu) * ln(1 - theta*nu - 0.5*sigma^2*nu)
    # We need 1 - theta*nu - 0.5*sigma^2*nu > 0 for the process to be well-defined
    arg = 1.0 - theta * nu - 0.5 * sigma**2 * nu
    if arg <= 0:
        st.error("Invalid parameters: 1 - θν - 0.5σ²ν must be > 0. Please reduce ν, θ, or σ.")
        return time, np.full((n_paths, steps + 1), S0)
        
    omega = (1.0 / nu) * np.log(arg)
    
    paths = np.zeros((n_paths, steps + 1))
    paths[:, 0] = S0
    
    # VG = theta*Gamma + sigma*W(Gamma)
    # Gamma increment dG ~ Gamma(dt/nu, nu)
    # Note: np.random.gamma takes (shape, scale). 
    # Here shape = dt/nu, scale = nu
    
    current_prices = np.full(n_paths, S0)
    for j in range(steps):
        dG = np.random.gamma(dt / nu, nu, n_paths)
        dW = np.random.standard_normal(n_paths) * np.sqrt(dG)
        
        # Risk-neutral price increment
        # S(t+dt) = S(t) * exp((r + omega)dt + theta*dG + sigma*dW)
        current_prices *= np.exp((r + omega) * dt + theta * dG + sigma * dW)
        paths[:, j+1] = current_prices
        
    return time, paths

def simulate_vg_ql(S0, r, sigma, nu, theta, T, steps, n_paths, seed):
    day_count = ql.Actual365Fixed()
    s0_handle = ql.QuoteHandle(ql.SimpleQuote(S0))
    r_handle = ql.YieldTermStructureHandle(ql.FlatForward(today, r, day_count))
    q_handle = ql.YieldTermStructureHandle(ql.FlatForward(today, 0.0, day_count))
    
    process = ql.VarianceGammaProcess(s0_handle, q_handle, r_handle, sigma, nu, theta)
    
    # GaussianPathGenerator works for VG if using the bridge or specific discretization
    # However, QuantLib's VG is typically used for pricing.
    # For simulation, we fall back to NumPy but provide the QL object context.
    st.info("QuantLib simulation for VG is falling back to the exact NumPy subordinator method.")
    return simulate_vg_np(S0, r, sigma, nu, theta, T, steps, n_paths, seed)

# --- Main App ---
st.title("📈 Variance Gamma (VG) Process")
st.write("A pure-jump Lévy process using a stochastic 'business clock'.")

tab_theory, tab_sim, tab_analysis, tab_insights = st.tabs([
    "📚 Theory", "⚡ Simulation", "📊 Analysis", "🧠 Insights"
])

# Run Simulation
time, data = simulate_vg_np(S0, r, sigma, nu, theta, T, steps, n_paths, seed)
final_prices = data[:, -1]

with tab_theory:
    st.markdown("""
    The **Variance Gamma (VG)** process is a pure-jump stochastic process widely used in quantitative finance. Unlike the EOU or EOU-J processes, which rely on a continuous Brownian motion component, the VG process is **infinite activity**, **finite variation**, and **path-discontinuous**.

    It is essentially a **subordinator** model: it replaces the physical time $t$ in a standard Brownian motion with a stochastic "business time" (a Gamma process), effectively capturing the reality that markets move at different speeds depending on trading activity.

    ### 1. Mathematical Construction
    The VG process $X_t$ can be defined as a Brownian motion with drift, evaluated at a random time given by a Gamma process.
    """)
    st.latex(r"X_t = \theta \Gamma_t + \sigma W_{\Gamma_t}")
    st.markdown("""
    Where:
    - **$\Gamma_t$**: A Gamma process with mean rate $t$ and variance rate $\nu$. It represents the **random clock** or business activity.
    - **$W_{\Gamma_t}$**: A standard Brownian motion evaluated at the time given by the Gamma process.
    - **$\theta$**: The drift (skewness) parameter.
    - **$\sigma$**: The volatility (scale) parameter.
    - **$\nu$**: The variance of the Gamma process, which controls the **kurtosis** (the "fatness" of the tails).

    ### 2. Key Characteristics
    - **Infinite Activity:** It has an infinite number of small jumps in any finite time interval, mimicking the "jittery" nature of high-frequency tick data.
    - **Finite Variation:** Despite the infinite jumps, the total variation of the path is finite, making it mathematically convenient for arbitrage-free pricing.
    - **Fat Tails & Skewness:** By adjusting $\nu$ and $\theta$, the model can perfectly fit the "smile" and "skew" observed in equity and FX option markets.
    - **No Diffusion Component:** There is no $dW_t$ term; all movement is jump-based. It is a **pure-jump Lévy process**.

    ### 3. Comparison with Jump-Diffusion
    While Jump-Diffusion (like Merton) combines a continuous component with occasional big jumps, the VG process assumes everything is a jump.

    | Feature | Jump-Diffusion (EOU-J) | Variance Gamma (VG) |
    | :--- | :--- | :--- |
    | **Path Nature** | Continuous with shocks | Continuous jumps (Infinite Activity) |
    | **Diffusion** | Yes (Brownian motion) | No |
    | **Model Type** | Jump-Diffusion | Pure-Jump Lévy Process |
    | **Calibration** | Fitting diffusion + jumps | Fitting $\sigma, \nu, \theta$ (Stable) |
    """)

with tab_sim:
    col_paths, col_stats = st.columns([2, 1])
    
    with col_paths:
        st.subheader("VG Stochastic Paths")
        fig_paths = go.Figure()
        
        n_display = min(n_paths, 30)
        for i in range(n_display):
            fig_paths.add_trace(go.Scatter(x=time, y=data[i], mode='lines', line=dict(width=1), opacity=0.4, showlegend=False))
            
        fig_paths.update_layout(template="plotly_dark", height=500, xaxis_title="Time", yaxis_title="Price", hovermode="closest")
        st.plotly_chart(fig_paths, use_container_width=True)

    with col_stats:
        st.subheader("Current Metrics")
        st.metric("Final Mean", f"${np.mean(final_prices):.2f}")
        st.metric("Final Std Dev", f"${np.std(final_prices):.2f}")
        
        # Theoretical mean in risk-neutral measure is S0 * exp(r*T)
        theoretical_mean = S0 * np.exp(r * T)
        st.metric("Theoretical Mean", f"${theoretical_mean:.2f}")

    st.markdown("---")
    st.subheader("Log-Returns Distribution (Skew & Kurtosis)")
    log_rets = np.log(data[:, 1:] / data[:, :-1]).flatten()
    
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(x=log_rets, histnorm='probability density', marker_color='#00d4ff', opacity=0.6, name="VG Returns"))
    
    # Normal Overlay for comparison
    mu_r = np.mean(log_rets)
    sd_r = np.std(log_rets)
    xr = np.linspace(min(log_rets), max(log_rets), 100)
    fig_hist.add_trace(go.Scatter(x=xr, y=norm.pdf(xr, mu_r, sd_r), line=dict(color='#ff007f', width=2), name="Normal Dist"))
    
    fig_hist.update_layout(template="plotly_dark", xaxis_title="Log-Return")
    st.plotly_chart(fig_hist, use_container_width=True)

with tab_analysis:
    st.subheader("Statistical Moments")
    
    # Calculate empirical skewness and kurtosis
    m3 = np.mean((log_rets - mu_r)**3)
    skew = m3 / (sd_r**3)
    
    m4 = np.mean((log_rets - mu_r)**4)
    kurt = m4 / (sd_r**4)
    
    col_a1, col_a2 = st.columns(2)
    col_a1.metric("Empirical Skewness", f"{skew:.3f}")
    col_a2.metric("Excess Kurtosis", f"{kurt - 3:.3f}")
    
    st.markdown("""
    **Understanding the VG Shape:**
    - **$\nu$ (Gamma Variance):** As $\nu \to 0$, the VG process converges to a standard Brownian motion. Higher $\nu$ increases the kurtosis (peakier center and fatter tails).
    - **$\theta$ (Skewness):** Negative $\theta$ creates a longer left tail (more frequent large drops), typical of equity markets.
    """)

with tab_insights:
    st.markdown("""
    ## 🧠 Applications in Quantitative Finance
    
    ### 1. Option Pricing
    The VG model provides a **closed-form solution for the characteristic function**, which allows for extremely fast **Fast Fourier Transform (FFT)** based option pricing. This makes it much faster to calibrate to market smiles than models requiring simulation.
    
    ### 2. High-Frequency Trading (HFT)
    Because the model is defined by "business time," it is excellent for modeling price discovery where liquidity and volatility spikes occur in clusters. It captures the "jittery" nature of tick data better than continuous diffusion models.
    
    ### 3. Risk Management
    It accurately estimates **Value at Risk (VaR)** and **Expected Shortfall** because it naturally accounts for the "crash" probability inherent in the fat tails of the VG distribution.
    
    ### 🛠️ Pro Quant Tip
    "Think of the Variance Gamma process as a Brownian motion that gets 'interrupted' by a random clock. When trading activity is high, the clock ticks faster, and the price moves more. This is why we call it a **time-changed Brownian motion**."
    """)

st.markdown("---")
st.caption("Quantitative Finance Portfolio | Variance Gamma (Pure-Jump Lévy) Implementation")