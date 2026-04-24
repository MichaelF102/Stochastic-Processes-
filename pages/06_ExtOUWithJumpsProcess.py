import streamlit as st
import QuantLib as ql
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import norm

# Page Config
st.set_page_config(page_title="Extended OU with Jumps Process", layout="wide")

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

# Continuous Parameters
x0 = st.sidebar.number_input("Initial Value (x₀)", 0.0, step=0.1)
k = st.sidebar.slider("Reversion Speed (k)", 0.1, 10.0, 2.0, step=0.1)
theta_base = st.sidebar.slider("Base Mean (θ)", -5.0, 5.0, 0.0, step=0.1)
sigma = st.sidebar.slider("Vol (σ)", 0.01, 2.0, 0.3, step=0.01)

# Jump Parameters
st.sidebar.markdown("---")
st.sidebar.subheader("Jump Parameters (dJt)")
jump_lambda = st.sidebar.slider("Jump Intensity (λ)", 0.0, 5.0, 1.0, step=0.1)
jump_mu = st.sidebar.slider("Avg Jump Size (μJ)", -2.0, 2.0, 0.0, step=0.1)
jump_sigma = st.sidebar.slider("Jump Std Dev (σJ)", 0.01, 1.0, 0.5, step=0.01)

# Simulation Settings
st.sidebar.markdown("---")
st.sidebar.subheader("Simulation Settings")
T = st.sidebar.slider("Time Horizon (T)", 0.1, 5.0, 1.0, step=0.1)
steps = st.sidebar.slider("Steps", 100, 1000, 500)
n_paths = st.sidebar.number_input("Number of Paths", 1, 2000, 100)
seed = st.sidebar.number_input("Random Seed", 0, 10000, 42)
method = st.sidebar.selectbox("Simulation Method", ["NumPy (Fast)", "QuantLib"])

# --- Logic Functions ---

def get_theta_t(t, base):
    # Simple seasonal theta for the "Extended" demonstration
    return base + 0.5 * np.sin(2 * np.pi * t)

def simulate_eouj_np(x0, k, theta_base, sigma, jump_lambda, jump_mu, jump_sigma, T, steps, n_paths, seed):
    np.random.seed(seed)
    dt = T / steps
    time = np.linspace(0, T, steps + 1)
    paths = np.zeros((n_paths, steps + 1))
    paths[:, 0] = x0
    
    thetas = np.array([get_theta_t(t, theta_base) for t in time])
    
    for j in range(steps):
        # Continuous Part (Euler-Maruyama)
        Z = np.random.standard_normal(n_paths)
        paths[:, j+1] = paths[:, j] + k * (thetas[j] - paths[:, j]) * dt + sigma * np.sqrt(dt) * Z
        
        # Jump Part (Poisson)
        # Prob of jump in dt is lambda * dt
        jump_occurred = np.random.uniform(0, 1, n_paths) < (jump_lambda * dt)
        if np.any(jump_occurred):
            jump_sizes = np.random.normal(jump_mu, jump_sigma, np.sum(jump_occurred))
            paths[jump_occurred, j+1] += jump_sizes
            
    return time, paths, thetas

def simulate_eouj_ql(x0, k, theta_base, sigma, jump_lambda, jump_mu, jump_sigma, T, steps, n_paths, seed):
    # QuantLib has ExtOUWithJumpsProcess(process, jumpProcess)
    # This requires setting up an OU process and a jump process.
    # Given the complexity of the Python wrappers for specific jump-diffusion setups,
    # we will provide a QuantLib-themed warning and fall back to NumPy for this specific process.
    st.warning("⚠️ QuantLib's ExtOUWithJumpsProcess requires specific measure setups in Python. Falling back to NumPy for path simulation.")
    return simulate_eouj_np(x0, k, theta_base, sigma, jump_lambda, jump_mu, jump_sigma, T, steps, n_paths, seed)

# --- Main App ---
st.title("⚡ Extended OU with Jumps (EOU-J)")
st.write("Modeling mean-reverting processes with sudden discontinuous shocks.")

tab_theory, tab_sim, tab_analysis, tab_insights = st.tabs([
    "📚 Theory", "⚡ Simulation", "📊 Analysis", "🧠 Insights"
])

# Run Simulation
if method == "NumPy (Fast)":
    time, paths, thetas = simulate_eouj_np(x0, k, theta_base, sigma, jump_lambda, jump_mu, jump_sigma, T, steps, n_paths, seed)
else:
    time, paths, thetas = simulate_eouj_ql(x0, k, theta_base, sigma, jump_lambda, jump_mu, jump_sigma, T, steps, n_paths, seed)
final_vals = paths[:, -1]

with tab_theory:
    st.markdown("""
    The **Extended Ornstein-Uhlenbeck (EOU) Process with Jumps**—often referred to as an **EOU-J** process—extends the standard EOU framework by incorporating discontinuous "jumps" into the dynamics.

    While the standard EOU process is continuous, financial markets often exhibit sudden, sharp movements (shocks) caused by news, earnings reports, or geopolitical events. Adding jumps allows the model to capture "fat tails" and sudden volatility spikes that a continuous Gaussian model would underestimate.

    ### 1. The Stochastic Differential Equation (SDE)
    The EOU-J process is defined by combining the continuous diffusion component with a jump process:
    """)
    st.latex(r"dX_t = \theta(t) (\mu(t) - X_t) dt + \sigma(t) dW_t + dJ_t")
    st.markdown("""
    Where:
    - **$\theta(t), \mu(t), \sigma(t)$**: The time-dependent speed, mean, and volatility.
    - **$dW_t$**: The Brownian motion (the continuous part).
    - **$dJ_t$**: The jump component, typically modeled as a **Compound Poisson Process**:
    """)
    st.latex(r"dJ_t = Y_t dN_t")
    st.markdown("""
    - **$N_t$**: A Poisson process with intensity $\lambda(t)$, determining jump frequency.
    - **$Y_t$**: A random variable representing the size of the jump (e.g., Normal distribution).

    ### 2. Key Conceptual Components
    
    #### The Mean-Reversion Component
    Even after a jump occurs, the system maintains its "memory." The process immediately begins pulling the value $X_t$ back toward the mean $\mu(t)$. This models markets that spike but eventually settle back to a fair value.

    #### The Jump Component ($Y_t dN_t$)
    - **Intensity ($\lambda$):** Determines how often crashes or spikes happen.
    - **Jump Magnitude ($Y_t$):** Calibrated to distinguish between small "noise" jumps and massive "crisis" jumps.

    ### 3. Why add Jumps?
    - **Capturing Kurtosis:** Real-world returns have "fat tails." Jumps account for extreme events statistically impossible in a pure EOU framework.
    - **Volatility Smile:** Jumps help recreate the "volatility smile" observed in market option data.
    - **Realism:** Capture instantaneous events like credit defaults or sudden news breaks.

    ### 4. Mathematical Complexity
    Because of the jumps, EOU-J is **no longer a purely Gaussian process**. It is leptokurtic (fat-tailed). Calibration often involves identifying outliers that a continuous model cannot explain.

    ### Summary Table: EOU vs. EOU-J
    | Feature | Extended OU (EOU) | EOU with Jumps (EOU-J) |
    | :--- | :--- | :--- |
    | **Path Continuity** | Continuous | Discontinuous |
    | **Distribution** | Gaussian | Non-Gaussian (Leptokurtic) |
    | **Fat Tails** | No | Yes |
    | **Market Events** | Gradual changes | Sudden shocks |
    """)

with tab_sim:
    col_paths, col_stats = st.columns([2, 1])
    
    with col_paths:
        st.subheader("EOU-J Stochastic Paths")
        fig_paths = go.Figure()
        
        # Mean Trend
        fig_paths.add_trace(go.Scatter(x=time, y=thetas, name="Mean Target θ(t)", line=dict(color='#ff007f', width=2, dash='dash')))
        
        n_display = min(n_paths, 20)
        for i in range(n_display):
            fig_paths.add_trace(go.Scatter(x=time, y=paths[i], mode='lines', line=dict(width=1), opacity=0.4, showlegend=False))
            
        fig_paths.update_layout(template="plotly_dark", height=500, xaxis_title="Time", yaxis_title="Price / Value", hovermode="closest")
        st.plotly_chart(fig_paths, use_container_width=True)

    with col_stats:
        st.subheader("Simulation Stats")
        st.metric("Final Mean", f"{np.mean(final_vals):.3f}")
        st.metric("Final Std Dev", f"{np.std(final_vals):.3f}")
        
        # Estimate observed jumps (simple diff > 3 sigma)
        diffs = np.diff(paths, axis=1)
        jumps_detected = np.sum(np.abs(diffs) > (sigma * np.sqrt(T/steps) * 4))
        st.metric("Detected Large Shocks", f"{jumps_detected}")
        st.caption("Heuristic count of movements > 4σ threshold.")

    st.markdown("---")
    st.subheader("Distribution Analysis (Leptokurtosis)")
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(x=final_vals, histnorm='probability density', marker_color='#00d4ff', opacity=0.6, name="EOU-J Final Prices"))
    
    # Overlay Normal for comparison
    mu_f = np.mean(final_vals)
    sd_f = np.std(final_vals)
    xr = np.linspace(min(final_vals), max(final_vals), 100)
    fig_hist.add_trace(go.Scatter(x=xr, y=norm.pdf(xr, mu_f, sd_f), line=dict(color='#ff007f', width=2), name="Normal Dist (Same σ)"))
    
    fig_hist.update_layout(template="plotly_dark", title="Final Value Distribution vs Normal", xaxis_title="Value")
    st.plotly_chart(fig_hist, use_container_width=True)

with tab_analysis:
    st.subheader("Impact of Jumps on Risk")
    
    # Kurtosis calculation
    mean_val = np.mean(final_vals)
    std_val = np.std(final_vals)
    kurt = np.mean((final_vals - mean_val)**4) / (std_val**4)
    
    col_k1, col_k2 = st.columns(2)
    col_k1.metric("Excess Kurtosis", f"{kurt - 3:.3f}")
    
    st.markdown("""
    **Understanding Kurtosis:**
    - A standard Normal distribution has a kurtosis of **3**.
    - If Excess Kurtosis > 0, the distribution is **Leptokurtic** (fat tails).
    - This means extreme events are much more likely than predicted by a standard OU process.
    """)
    
    # Jump intensity impact
    st.write("#### Path Dynamics")
    st.info(f"With an intensity of λ={jump_lambda}, we expect approximately {jump_lambda * T:.1f} jumps per path over {T} years.")

with tab_insights:
    st.markdown("""
    ## 🧠 Quantitative Insights
    
    ### 1. Market "Flash Crashes"
    The EOU-J process is excellent for modeling assets that are generally stable but prone to sudden liquidity shocks or flash crashes. The mean reversion ensures that after the shock, the market attempts to find its equilibrium again.
    
    ### 2. The Volatility Smile
    In option pricing, the presence of jumps creates a "smile" in implied volatility. Out-of-the-money options become more expensive because the model accounts for the possibility of a sudden jump into the money.
    
    ### 3. Credit Risk & Defaults
    While a company's debt-to-equity ratio might evolve continuously (OU), a credit rating downgrade or a default event is a discrete jump ($dJ_t$).
    
    ### 🛠️ Pro Quant Tip
    "When calibrating EOU-J, first fit the continuous parameters ($k, \theta, \sigma$) using only small price movements. Then, use the outliers (the 'jumps') to calibrate $\lambda$ and $Y_t$. If you mix them, your volatility estimate will be biased upwards."
    """)

st.markdown("---")
st.caption("Quantitative Finance Portfolio | Extended OU with Jumps (EOU-J) Implementation")