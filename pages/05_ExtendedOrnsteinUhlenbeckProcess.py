import streamlit as st
import QuantLib as ql
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import norm

# Page Config
st.set_page_config(page_title="Extended Ornstein-Uhlenbeck Process", layout="wide")

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
x0 = st.sidebar.number_input("Initial Value (x₀)", 0.0, step=0.1)
k = st.sidebar.slider("Speed of Reversion (k)", 0.01, 10.0, 1.0, step=0.1)
theta_base = st.sidebar.slider("Base Mean (θ)", -5.0, 5.0, 0.0, step=0.1)
sigma = st.sidebar.slider("Volatility (σ)", 0.01, 2.0, 0.3, step=0.01)
T = st.sidebar.slider("Time Horizon (T)", 0.1, 10.0, 1.0, step=0.1)

# Extension: Time-dependent Mean
st.sidebar.markdown("---")
st.sidebar.subheader("Extended Parameters")
mean_trend = st.sidebar.selectbox("Long-term Mean Trend θ(t)", ["Flat", "Upward", "Downward", "Sinusoidal"])

st.sidebar.markdown("---")
st.sidebar.subheader("Simulation Settings")
steps = st.sidebar.slider("Steps", 50, 2000, 500)
n_paths = st.sidebar.number_input("Number of Paths", 1, 5000, 100)
seed = st.sidebar.number_input("Random Seed", 0, 10000, 42)
method = st.sidebar.selectbox("Simulation Method", ["NumPy (Fast)", "QuantLib"])

# --- Logic Functions ---

def get_theta_t(t, base, trend):
    if trend == "Flat":
        return base
    elif trend == "Upward":
        return base + 0.5 * t
    elif trend == "Downward":
        return base - 0.5 * t
    elif trend == "Sinusoidal":
        return base + np.sin(2 * np.pi * t)
    return base

def simulate_ou_np(x0, k, theta_base, mean_trend, sigma, T, steps, n_paths, seed):
    np.random.seed(seed)
    dt = T / steps
    time = np.linspace(0, T, steps + 1)
    paths = np.zeros((n_paths, steps + 1))
    paths[:, 0] = x0
    
    # Pre-calculate theta(t) for all steps
    thetas = np.array([get_theta_t(t, theta_base, mean_trend) for t in time])
    
    # Exact discretization for OU: 
    # x(t+dt) = x(t)*exp(-k*dt) + theta(t)*(1 - exp(-k*dt)) + sigma * sqrt((1 - exp(-2*k*dt))/(2*k)) * Z
    exp_kdt = np.exp(-k * dt)
    vol_adj = sigma * np.sqrt((1 - np.exp(-2 * k * dt)) / (2 * k))
    
    for j in range(steps):
        Z = np.random.standard_normal(n_paths)
        paths[:, j+1] = paths[:, j] * exp_kdt + thetas[j] * (1 - exp_kdt) + vol_adj * Z
        
    return time, paths, thetas

def simulate_ou_ql(x0, k, theta_base, mean_trend, sigma, T, steps, n_paths, seed):
    # For a constant mean theta_base, we can use ql.OrnsteinUhlenbeckProcess
    # For the extended version with time-dependent theta, we'd typically use ql.ExtendedOrnsteinUhlenbeckProcess
    # but that requires a level function which is complex in the SWIG wrapper.
    # We will use ql.OrnsteinUhlenbeckProcess for the constant mean case if trend is Flat,
    # otherwise we use NumPy and add a warning.
    
    if mean_trend == "Flat":
        process = ql.OrnsteinUhlenbeckProcess(k, sigma, x0, theta_base)
        rsg = ql.GaussianRandomSequenceGenerator(
            ql.UniformRandomSequenceGenerator(steps, ql.UniformRandomGenerator(seed))
        )
        path_generator = ql.GaussianPathGenerator(process, T, steps, rsg, False)
        
        paths = np.zeros((n_paths, steps + 1))
        for i in range(n_paths):
            sample_path = path_generator.next().value()
            paths[i, :] = [sample_path[j] for j in range(len(sample_path))]
            
        time = np.linspace(0, T, steps + 1)
        thetas = np.full(steps + 1, theta_base)
        return time, paths, thetas
    else:
        st.warning("⚠️ QuantLib's Extended OU simulation with time-dependent mean is complex in Python. Falling back to NumPy.")
        return simulate_ou_np(x0, k, theta_base, mean_trend, sigma, T, steps, n_paths, seed)

# --- Main App ---
st.title("📈 Extended Ornstein-Uhlenbeck Process")
st.write("A mean-reverting stochastic process with time-dependent parameters.")

tab_theory, tab_sim, tab_analysis, tab_insights = st.tabs([
    "📚 Theory", "⚡ Simulation", "📊 Analysis", "🧠 Insights"
])

# Run Simulation
if method == "NumPy (Fast)":
    time, paths, thetas = simulate_ou_np(x0, k, theta_base, mean_trend, sigma, T, steps, n_paths, seed)
else:
    time, paths, thetas = simulate_ou_ql(x0, k, theta_base, mean_trend, sigma, T, steps, n_paths, seed)

with tab_theory:
    st.markdown("""
    The **Extended Ornstein-Uhlenbeck (EOU)** process is a stochastic process that generalizes the classic Ornstein-Uhlenbeck (OU) process by allowing the mean-reversion level, volatility, and even the speed of mean reversion to be time-dependent deterministic functions, rather than constants.

    In quantitative finance, this is a powerful tool for modeling interest rates, commodity prices, and credit spreads where the long-term trend (the "mean") is not stationary but evolves over time.

    ### 1. The Classic OU Process (The Foundation)
    To understand the extension, recall the standard OU process (a mean-reverting process) defined by the Stochastic Differential Equation (SDE):
    """)
    st.latex(r"dX_t = \theta (\mu - X_t) dt + \sigma dW_t")
    st.markdown("""
    Where:
    - **$\theta$**: The speed of mean reversion.
    - **$\mu$**: The long-term mean level.
    - **$\sigma$**: The volatility (diffusion coefficient).
    - **$dW_t$**: The Wiener process (Brownian motion).

    ### 2. The Extended Ornstein-Uhlenbeck Process
    The EOU process replaces these constants with time-dependent functions: $\theta(t)$, $\mu(t)$, and $\sigma(t)$. The SDE becomes:
    """)
    st.latex(r"dX_t = \theta(t) (\mu(t) - X_t) dt + \sigma(t) dW_t")
    st.markdown("""
    **Why this extension matters:**
    - **Time-Varying Mean $\mu(t)$:** Allows the model to calibrate to the current term structure of forward rates or expected future price trajectories.
    - **Time-Varying Volatility $\sigma(t)$:** Captures "volatility clustering" or known seasonal effects (e.g., higher volatility in commodity prices during specific contract expiration windows).
    - **Time-Varying Speed $\theta(t)$:** Allows the strength of the pull toward the mean to weaken or strengthen based on market regimes.

    ### 3. Mathematical Properties
    The EOU process belongs to the class of **Gaussian Processes**. This means that if $X_0$ is constant or normally distributed, $X_t$ remains normally distributed for all $t > 0$.

    #### Solving the SDE
    Using the method of integrating factors, the solution for $X_t$ given $X_0$ is:
    """)
    st.latex(r"X_t = X_0 e^{-\int_0^t \theta(s) ds} + \int_0^t e^{-\int_s^t \theta(u) du} \theta(s) \mu(s) ds + \int_0^t e^{-\int_s^t \theta(u) du} \sigma(s) dW_s")
    st.markdown("""
    This solution shows that $X_t$ is a weighted sum of its initial state, its historical mean trajectory, and an accumulated stochastic noise component.

    #### Moments
    **Conditional Expectation:**
    """)
    st.latex(r"E[X_t | X_0] = X_0 e^{-\int_0^t \theta(s) ds} + \int_0^t e^{-\int_s^t \theta(u) du} \theta(s) \mu(s) ds")
    st.markdown("**Conditional Variance:**")
    st.latex(r"Var[X_t | X_0] = \int_0^t e^{-2\int_s^t \theta(u) du} \sigma^2(s) ds")

    st.markdown("""
    ### 4. Applications in Finance
    - **Interest Rate Modeling:** Models like the **Hull-White model** are essentially EOU processes applied to short rates. By choosing $\mu(t)$ correctly, the model can perfectly fit the initial term structure of interest rates observed in the market.
    - **Commodity Pricing:** Used to model prices that are mean-reverting but exhibit seasonality (where $\mu(t)$ reflects the seasonal demand cycle).
    - **Spread Trading:** In pairs trading, the spread between two assets is often modeled as an OU process. The Extended version is used when the relationship between the two assets shifts over time due to macroeconomic changes.

    ### Summary Table
    | Feature | Standard OU Process | Extended OU Process |
    | :--- | :--- | :--- |
    | **Mean Reversion Level** | Constant ($\mu$) | Time-dependent ($\mu(t)$) |
    | **Reversion Speed** | Constant ($\theta$) | Time-dependent ($\theta(t)$) |
    | **Volatility** | Constant ($\sigma$) | Time-dependent ($\sigma(t)$) |
    | **Stationarity** | Stationary | Non-stationary (usually) |
    | **Primary Use** | Theoretical modeling | Market calibration |
    """)

with tab_sim:
    col_paths, col_stats = st.columns([2, 1])
    
    with col_paths:
        st.subheader("Mean Reverting Paths")
        fig_paths = go.Figure()
        
        # Plot mean trend theta(t)
        fig_paths.add_trace(go.Scatter(x=time, y=thetas, name="Long-term Mean θ(t)", line=dict(color='#ff007f', width=3, dash='dash')))
        
        n_display = min(n_paths, 30)
        for i in range(n_display):
            fig_paths.add_trace(go.Scatter(x=time, y=paths[i], mode='lines', line=dict(width=1), opacity=0.3, showlegend=False))
            
        p5 = np.percentile(paths, 5, axis=0)
        p95 = np.percentile(paths, 95, axis=0)
        fig_paths.add_trace(go.Scatter(x=time, y=p5, line=dict(color='rgba(255,255,255,0)'), showlegend=False, hoverinfo='skip'))
        fig_paths.add_trace(go.Scatter(x=time, y=p95, fill='tonexty', fillcolor='rgba(0, 212, 255, 0.1)', line=dict(color='rgba(255,255,255,0)'), name="90% Confidence Interval"))
        
        fig_paths.update_layout(template="plotly_dark", height=500, xaxis_title="Time", yaxis_title="x(t)", hovermode="closest")
        st.plotly_chart(fig_paths, use_container_width=True)

    with col_stats:
        st.subheader("Current Metrics")
        final_vals = paths[:, -1]
        st.metric("Final Mean", f"{np.mean(final_vals):.3f}")
        st.metric("Final Std Dev", f"{np.std(final_vals):.3f}")
        
        # Half-life
        half_life = np.log(2) / k
        st.metric("Half-life of Reversion", f"{half_life:.3f} units")
        st.caption("Time required for the process to return halfway to the mean.")

    st.markdown("---")
    st.subheader("Final Value Distribution")
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(x=final_vals, histnorm='probability density', marker_color='#00d4ff', opacity=0.6, name="Simulated"))
    
    # Theoretical Normal at T (for constant theta)
    if mean_trend == "Flat":
        mu_T = x0 * np.exp(-k * T) + theta_base * (1 - np.exp(-k * T))
        var_T = (sigma**2 / (2 * k)) * (1 - np.exp(-2 * k * T))
        sd_T = np.sqrt(var_T)
        xr = np.linspace(min(final_vals), max(final_vals), 100)
        fig_hist.add_trace(go.Scatter(x=xr, y=norm.pdf(xr, mu_T, sd_T), line=dict(color='#ff007f', width=2), name="Normal PDF (Theory)"))
    
    fig_hist.update_layout(template="plotly_dark", xaxis_title="Value", yaxis_title="Density")
    st.plotly_chart(fig_hist, use_container_width=True)

with tab_analysis:
    st.subheader("Stationarity & Reversion Analysis")
    
    # Autocorrelation of a single path
    path_data = pd.Series(paths[0])
    autocorr = [path_data.autocorr(lag=i) for i in range(1, min(steps, 50))]
    
    fig_ac = go.Figure()
    fig_ac.add_trace(go.Bar(y=autocorr, marker_color='#00d4ff'))
    fig_ac.update_layout(template="plotly_dark", title="Path Autocorrelation (Path 1)", xaxis_title="Lag", yaxis_title="Correlation", height=300)
    st.plotly_chart(fig_ac, use_container_width=True)
    
    st.markdown("""
    ### 📈 Key Properties
    1.  **Stationarity:** If $k > 0$, the process is mean-reverting and has a stationary distribution $\mathcal{N}(\theta, \frac{\sigma^2}{2k})$.
    2.  **Memory:** Unlike Brownian Motion, the OU process has "memory"—its future state depends on its distance from the mean.
    3.  **Negative Correlation:** When $x_t > \theta$, the expected change $dx_t$ is negative, creating a "rubber band" effect.
    """)

with tab_insights:
    st.markdown("""
    ## 🧠 Where is OU Used?
    
    The Ornstein-Uhlenbeck process is the workhorse of **Mean Reversion** strategies.
    
    ### 1. Pairs Trading (Statistical Arbitrage)
    When two stocks are co-integrated (e.g., Coke and Pepsi), their **price spread** often follows an OU process. Quants trade the "spread" by going long when it's below the mean and short when it's above.
    
    ### 2. Interest Rate Modeling (Hull-White)
    The **Hull-White model** is essentially an extended OU process applied to short-term interest rates. The "Extension" ($\theta(t)$) allows the model to fit the current yield curve perfectly.
    
    ### 3. Commodity Prices
    Unlike stocks, commodities (like Electricity or Natural Gas) often have mean-reverting prices because high prices attract more supply, which eventually pulls the price back down.
    

    """)
