import streamlit as st
import QuantLib as ql
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import norm, lognorm

# Page Config
st.set_page_config(page_title="Black-Scholes Process ", layout="wide")

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

# S0, r, sigma, T
S0 = st.sidebar.number_input("Initial Price (S₀)", 100.0, step=1.0)
r = st.sidebar.slider("Risk-free Rate (r)", 0.0, 0.2, 0.05, step=0.01)
sigma = st.sidebar.slider("Volatility (σ)", 0.01, 1.0, 0.2, step=0.01)
T = st.sidebar.slider("Time Horizon (T in years)", 0.1, 5.0, 1.0, step=0.1)

st.sidebar.markdown("---")
st.sidebar.subheader("Simulation Settings")
steps = st.sidebar.slider("Steps", 50, 1000, 252)
n_paths = st.sidebar.number_input("Number of Paths", 1, 10000, 100)
seed = st.sidebar.number_input("Random Seed", 0, 10000, 42)
method = st.sidebar.selectbox("Simulation Method", ["NumPy (Fast)", "QuantLib"])

if n_paths > 2000:
    st.sidebar.warning("⚠️ High path count may affect responsiveness.")

# Global Setup
today = ql.Date.todaysDate()
ql.Settings.instance().evaluationDate = today

# --- Logic Functions ---

def simulate_bs_np(S0, r, sigma, T, steps, n_paths, seed):
    np.random.seed(seed)
    dt = T / steps
    # Vectorized simulation
    # S(t+dt) = S(t) * exp((r - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
    Z = np.random.standard_normal((n_paths, steps))
    log_returns = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    
    # Cumulative sum of log returns to get price paths
    price_paths = np.zeros((n_paths, steps + 1))
    price_paths[:, 0] = S0
    price_paths[:, 1:] = S0 * np.exp(np.cumsum(log_returns, axis=1))
    
    time = np.linspace(0, T, steps + 1)
    return time, price_paths

def simulate_bs_ql(S0, r, sigma, T, steps, n_paths, seed):
    # QuantLib setup
    day_count = ql.Actual365Fixed()
    calendar = ql.NullCalendar()
    
    s0_handle = ql.QuoteHandle(ql.SimpleQuote(S0))
    r_handle = ql.YieldTermStructureHandle(ql.FlatForward(today, r, day_count))
    v_handle = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, calendar, sigma, day_count))
    
    process = ql.BlackScholesProcess(s0_handle, r_handle, v_handle)
    
    # Random sequence generator with seed
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
st.title("📈 Black-Scholes Process")

tab_theory, tab_sim, tab_pricing, tab_insights = st.tabs([
    "📚 Theory & Math", "⚡ Simulation", "🎯 Option Pricing", "🧠 Insights"
])

# Run Simulation for use in multiple tabs
if method == "NumPy (Fast)":
    time, data = simulate_bs_np(S0, r, sigma, T, steps, n_paths, seed)
else:
    time, data = simulate_bs_ql(S0, r, sigma, T, steps, n_paths, seed)

final_prices = data[:, -1]


with tab_theory:  
    st.markdown("## 🔍 The Black–Scholes Process Framework")
    st.markdown("""
    The **Black–Scholes process** describes the evolution of an asset price under the **risk-neutral measure**, forming the foundation of modern derivative pricing. This framework ensures **arbitrage-free pricing** by transforming the probability measure rather than modeling real-world returns.

    ### 🧠 Risk-Neutral Framework (Core Idea)
    In the risk-neutral world:
    - All assets grow at the **risk-free rate $r$**
    - Investor risk preferences are ignored
    - Prices are computed as **discounted expectations**

    > This is a mathematical construct used for pricing—not a description of reality.
    """)
    st.latex(r"e^{-rt} S_t \text{ is a martingale}")
    st.markdown("This martingale condition guarantees no-arbitrage and consistent derivative pricing.")


    st.markdown("### ⚖️ Real-World vs Risk-Neutral (Key Distinction)")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🌍 Real-World Measure ($\\mathbb{P}$)")
        st.markdown("- Drift = $\\mu$ \n- Used for forecasting \n- Reflects actual expected returns")
    with col2:
        st.markdown("#### ⚖️ Risk-Neutral Measure ($\\mathbb{Q}$)")
        st.markdown("- Drift = $r$ \n- Used for pricing \n- Ensures no-arbitrage")
    st.info("In the Black–Scholes framework, we always work under the risk-neutral measure ($\\mathbb{Q}$).")


    st.markdown("### 📐 Black–Scholes Process (SDE)")
    st.latex(r"dS_t = r S_t \, dt + \sigma S_t \, dW_t")
    st.markdown("Where $S_t$ is the asset price, $r$ the risk-free rate, $\\sigma$ constant volatility, and $W_t$ Brownian motion.")
    
    st.markdown("---")
    st.markdown("### 📌 Analytical Solution")
    st.latex(r"S_t = S_0 \exp\left((r - \tfrac{1}{2}\sigma^2)t + \sigma W_t\right)")
    st.success("""
    Interpretation:
    - $rt$: Deterministic growth at risk-free rate
    - $-\\frac{1}{2}\\sigma^2 t$: Volatility adjustment (Itô correction)
    - $\\sigma W_t$: Randomness
    """)


    st.markdown("### 🎯 Pricing Implication")
    st.latex(r"\text{Price} = e^{-rT} \mathbb{E}^{\mathbb{Q}}[\text{Payoff}]")
    st.markdown("Allows pricing without knowing $\\mu$—only volatility and the risk-free rate are required.")
    
    st.markdown("### 📊 Distribution Insight")
    st.markdown("Since $\\log(S_t)$ is normally distributed, $S_t$ is **log-normally distributed**, ensuring prices remain positive.")


    st.markdown("### ⚠️ Model Assumptions")
    st.markdown("- Constant volatility ($\\sigma$) and risk-free rate ($r$) \n- No dividends \n- Continuous trading \n- No transaction costs")
    
    st.markdown("### 🚧 Limitations")
    st.markdown("Real markets exhibit stochastic volatility, price jumps, and fat-tailed distributions, motivating more advanced models.")
    
    st.markdown("### 🧭 Mental Model")
    st.info("Think of the Black–Scholes process as an arbitrage-consistent pricing framework where uncertainty is priced via expectation, not prediction.")


with tab_sim:
    col_paths, col_stats = st.columns([2, 1])
    
    with col_paths:
        st.subheader("Price Path Simulation")
        fig_paths = go.Figure()
        
        # Plot a subset of paths for performance
        n_display = min(n_paths, 50)
        for i in range(n_display):
            fig_paths.add_trace(go.Scatter(
                x=time, y=data[i], mode='lines', 
                line=dict(width=1), opacity=0.4,
                name=f"Path {i+1}", hoverinfo='name+y'
            ))
            
        # Add Fan Chart (Percentiles)
        p5 = np.percentile(data, 5, axis=0)
        p50 = np.percentile(data, 50, axis=0)
        p95 = np.percentile(data, 95, axis=0)
        
        fig_paths.add_trace(go.Scatter(x=time, y=p5, line=dict(color='rgba(255,255,255,0)', dash='dash'), showlegend=False, hoverinfo='skip'))
        fig_paths.add_trace(go.Scatter(x=time, y=p95, fill='tonexty', fillcolor='rgba(0, 212, 255, 0.1)', line=dict(color='rgba(255,255,255,0)'), name="90% Confidence Interval"))
        fig_paths.add_trace(go.Scatter(x=time, y=p50, line=dict(color='#00d4ff', width=3), name="Median Path (P50)"))
        
        fig_paths.update_layout(
            template="plotly_dark", height=500,
            xaxis_title="Time (T)", yaxis_title="Price ($)",
            hovermode="closest",
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_paths, use_container_width=True)

    with col_stats:
        st.subheader("Simulation Metrics")
        st.metric("Final Mean", f"${np.mean(final_prices):.2f}")
        st.metric("Final Std Dev", f"${np.std(final_prices):.2f}")
        st.metric("Max Price Observed", f"${np.max(data):.2f}")
        st.metric("Min Price Observed", f"${np.min(data):.2f}")

    st.markdown("---")
    
    col_returns, col_dist = st.columns(2)
    
    with col_returns:
        st.subheader("📊 Log-Returns Analysis")
        # Calculate log returns: ln(S_t / S_{t-1})
        # data is (n_paths, steps+1)
        log_rets = np.log(data[:, 1:] / data[:, :-1]).flatten()
        
        fig_rets = go.Figure()
        fig_rets.add_trace(go.Histogram(
            x=log_rets, histnorm='probability density', 
            marker_color='#00d4ff', opacity=0.6, name="Simulated"
        ))
        
        # Overlay Normal PDF
        mu_ret = (r - 0.5 * sigma**2) * (T/steps)
        std_ret = sigma * np.sqrt(T/steps)
        x_range = np.linspace(min(log_rets), max(log_rets), 100)
        y_pdf = norm.pdf(x_range, mu_ret, std_ret)
        fig_rets.add_trace(go.Scatter(x=x_range, y=y_pdf, line=dict(color='#ff007f', width=2), name="Normal PDF (Theory)"))
        
        fig_rets.update_layout(template="plotly_dark", title="Distribution of Log-Returns", xaxis_title="Log-Return", showlegend=True)
        st.plotly_chart(fig_rets, use_container_width=True)

    with col_dist:
        st.subheader("🎯 Terminal Price Distribution")
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(
            x=final_prices, histnorm='probability density', 
            marker_color='#00d4ff', opacity=0.6, name="Simulated"
        ))
        
        # Overlay Log-Normal PDF
        # Log-normal parameters: meanlog = ln(S0) + (r - 0.5*sigma^2)*T, sdlog = sigma*sqrt(T)
        mean_log = np.log(S0) + (r - 0.5 * sigma**2) * T
        sd_log = sigma * np.sqrt(T)
        
        x_range_p = np.linspace(min(final_prices), max(final_prices), 100)
        # scipy.stats.lognorm takes s=sd_log, scale=exp(mean_log)
        y_pdf_p = lognorm.pdf(x_range_p, s=sd_log, scale=np.exp(mean_log))
        fig_dist.add_trace(go.Scatter(x=x_range_p, y=y_pdf_p, line=dict(color='#ff007f', width=2), name="Log-Normal PDF (Theory)"))
        
        fig_dist.update_layout(template="plotly_dark", title="Final Price Distribution", xaxis_title="Price ($)")
        st.plotly_chart(fig_dist, use_container_width=True)

with tab_pricing:
    st.subheader("European Option Workbench")
    
    col_p1, col_p2 = st.columns([1, 2])
    
    with col_p1:
        st.markdown("### Parameters")
        K = st.number_input("Strike Price (K)", value=S0, step=1.0)
        opt_type = st.radio("Option Type", ["Call", "Put"])
        
        # QuantLib Analytical Pricing
        day_count = ql.Actual365Fixed()
        calendar = ql.NullCalendar()
        exercise_date = today + ql.Period(int(T*365), ql.Days)
        
        s0_handle = ql.QuoteHandle(ql.SimpleQuote(S0))
        r_handle = ql.YieldTermStructureHandle(ql.FlatForward(today, r, day_count))
        v_handle = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, calendar, sigma, day_count))
        process = ql.BlackScholesProcess(s0_handle, r_handle, v_handle)
        
        type_enum = ql.Option.Call if opt_type == "Call" else ql.Option.Put
        payoff = ql.PlainVanillaPayoff(type_enum, K)
        exercise = ql.EuropeanExercise(exercise_date)
        option = ql.EuropeanOption(payoff, exercise)
        option.setPricingEngine(ql.AnalyticEuropeanEngine(process))
        
        bs_price = option.NPV()
        delta = option.delta()
        vega = option.vega() / 100.0 # Standardize to per 1% vol
        
        # Monte Carlo Price
        if opt_type == "Call":
            payoffs = np.maximum(final_prices - K, 0)
        else:
            payoffs = np.maximum(K - final_prices, 0)
        mc_price = np.mean(payoffs) * np.exp(-r * T)
        mc_error = np.std(payoffs) / np.sqrt(n_paths) * np.exp(-r * T)
        
        st.markdown("---")
        st.metric("Analytical BS Price", f"${bs_price:.4f}")
        st.metric("Monte Carlo Price", f"${mc_price:.4f}", delta=f"Err: {mc_error:.4f}")

    with col_p2:
        st.markdown("### Greeks & Performance")
        col_g1, col_g2 = st.columns(2)
        col_g1.metric("Delta (Δ)", f"{delta:.4f}")
        col_g2.metric("Vega (ν)", f"{vega:.4f}")
        
        st.markdown("""
        **Delta ($\Delta$):** Sensitivity of the option price to a $1 change in the underlying.  
        **Vega ($\nu$):** Sensitivity of the option price to a 1% change in volatility.
        """)
        
        # Convergence Chart
        st.write("#### MC Convergence")
        cum_payoffs = np.cumsum(payoffs)
        cum_mc = (cum_payoffs / (np.arange(n_paths) + 1)) * np.exp(-r * T)
        
        fig_conv = go.Figure()
        fig_conv.add_trace(go.Scatter(y=cum_mc, name="MC Estimate", line=dict(color='#00d4ff')))
        fig_conv.add_hline(y=bs_price, line_dash="dash", line_color="#ff007f", annotation_text="BS Analytical")
        fig_conv.update_layout(template="plotly_dark", height=300, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_conv, use_container_width=True)

with tab_insights:
    st.markdown("""
    ## 🧠 Beyond the Textbook
    
    ### ⚠️ What Breaks the Black-Scholes Model?
    While the BS process is a standard benchmark, it fails to capture several real-world phenomena:
    
    1.  **Volatility Clustering:** Real markets exhibit periods of high volatility followed by quiet periods (ARCH/GARCH effects). BS assumes constant volatility.
    2.  **Fat Tails (Kurtosis):** Returns are not strictly normal; extreme events (market crashes) happen more often than the model predicts.
    3.  **Leverage Effect:** Volatility often increases when stock prices fall—a negative correlation not captured here.
    4.  **The Volatility Smile:** If the model were perfect, all strikes would have the same implied volatility. In reality, they form a "smile" or "skew."
    
    ### 🚀 Advanced Alternatives
    If you find GBM too limiting, explore:
    - **Heston Model:** Adds stochastic volatility.
    - **Merton Jump Diffusion:** Adds random "jumps" to price paths.
    - **Bates Model:** Combines Heston and Jumps.
    
    ### 🛠️ Pro Tips for Recruiters
    - "I used **Vectorized NumPy** for performance but cross-validated with **QuantLib** for analytical precision."
    - "I implemented **Monte Carlo convergence analysis** to ensure simulation stability."
    - "I distinguished between **$\mathbb{P}$ and $\mathbb{Q}$ measures**, showing a deep understanding of derivative fundamental theorems."
    """)

