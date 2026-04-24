import streamlit as st
import QuantLib as ql
import numpy as np
import plotly.graph_objects as go



st.set_page_config(page_title="Geometric Brownian Motion Process", layout="wide")

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
    .stButton>button {
        background: linear-gradient(90deg, #00d4ff, #0055ff);
        color: white;
        border: none;
        padding: 10px 25px;
        border-radius: 8px;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 0 15px rgba(0, 212, 255, 0.5);
    }
</style>
""", unsafe_allow_html=True)

st.title("📈 Geometric Brownian Motion Process")
st.write("Documentation and implementation of the Geometric Brownian Motion Process in QuantLib.")

st.markdown("""
## 🔍 What is Geometric Brownian Motion?

**Geometric Brownian Motion (GBM)** is the most widely used stochastic process in finance.

It models asset prices under the assumptions:
- Continuous time evolution
- Log-normal distribution of prices
- Random shocks driven by Brownian motion

It is the foundation of:
- Black-Scholes option pricing
- Risk-neutral valuation
- Monte Carlo simulations

---

## 🧠 Intuition

Think of GBM as:

> “A price that grows exponentially, but with random noise at every instant.”

- Drift (μ) → average growth  
- Volatility (σ) → randomness  
- Brownian Motion → uncertainty  

---

## 📐 Mathematical Formulation
""")

st.markdown("### Stochastic Differential Equation (SDE)")
st.latex(r"dS_t = \mu S_t \, dt + \sigma S_t \, dW_t")

st.latex(r"""
\begin{aligned}
\text{Where:}\\
S_t & : \text{Asset price} \\
\mu & : \text{Drift} \\
\sigma & : \text{Volatility} \\
W_t & : \text{Brownian Motion}
\end{aligned}
""")

st.markdown("### 📌 Solution of GBM")

st.latex(r"""
S_t = S_0 \exp\left((\mu - \tfrac{1}{2}\sigma^2)t + \sigma W_t\right)
""")
st.markdown("""
Why the $-\\frac{1}{2}\\sigma^2$ Term?

This term comes from **Itô's Lemma**.

Because Brownian motion has variance growing with time:
- $(dW_t)^2 = dt$

When applying Itô calculus to $\\log S_t$, an adjustment appears:
- This reduces the drift slightly

""")



st.markdown("""

## ⚙️ QuantLib Implementation

QuantLib provides a direct implementation of GBM:

```python
import QuantLib as ql

initialValue = 100
mu = 0.01
sigma = 0.2

process = ql.GeometricBrownianMotionProcess(initialValue, mu, sigma)

""")




S0 = st.sidebar.number_input("Initial Price (S₀)", 100.0)
mu = st.sidebar.slider("Drift (μ)", -0.5, 0.5, 0.05)
sigma = st.sidebar.slider("Volatility (σ)", 0.01, 1.0, 0.2)
T = st.sidebar.slider("Time Horizon (T)", 0.1, 2.0, 1.0)
steps = st.sidebar.slider("Steps", 50, 500, 200)
n_paths = st.sidebar.slider("Number of Paths", 1, 50, 10)
def simulate_gbm_ql(S0, mu, sigma, T, steps, n_paths):
    dt = T / steps

    # Time grid
    time_grid = ql.TimeGrid(T, steps)

    # GBM process
    process = ql.GeometricBrownianMotionProcess(S0, mu, sigma)

    # Random generator
    dimension = len(time_grid) - 1
    rng = ql.GaussianRandomSequenceGenerator(
        ql.UniformRandomSequenceGenerator(dimension, ql.UniformRandomGenerator())
    )

    # Path generator
    path_generator = ql.GaussianPathGenerator(process, T, steps, rng, False)

    paths = np.zeros((n_paths, steps + 1))

    for i in range(n_paths):
        sample_path = path_generator.next().value()
        for j in range(len(sample_path)):
            paths[i, j] = sample_path[j]

    time = np.linspace(0, T, steps + 1)
    return time, paths

time, data = simulate_gbm_ql(S0, mu, sigma, T, steps, n_paths)

st.markdown(""" 
## 📊 Distribution Properties

At any time t:
- $\log(S_t) \sim \mathcal{N}$ (normal distribution)

- $S_t \sim lognormal$

📌 Key implication:

- Heavy right tail → rare big gains
- No negative prices

""")

st.markdown(" --- ")

fig = go.Figure()

for i in range(n_paths):
    fig.add_trace(go.Scatter(
    x=time,
    y=data[i],
    mode='lines',
    name=f'Path {i+1}'
    ))

fig.update_layout(
title="GBM Simulated Paths",
xaxis_title="Time",
yaxis_title="Price",
template="plotly_dark"
)

st.plotly_chart(fig, use_container_width=True)

theoretical_mean = S0 * np.exp(mu * T)
theoretical_var = (S0**2) * np.exp(2 * mu * T) * (np.exp(sigma**2 * T) - 1)

st.write(f"Theoretical Mean: {theoretical_mean:.2f}")
st.write(f"Theoretical Variance: {theoretical_var:.2f}")

st.subheader("📊 Distribution of Final Prices")

final_vals = data[:, -1]

hist_fig = go.Figure()
hist_fig.add_trace(go.Histogram(x=final_vals))

hist_fig.update_layout(
title="Final Price Distribution (Log-Normal)",
template="plotly_dark"
)

st.plotly_chart(hist_fig, use_container_width=True)

st.write(f"Mean: {np.mean(final_vals):.2f}")
st.write(f"Std Dev: {np.std(final_vals):.2f}")

st.markdown("""

📌 Key Observations
- Higher μ → upward drift
- Higher σ → wider spread of paths
- Distribution becomes right-skewed
- Paths never go negative


⚠️ Limitations of GBM
- Assumes constant volatility
- No jumps or crashes
- No mean reversion
- Not realistic for long-term markets


🚀 Where GBM is Used
- Black-Scholes Model
- Monte Carlo Option Pricing
- Risk-neutral simulations

""")