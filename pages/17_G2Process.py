import streamlit as st
import QuantLib as ql
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Page Config
st.set_page_config(page_title="G2++ Two-Factor Model", layout="wide")

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

r0 = st.sidebar.number_input("Initial Rate (r₀)", 0.03, step=0.005, format="%.3f")

st.sidebar.subheader("Factor 1 (x)")
a = st.sidebar.slider("Reversion Speed (a)", 0.01, 1.0, 0.1)
sigma = st.sidebar.slider("Volatility (σ)", 0.001, 0.1, 0.01, format="%.3f")

st.sidebar.subheader("Factor 2 (y)")
b = st.sidebar.slider("Reversion Speed (b)", 0.01, 1.0, 0.5)
eta = st.sidebar.slider("Volatility (η)", 0.001, 0.1, 0.008, format="%.3f")

st.sidebar.subheader("Joint Parameters")
rho = st.sidebar.slider("Correlation (ρ)", -1.0, 1.0, -0.7)

# Settings
st.sidebar.markdown("---")
T = st.sidebar.slider("Horizon (Years)", 1, 30, 10)
steps = st.sidebar.slider("Steps", 100, 500, 252)
n_paths = st.sidebar.number_input("Paths", 1, 500, 50)
seed = st.sidebar.number_input("Seed", 0, 1000, 42)

# Global QuantLib Setup
today = ql.Date.todaysDate()
ql.Settings.instance().evaluationDate = today
day_count = ql.Actual365Fixed()
ts = ql.YieldTermStructureHandle(ql.FlatForward(today, r0, day_count))

# --- Logic Functions ---

def simulate_g2_ql(a, sigma, b, eta, rho, T, steps, n_paths, seed):
    # G2 Process requires 2D shocks
    process = ql.G2Process(a, sigma, b, eta, rho, ts)
    
    # GaussianMultiPathGenerator for 2 factors
    rsg = ql.GaussianMultiPathGenerator(process, ql.TimeGrid(T, steps), 
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
        # In G2++, r(t) = x(t) + y(t) + phi(t). 
        # QuantLib's G2Process.x0() etc are already adjusted for phi if ts is provided.
        # But for path generation, multi_path[0] and multi_path[1] are the factors.
        r_paths[i, :] = x_paths[i, :] + y_paths[i, :] + r0 # Simplified phi as r0 for flat curve
        
    time = np.linspace(0, T, steps + 1)
    return time, x_paths, y_paths, r_paths

# --- Main App ---
st.title("📈 G2++ Two-Factor Model")
st.write("Modeling interest rate dynamics with two correlated stochastic factors.")

tab_theory, tab_sim, tab_analytics, tab_insights = st.tabs([
    "📚 Theory", "⚡ Simulation", "🎯 Analytics", "🧠 Insights"
])

# Run Simulation
time, x_data, y_data, r_data = simulate_g2_ql(a, sigma, b, eta, rho, float(T), steps, n_paths, seed)

with tab_theory:
    st.markdown("""
    ## 🔍 Beyond One Factor: The G2++ Model
    
    The **G2++ Model** (Two-Factor Gaussian Short Rate Model) is a more advanced interest rate model that addresses the major limitation of models like Hull-White: the assumption that all points on the yield curve are perfectly correlated.
    
    ### 🏗️ Why Two Factors?
    In a one-factor model, if the 1-year rate rises, the 10-year rate must rise by a proportional amount. In reality, the yield curve can **twist** (short rates rise, long rates fall) or **butterfly** (middle rates move differently than the wings). Two factors allow the model to capture these independent movements.
    
    ### 📐 The SDEs
    The short rate $r_t$ is the sum of two correlated Gaussian factors:
    """)
    st.latex(r"r_t = x_t + y_t + \phi(t)")
    st.latex(r"dx_t = -a x_t dt + \sigma dW_{1,t}")
    st.latex(r"dy_t = -b y_t dt + \eta dW_{2,t}")
    st.markdown("""
    With correlation $\mathbb{E}[dW_{1,t} dW_{2,t}] = \rho dt$.
    
    ### 🎯 Key Advantage
    The G2++ model can price products that depend on the **spread** between two rates (e.g., CMS Spread Options), which one-factor models fundamentally misprice.
    """)

with tab_sim:
    st.subheader("Resulting Short Rate Paths (r = x + y + φ)")
    fig_r = go.Figure()
    for i in range(min(n_paths, 15)):
        fig_r.add_trace(go.Scatter(x=time, y=r_data[i], mode='lines', line=dict(width=1.5), showlegend=False))
    fig_r.update_layout(template="plotly_dark", height=450, xaxis_title="Years", yaxis_title="Short Rate")
    st.plotly_chart(fig_r, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Factor 1 (x) Paths")
        fig_x = go.Figure()
        for i in range(min(n_paths, 10)):
            fig_x.add_trace(go.Scatter(x=time, y=x_data[i], mode='lines', line=dict(width=1, color='#00d4ff'), opacity=0.5, showlegend=False))
        fig_x.update_layout(template="plotly_dark", height=300)
        st.plotly_chart(fig_x, use_container_width=True)
    with col2:
        st.subheader("Factor 2 (y) Paths")
        fig_y = go.Figure()
        for i in range(min(n_paths, 10)):
            fig_y.add_trace(go.Scatter(x=time, y=y_data[i], mode='lines', line=dict(width=1, color='#ff007f'), opacity=0.5, showlegend=False))
        fig_y.update_layout(template="plotly_dark", height=300)
        st.plotly_chart(fig_y, use_container_width=True)

with tab_analytics:
    st.subheader("Factor Interaction Analysis")
    
    # Correlation Check
    emp_rho = np.corrcoef(x_data.flatten(), y_data.flatten())[0, 1]
    
    col_a1, col_a2 = st.columns(2)
    with col_a1:
        st.metric("Target Correlation (ρ)", f"{rho:.2f}")
        st.metric("Empirical Correlation", f"{emp_rho:.2f}")
        st.info("The correlation between factors determines how often the curve 'twists' vs. moves 'in parallel'.")
        
    with col_a2:
        # Scatter of factor increments
        dx = np.diff(x_data, axis=1).flatten()
        dy = np.diff(y_data, axis=1).flatten()
        fig_scat = go.Figure()
        fig_scat.add_trace(go.Scatter(x=dx, y=dy, mode='markers', marker=dict(size=2, opacity=0.3, color='#00d4ff')))
        fig_scat.update_layout(template="plotly_dark", title="Factor Increment Correlation", xaxis_title="Δx", yaxis_title="Δy")
        st.plotly_chart(fig_scat, use_container_width=True)

with tab_insights:
    st.markdown("""
    ## 🧠 High-Level Insights
    
    ### 1. Pricing Curve Spreads
    If you trade the spread between the 2-year and 10-year rate (a **Steepener** or **Flattener**), you are implicitly betting on the correlation between different parts of the curve. G2++ is the minimum requirement for modeling this risk correctly.
    
    ### 2. Imperfect Correlation
    In the G2++ model, the instantaneous correlation between two forward rates is no longer 1.0. This "decorrelation" makes the model much more realistic for large portfolios of interest rate swaps and options.
    
    ### 3. Analytical Tractability
    Despite having two factors, G2++ still allows for **analytical pricing** of Zero-Coupon Bonds and European Swaptions. This makes calibration just as fast as the one-factor Hull-White model.
    
    ### 🛠️ Pro Quant Tip
    "When calibrating G2++, look for a negative $\rho$ (around -0.7). This usually helps the model capture the 'hump' in the volatility term structure and the fact that short rates and long rates often move independently during economic shifts."
    """)

st.markdown("---")
st.caption("Quantitative Finance Portfolio | G2++ Two-Factor Process Implementation")