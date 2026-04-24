import streamlit as st
import QuantLib as ql
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Page Config
st.set_page_config(page_title="Quant Stochastic Processes", layout="wide")

# Custom CSS for Premium Design
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
        padding: 30px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 25px;
        transition: 0.3s;
    }
    .glass-card:hover {
        background: rgba(255, 255, 255, 0.08);
        border-color: rgba(0, 212, 255, 0.4);
        transform: translateY(-5px);
    }
    h1, h2, h3 {
        color: #00d4ff !important;
        font-family: 'Outfit', sans-serif;
    }
    .hero-text {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00d4ff, #0055ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
    }
    .category-label {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: #ff007f;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        background: rgba(0, 212, 255, 0.1);
        border: 1px solid #00d4ff;
        color: #00d4ff;
        margin-right: 10px;
    }
</style>
""", unsafe_allow_html=True)

# --- Hero Section ---
col_hero1, col_hero2 = st.columns([2, 1])

with col_hero1:
    st.markdown("""
    ### 🌌 Stochastic Processes & Financial Engineering
    Welcome to the ultimate interactive laboratory for quantitative finance. This workbench bridges the gap between 
    **Stochastic Calculus** and **Real-World Markets**, providing professional-grade simulations and analytics.
    
    Built with **QuantLib**, **NumPy**, and **Plotly**, this guide explores the evolution of randomness from basic Brownian Motion 
    to complex Multi-Factor Term Structure models.
    """)
    
    st.markdown("""
    <div style="margin-top: 20px;">
        <span class="badge">QuantLib 1.34</span>
        <span class="badge">Monte Carlo Engines</span>
        <span class="badge">Analytical Greeks</span>
        <span class="badge">Measure Theory</span>
    </div>
    """, unsafe_allow_html=True)

with col_hero2:
    # A small live simulation to "wow" the user
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.caption("Live Preview: Geometric Brownian Motion")
    t = np.linspace(0, 1, 100)
    bm = np.cumsum(np.random.standard_normal(100)) * 0.1
    price = 100 * np.exp(0.05 * t + bm)
    
    fig_hero = go.Figure()
    fig_hero.add_trace(go.Scatter(x=t, y=price, line=dict(color='#00d4ff', width=3)))
    fig_hero.update_layout(
        template="plotly_dark", height=150, margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig_hero, use_container_width=True, config={'displayModeBar': False})
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

# --- Categorized Gallery ---
st.markdown("## 🏗️ Explore the Model Library")

def process_card(title, description, page_link, category):
    st.markdown(f"""
    <div class="glass-card">
        <div class="category-label">{category}</div>
        <h3>{title}</h3>
        <p style="font-size: 0.9rem; color: #ccc;">{description}</p>
        <a href="{page_link}" target="_self" style="text-decoration: none;">
            <button style="background: rgba(0, 212, 255, 0.1); border: 1px solid #00d4ff; color: #00d4ff; padding: 5px 15px; border-radius: 5px; cursor: pointer;">
                Launch Module →
            </button>
        </a>
    </div>
    """, unsafe_allow_html=True)

# Grid Layout
col_cat1, col_cat2, col_cat3 = st.columns(3)

with col_cat1:
    st.markdown("#### 📈 Equity & FX")
    process_card("Geometric Brownian Motion", "The fundamental building block of finance. Constant drift and volatility.", "/GeometricBrownianMotionProcess", "Classical")
    process_card("Black-Scholes-Merton", "Pricing with continuous dividend yields and cost-of-carry.", "/BlackScholesMertonProcess", "Standard")
    process_card("Heston Model", "Advanced stochastic volatility with mean reversion and leverage effects.", "/HestonProcess", "Stochastic Vol")
    process_card("Bates Model", "The 'Super-Model' combining Heston vol with Merton jumps.", "/BatesProcess", "Hybrid")

with col_cat2:
    st.markdown("#### ⚡ Jumps & Lévy")
    process_card("Merton Jump Diffusion", "Continuous diffusion punctuated by Poisson-driven crashes.", "/MertonJumpDiffusionProcess", "Discontinuous")
    process_card("Variance Gamma", "A pure-jump Lévy process using stochastic 'business time'.", "/VarianceGammaProcess", "Infinite Activity")
    process_card("Garman-Kohlhagen", "Tailored for the FX market with dual interest rate logic.", "/GarmanKohlagenProcess", "FX Special")
    process_card("Heston SLV", "Hybrid Stochastic-Local Volatility for exotic barrier options.", "/HestonSLVProcess", "Exotic")

with col_cat3:
    st.markdown("#### 🏛️ Fixed Income & Rates")
    process_card("Hull-White Model", "No-arbitrage short rate model calibrated to the yield curve.", "/HullWhiteProcess", "Short Rate")
    process_card("GSR Process", "Generalized short rate model with piecewise constant parameters.", "/GSRProcess", "Professional")
    process_card("G2++ Model", "Two-factor Gaussian model capturing yield curve twists and spreads.", "/G2Process", "Multi-Factor")
    process_card("Forward Measures", "Modeling dynamics under the T-Forward numeraire.", "/HullWhiteForwardProcess", "Measure Theory")

st.markdown("---")

# --- Footer / Quick Start ---
col_f1, col_f2 = st.columns([2, 1])

with col_f1:
    st.markdown("""
    ### 🛠️ How to use the Workbench
    1.  **Sidebar Controls**: Every module features a dynamic sidebar for parameter adjustments (Drift, Vol, Jumps, etc.).
    2.  **Tabbed UI**: Each page is organized into **Theory**, **Simulation**, **Analytics**, and **Insights**.
    3.  **Simulation Toggles**: Choose between high-speed **NumPy** vectorization or official **QuantLib** engines.
    4.  **Interactive Plots**: Use the Plotly toolbar to zoom, pan, and inspect individual sample paths.
    """)

with col_f2:
    st.markdown('<div class="glass-card" style="text-align: center;">', unsafe_allow_html=True)
    st.markdown("#### Ready to Start?")
    st.write("Jump into the most fundamental model:")
    st.markdown("""
    <a href="/GeometricBrownianMotionProcess" target="_self">
        <button style="width: 100%; padding: 12px; background: linear-gradient(90deg, #00d4ff, #0055ff); color: white; border: none; border-radius: 8px; font-weight: bold; cursor: pointer;">
            Start with GBM →
        </button>
    </a>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
## 👨‍💻 About the Creator

This project is built by **Michael Fernandes** — a passionate developer with strong interests in:

- 📊 Data Science  
- ⚙️ Automation  
- 💹 Quantitative Finance  

I enjoy building **interactive tools and systems** that simplify complex concepts and make learning more intuitive.

🔗 **Explore more of my work:**
- GitHub: https://github.com/MichaelF102  
- LinkedIn: https://www.linkedin.com/in/michael-fernandes-7a3b6227a/
""")