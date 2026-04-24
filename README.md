# 🌌 Quant Workbench: Stochastic Processes Explorer

[![QuantLib](https://img.shields.io/badge/QuantLib-1.34-blue.svg)](https://www.quantlib.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-ff4b4b.svg)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.9+-3776ab.svg)](https://www.python.org/)

A professional-grade interactive laboratory for exploring **Stochastic Processes** and **Financial Engineering**. This workbench bridges the gap between complex mathematical theory and visual intuition, leveraging the power of **QuantLib** for industry-standard financial modeling.

---

## 🏗️ Core Architecture

This project is built using a modular, high-performance architecture designed for quantitative research and education:

*   **Financial Engine**: Powered by [QuantLib](https://www.quantlib.org/), the gold standard for quantitative finance.
*   **Simulation Suite**: High-performance vectorized simulations using **NumPy** for real-time interactivity.
*   **Visualization**: Dynamic, interactive charts powered by **Plotly** with support for zoomed path inspection and terminal distribution analysis.
*   **User Interface**: A premium **Glassmorphism UI** built with **Streamlit**, featuring a consistent 4-tab framework for every module.

---

## 🧪 Model Library

The workbench covers a wide spectrum of stochastic dynamics, grouped into specialized categories:

### 📈 Equity & FX Models
*   **Geometric Brownian Motion (GBM)**: The classical foundation.
*   **Black-Scholes-Merton**: Continuous dividend yields and cost-of-carry.
*   **Garman-Kohlhagen**: Specialized dual interest rate logic for FX.
*   **Heston Stochastic Volatility**: Mean-reverting vol with leverage effects.
*   **Heston SLV**: Hybrid Stochastic-Local Volatility for exotic pricing.
*   **Bates Model**: Combining Heston dynamics with Merton jumps.

### ⚡ Jump & Lévy Processes
*   **Merton Jump Diffusion**: Poisson-driven market shocks and fat tails.
*   **Variance Gamma**: Pure-jump Lévy process using stochastic "business time."
*   **EOU with Jumps**: Extended Ornstein-Uhlenbeck dynamics for mean-reverting shocks.

### 🏛️ Fixed Income & Interest Rates
*   **Hull-White Model**: No-arbitrage short rate model with yield curve fitting.
*   **GSR Process**: Generalized Gaussian Short Rate with piecewise parameters.
*   **G2++ Model**: Two-factor dynamics capturing yield curve twists and spreads.
*   **Forward Measure Logic**: Modeling dynamics under various $T$-Forward numeraires.

---

## 🛠️ The 4-Tab Framework

Every module in this workbench follows a standardized academic-to-applied workflow:

1.  **📚 Theory**: Detailed SDEs, analytical solutions, and conceptual context.
2.  **⚡ Simulation**: Real-time sample path generation with interactive parameter controls.
3.  **🎯 Analytics/Pricing**: Benchmarking Monte Carlo results against analytical formulas and Greeks.
4.  **🧠 Insights**: High-level qualitative takeaways and "Pro Quant Tips" for industry application.

---

## 🚀 Getting Started

### Prerequisites
*   Python 3.9+
*   QuantLib SWIG bindings
*   Streamlit

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/MichaelF102/Stochastic_Processes.git
   cd Stochastic_Processes
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Launch the workbench:
   ```bash
   streamlit run Introduction.py
   ```

---

## 👨‍💻 About the Creator

**Michael Fernandes**  
*Quantitative Finance Enthusiast | Data Science | Automation*

I enjoy building interactive tools that simplify complex systems and make learning more intuitive.

🔗 **Connect with me:**
*   [GitHub](https://github.com/MichaelF102)
*   [LinkedIn](https://www.linkedin.com/in/michael-fernandes-7a3b6227a/)

---

## ⚖️ License
This project is licensed under the MIT License - see the LICENSE file for details.
