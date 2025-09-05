import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# --- Configuración de la página ---
st.set_page_config(page_title="Simulación BTC y Futuros", layout="wide")
st.title("📈 Simulación de Bitcoin y Contratos de Futuros")

# --- Parámetros desde el sidebar ---
st.sidebar.header("⚙️ Parámetros de simulación")

S0 = st.sidebar.number_input("Precio actual de BTC (USD)", value=50000.0, min_value=100.0)
mu = st.sidebar.slider("Rendimiento esperado anual (%)", -50, 100, 10) / 100
sigma = st.sidebar.slider("Volatilidad anual (%)", 10, 200, 60) / 100
T_days = st.sidebar.slider("Horizonte (días)", 30, 730, 180)
n_sim = st.sidebar.slider("Simulaciones", 10, 500, 100)
r = st.sidebar.slider("Tasa libre de riesgo (%)", 0.0, 10.0, 3.0) / 100

# --- Simular precios spot con GBM ---
def simulate_gbm(S0, mu, sigma, T_days, n_sim):
    dt = 1 / 365
    steps = T_days
    prices = np.zeros((steps + 1, n_sim))
    prices[0] = S0
    for t in range(1, steps + 1):
        z = np.random.standard_normal(n_sim)
        prices[t] = prices[t - 1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
    return prices

spot_prices = simulate_gbm(S0, mu, sigma, T_days, n_sim)
days = np.arange(spot_prices.shape[0])
spot_df = pd.DataFrame(spot_prices, index=days)

# --- Calcular precios teóricos de futuros ---
T_array = (T_days - days) / 365  # array con T - t
adjustment_factors = np.exp(r * T_array)  # exp(r(T - t))
futures_df = spot_df.mul(adjustment_factors[:, np.newaxis], axis=0)

# --- Percentiles ---
def get_percentiles(df):
    return {
        'P10': df.quantile(0.10, axis=1),
        'P25': df.quantile(0.25, axis=1),
        'P50': df.quantile(0.50, axis=1),
        'P75': df.quantile(0.75, axis=1),
        'P90': df.quantile(0.90, axis=1),
    }

spot_percentiles = get_percentiles(spot_df)
futures_percentiles = get_percentiles(futures_df)

# --- Gráfico ---
st.subheader("📉 Simulación de precios Spot vs Futuros")

fig, ax = plt.subplots(figsize=(14, 6))

# Spot
ax.plot(days, spot_percentiles['P50'], label='Spot P50', color='blue', linewidth=2)
ax.fill_between(days, spot_percentiles['P10'], spot_percentiles['P90'], color='blue', alpha=0.1, label='Spot P10–P90')

# Futuro
ax.plot(days, futures_percentiles['P50'], label='Futuro P50', color='green', linestyle='--', linewidth=2)
ax.fill_between(days, futures_percentiles['P10'], futures_percentiles['P90'], color='green', alpha=0.1, label='Futuro P10–P90')

ax.set_xlabel("Día")
ax.set_ylabel("Precio (USD)")
ax.set_title("Simulación de BTC Spot y Futuros con tasa libre de riesgo")
ax.grid(True)
ax.legend()
st.pyplot(fig)

# --- Tabla final ---
st.subheader("📊 Resumen de precios al último día")

summary = pd.DataFrame({
    "Spot P10": [spot_percentiles["P10"].iloc[-1]],
    "Spot P50": [spot_percentiles["P50"].iloc[-1]],
    "Spot P90": [spot_percentiles["P90"].iloc[-1]],
    "Futuro P10": [futures_percentiles["P10"].iloc[-1]],
    "Futuro P50": [futures_percentiles["P50"].iloc[-1]],
    "Futuro P90": [futures_percentiles["P90"].iloc[-1]],
})

st.dataframe(summary.style.format("${:,.0f}"))

# --- Descargar CSV ---
st.download_button(
    "⬇️ Descargar precios simulados (Spot)",
    data=spot_df.to_csv().encode("utf-8"),
    file_name="btc_spot_simulaciones.csv",
    mime="text/csv"
)

st.download_button(
    "⬇️ Descargar precios simulados (Futuros)",
    data=futures_df.to_csv().encode("utf-8"),
    file_name="btc_futuros_simulaciones.csv",
    mime="text/csv"
)
