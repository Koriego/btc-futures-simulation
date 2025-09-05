import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# --- Configuración de la página ---
st.set_page_config(page_title="Simulación BTC + Futuros", layout="wide")
st.title("📈 Simulación de Bitcoin y contratos de futuros (GBM + tasa libre de riesgo)")

# --- Parámetros de simulación ---
st.sidebar.header("⚙️ Parámetros de simulación")

S0 = st.sidebar.number_input("Precio actual de BTC (USD)", value=50000.0, min_value=100.0)
mu = st.sidebar.slider("Rendimiento esperado anual (%)", -50, 100, 10) / 100
sigma = st.sidebar.slider("Volatilidad anual (%)", 10, 200, 60) / 100
T_days = st.sidebar.slider("Horizonte de simulación (días)", 30, 730, 180)
n_sim = st.sidebar.slider("Número de simulaciones", 10, 500, 100)
r = st.sidebar.slider("Tasa libre de riesgo anual (%)", 0.0, 10.0, 3.0) / 100

target_prices_input = st.sidebar.text_input("🎯 Precios objetivo (coma)", "100000,120000")
try:
    target_prices = [float(x.strip()) for x in target_prices_input.split(",") if x.strip()]
except:
    target_prices = []

# --- Simulación GBM ---
def simulate_gbm(S0, mu, sigma, T_days, n_sim):
    dt = 1 / 365
    steps = int(T_days)
    prices = np.zeros((steps + 1, n_sim))
    prices[0] = S0
    for t in range(1, steps + 1):
        z = np.random.standard_normal(n_sim)
        prices[t] = prices[t - 1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
    return prices

# --- Simular precios spot ---
prices = simulate_gbm(S0, mu, sigma, T_days, n_sim)
days = np.arange(prices.shape[0])
prices_df = pd.DataFrame(prices, index=days)

# --- Calcular precios de futuros ---
# --- Calcular precios de futuros (vectorizado) ---
T_array = (T_days - days) / 365  # shape (n_days,)
adjustment_factors = np.exp(r * T_array)  # shape (n_days,)

# Expand to shape (n_days, n_sim)
futures_df = prices_df.mul(adjustment_factors[:, np.newaxis], axis=0)


# --- Calcular percentiles ---
def get_percentiles(df):
    return {
        'P10': df.quantile(0.10, axis=1),
        'P25': df.quantile(0.25, axis=1),
        'P50': df.quantile(0.50, axis=1),
        'P75': df.quantile(0.75, axis=1),
        'P90': df.quantile(0.90, axis=1),
    }

spot_percentiles = get_percentiles(prices_df)
futures_percentiles = get_percentiles(futures_df)

# --- Gráfico spot y futuros ---
st.subheader("📉 Simulación de precios de BTC (Spot y Futuros)")

fig, ax = plt.subplots(figsize=(14, 6))

# Spot
ax.plot(days, spot_percentiles['P50'], label='Spot P50', color='blue', linewidth=2)
ax.fill_between(days, spot_percentiles['P10'], spot_percentiles['P90'], color='blue', alpha=0.1, label='Spot P10–P90')

# Futuros
ax.plot(days, futures_percentiles['P50'], label='Futuro P50', color='green', linestyle='--', linewidth=2)
ax.fill_between(days, futures_percentiles['P10'], futures_percentiles['P90'], color='green', alpha=0.1, label='Futuro P10–P90')

ax.set_xlabel("Día")
ax.set_ylabel("Precio (USD)")
ax.set_title("Simulación de precios de BTC y contratos de futuros")
ax.grid(True)
ax.legend()
st.pyplot(fig)

# --- Probabilidades de superar precios objetivo ---
st.subheader("🎯 Probabilidad de que BTC supere precios objetivo (Spot final)")

final_spot = prices_df.iloc[-1]
for target in target_prices:
    prob = (final_spot > target).mean() * 100
    st.write(f"📌 Probabilidad de que Spot > ${target:,.0f}: **{prob:.2f}%**")

# --- Tabla resumen final ---
st.subheader("📊 Valores simulados al día final")

summary_df = pd.DataFrame({
    'Spot P10': [spot_percentiles['P10'].iloc[-1]],
    'Spot P50': [spot_percentiles['P50'].iloc[-1]],
    'Spot P90': [spot_percentiles['P90'].iloc[-1]],
    'Futuro P10': [futures_percentiles['P10'].iloc[-1]],
    'Futuro P50': [futures_percentiles['P50'].iloc[-1]],
    'Futuro P90': [futures_percentiles['P90'].iloc[-1]],
})

st.dataframe(summary_df.style.format("${:,.0f}"))

# --- Botón de descarga ---
st.download_button(
    label="⬇️ Descargar precios simulados (Spot)",
    data=prices_df.to_csv().encode('utf-8'),
    file_name="simulaciones_spot.csv",
    mime="text/csv"
)

st.download_button(
    label="⬇️ Descargar precios simulados (Futuros)",
    data=futures_df.to_csv().encode('utf-8'),
    file_name="simulaciones_futuros.csv",
    mime="text/csv"
)

