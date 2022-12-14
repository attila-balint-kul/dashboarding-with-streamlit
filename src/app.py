from datetime import timedelta

import numpy as np
import pandas as pd
import streamlit as st

import business_login as bl

np.random.seed(42)  # Make random control pseudorandom


@st.cache
def load_data() -> pd.DataFrame:
    """Loads the full dataset once, and keeps in cached to speed up subsequent runs."""
    return pd.read_csv(
        "./data/data.csv",
        parse_dates=['timestamp'],
        index_col='timestamp'
    )[['demand.power', 'pv.power']]


st.set_page_config(
    page_title="Battery Control Dashboard",
    layout="wide",
)

# Load the dataset
dataset = load_data()

# Initialize factory
algorithm_factory = bl.AlgorithmFactory()


# -------------------------------- DASHBOARD SIDEBAR -----------------------------------
with st.sidebar:
    # Builds the sidebar with the configuration.
    st.header('Configuration')
    st.text(f"Available data:\n    {dataset.index[0].date()} - {dataset.index[-1].date()}")
    optimization_date = st.date_input(
        label="Date of optimization",
        value=dataset.index[0].date() + timedelta(days=1),
        help="Select the date for which you want to run the optimization.",
        key="optimization_date",
    )
    horizon_days = st.slider(
        "Horizon",
        min_value=1,
        max_value=7,
        value=1,
        step=1,
        help="Horizon in number of days.",
    )

    st.subheader('Tariff')
    day_tariff = st.number_input(
        label="Day tariff [€/kWh]",
        value=0.30,
        step=0.1,
        min_value=0.,
        help="Consumption price during the day tariff period.",
    )
    night_tariff = st.number_input(
        label="Night tariff [€/kWh]",
        value=0.20,
        step=0.1,
        min_value=0.,
        help="Consumption price during the night tariff period.",
    )
    injection_tariff = st.number_input(
        label="Injection tariff [€/kWh]",
        value=0.,
        step=0.1,
        min_value=0.,
        help="Injection price during the day and night tariff period.",
    )

    st.subheader('Battery')
    battery_power = st.number_input(
        label="Rated power [kW]",
        value=5.,
        step=0.1,
        min_value=0.,
        help="Rated power of the battery inverter."
    )
    battery_capacity = st.number_input(
        label="Rated capacity [kWh]",
        value=5.,
        step=0.1,
        min_value=0.,
        help="Rated capacity of the battery."
    )
    battery_soc = st.number_input(
        label="Initial state of charge [%]",
        value=50,
        step=1,
        min_value=0,
        max_value=100,
        help="State of charge of the battery at the start of the optimization horizon."
    )
    # Build battery object
    battery = bl.Battery(
        rated_power__kW=battery_power,
        rated_capacity__kWh=battery_capacity,
        initial_state_of_charge__=battery_soc / 100,
    )

    st.subheader('Control')
    algorithm_A = st.selectbox(
        label="Control algorithm A",
        options=algorithm_factory.choices,
        help="Select the battery control optimization algorithm."
    )

    algorithm_B = st.selectbox(
        label="Control algorithm B",
        options=algorithm_factory.choices,
    )

    debug = st.checkbox(
        label="Debug",
        help="Enable to print out the various dataframes.",
    )

# -------------------------------- BUSINESS LOGIC --------------------------------------

# Slice the dataset to history and future
df_yesterday = bl.get_historical_values(dataset, optimization_date, n_days=1)
df_history = bl.get_historical_values(dataset, optimization_date)
df_true = bl.get_true_values(dataset, optimization_date, n_days=horizon_days)

# Make forecasts
df_forecast = bl.make_forecasts(
    history=df_history,
    n_days=horizon_days,
)

# Make prices
df_prices = bl.calculate_day_night_prices(
    timestamps=df_forecast.index,
    day_tariff=day_tariff,
    night_tariff=night_tariff,
    injection_tariff=injection_tariff,
)

# Optimize battery control
battery_control_A = algorithm_factory(algorithm_A)(battery, df_prices, df_forecast)
df_control_A = bl.simulate_battery(battery, battery_control_A)
df_costs_A = bl.calculate_costs(df_true, df_forecast, df_control_A, df_prices)
metrics_A = bl.calculate_control_metrics(df_true, df_forecast, df_costs_A)

battery_control_B = algorithm_factory(algorithm_B)(battery, df_prices, df_forecast)
df_control_B = bl.simulate_battery(battery, battery_control_B)
df_costs_B = bl.calculate_costs(df_true, df_forecast, df_control_B, df_prices)
metrics_B = bl.calculate_control_metrics(df_true, df_forecast, df_costs_B)

df_control = df_control_A.merge(df_control_B, left_index=True, right_index=True, suffixes=('_A', '_B'))
df_costs = df_costs_A.merge(df_costs_B, left_index=True, right_index=True, suffixes=('_A', '_B'))

# Calculate metrics
forecast_metrics = bl.calculate_forecast_metrics(df_true, df_forecast)
ab_metrics = bl.calculate_ab_metrics(metrics_A, metrics_B)

# ------------------------------ MAIN DASHBOARD AREA -----------------------------------

try:
    # Title
    st.title('Battery Control Dashboard')

    # Metrics row
    metrics = forecast_metrics + ab_metrics
    columns = st.columns(len(metrics))
    for col, metric in zip(columns, metrics):
        col.metric(
            label=metric['label'], value=metric['value'], delta=metric.get("delta")
        )

    # Main plot
    fig = bl.plot_results(
        df_yesterday, df_true, df_forecast, df_prices, df_costs, df_control, battery
    )
    st.plotly_chart(fig, use_container_width=True)
except Exception as e:
    raise e
finally:
    # Show debugging purposes even if there is an error
    # ---------------------------------- DEBUGGING AREA --------------------------------
    if debug:
        st.header("Debugging:")

        st.subheader("Historical values")
        st.dataframe(df_history)

        st.subheader("True future values")
        st.dataframe(df_true)

        st.subheader("Forecasts")
        st.dataframe(df_forecast)

        st.subheader("Prices")
        st.dataframe(df_prices)

        st.subheader("Control")
        left_col, right_col = st.columns(2)
        with left_col:
            st.text("A")
            st.dataframe(df_control_A)
        with right_col:
            st.text("B")
            st.dataframe(df_control_B)
        st.text("Merged")
        st.dataframe(df_control.head(2))

        st.subheader("Costs")
        left_col, right_col = st.columns(2)
        with left_col:
            st.text("A")
            st.dataframe(df_costs_A)
        with right_col:
            st.text("B")
            st.dataframe(df_costs_B)
        st.text("Merged")
        st.dataframe(df_costs.head(2))
