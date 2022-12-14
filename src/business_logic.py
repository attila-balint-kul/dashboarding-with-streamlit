from dataclasses import dataclass
from datetime import date, timedelta
from typing import Callable, List

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from darts import TimeSeries
from darts.metrics import rmse
from darts.models import NaiveSeasonal
from plotly.graph_objs import Figure
from plotly.subplots import make_subplots


# ------------------------------------ DATA MODELS -------------------------------------

@dataclass
class Battery:
    """Simple battery class to hold asset parameters."""
    rated_power__kW: float
    rated_capacity__kWh: float
    initial_state_of_charge__: float = 1.0
    round_trip_efficiency__: float = 1.0


# ----------------------------- BUSINESS LOGIC FUNCTIONS -------------------------------


def get_historical_values(
    dataset: pd.DataFrame,
    opt_date: date,
    n_days: int = None,
) -> pd.DataFrame:
    """Returns the historical values before the day of the optimization."""
    if n_days:
        from_ = opt_date - timedelta(days=n_days)
    else:
        from_ = None
    until_ = opt_date
    return dataset.loc[from_:until_][:-1]


def get_true_values(dataset: pd.DataFrame, opt_date: date, n_days: int = 1) -> pd.DataFrame:
    """Returns the actual values on the day of the optimization."""
    from_ = opt_date
    until_ = opt_date + timedelta(days=n_days)
    return dataset.loc[from_:until_][:-1][['pv.power', 'demand.power']]


def plot_results(
    history: pd.DataFrame,
    true: pd.DataFrame,
    forecasts: pd.DataFrame,
    prices: pd.DataFrame,
    costs: pd.DataFrame,
    control: pd.DataFrame,
    battery: Battery,
) -> Figure:
    """Plots the results of optimization."""
    n_rows = 4
    n_cols = 1
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=['PV & Demand', 'Prices', 'Battery', 'Costs & Savings'],
        shared_xaxes=True,
        row_heights=[1/6, 1/6, 2/6, 2/6],
        vertical_spacing=0.05,
        specs=[
            [{"secondary_y": False}],
            [{"secondary_y": False}],
            [{"secondary_y": True}],
            [{"secondary_y": False}],
        ],
    )

    def plot_demand_and_pv(row, col=1):
        # Historical PV & demand
        fig.add_trace(
            go.Scatter(
                x=history.index, y=-history['pv.power'],
                line=dict(color='#F5B041', width=3, dash='dot'),
                name="pv", legendgroup="history", legendgrouptitle_text="Historical values",
            ),
            row=row, col=col
        )
        fig.add_trace(
            go.Scatter(
                x=history.index, y=history['demand.power'],
                line=dict(color='#2E86C1', width=3, dash='dot'),
                name="demand", legendgroup="history"),
            row=row, col=col
        )

        # True future PV & demand values
        fig.add_trace(
            go.Scatter(
                x=true.index, y=-true['pv.power'],
                line=dict(color='#F4D03F', width=2, dash='dot'),
                name="pv", legendgroup="true", legendgrouptitle_text="True values"),
            row=row, col=col
        )
        fig.add_trace(
            go.Scatter(
                x=true.index, y=true['demand.power'],
                line=dict(color='#7FB3D5', width=2, dash='dot'),
                name="demand", legendgroup="true"),
            row=row, col=col
        )

        # Forecasted future PV & demand values
        # PV forecast
        fig.add_trace(
            go.Scatter(
                x=forecasts.index, y=-forecasts['pv.power'],
                line=dict(color='#F5B041', width=2),
                name="pv", legendgroup="forecast", legendgrouptitle_text="Forecasted values"),
            row=row, col=col
        )
        # Demand forecast
        fig.add_trace(
            go.Scatter(
                x=forecasts.index, y=forecasts['demand.power'],
                line=dict(color='#2E86C1', width=2),
                name="demand", legendgroup="forecast"),
            row=row, col=col
        )

        # PV & demand forecast errors
        fig.add_trace(
            go.Scatter(
                x=true.index, y=-true['pv.power'],
                line=dict(width=0, ),
                legendgroup="forecast-errors", showlegend=False, ),
            row=row, col=col
        )
        fig.add_trace(
            go.Scatter(
                x=forecasts.index, y=-forecasts['pv.power'],
                fill='tonexty',
                line=dict(color='#F5B041', width=0),
                name="pv", legendgroup="forecast-errors", legendgrouptitle_text="Forecast errors"),
            row=row, col=col
        )
        fig.add_trace(
            go.Scatter(
                x=true.index, y=true['demand.power'],
                line=dict(width=0),
                legendgroup="forecast-errors", showlegend=False, ),
            row=row, col=col
        )
        fig.add_trace(
            go.Scatter(
                x=forecasts.index, y=forecasts['demand.power'],
                fill='tonexty',
                line=dict(color='#2E86C1', width=0),
                name="demand", legendgroup="forecast-errors"),
            row=row, col=col
        )
        fig.update_yaxes(title_text="Power [kW]", row=row, col=col)

    def plot_battery(row, col=1):
        # A
        fig.add_trace(
            go.Scatter(
                x=control.index, y=control['battery.forecasted_soc_A'] * 100,
                line=dict(color="#CD6155"), opacity=0.6,
                name="state of charge", legendgroup="control-a",
            ),
            secondary_y=True,
            row=row, col=col
        )
        fig.add_trace(
            go.Scatter(
                x=control.index, y=control['battery.power_setpoint_A'],
                line=dict(color="#2E86C1", shape='hv'), opacity=0.8,
                name="setpoint", legendgroup="control-a", legendgrouptitle_text="Control A",
            ),
            secondary_y=False,
            row=row, col=col
        )

        # B
        fig.add_trace(
            go.Scatter(
                x=control.index, y=control['battery.forecasted_soc_B'] * 100,
                line=dict(color="#F5B041"), opacity=0.8,
                name="state of charge", legendgroup="control-b",
            ),
            secondary_y=True,
            row=row, col=col
        )
        fig.add_trace(
            go.Scatter(
                x=control.index, y=control['battery.power_setpoint_B'],
                line=dict(color="#52BE80", shape='hv'), opacity=0.7,
                name="setpoint", legendgroup="control-b", legendgrouptitle_text="Control B",
            ),
            secondary_y=False,
            row=row, col=col
        )

        # Red lines for power bounds of battery
        fig.add_hline(
            y=battery.rated_power__kW,
            line_width=1, line_dash="dot", line_color="red",
            secondary_y=False,
            row=row, col=col
        )
        fig.add_hline(
            y=-1 * battery.rated_power__kW,
            line_width=1, line_dash="dot", line_color="red",
            secondary_y=False,
            row=row, col=col
        )
        # Red lines for state of charge bounds of battery
        fig.add_hline(
            y=0,
            line_width=1, line_dash="dot", line_color="red",
            secondary_y=True,
            row=row, col=col
        )
        fig.add_hline(
            y=100,
            line_width=1, line_dash="dot", line_color="red",
            secondary_y=True,
            row=row, col=col
        )

        fig.update_yaxes(title_text="Power [kW]", row=row, col=col,
                         range=[-battery.rated_power__kW, battery.rated_power__kW],
                         secondary_y=False)
        fig.update_yaxes(title_text="State of Charge [%]", row=row, col=col,
                         range=[0, 100], secondary_y=True)

    def plot_prices(row, col=1):
        fig.add_trace(
            go.Scatter(
                x=prices.index, y=prices['consumption'],
                line=dict(color="#A569BD", shape="hv"),
                name="consumption", legendgroup="prices", legendgrouptitle_text="Prices",
            ),
            secondary_y=False,
            row=row, col=col
        )
        fig.add_trace(
            go.Scatter(
                x=prices.index, y=prices['injection'],
                line=dict(color="#EB984E", shape="hv"),
                name="injection", legendgroup="prices",
            ),
            secondary_y=False,
            row=row, col=col
        )

        fig.update_yaxes(title_text="Price [€/kWh]", row=row, col=col)

    def plot_costs(row, col=1):
        # A
        fig.add_trace(
            go.Scatter(
                x=costs.index, y=costs['realized_A'].cumsum(),
                line=dict(color="#2E86C1"), opacity=0.8,
                name="realized", legendgroup="costs-a",
            ),
            row=row, col=col
        )
        fig.add_trace(
            go.Scatter(
                x=costs.index, y=costs['forecasted_A'].cumsum(),
                line=dict(color="#2E86C1", width=2, dash='dot'), opacity=0.8,
                name="forecasted", legendgroup="costs-a", legendgrouptitle_text="Costs A",
            ),
            row=row, col=col
        )
        fig.add_trace(
            go.Scatter(
                x=costs.index, y=costs['baseline_A'].cumsum(),
                line=dict(color="#AAB7B8"),
                name="baseline", legendgroup="costs-a",
            ),
            row=row, col=col
        )

        # B
        fig.add_trace(
            go.Scatter(
                x=costs.index, y=costs['realized_B'].cumsum(),
                line=dict(color="#52BE80"), opacity=0.8,
                name="realized", legendgroup="costs-b",
            ),
            row=row, col=col
        )
        fig.add_trace(
            go.Scatter(
                x=costs.index, y=costs['forecasted_B'].cumsum(),
                line=dict(color="#52BE80", width=2, dash='dot'), opacity=0.8,
                name="forecasted", legendgroup="costs-b", legendgrouptitle_text="Costs B",
            ),
            row=row, col=col
        )
        fig.add_trace(
            go.Scatter(
                x=costs.index, y=costs['baseline_B'].cumsum(),
                line=dict(color="#AAB7B8"),
                name="baseline", legendgroup="costs-b",
            ),
            row=row, col=col
        )

        fig.update_yaxes(title_text="Cost [€]", row=row, col=col)

    plot_demand_and_pv(row=1)
    plot_prices(row=2)
    plot_battery(row=3)
    plot_costs(row=4)

    # Add vertical lines for today
    fig.add_vline(
        x=forecasts.index[0],
        line_width=1, line_dash="dash", line_color="black",
    )

    fig.update_xaxes(row=n_rows, col=1, range=[history.index[0], forecasts.index[-1]])
    fig.update_layout(height=900, margin=dict(l=0, r=0, t=30, b=0))
    return fig


def make_forecasts(history: pd.DataFrame, n_days: int) -> pd.DataFrame:
    """Creates all necessary forecasts and returns them in a DataFrame."""
    # Call the models to create forecast
    pv_forecast = naive_seasonal_forecast(history['pv.power'], n_days=n_days)
    demand_forecast = naive_seasonal_forecast(history['demand.power'], n_days=n_days)

    # Assemble and return forecast DataFrame
    return pd.DataFrame(
        data={
            "demand.power": demand_forecast,
            "pv.power": pv_forecast,
        }
    )


def calculate_day_night_prices(
    timestamps: pd.DatetimeIndex,
    day_tariff: float,
    night_tariff: float,
    injection_tariff: float
) -> pd.DataFrame:
    """Calculates the prices based on day-night tariff structure.

    Day tariff start every weekday at 9 AM and lasts until 22 PM.
    Weekends belong to night tariff.
    """
    prices = pd.DataFrame(index=timestamps)
    prices['injection'] = injection_tariff
    prices['consumption'] = night_tariff
    prices.loc[
        (prices.index.hour >= 6) & (prices.index.hour < 22)
        & (prices.index.dayofweek < 5), 'consumption'
    ] = day_tariff
    return prices


def simulate_battery(battery: Battery, control: pd.Series) -> pd.DataFrame:
    """Calculated the battery parameters based on the battery characteristics and the power setpoints."""
    control_idx = pd.date_range(control.index[0].date(), freq="15T", periods=len(control) + 1)

    # Calculate the battery energy content
    energy_content = [battery.rated_capacity__kWh * battery.initial_state_of_charge__]
    previous_energy = energy_content[0]
    actual_power = []
    for control_setpoint in control:
        new_energy = energy_content[-1] + control_setpoint / 4  # 15min resolution
        power = control_setpoint
        if new_energy > battery.rated_capacity__kWh:
            new_energy = battery.rated_capacity__kWh
            power = (battery.rated_capacity__kWh - previous_energy) / 4
        if new_energy < 0:
            new_energy = 0
            power = (0 - previous_energy) / 4

        energy_content.append(new_energy)
        actual_power.append(power)
        previous_energy = new_energy

    # TODO: Add realized power and state of charge in case control is not physically possible

    df = pd.DataFrame(
        index=control_idx,
        data={
            "battery.power_setpoint": np.append(control.values, [np.nan]),
            "battery.power": np.append(np.array(actual_power), [np.nan]),
            "battery.forecasted_soc": np.array(energy_content) / battery.rated_capacity__kWh,
        }
    )
    return df


def calculate_costs(
    true: pd.DataFrame,
    forecast: pd.DataFrame,
    control: pd.DataFrame,
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """Calculated the energy costs for the day of the optimization."""
    baseline_energy_balance = (true['demand.power'] - true['pv.power']) / 4  # 15min resolution
    baseline_consumption = baseline_energy_balance.clip(lower=0)
    baseline_injection = (-1 * baseline_energy_balance).clip(lower=0)

    # Costs based on forecasts and optimization
    forecasted_energy_balance = (forecast['demand.power'] - forecast['pv.power'] + control[
        'battery.power']) / 4  # 15min resolution
    forecasted_consumption = forecasted_energy_balance.clip(lower=0)
    forecasted_injection = (-1 * forecasted_energy_balance).clip(lower=0)

    # TODO: Calculate realized costs, not just based on forecasts
    realized_consumption = np.full_like(baseline_consumption, np.nan)
    realized_injection = np.full_like(baseline_consumption, np.nan)

    costs = pd.DataFrame(index=control.index)
    costs['baseline'] = baseline_consumption * prices['consumption'] \
                        - baseline_injection * prices['injection']
    costs['forecasted'] = forecasted_consumption * prices['consumption'] \
                          - forecasted_injection * prices['injection']
    costs['realized'] = realized_consumption * prices['consumption'] \
                          - realized_injection * prices['injection']
    return costs


def calculate_forecast_metrics(true: pd.DataFrame, forecasts: pd.DataFrame):
    """Calculates the forecast metrics.

    Metrics:
        RMSE of PV forecast error.
        RMSE of demand forecast error.
    """
    pv_forecast_rmse = rmse(
        actual_series=TimeSeries.from_series(true['pv.power']),
        pred_series=TimeSeries.from_series(forecasts['pv.power']),
    )
    demand_forecast_rmse = rmse(
        actual_series=TimeSeries.from_series(true['demand.power']),
        pred_series=TimeSeries.from_series(forecasts['demand.power']),
    )

    return [
        {
            "label": "PV Forecast RMSE",
            "value": f"{pv_forecast_rmse:.3f} kW",
        },
        {
            "label": "Demand Forecast RMSE",
            "value": f"{demand_forecast_rmse:.3f} kW",
        },
    ]


def calculate_control_metrics(true: pd.DataFrame, forecasts: pd.DataFrame, costs: pd.DataFrame):
    """Calculates the control metrics.

    Metrics:
        Forecasted costs savings (positive means we forecast a reduction in energy bill.)
        Actual costs savings (positive means we reduced the energy bill.)
    """
    baseline_costs = costs['baseline'].sum()
    forecasted_costs = costs['forecasted'].sum()
    absolute_cost_savings = baseline_costs - forecasted_costs
    relative_cost_savings = absolute_cost_savings / baseline_costs

    # TODO: Calculate realised savings

    return [
        {
            "label": "Forecasted Savings",
            "value": f"{absolute_cost_savings:.2f} €",
            "delta": f"{relative_cost_savings * 100:.2f}%",
        },
    ]


def calculate_ab_metrics(a_metrics: list, b_metrics: list) -> list:
    # TODO: Add comparing metrics to decide which control is better and by how much.
    # Return a dictionary as follows
    # {
    #     "label": "string", Display name of the metric
    #     "value": "string", Value of the metric
    #     "delta": "string", Change in the metric (optional)
    # }
    return []


# ------------------------------ FORECASTING MODELS ------------------------------------

def naive_seasonal_forecast(history: pd.Series, n_days: int) -> pd.Series:
    """Example naive seasonal forecast.

    :param history: Historical data.
    :return: A pandas Series with the forecast values and datetime index.
    """
    model = NaiveSeasonal(K=96)
    model.fit(TimeSeries.from_series(history))
    forecast = model.predict(n=n_days * 96)
    return forecast.pd_series()


# --------------------------- OPTIMIZATION ALGORITHMS ----------------------------------

def random_algo(battery: Battery, prices: pd.DataFrame, forecasts: pd.DataFrame) -> pd.Series:
    """Example battery optimization algorithm.

    This algorithm just return random control setpoints limited by the battery's rated power.

    :param battery: The battery object with it's parameters.
    :param prices: The prices dataframe with the consumption and injection prices
        for the optimization horizon.
    :param forecasts: The forecasts dataframe with the PV and demand forecasts
        for the optimization horizon.
    :return: A pandas Series with the battery power setpoints in kWs
        for the optimization horizon.
    """
    horizon = len(forecasts)
    control_values = (np.random.random(horizon) - 0.5) * battery.rated_power__kW * 2
    return pd.Series(control_values, index=forecasts.index)


def self_consumption(battery: Battery, prices: pd.DataFrame, forecasts: pd.DataFrame) -> pd.Series:
    consol_values = forecasts['pv.power'] - forecasts['demand.power']
    return pd.Series(consol_values, index=forecasts.index)


class AlgorithmFactory:
    _algorithms = {
        "Random": random_algo,
        "Self Consumption": self_consumption,
        # TODO: Optionally add your own algorithm here.
    }

    @property
    def choices(self) -> List[str]:
        return list(self._algorithms.keys())

    def __call__(self, algorithm: str) -> Callable:
        """Simply returns the algorithm function based on the registered name."""
        try:
            return self._algorithms[algorithm]
        except KeyError:
            raise ValueError(
                f"Algorithm '{algorithm}' is not registered, possible choices are: {self._algorithms.keys()}")
