import pandas as pd
import matplotlib.pyplot as plt

def plot_timeseries(df: pd.DataFrame, date_col: str, value_col: str):
    """
    Plots a time series with the date on the x-axis and a specified value on the y-axis.

    Args:
        df (pd.DataFrame): The input DataFrame.
        date_col (str): The name of the date column.
        value_col (str): The name of the column with values to plot.

    Returns:
        matplotlib.figure.Figure: The figure object containing the plot.
    """
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.plot(df[date_col], df[value_col], label=value_col)
    ax.set_xlabel('Date')
    ax.set_ylabel(value_col)
    ax.set_title(f'Time Series Plot of {value_col}')
    ax.legend()
    plt.grid(True)
    return fig

def plot_acf_pacf(series: pd.Series, lags: int = 40):
    """
    Plots the AutoCorrelation Function (ACF) and Partial AutoCorrelation Function (PACF)
    for a time series.

    Args:
        series (pd.Series): The time series data.
        lags (int, optional): The number of lags to plot. Defaults to 40.

    Returns:
        matplotlib.figure.Figure: The figure object containing the ACF and PACF plots.

    Raises:
        ImportError: If statsmodels is not installed.
    """
    try:
        from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    except ImportError:
        raise ImportError("statsmodels is required for plot_acf_pacf. Please install it using 'pip install statsmodels'.")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    plot_acf(series.dropna(), lags=lags, ax=ax1)
    ax1.set_title('Autocorrelation Function (ACF)')
    
    plot_pacf(series.dropna(), lags=lags, ax=ax2)
    ax2.set_title('Partial Autocorrelation Function (PACF)')
    
    return fig

def plot_seasonality_decomposition(series: pd.Series, period: int):
    """
    Performs seasonal decomposition and plots the trend, seasonal, and residual components.

    Args:
        series (pd.Series): The time series data.
        period (int): The period of the seasonality (e.g., 12 for monthly data with yearly seasonality).

    Returns:
        matplotlib.figure.Figure: The figure object containing the decomposition plots.

    Raises:
        ImportError: If statsmodels is not installed.
    """
    try:
        from statsmodels.tsa.seasonal import seasonal_decompose
    except ImportError:
        raise ImportError("statsmodels is required for plot_seasonality_decomposition. Please install it using 'pip install statsmodels'.")

    decomposition = seasonal_decompose(series.dropna(), model='additive', period=period)
    
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    
    decomposition.observed.plot(ax=ax1)
    ax1.set_ylabel('Observed')
    
    decomposition.trend.plot(ax=ax2)
    ax2.set_ylabel('Trend')
    
    decomposition.seasonal.plot(ax=ax3)
    ax3.set_ylabel('Seasonal')
    
    decomposition.resid.plot(ax=ax4)
    ax4.set_ylabel('Residual')
    
    plt.tight_layout()
    return fig

def plot_rolling_statistics(df: pd.DataFrame, col: str, window: int):
    """
    Plots the original time series along with its rolling mean and standard deviation.

    Args:
        df (pd.DataFrame): The input DataFrame.
        col (str): The name of the column to analyze.
        window (int): The size of the rolling window.

    Returns:
        matplotlib.figure.Figure: The figure object containing the plot.
    """
    fig, ax = plt.subplots(figsize=(15, 7))
    
    rolling_mean = df[col].rolling(window=window).mean()
    rolling_std = df[col].rolling(window=window).std()
    
    ax.plot(df.index, df[col], color='blue', label='Original')
    ax.plot(rolling_mean.index, rolling_mean, color='red', label=f'Rolling Mean (window={window})')
    ax.plot(rolling_std.index, rolling_std, color='black', label=f'Rolling Std (window={window})')
    
    ax.set_title(f'Rolling Statistics for {col}')
    ax.legend()
    return fig
