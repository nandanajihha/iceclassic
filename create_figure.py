import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt
from scipy.stats import norm,gaussian_kde
import matplotlib.patches as mpatches

def get_path(filename, subdir=None):
    if subdir==None:
        directory = os.path.join(os.path.dirname(__file__))
    else:
        directory = os.path.join(os.path.dirname(__file__), subdir)
    filepath = os.path.join(directory, filename)
    return os.path.normpath(filepath)

def plot_date_time_distribution(df_1: pd.DataFrame,
                                datetime_col_1: str,
                                decimal_time_col_1: str,
                                df_2: pd.DataFrame, 
                                datetime_col_2: str,
                                decimal_time_col_2: str,
                                estimator: str,
                                title,
                                plot_limits: tuple = None,
                                plot_size: tuple = (15, 10)) -> plt.Figure:
    """
    Scatter plot of (datetime_col, decimal_time_col), and density plot of (decimal_time_col) and (datetime_col).
    Option to use day of the year for datetime on x-axis.

    Parameters
    ----------
    df_1 : pd.DataFrame
        DataFrame containing the dates.
    datetime_col_1 : str
        Column name containing the datetime values.
    decimal_time_col_1 : str
        Column name containing the decimal time values.
    df_2 : pd.DataFrame
        DataFrame containing the dates.
    datetime_col_2 : str
        Column name containing the datetime values.
    decimal_time_col_2 : str
        Column name containing the decimal time values.
    estimator : str 
        Estimator to use for the density plot. Options are 'mle' (maximum likelihood), 'mm' (moments).
    title : str, optional
        Title of the plot. The default is None.
    plot_size : tuple, optional
        Size of the plot. The default is (15, 10).
    
    Returns
    -------
    plt.Figure
        Figure object containing the plots.
    
    """
    fig, axs = plt.subplots(2, 2, figsize=plot_size, gridspec_kw={'width_ratios': [1, 6], 'height_ratios': [4, 1]})

    # Scatter plot in top right corner (axs[0,1])
    x_values_1 = df_1[datetime_col_1].dt.dayofyear
    x_values_2 = df_2[datetime_col_2].dt.dayofyear
    x_values = x_values_2
    mean_x = x_values.mean()

    y_values_1 = df_1[decimal_time_col_1]
    y_values_2 = df_2[decimal_time_col_2]
    y_values = y_values_2
    mean_y = y_values.mean()

    title += f", N={len(y_values_1)}"

    axs[0, 1].scatter(x_values_1, y_values_1,
                      color='red', label='2025 Predictions',
                      edgecolors='black', linewidths=1.0,
                      s=100)
    axs[0, 1].scatter(x_values_2, y_values_2,
                      color='gray', label='Historic Breakup',
                      edgecolors='black', linewidths=1.0,
                      s=20)
    
    if plot_limits is not None:
        axs[0, 1].set_xlim(plot_limits)
        title += f" (only showing days {plot_limits[0]} to {plot_limits[1]})"
        
    axs[0, 1].axvline(x=90, color='black', linestyle='--', alpha=0.5, linewidth=1.5, label='Betting Deadline')

    axs[0, 1].grid(True)
    axs[0, 1].legend()

    axs[0, 1].set_xlabel('Day of Year')
    axs[0, 1].set_ylabel('Time (Decimal)')
    axs[0, 1].set_title(title)

    # get xlimits from plot
    xlimits = axs[0, 1].get_xlim()

    # Density plot for datetime_col (Day of year) in position (1, 1)
    mu, std = norm.fit(x_values)
    x_values_density = np.linspace(np.min(x_values), np.max(x_values), 500)
    y_norm = norm.pdf(x_values_density, mu, std)

    kde_x = gaussian_kde(x_values)
    kde_values_x = kde_x(x_values_density)

    axs[1, 1].hist(x_values, bins='auto', density=True, alpha=0.5, color='gray', label='Histogram')
    axs[1, 1].plot(x_values_density, y_norm, label=f"Normal estimate", color='blue')
    axs[1, 1].plot(x_values_density, kde_values_x, label='KDE Estimate', color='red', alpha=0.3)
    axs[1, 1].set_ylabel('Density')
    axs[1, 1].invert_yaxis()
    axs[1, 1].legend()
    axs[1, 1].set_xlim(xlimits)
    axs[1, 1].set_xticks([])
    axs[1, 1].set_yticks([])

    # Density plot for decimal time in position (0, 0) (rotated)
    mu_time, std_time = norm.fit(y_values)
    y_values_time = np.linspace(np.min(y_values), np.max(y_values), 500)
    y_norm_time = norm.pdf(y_values_time, mu_time, std_time)

    kde_time = gaussian_kde(y_values)
    kde_values_time = kde_time(y_values_time)

    axs[0, 0].barh(np.histogram(y_values, bins='auto', density=True)[1][:-1], 
                   np.histogram(y_values, bins='auto', density=True)[0], 
                   height=np.diff(np.histogram(y_values, bins='auto')[1]), 
                   alpha=0.5, color='gray', label='Histogram')
    axs[0, 0].plot(kde_values_time, y_values_time, label='KDE Estimate', color='red', alpha=0.3)
    axs[0, 0].plot(y_norm_time, y_values_time, label=f"Norm estimate", color='blue')
    #axs[0, 0].invert_yaxis()
    axs[0, 0].invert_xaxis()
    axs[0, 0].legend()
    axs[0, 0].set_xticks([])

    # Hide the empty subplot
    axs[1, 0].axis('off')

    plt.tight_layout()  
    return fig


def plot_location_map(plot_size: tuple = (10, 8)) -> plt.Figure:
    """
    Create a simple map showing the location of Nenana, Alaska.
    
    Parameters
    ----------
    plot_size : tuple, optional
        Size of the plot. The default is (10, 8).
    
    Returns
    -------
    plt.Figure
        Figure object containing the location map.
    """
    # Nenana, Alaska coordinates
    nenana_lat = 64.5633
    nenana_lon = -149.0933
    
    # Alaska map boundaries (latitude and longitude ranges)
    ALASKA_LAT_MIN, ALASKA_LAT_MAX = 51, 72
    ALASKA_LON_MIN, ALASKA_LON_MAX = -170, -130
    
    # Highlight circle radius (in degrees)
    HIGHLIGHT_RADIUS = 2.0
    
    fig, ax = plt.subplots(figsize=plot_size)
    
    # Set map boundaries
    ax.set_xlim(ALASKA_LON_MIN, ALASKA_LON_MAX)
    ax.set_ylim(ALASKA_LAT_MIN, ALASKA_LAT_MAX)
    
    # Plot Nenana location
    ax.plot(nenana_lon, nenana_lat, 'ro', markersize=15, label='Nenana', zorder=5)
    
    # Add a circle around Nenana to highlight it
    circle = mpatches.Circle((nenana_lon, nenana_lat), HIGHLIGHT_RADIUS, 
                             fill=False, edgecolor='red', linewidth=2, linestyle='--')
    ax.add_patch(circle)
    
    # Add text annotation
    ax.annotate('Nenana, Alaska\nIce Classic Location', 
                xy=(nenana_lon, nenana_lat), 
                xytext=(nenana_lon + 5, nenana_lat + 3),
                fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', lw=2, color='red'))
    
    # Add reference cities for context
    # Anchorage
    anchorage_lat, anchorage_lon = 61.2181, -149.9003
    ax.plot(anchorage_lon, anchorage_lat, 'b^', markersize=10, label='Anchorage')
    ax.text(anchorage_lon - 1, anchorage_lat - 2, 'Anchorage', fontsize=9)
    
    # Fairbanks
    fairbanks_lat, fairbanks_lon = 64.8378, -147.7164
    ax.plot(fairbanks_lon, fairbanks_lat, 'bs', markersize=10, label='Fairbanks')
    ax.text(fairbanks_lon + 1, fairbanks_lat, 'Fairbanks', fontsize=9)
    
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title('Nenana Ice Classic - Event Location', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
    # Add coordinate information
    coord_text = f'Coordinates: {nenana_lat}°N, {abs(nenana_lon)}°W'
    ax.text(0.02, 0.02, coord_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    return fig


past_break_up_dates = pd.read_csv(get_path('breakup_dates.csv', subdir='data'))
past_break_up_dates['Break up dates'] = pd.to_datetime(past_break_up_dates['Break up dates'], errors='coerce')
past_break_up_dates['decimal time'] =past_break_up_dates['Break up dates'].dt.hour +past_break_up_dates['Break up dates'].dt.minute / 60

predictions = pd.read_csv(get_path('predictions.txt', subdir='data'), sep=';')
predictions['Prediction'] = pd.to_datetime(predictions['Prediction'], errors='coerce')
predictions['decimal time'] =predictions['Prediction'].dt.hour +predictions['Prediction'].dt.minute / 60
predictions.dropna(inplace=True)

fig = plot_date_time_distribution(predictions, 'Prediction', 'decimal time',
                            past_break_up_dates, 'Break up dates', 'decimal time',
                            estimator='mle',
                            title='Historic Breakup (with density estimates) and 2025 Predictions');
fig.savefig(get_path('predictions_all.svg'))

fig = plot_date_time_distribution(predictions, 'Prediction', 'decimal time',
                            past_break_up_dates, 'Break up dates', 'decimal time',
                            estimator='mle',
                            title='Historic Breakup (with density estimates) and 2025 Predictions',
                            plot_limits=(60, 155));
fig.savefig(get_path('predictions.svg'))

# Generate location map
fig_map = plot_location_map()
fig_map.savefig(get_path('location_map.svg'))