import gradio as gr
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
from bs4 import BeautifulSoup
import json
import warnings
import openpyxl
from fredapi import Fred
import os
import dotenv

warnings.filterwarnings('ignore')
import pandas as pd

dotenv.load_dotenv()

def get_interest_rates(currencies, start_date, end_date):
    """
    Retrieve 10-year government bond yields from FRED for the specified currencies.
    The function returns a DataFrame of yields (percent) indexed by date.
    
    Parameters:
    -----------
    currencies : list
        List of currency codes or country identifiers (e.g., ['USD','EUR', ...]).
    start_date : str or datetime-like
        Start date for data retrieval, in YYYY-MM-DD format or a datetime object.
    end_date : str or datetime-like
        End date for data retrieval, in YYYY-MM-DD format or a datetime object.
        
    Returns:
    --------
    interest_rates : pd.DataFrame
        DataFrame indexed by date, with one column per currency (country) representing the yield time series.
    """
    
    # FRED series IDs for 10-year government bond yields.
    # These IDs are subject to change/availability. Check FRED for the latest info.
    all_proxies_fred = {
        'USD': 'DGS10',               # 10-Year Treasury Constant Maturity (Percent)
        'EUR': 'IRLTLT01EZM156N',     # 10-Year Govt Bond Yields: Euro Area
        'GBP': 'IRLTLT01GBM156N',     # 10-Year Govt Bond Yields: United Kingdom
        'CNY': 'IRLTLT01CNM156N',     # 10-Year Govt Bond Yields: China
        'AUD': 'IRLTLT01AUM156N',     # 10-Year Govt Bond Yields: Australia
        'CAD': 'IRLTLT01CAM156N',     # 10-Year Govt Bond Yields: Canada
    }
    
    # Initialize FRED client. Replace with your own API key.
    fred = Fred(api_key=os.getenv('FRED_API_KEY'))
    
    # If no currencies specified, use the entire list
    if not currencies:
        currencies = list(all_proxies_fred.keys())
    
    interest_rates = pd.DataFrame()
    
    # Loop over each requested currency
    for currency in currencies:
        series_id = all_proxies_fred.get(currency)
        if not series_id:
            print(f"No FRED series found for {currency}, skipping.")
            continue
        
        print(f"Downloading FRED data for {currency} (Series ID: {series_id})")
        
        try:
            # fred.get_series returns a pandas Series with a DatetimeIndex by default
            data_series = fred.get_series(series_id,
                                          observation_start=start_date,
                                          observation_end=end_date)
            
            if data_series is None or data_series.empty:
                print(f"No data returned for {currency}")
                continue
            
            # Add this series to the interest_rates DataFrame
            interest_rates[currency] = data_series
            print(f"Successfully retrieved data for {currency}")
        
        except Exception as e:
            print(f"Error retrieving data for {currency}: {e}")
    
    # The resulting DataFrame is indexed by date, with columns = [currency codes]
    # Each value is the daily (or monthly) yield for that date, depending on series frequency.
    return interest_rates
    

def get_gdp_data(countries, start_year, end_year):
    """
    Get GDP data for selected countries from CSV file with year range filter
    
    Parameters:
    countries (list): List of country codes to filter
    start_year (int): Start year for the data range
    end_year (int): End year for the data range
    
    Returns:
    pandas.DataFrame: Filtered GDP data for the selected countries and years
    """
    try:
        # Read GDP data from CSV file
        gdp_df = pd.read_csv('./data/gdp_data.csv', skiprows=4)
        
        # Map country codes to full names for filtering
        country_mapping = {
            'US': 'United States',
            'UK': 'United Kingdom',
            'China': 'China',
            'Canada': 'Canada',
            'Australia': 'Australia'
        }
        
        # Convert selected country codes to full names
        selected_countries = [country_mapping.get(country) for country in countries if country in country_mapping]
        
        # Filter data for selected countries
        filtered_data = gdp_df[gdp_df['Country Name'].isin(selected_countries)]
        
        if filtered_data.empty:
            return None
            
        # Select only the year columns within the specified range
        year_columns = [str(year) for year in range(start_year, end_year + 1)]
        available_year_columns = [col for col in year_columns if col in filtered_data.columns]
        
        if not available_year_columns:
            return None
            
        # Select country name and year columns
        result_df = filtered_data[['Country Name'] + available_year_columns].copy()
        
        # Melt the dataframe to convert years from columns to rows
        melted_df = pd.melt(
            result_df, 
            id_vars=['Country Name'], 
            value_vars=available_year_columns,
            var_name='Year', 
            value_name='Value'
        )
        
        # Convert Year column to integer
        melted_df['Year'] = melted_df['Year'].astype(int)
        
        # Rename Country Name to Country for consistency
        melted_df = melted_df.rename(columns={'Country Name': 'Country'})
        
        # Pivot the data to have years as columns and countries as rows for easier plotting
        pivot_df = melted_df.pivot(index='Country', columns='Year', values='Value').reset_index()
        
        return pivot_df
        
    except Exception as e:
        print(f"Error reading GDP data from CSV: {e}")
        return None

def get_inflation_data(countries, start_year, end_year):
    """
    Get inflation data for selected countries from CSV file with year range filter
    
    Parameters:
    countries (list): List of country codes to filter
    start_year (int): Start year for the data range
    end_year (int): End year for the data range
    
    Returns:
    pandas.DataFrame: Filtered GDP data for the selected countries and years
    """
    try:
        # Read GDP data from CSV file
        inflation_df = pd.read_csv('./data/inflation_data.csv', skiprows=4)
        
        # Map country codes to full names for filtering
        country_mapping = {
            'US': 'United States',
            'UK': 'United Kingdom',
            'China': 'China',
            'Canada': 'Canada',
            'Australia': 'Australia'
        }
        
        # Convert selected country codes to full names
        selected_countries = [country_mapping.get(country) for country in countries if country in country_mapping]
        
        # Filter data for selected countries
        filtered_data = inflation_df[inflation_df['Country Name'].isin(selected_countries)]
        
        if filtered_data.empty:
            return None
            
        # Select only the year columns within the specified range
        year_columns = [str(year) for year in range(start_year, end_year + 1)]
        available_year_columns = [col for col in year_columns if col in filtered_data.columns]
        
        if not available_year_columns:
            return None
            
        # Select country name and year columns
        result_df = filtered_data[['Country Name'] + available_year_columns].copy()
        
        # Melt the dataframe to convert years from columns to rows
        melted_df = pd.melt(
            result_df, 
            id_vars=['Country Name'], 
            value_vars=available_year_columns,
            var_name='Year', 
            value_name='Value'
        )
        
        # Convert Year column to integer
        melted_df['Year'] = melted_df['Year'].astype(int)
        
        # Rename Country Name to Country for consistency
        melted_df = melted_df.rename(columns={'Country Name': 'Country'})
        
        # Pivot the data to have years as columns and countries as rows for easier plotting
        pivot_df = melted_df.pivot(index='Country', columns='Year', values='Value').reset_index()
        
        return pivot_df
        
    except Exception as e:
        print(f"Error reading inflation data from CSV: {e}")
        return None

def get_commodity_data(commodities, start_date, end_date):
    """
    Get price data for selected commodities
    """
    # Tickers for commodities
    tickers = {
        'Oil': 'CL=F',  # Crude Oil Futures
        'Gold': 'GC=F',  # Gold Futures
        'Silver': 'SI=F',  # Silver Futures
        'BTC': 'BTC-USD',  # Bitcoin
        'ETH': 'ETH-USD'   # Ethereum
    }
    
    commodity_data = pd.DataFrame()
    
    for commodity in commodities:
        if commodity in tickers:
            try:
                data = yf.download(tickers[commodity], start=start_date, end=end_date)
                if not data.empty:
                    commodity_data[commodity] = data['Close']
            except Exception as e:
                print(f"Error getting data for {commodity}: {e}")
    
    return commodity_data

# Plotting functions
def plot_interest_rates(interest_data, currencies_to_display):
    """
    Plot interest rates for the selected currencies
    
    Parameters:
    interest_data (pandas.DataFrame): DataFrame with interest rate data for all currencies
    currencies_to_display (list): List of currencies to display in the plot
    
    Returns:
    plotly.graph_objects.Figure: The plotted figure
    """
    if interest_data is None or interest_data.empty:
        fig = go.Figure()
        fig.update_layout(
            title="No Interest Rate Data Available",
            template="plotly_dark",
            paper_bgcolor="black",
            plot_bgcolor="black",
            font=dict(color="white"),
            autosize=True
        )
        return fig
    
    # Debug information
    print(f"Interest data columns: {interest_data.columns.tolist()}")
    print(f"Currencies to display: {currencies_to_display}")
    print(f"Interest data shape: {interest_data.shape}")
    print(f"Interest data sample:\n{interest_data.head()}")
    
    fig = go.Figure()
    
    # Filter to include only the currencies we have data for
    available_currencies = [c for c in currencies_to_display if c in interest_data.columns]
    
    # If no selected currencies have data, show all available currencies
    if not available_currencies and not interest_data.empty:
        available_currencies = interest_data.columns.tolist()
    
    print(f"Available currencies for plotting: {available_currencies}")
    
    # Use different colors for each currency for better visibility
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan']
    
    for i, currency in enumerate(available_currencies):
        # Get color index with wraparound if we have more currencies than colors
        color_idx = i % len(colors)
        
        # Make sure the data for this currency is not all NaN
        if interest_data[currency].notna().any():
            # Create a copy without NaN values for this trace
            plot_data = interest_data[currency].dropna()
            
            fig.add_trace(go.Scatter(
                x=plot_data.index,
                y=plot_data.values,
                mode='lines',
                name=f"{currency} Interest Rate",
                line=dict(color=colors[color_idx], width=2)
            ))
            print(f"Added trace for {currency} with {len(plot_data)} points")
        else:
            print(f"No non-NaN data for {currency}, skipping")
    
    fig.update_layout(
        title="Interest Rates by Currency",
        xaxis_title="Date",
        yaxis_title="Rate (%)",
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor="black",
        plot_bgcolor="black",
        font=dict(color="white"),
        autosize=True
    )
    
    # Enable autoscaling for both axes
    fig.update_xaxes(autorange=True)
    fig.update_yaxes(autorange=True)
    
    return fig

def plot_gdp_comparison(gdp_data, start_year, end_year):
    """
    Plot GDP comparison as a step function across years
    
    Parameters:
    gdp_data (pandas.DataFrame): DataFrame with GDP data
    start_year (int): Start year for x-axis
    end_year (int): End year for x-axis
    
    Returns:
    plotly.graph_objects.Figure: The plotted figure
    """
    if gdp_data is None or gdp_data.empty:
        fig = go.Figure()
        fig.update_layout(
            title="No GDP Data Available",
            template="plotly_dark",
            paper_bgcolor="black",
            plot_bgcolor="black",
            font=dict(color="white"),
            autosize=True
        )
        return fig
    
    fig = go.Figure()
    
    # Get all year columns (excluding the 'Country' column)
    year_columns = [col for col in gdp_data.columns if col != 'Country']
    
    # Filter years based on the range
    year_columns = [year for year in year_columns if start_year <= year <= end_year]
    
    # For each country, add a step line
    for _, row in gdp_data.iterrows():
        country = row['Country']
        
        # Extract values for the selected years
        years = [year for year in year_columns]
        values = [row[year] for year in year_columns]
        
        fig.add_trace(go.Scatter(
            x=years,
            y=values,
            mode='lines+markers',
            name=country,
            line=dict(shape='hv')  # 'hv' creates horizontal first, then vertical line (step function)
        ))
    
    fig.update_layout(
        title="GDP Trends by Country",
        xaxis_title="Year",
        yaxis_title="GDP Value",
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor="black",
        plot_bgcolor="black",
        font=dict(color="white"),
        autosize=True
    )
    
    # Enable autoscaling for both axes
    fig.update_xaxes(autorange=True)
    fig.update_yaxes(autorange=True)
    
    return fig


def plot_inflation_comparison(inflation_data, start_year, end_year):
    """
    Plot inflation comparison as a step function across years
    
    Parameters:
    inflation_data (pandas.DataFrame): DataFrame with inflation data
    start_year (int): Start year for x-axis
    end_year (int): End year for x-axis
    
    Returns:
    plotly.graph_objects.Figure: The plotted figure
    """
    if inflation_data is None or inflation_data.empty:
        fig = go.Figure()
        fig.update_layout(
            title="No Inflation Data Available",
            template="plotly_dark",
            paper_bgcolor="black",
            plot_bgcolor="black",
            font=dict(color="white"),
            autosize=True
        )
        return fig
    
    fig = go.Figure()
    
    # Get all year columns (excluding the 'Country' column)
    year_columns = [col for col in inflation_data.columns if col != 'Country']
    
    # Filter years based on the range
    year_columns = [year for year in year_columns if start_year <= year <= end_year]
    
    # For each country, add a step line
    for _, row in inflation_data.iterrows():
        country = row['Country']
        
        # Extract values for the selected years
        years = [year for year in year_columns]
        values = [row[year] for year in year_columns]
        
        fig.add_trace(go.Scatter(
            x=years,
            y=values,
            mode='lines+markers',
            name=country,
            line=dict(shape='hv')  # 'hv' creates horizontal first, then vertical line (step function)
        ))
    
    fig.update_layout(
        title="Inflation Trends by Country",
        xaxis_title="Year",
        yaxis_title="Inflation Rate",
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor="black",
        plot_bgcolor="black",
        font=dict(color="white"),
        autosize=True
    )
    
    # Enable autoscaling for both axes
    fig.update_xaxes(autorange=True)
    fig.update_yaxes(autorange=True)
    
    return fig

def plot_commodities(commodity_data):
    """Plot commodity prices"""
    if commodity_data is None or commodity_data.empty:
        fig = go.Figure()
        fig.update_layout(
            title="No Commodity Data Available",
            template="plotly_dark",
            paper_bgcolor="black",
            plot_bgcolor="black",
            font=dict(color="white"),
            autosize=True
        )
        return fig
        
    # Create subplots: one for each commodity to better visualize different scales
    fig = make_subplots(
        rows=len(commodity_data.columns), 
        cols=1,
        subplot_titles=[f"{commodity} Price" for commodity in commodity_data.columns],
        shared_xaxes=True,
        vertical_spacing=0.05
    )
    
    for i, commodity in enumerate(commodity_data.columns):
        fig.add_trace(
            go.Scatter(
                x=commodity_data.index,
                y=commodity_data[commodity],
                mode='lines',
                name=commodity
            ),
            row=i+1, 
            col=1
        )
    
    fig.update_layout(
        height=300 * len(commodity_data.columns),
        title_text="Commodity Price Trends",
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        showlegend=True,
        paper_bgcolor="black",
        plot_bgcolor="black",
        font=dict(color="white"),
        autosize=True
    )
    
    # Enable autoscaling for each subplot
    for i in range(1, len(commodity_data.columns) + 1):
        fig.update_xaxes(autorange=True, row=i, col=1)
        fig.update_yaxes(autorange=True, row=i, col=1)
    
    return fig

# Helper function to convert date options to actual dates
def convert_date_option(date_option):
    date_options = {
        "Today": datetime.now().strftime('%Y-%m-%d'),
        "Yesterday": (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d'),
        "1 Week Ago": (datetime.now() - timedelta(weeks=1)).strftime('%Y-%m-%d'),
        "1 Month Ago": (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
        "3 Months Ago": (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d'),
        "6 Months Ago": (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d'),
        "1 Year Ago": (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'),
        "5 Years Ago": (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d'),
        "10 Years Ago": (datetime.now() - timedelta(days=10*365)).strftime('%Y-%m-%d'),
        "15 Years Ago": (datetime.now() - timedelta(days=15*365)).strftime('%Y-%m-%d'),
        "20 Years Ago": (datetime.now() - timedelta(days=20*365)).strftime('%Y-%m-%d')
    }
    return date_options[date_option]

def update_interest_rates(start_date_option, end_date_option, currencies_to_display):
    start_date = convert_date_option(start_date_option)
    end_date = convert_date_option(end_date_option)
    
    print(f"Retrieving interest rates from {start_date} to {end_date} for {currencies_to_display}")
    
    # Get data only for the selected currencies
    interest_data = get_interest_rates(currencies_to_display, start_date, end_date)
    
    print(f"Retrieved interest rate data with shape: {interest_data.shape if interest_data is not None else 'None'}")
    
    return plot_interest_rates(interest_data, currencies_to_display)

# Main Gradio interface
def create_dashboard():
    with gr.Blocks(theme="dark") as dashboard:
        gr.Markdown("# International Finance Dashboard", elem_id="dashboard-title")
        
        # Add custom CSS for complete black background
        gr.HTML("""
        <style>
            body, .gradio-container, .dark {
                background-color: black !important;
            }
            
            .tabs {
                background-color: black !important;
                border-color: #333 !important;
            }
            
            .tab-nav {
                background-color: black !important;
            }
            
            .tab-nav button {
                background-color: black !important;
                color: white !important;
                border-color: #333 !important;
            }
            
            .tab-nav button.selected {
                border-bottom-color: white !important;
            }
            
            #dashboard-title {
                color: white !important;
            }
            
            .form {
                background-color: black !important;
                border-color: #333 !important;
            }
            
            input, select, button {
                background-color: #111 !important;
                color: white !important;
                border-color: #333 !important;
            }
            
            label, p {
                color: white !important;
            }
        </style>
        """)
        
        # Create tabs for different visualizations
        with gr.Tabs() as tabs:
            # Interest Rates Tab
            with gr.TabItem("Interest Rates"):
                with gr.Row():
                    with gr.Column(scale=1):
                        interest_start_date = gr.Dropdown(
                            label="Start Date",
                            choices=["5 Years Ago", "10 Years Ago", "15 Years Ago", "20 Years Ago"],
                            value="10 Years Ago"
                        )
                    
                    with gr.Column(scale=1):
                        interest_end_date = gr.Dropdown(
                            label="End Date",
                            choices=["Today", "1 Year Ago", "5 Years Ago", "10 Years Ago", "15 Years Ago", "20 Years Ago"],
                            value="Today"
                        )
                
                with gr.Row():
                    interest_currency_selection = gr.CheckboxGroup(
                        label="Select Currencies to Display",
                        choices=["USD", "EUR", "GBP", "CNY", "AUD", "CAD"],
                        value=["USD", "EUR", "GBP", "CNY", "AUD", "CAD"]  # Select all by default
                    )
                
                with gr.Row():
                    interest_update_btn = gr.Button("Update Interest Rates")
                
                interest_plot = gr.Plot(label="Interest Rates")
                
                interest_update_btn.click(
                    fn=update_interest_rates,
                    inputs=[interest_start_date, interest_end_date, interest_currency_selection],
                    outputs=interest_plot
                )
            
            # GDP Comparison Tab
            with gr.TabItem("GDP Comparison"):
                with gr.Row():
                    gdp_country_selection = gr.CheckboxGroup(
                        label="Select Countries for GDP Comparison",
                        choices=["US", "China", "UK", "Canada", "Australia"],
                        value=["US", "China", "UK"]
                    )
                
                with gr.Row():
                    with gr.Column(scale=1):
                        start_year_slider = gr.Slider(
                            minimum=1968,
                            maximum=2022,
                            value=2000,
                            step=1,
                            label="Start Year"
                        )
                    with gr.Column(scale=1):
                        end_year_slider = gr.Slider(
                            minimum=1968,
                            maximum=2022,
                            value=2022,
                            step=1,
                            label="End Year"
                        )
                
                with gr.Row():
                    gdp_update_btn = gr.Button("Update GDP Comparison")
                
                gdp_plot = gr.Plot(label="GDP by Country Over Time")
                
                def update_gdp_comparison(countries, start_year, end_year):
                    # Ensure start_year is not greater than end_year
                    if start_year > end_year:
                        start_year, end_year = end_year, start_year
                    
                    gdp_data = get_gdp_data(countries, start_year, end_year)
                    return plot_gdp_comparison(gdp_data, start_year, end_year)
                
                gdp_update_btn.click(
                    fn=update_gdp_comparison,
                    inputs=[gdp_country_selection, start_year_slider, end_year_slider],
                    outputs=gdp_plot
                )

            # Inflation Comparison Tab
            with gr.TabItem("Inflation Comparison"):
                with gr.Row():
                    inflation_country_selection = gr.CheckboxGroup(
                        label="Select Countries for Inflation Comparison",
                        choices=["US", "China", "UK", "Canada", "Australia"],
                        value=["US", "China", "UK"]
                    )
                
                with gr.Row():
                    with gr.Column(scale=1):
                        start_year_slider = gr.Slider(
                            minimum=1968,
                            maximum=2022,
                            value=2000,
                            step=1,
                            label="Start Year"
                        )
                    with gr.Column(scale=1):
                        end_year_slider = gr.Slider(
                            minimum=1968,
                            maximum=2022,
                            value=2022,
                            step=1,
                            label="End Year"
                        )
                
                with gr.Row():
                    inflation_update_btn = gr.Button("Update Inflation Comparison")
                
                inflation_plot = gr.Plot(label="Inflation by Country Over Time")
                
                def update_inflation_comparison(countries, start_year, end_year):
                    # Ensure start_year is not greater than end_year
                    if start_year > end_year:
                        start_year, end_year = end_year, start_year
                    
                    inflation_data = get_inflation_data(countries, start_year, end_year)
                    return plot_inflation_comparison(inflation_data, start_year, end_year)
                
                inflation_update_btn.click(
                    fn=update_inflation_comparison,
                    inputs=[inflation_country_selection, start_year_slider, end_year_slider],
                    outputs=inflation_plot
                )
            
            # Commodities Tab
            with gr.TabItem("Commodities"):
                with gr.Row():
                    with gr.Column(scale=1):
                        commodity_start_date = gr.Dropdown(
                            label="Start Date",
                            choices=["1 Week Ago", "1 Month Ago", "3 Months Ago", "6 Months Ago", "1 Year Ago"],
                            value="3 Months Ago"
                        )
                    
                    with gr.Column(scale=1):
                        commodity_end_date = gr.Dropdown(
                            label="End Date",
                            choices=["Today", "Yesterday", "1 Week Ago"],
                            value="Today"
                        )
                
                with gr.Row():
                    commodity_selection = gr.CheckboxGroup(
                        label="Select Commodities",
                        choices=["Oil", "Gold", "Silver", "BTC", "ETH"],
                        value=["Oil", "Gold", "BTC"]
                    )
                
                with gr.Row():
                    commodity_update_btn = gr.Button("Update Commodities")
                
                commodity_plot = gr.Plot(label="Commodity Prices")
                
                def update_commodities(start_date_option, end_date_option, commodities):
                    start_date = convert_date_option(start_date_option)
                    end_date = convert_date_option(end_date_option)
                    
                    commodity_data = get_commodity_data(commodities, start_date, end_date)
                    return plot_commodities(commodity_data)
                
                commodity_update_btn.click(
                    fn=update_commodities,
                    inputs=[commodity_start_date, commodity_end_date, commodity_selection],
                    outputs=commodity_plot
                )
        
        # Load initial data for each tab when dashboard starts
        dashboard.load(
            fn=update_interest_rates,
            inputs=[interest_start_date, interest_end_date, interest_currency_selection],
            outputs=interest_plot
        )
        
        dashboard.load(
            fn=update_gdp_comparison,
            inputs=[gdp_country_selection, start_year_slider, end_year_slider],
            outputs=gdp_plot
        )
        
        dashboard.load(
            fn=update_commodities,
            inputs=[commodity_start_date, commodity_end_date, commodity_selection],
            outputs=commodity_plot
        )
    
    return dashboard

# Launch the dashboard
if __name__ == "__main__":
    dashboard = create_dashboard()
    dashboard.launch()