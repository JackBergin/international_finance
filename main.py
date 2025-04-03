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

warnings.filterwarnings('ignore')

# Helper functions for data retrieval
def get_interest_rates(currencies, start_date, end_date):
    """
    Get approximate interest rates for selected currencies.
    Note: This is simplified as real-time interest rate data requires specialized data providers.
    """
    # Mapping of currencies to their interest rate proxies (bonds or rates)
    proxies = {
        'USD': '^TNX',  # 10-Year Treasury Yield
        'EUR': 'EURIBOR3M.SW',  # Euro Interbank Offered Rate
        'GBP': 'GB10Y.L',  # UK 10Y Gilt
        'CNY': 'CHIBON.SS',  # China 1Y bond
        'AUD': 'GSBG10.AX',  # Australia 10Y Bond
        'CAD': 'CA10YT.TO',  # Canada 10Y Bond
    }
    
    interest_rates = pd.DataFrame()
    
    for currency in currencies:
        if currency in proxies:
            try:
                data = yf.download(proxies[currency], start=start_date, end=end_date)
                if not data.empty:
                    interest_rates[currency] = data['Close']
            except Exception as e:
                print(f"Error getting data for {currency}: {e}")
    
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

def get_inflation_data(countries, start_date, end_date):
    """
    Get inflation proxy data for selected countries
    """
    # Mapping of countries to inflation proxies (ETFs or indices)
    proxies = {
        'US': 'RINF',  # ProShares Inflation Expectations ETF
        'UK': 'UKRPI.L',  # UK Retail Price Index
        'China': '000922.SS',  # China CSI 300 Non-Ferrous Metal Index
        'Canada': 'CACPI.TO',  # Canada CPI Index
        'Australia': 'AUDUSD=X'  # Using currency as a very rough proxy
    }
    
    inflation_data = pd.DataFrame()
    
    for country in countries:
        if country in proxies:
            try:
                data = yf.download(proxies[country], start=start_date, end=end_date)
                if not data.empty:
                    inflation_data[f"{country} Inflation Proxy"] = data['Close']
            except Exception as e:
                print(f"Error getting inflation data for {country}: {e}")
    
    return inflation_data

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
def plot_interest_rates(interest_data, dark_mode=False):
    """Plot interest rates for selected currencies"""
    if interest_data is None or interest_data.empty:
        fig = go.Figure()
        fig.update_layout(
            title="No Interest Rate Data Available",
            template="plotly_dark" if dark_mode else "plotly_white"
        )
        return fig
        
    fig = go.Figure()
    
    for currency in interest_data.columns:
        fig.add_trace(go.Scatter(
            x=interest_data.index,
            y=interest_data[currency],
            mode='lines',
            name=f"{currency} Interest Rate"
        ))
    
    fig.update_layout(
        title="Interest Rates by Currency",
        xaxis_title="Date",
        yaxis_title="Rate (%)",
        template="plotly_dark" if dark_mode else "plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

def plot_gdp_comparison(gdp_data, start_year, end_year, dark_mode=False):
    """
    Plot GDP comparison as a step function across years
    
    Parameters:
    gdp_data (pandas.DataFrame): DataFrame with GDP data
    start_year (int): Start year for x-axis
    end_year (int): End year for x-axis
    dark_mode (bool): Whether to use dark mode for the plot
    
    Returns:
    plotly.graph_objects.Figure: The plotted figure
    """
    if gdp_data is None or gdp_data.empty:
        fig = go.Figure()
        fig.update_layout(
            title="No GDP Data Available",
            template="plotly_dark" if dark_mode else "plotly_white"
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
        template="plotly_dark" if dark_mode else "plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

def plot_inflation_cpi(inflation_data, dark_mode=False):
    """Plot inflation and CPI trends"""
    if inflation_data is None or inflation_data.empty:
        fig = go.Figure()
        fig.update_layout(
            title="No Inflation Data Available",
            template="plotly_dark" if dark_mode else "plotly_white"
        )
        return fig
        
    fig = go.Figure()
    
    for column in inflation_data.columns:
        fig.add_trace(go.Scatter(
            x=inflation_data.index,
            y=inflation_data[column],
            mode='lines',
            name=column
        ))
    
    fig.update_layout(
        title="Inflation Proxy Indicators by Country",
        xaxis_title="Date",
        yaxis_title="Value",
        template="plotly_dark" if dark_mode else "plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

def plot_commodities(commodity_data, dark_mode=False):
    """Plot commodity prices"""
    if commodity_data is None or commodity_data.empty:
        fig = go.Figure()
        fig.update_layout(
            title="No Commodity Data Available",
            template="plotly_dark" if dark_mode else "plotly_white"
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
        template="plotly_dark" if dark_mode else "plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        showlegend=True
    )
    
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
        "3 Years Ago": (datetime.now() - timedelta(days=3*365)).strftime('%Y-%m-%d')
    }
    return date_options[date_option]

# Main Gradio interface
def create_dashboard():
    with gr.Blocks(theme=gr.themes.Default()) as dashboard:
        gr.Markdown("# Financial Markets Dashboard")
        
        # Global theme toggle
        with gr.Row():
            theme_toggle = gr.Checkbox(label="Dark Mode", value=False)
        
        # Create tabs for different visualizations
        with gr.Tabs() as tabs:
            # Interest Rates Tab
            with gr.TabItem("Interest Rates"):
                with gr.Row():
                    with gr.Column(scale=1):
                        interest_start_date = gr.Dropdown(
                            label="Start Date",
                            choices=["1 Week Ago", "1 Month Ago", "3 Months Ago", "6 Months Ago", "1 Year Ago", "3 Years Ago"],
                            value="1 Month Ago"
                        )
                    
                    with gr.Column(scale=1):
                        interest_end_date = gr.Dropdown(
                            label="End Date",
                            choices=["Today", "Yesterday", "1 Week Ago"],
                            value="Today"
                        )
                
                with gr.Row():
                    interest_currency_selection = gr.CheckboxGroup(
                        label="Select Currencies for Interest Rates",
                        choices=["USD", "EUR", "GBP", "CNY", "AUD", "CAD"],
                        value=["USD", "EUR", "GBP"]
                    )
                
                with gr.Row():
                    interest_update_btn = gr.Button("Update Interest Rates")
                
                interest_plot = gr.Plot(label="Interest Rates")
                
                def update_interest_rates(start_date_option, end_date_option, currencies, dark_mode):
                    start_date = convert_date_option(start_date_option)
                    end_date = convert_date_option(end_date_option)
                    
                    interest_data = get_interest_rates(currencies, start_date, end_date)
                    return plot_interest_rates(interest_data, dark_mode)
                
                interest_update_btn.click(
                    fn=update_interest_rates,
                    inputs=[interest_start_date, interest_end_date, interest_currency_selection, theme_toggle],
                    outputs=interest_plot
                )
            
            # GDP Comparison Tab - MODIFIED
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
                
                def update_gdp_comparison(countries, start_year, end_year, dark_mode):
                    # Ensure start_year is not greater than end_year
                    if start_year > end_year:
                        start_year, end_year = end_year, start_year
                    
                    gdp_data = get_gdp_data(countries, start_year, end_year)
                    return plot_gdp_comparison(gdp_data, start_year, end_year, dark_mode)
                
                gdp_update_btn.click(
                    fn=update_gdp_comparison,
                    inputs=[gdp_country_selection, start_year_slider, end_year_slider, theme_toggle],
                    outputs=gdp_plot
                )
            
            # Inflation Indicators Tab
            with gr.TabItem("Inflation Indicators"):
                with gr.Row():
                    with gr.Column(scale=1):
                        inflation_start_date = gr.Dropdown(
                            label="Start Date",
                            choices=["1 Month Ago", "3 Months Ago", "6 Months Ago", "1 Year Ago", "3 Years Ago"],
                            value="6 Months Ago"
                        )
                    
                    with gr.Column(scale=1):
                        inflation_end_date = gr.Dropdown(
                            label="End Date",
                            choices=["Today", "Yesterday", "1 Week Ago"],
                            value="Today"
                        )
                
                with gr.Row():
                    inflation_country_selection = gr.CheckboxGroup(
                        label="Select Countries for Inflation Indicators",
                        choices=["US", "China", "UK", "Canada", "Australia"],
                        value=["US", "UK"]
                    )
                
                with gr.Row():
                    inflation_update_btn = gr.Button("Update Inflation Indicators")
                
                inflation_plot = gr.Plot(label="Inflation Indicators")
                
                def update_inflation_indicators(start_date_option, end_date_option, countries, dark_mode):
                    start_date = convert_date_option(start_date_option)
                    end_date = convert_date_option(end_date_option)
                    
                    inflation_data = get_inflation_data(countries, start_date, end_date)
                    return plot_inflation_cpi(inflation_data, dark_mode)
                
                inflation_update_btn.click(
                    fn=update_inflation_indicators,
                    inputs=[inflation_start_date, inflation_end_date, inflation_country_selection, theme_toggle],
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
                
                def update_commodities(start_date_option, end_date_option, commodities, dark_mode):
                    start_date = convert_date_option(start_date_option)
                    end_date = convert_date_option(end_date_option)
                    
                    commodity_data = get_commodity_data(commodities, start_date, end_date)
                    return plot_commodities(commodity_data, dark_mode)
                
                commodity_update_btn.click(
                    fn=update_commodities,
                    inputs=[commodity_start_date, commodity_end_date, commodity_selection, theme_toggle],
                    outputs=commodity_plot
                )
        
        # Update theme for all plots when dark mode is toggled
        def update_all_plots_theme(dark_mode):
            # Trigger all update functions with current values
            interest_fig = update_interest_rates(
                interest_start_date.value, 
                interest_end_date.value, 
                interest_currency_selection.value, 
                dark_mode
            )
            
            gdp_fig = update_gdp_comparison(
                gdp_country_selection.value, 
                start_year_slider.value,
                end_year_slider.value,
                dark_mode
            )
            
            inflation_fig = update_inflation_indicators(
                inflation_start_date.value, 
                inflation_end_date.value, 
                inflation_country_selection.value, 
                dark_mode
            )
            
            commodity_fig = update_commodities(
                commodity_start_date.value, 
                commodity_end_date.value, 
                commodity_selection.value, 
                dark_mode
            )
            
            return interest_fig, gdp_fig, inflation_fig, commodity_fig
        
        theme_toggle.change(
            fn=update_all_plots_theme,
            inputs=[theme_toggle],
            outputs=[interest_plot, gdp_plot, inflation_plot, commodity_plot]
        )
    
    return dashboard

# Launch the dashboard
if __name__ == "__main__":
    dashboard = create_dashboard()
    dashboard.launch()