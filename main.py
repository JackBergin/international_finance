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

def get_gdp_data(countries, end_date):
    """
    Get GDP data for selected countries
    Using latest annual data (simplified)
    """
    # Simplified mock GDP data
    gdp_mock = {
        'US': 25.46,
        'China': 17.96,
        'UK': 3.07,
        'Canada': 2.14,
        'Australia': 1.75
    }
    
    selected_data = {country: gdp_mock.get(country, 0) for country in countries if country in gdp_mock}
    
    if not selected_data:
        return None
        
    df = pd.DataFrame({
        'Country': list(selected_data.keys()),
        'GDP (Trillion USD)': list(selected_data.values())
    })
    
    return df

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

def plot_gdp_comparison(gdp_data, dark_mode=False):
    """Plot GDP comparison as a bar chart"""
    if gdp_data is None or gdp_data.empty:
        fig = go.Figure()
        fig.update_layout(
            title="No GDP Data Available",
            template="plotly_dark" if dark_mode else "plotly_white"
        )
        return fig
        
    fig = px.bar(
        gdp_data,
        x='Country',
        y='GDP (Trillion USD)',
        color='Country',
        title="GDP Comparison by Country"
    )
    
    fig.update_layout(
        template="plotly_dark" if dark_mode else "plotly_white",
        xaxis_title="Country",
        yaxis_title="GDP (Trillion USD)"
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
            
            # GDP Comparison Tab
            with gr.TabItem("GDP Comparison"):
                with gr.Row():
                    gdp_country_selection = gr.CheckboxGroup(
                        label="Select Countries for GDP Comparison",
                        choices=["US", "China", "UK", "Canada", "Australia"],
                        value=["US", "China", "UK"]
                    )
                
                with gr.Row():
                    gdp_update_btn = gr.Button("Update GDP Comparison")
                
                gdp_plot = gr.Plot(label="GDP by Country")
                
                def update_gdp_comparison(countries, dark_mode):
                    gdp_data = get_gdp_data(countries, datetime.now().strftime('%Y-%m-%d'))
                    return plot_gdp_comparison(gdp_data, dark_mode)
                
                gdp_update_btn.click(
                    fn=update_gdp_comparison,
                    inputs=[gdp_country_selection, theme_toggle],
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