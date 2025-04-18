import gradio as gr
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
import warnings
import dotenv
from data import get_interest_rates, get_gdp_data, get_inflation_data, get_cpi_data, get_commodity_data, get_commodity_data_with_volume, get_trading_advice
warnings.filterwarnings('ignore')
import pandas as pd
import yfinance as yf
from plotly.subplots import make_subplots
from openai import OpenAI
import os

dotenv.load_dotenv()

SVG_LOGO = """
<svg id='dashboard-logo-svg' width='48' height='48' viewBox='0 0 48 48' fill='none' xmlns='http://www.w3.org/2000/svg'>
  <circle cx='24' cy='24' r='22' stroke='white' stroke-width='2'/>
  <ellipse cx='24' cy='24' rx='22' ry='8' stroke='white' stroke-width='1.5'/>
  <ellipse cx='24' cy='24' rx='22' ry='16' stroke='white' stroke-width='1'/>
  <polyline points='10,30 18,22 24,26 32,14 38,18' fill='none' stroke='white' stroke-width='3' stroke-linecap='round' stroke-linejoin='round'/>
  <circle cx='10' cy='30' r='2.4' fill='white'/>
  <circle cx='18' cy='22' r='2.4' fill='white'/>
  <circle cx='24' cy='26' r='2.4' fill='white'/>
  <circle cx='32' cy='14' r='2.4' fill='white'/>
  <circle cx='38' cy='18' r='2.4' fill='white'/>
</svg>
"""

# Plotting functions
def plot_interest_rates(interest_data, currencies_to_display):
    """
    Plot interest rates for the selected currencies
    """
    if interest_data is None or interest_data.empty:
        fig = go.Figure()
        fig.update_layout(
            title="No Interest Rate Data Available",
            template="plotly_dark",
            paper_bgcolor="black",
            plot_bgcolor="black",
            font=dict(color="white"),
            autosize=True,
            height=500,  # Set a consistent height
            margin=dict(l=50, r=50, t=80, b=50)  # Adjust margins for better use of space
        )
        return fig
    
    # Debug information
    print(f"Plotting interest rates with shape: {interest_data.shape}")
    print(f"Columns: {interest_data.columns.tolist()}")
    
    # Thin out the US data if it has more points than other currencies
    if 'USD' in interest_data.columns:
        # Count non-NaN values for each currency
        data_counts = {currency: interest_data[currency].dropna().shape[0] 
                      for currency in interest_data.columns}
        print(f"Data points per currency: {data_counts}")
        
        # Find the median number of points (excluding USD)
        non_usd_counts = [count for curr, count in data_counts.items() if curr != 'USD']
        if non_usd_counts:
            target_count = int(sum(non_usd_counts) / len(non_usd_counts))
            
            # If USD has significantly more points, thin it out
            if 'USD' in data_counts and data_counts['USD'] > target_count * 1.5:
                print(f"Thinning USD data from {data_counts['USD']} to ~{target_count} points")
                
                # Get USD data without NaN values
                usd_data = interest_data['USD'].dropna()
                
                # Calculate the step size to get approximately target_count points
                step = max(1, len(usd_data) // target_count)
                
                # Create a mask for the rows to keep
                mask = pd.Series(False, index=interest_data.index)
                mask[usd_data.index[::step]] = True
                
                # Apply the mask to USD data
                interest_data.loc[~mask, 'USD'] = None
                print(f"USD data thinned to {interest_data['USD'].dropna().shape[0]} points")
    
    fig = go.Figure()
    
    # Use different colors for each currency for better visibility
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan']
    
    for i, currency in enumerate(currencies_to_display):
        # Get color index with wraparound if we have more currencies than colors
        color_idx = i % len(colors)
        
        # Make sure the currency is in the data and not all NaN
        if currency in interest_data.columns and interest_data[currency].notna().any():
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
            print(f"No data for {currency}, skipping")
    
    fig.update_layout(
        title="Interest Rates by Currency",
        xaxis_title="Date",
        yaxis_title="Interest Rate (%)",
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor="black",
        plot_bgcolor="black",
        font=dict(color="white"),
        autosize=True,
        height=500,  # Set a consistent height
        margin=dict(l=50, r=50, t=80, b=50)  # Adjust margins for better use of space
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
    Plot inflation comparison across countries as a line graph
    
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
            autosize=True,
            height=500,  # Set a consistent height
            margin=dict(l=50, r=50, t=80, b=50)  # Adjust margins for better use of space
        )
        return fig
    
    # Create a line chart for inflation data
    fig = go.Figure()
    
    # Get years as columns (excluding 'Country')
    years = [col for col in inflation_data.columns if col != 'Country']
    
    # Use different colors for each country
    colors = ['red', 'blue', 'green', 'purple', 'orange']
    
    # Add a trace for each country
    for i, country in enumerate(inflation_data['Country'].unique()):
        color_idx = i % len(colors)
        
        country_data = inflation_data[inflation_data['Country'] == country]
        
        # Extract values for this country across all years
        x_values = []
        y_values = []
        
        for year in years:
            if year in country_data.columns:
                value = country_data[year].values[0]
                if pd.notna(value):  # Only include non-NaN values
                    x_values.append(int(year))
                    y_values.append(value)
        
        # Sort by year (x-axis)
        sorted_indices = sorted(range(len(x_values)), key=lambda k: x_values[k])
        x_values = [x_values[i] for i in sorted_indices]
        y_values = [y_values[i] for i in sorted_indices]
        
        fig.add_trace(go.Scatter(
            x=x_values,
            y=y_values,
            mode='lines+markers',
            name=country,
            line=dict(color=colors[color_idx], width=2)
        ))
    
    fig.update_layout(
        title=f"Inflation Rates by Country ({start_year}-{end_year})",
        xaxis_title="Year",
        yaxis_title="Inflation Rate (%)",
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor="black",
        plot_bgcolor="black",
        font=dict(color="white"),
        autosize=True,
        height=500,  # Set a consistent height
        margin=dict(l=50, r=50, t=80, b=50)  # Adjust margins for better use of space
    )
    
    # Enable autoscaling for both axes
    fig.update_xaxes(autorange=True)
    fig.update_yaxes(autorange=True)
    
    return fig

def plot_commodities(commodity_data):
    """
    Plot commodity prices
    """
    if commodity_data is None or commodity_data.empty:
        fig = go.Figure()
        fig.update_layout(
            title="No Commodity Data Available",
            template="plotly_dark",
            paper_bgcolor="black",
            plot_bgcolor="black",
            font=dict(color="white"),
            autosize=True,
            height=500,  # Set a consistent height
            margin=dict(l=50, r=50, t=80, b=50)  # Adjust margins for better use of space
        )
        return fig
    
    fig = go.Figure()
    
    # Use different colors for each commodity
    colors = {
        'Oil': 'red',
        'Gold': 'gold',
        'Silver': 'silver',
        'BTC': 'orange',
        'ETH': 'blue'
    }
    
    # Normalize all prices to start at 100 for easier comparison
    normalized_data = pd.DataFrame()
    
    for commodity in commodity_data.columns:
        if not commodity_data[commodity].empty and commodity_data[commodity].notna().any():
            # Get the first valid value
            first_valid = commodity_data[commodity].dropna().iloc[0]
            if first_valid != 0:  # Avoid division by zero
                # Normalize to start at 100
                normalized_data[commodity] = (commodity_data[commodity] / first_valid) * 100
    
    # Plot each commodity
    for commodity in normalized_data.columns:
        color = colors.get(commodity, 'gray')  # Default to gray if commodity not in colors dict
        
        fig.add_trace(go.Scatter(
            x=normalized_data.index,
            y=normalized_data[commodity],
            mode='lines',
            name=commodity,
            line=dict(color=color, width=2)
        ))
    
    fig.update_layout(
        title="Commodity Price Performance (Normalized to 100)",
        xaxis_title="Date",
        yaxis_title="Normalized Price",
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor="black",
        plot_bgcolor="black",
        font=dict(color="white"),
        autosize=True,
        height=500,  # Set a consistent height
        margin=dict(l=50, r=50, t=80, b=50)  # Adjust margins for better use of space
    )
    
    # Enable autoscaling for both axes
    fig.update_xaxes(autorange=True)
    fig.update_yaxes(autorange=True)
    
    return fig

# Helper function to convert date options to actual dates
def convert_date_option(option: str) -> str:
    """Convert preset strings (e.g. '1 Month Ago') to ISO‑8601 dates."""
    offsets = {
        'Today': 0,
        'Yesterday': 1,
        '1 Week Ago': 7,
        '1 Month Ago': 30,
        '3 Months Ago': 90,
        '6 Months Ago': 180,
        '1 Year Ago': 365,
        '5 Years Ago': 5 * 365,
        '10 Years Ago': 10 * 365,
        '15 Years Ago': 15 * 365,
        '20 Years Ago': 20 * 365,
    }
    days = offsets[option]
    return (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

def update_interest_rates(start_date_option, end_date_option, currencies_to_display):
    start_date = convert_date_option(start_date_option)
    end_date = convert_date_option(end_date_option)
    
    print(f"Retrieving interest rates from {start_date} to {end_date} for {currencies_to_display}")
    
    # Get data only for the selected currencies
    interest_data = get_interest_rates(currencies_to_display, start_date, end_date)
    
    print(f"Retrieved interest rate data with shape: {interest_data.shape if interest_data is not None else 'None'}")
    
    return plot_interest_rates(interest_data, currencies_to_display)

def plot_cpi_comparison(cpi_data):
    """
    Plot CPI comparison across countries
    """
    if cpi_data is None or cpi_data.empty:
        fig = go.Figure()
        fig.update_layout(
            title="No CPI Data Available",
            template="plotly_dark",
            paper_bgcolor="black",
            plot_bgcolor="black",
            font=dict(color="white"),
            autosize=True,
            height=500,  # Set a consistent height
            margin=dict(l=50, r=50, t=80, b=50)  # Adjust margins for better use of space
        )
        return fig
    
    fig = go.Figure()
    
    # Use different colors for each country for better visibility
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan']
    
    for i, country in enumerate(cpi_data.columns):
        # Get color index with wraparound if we have more countries than colors
        color_idx = i % len(colors)
        
        # Make sure the data for this country is not all NaN
        if cpi_data[country].notna().any():
            # Create a copy without NaN values for this trace
            plot_data = cpi_data[country].dropna()
            
            fig.add_trace(go.Scatter(
                x=plot_data.index,
                y=plot_data.values,
                mode='lines',
                name=f"{country} CPI",
                line=dict(color=colors[color_idx], width=2)
            ))
            print(f"Added CPI trace for {country} with {len(plot_data)} points")
        else:
            print(f"No non-NaN CPI data for {country}, skipping")
    
    fig.update_layout(
        title="Consumer Price Index by Country",
        xaxis_title="Date",
        yaxis_title="CPI Value",
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor="black",
        plot_bgcolor="black",
        font=dict(color="white"),
        autosize=True,
        height=500,  # Set a consistent height
        margin=dict(l=50, r=50, t=80, b=50)  # Adjust margins for better use of space
    )
    
    # Enable autoscaling for both axes
    fig.update_xaxes(autorange=True)
    fig.update_yaxes(autorange=True)
    
    return fig

def plot_commodity_detail(commodity_name, commodity_df):
    """
    Create a detailed plot for a single commodity with price and volume
    """
    if commodity_df is None:
        print(f"No data available for {commodity_name}")
        fig = go.Figure()
        fig.update_layout(
            title=f"No Data Available for {commodity_name}",
            template="plotly_dark",
            paper_bgcolor="black",
            plot_bgcolor="black",
            font=dict(color="white"),
            autosize=True,
            height=500,
            margin=dict(l=50, r=50, t=80, b=50)
        )
        return fig
    
    print(f"Creating detail plot for {commodity_name} with {len(commodity_df)} data points")
    print(f"Columns available: {commodity_df.columns.tolist()}")
    
    # Handle MultiIndex columns if present
    if isinstance(commodity_df.columns, pd.MultiIndex):
        # Check if Volume is in the second level of the MultiIndex
        has_volume = any('Volume' in col for col in commodity_df.columns)
        
        # Flatten MultiIndex columns for easier access
        commodity_df.columns = [col[0] if isinstance(col, tuple) else col for col in commodity_df.columns]
    else:
        # Standard check for Volume column
        has_volume = 'Volume' in commodity_df.columns
    
    # Make sure Volume has data if the column exists
    if has_volume and 'Volume' in commodity_df.columns:
        has_volume = commodity_df['Volume'].sum() > 0
    
    if has_volume:
        # Create figure with secondary y-axis for volume
        fig = make_subplots(rows=2, cols=1, 
                            shared_xaxes=True, 
                            vertical_spacing=0.05, 
                            row_heights=[0.7, 0.3])
    else:
        # Create a single plot for price only
        fig = go.Figure()
    
    # Colors based on commodity
    colors = {
        'Oil': 'red',
        'Gold': 'gold',
        'Silver': 'silver',
        'BTC': 'orange',
        'ETH': 'blue'
    }
    
    color = colors.get(commodity_name, 'white')
    
    # Make sure we have the Close column
    if 'Close' not in commodity_df.columns:
        # Try to find a column that might contain close prices
        close_candidates = [col for col in commodity_df.columns if 'close' in str(col).lower()]
        if close_candidates:
            close_column = close_candidates[0]
        else:
            # If no close column found, use the first numeric column
            numeric_cols = commodity_df.select_dtypes(include=['number']).columns
            close_column = numeric_cols[0] if len(numeric_cols) > 0 else None
            
        if close_column is None:
            print(f"No suitable price data found for {commodity_name}")
            fig = go.Figure()
            fig.update_layout(
                title=f"No Price Data Available for {commodity_name}",
                template="plotly_dark",
                paper_bgcolor="black",
                plot_bgcolor="black",
                font=dict(color="white"),
                autosize=True,
                height=500,
                margin=dict(l=50, r=50, t=80, b=50)
            )
            return fig
    else:
        close_column = 'Close'
    
    if has_volume:
        # Add price trace to subplot
        fig.add_trace(
            go.Scatter(
                x=commodity_df.index,
                y=commodity_df[close_column],
                mode='lines',
                name=f'{commodity_name} Price',
                line=dict(color=color, width=2)
            ),
            row=1, col=1
        )
        
        # Add volume trace as bar chart
        fig.add_trace(
            go.Bar(
                x=commodity_df.index,
                y=commodity_df['Volume'],
                name='Volume',
                marker=dict(color='rgba(100, 100, 100, 0.5)')
            ),
            row=2, col=1
        )
        
        # Update y-axes labels
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        
        # Update x-axis
        fig.update_xaxes(title_text="Date", row=2, col=1)
    else:
        # Add price trace to main figure
        fig.add_trace(
            go.Scatter(
                x=commodity_df.index,
                y=commodity_df[close_column],
                mode='lines',
                name=f'{commodity_name} Price',
                line=dict(color=color, width=2)
            )
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Date")
        fig.update_yaxes(title_text="Price")
    
    # Update layout
    fig.update_layout(
        title=f"{commodity_name} Price" + (" and Volume" if has_volume else ""),
        template="plotly_dark",
        paper_bgcolor="black",
        plot_bgcolor="black",
        font=dict(color="white"),
        autosize=True,
        height=600,
        margin=dict(l=50, r=50, t=80, b=50),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # Enable autoscaling for all axes
    fig.update_xaxes(autorange=True)
    fig.update_yaxes(autorange=True)
    
    return fig

# Main Gradio interface
def create_dashboard():
    with gr.Blocks(theme='dark') as dashboard:

        # Add the updated header styling
        gr.HTML("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
            body, .gradio-container, .dark {background:#000!important;color:#fff;font-family:'Inter',sans-serif;}
            #dashboard-header {
            display: flex;
            align-items: center;
            border-bottom: 1px solid #333;
            padding-bottom: 8px;
            margin-bottom: 12px;
            }
            #dashboard-logo-container {
            display: flex;
            align-items: center;
            flex-shrink: 0;
            margin-right: auto;
            }
            #dashboard-logo-svg {
            height: 36px;
            width: 36px;
            display: inline-block;
            margin-right: 8px;
            flex-shrink: 0;
            }
            #dashboard-title {
            font-size: 1.5rem!important;
            font-weight: 700;
            margin: 0;
            white-space: nowrap;
            display: inline-block;
            }
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

        # ---- Header row with fixed structure ----
        with gr.Row(elem_id='dashboard-header'):
            with gr.Column(scale=1, min_width=0):
                with gr.Row():
                    # Wrap logo and title in a container to ensure they stay together
                    gr.HTML(f"""
                    <div id="dashboard-logo-container">
                        {SVG_LOGO}
                        <h2 id="dashboard-title">International Finance</h2>
                    </div>
                    """)
            
            # Empty column to push content to the left
            with gr.Column(scale=3, min_width=0):
                gr.HTML("")

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
                
                # Main comparison chart
                commodity_plot = gr.Plot(label="Commodity Price Comparison")
                
                # Create a tabbed interface for detailed charts
                with gr.Tabs(elem_id="commodity_detail_tabs") as commodity_detail_tabs:
                    with gr.Tab("Oil") as oil_tab:
                        oil_detail_plot = gr.Plot(label="Oil Detail")
                    
                    with gr.Tab("Gold") as gold_tab:
                        gold_detail_plot = gr.Plot(label="Gold Detail")
                    
                    with gr.Tab("Silver") as silver_tab:
                        silver_detail_plot = gr.Plot(label="Silver Detail")
                    
                    with gr.Tab("BTC") as btc_tab:
                        btc_detail_plot = gr.Plot(label="Bitcoin Detail")
                    
                    with gr.Tab("ETH") as eth_tab:
                        eth_detail_plot = gr.Plot(label="Ethereum Detail")
                
                def update_commodities(start_date_option, end_date_option, commodities):
                    start_date = convert_date_option(start_date_option)
                    end_date = convert_date_option(end_date_option)
                    
                    print(f"Updating commodities from {start_date} to {end_date} for {commodities}")
                    
                    # Get data for comparison chart
                    commodity_data = get_commodity_data(commodities, start_date, end_date)
                    comparison_chart = plot_commodities(commodity_data)
                    
                    # Get detailed data including volume
                    detailed_data = get_commodity_data_with_volume(commodities, start_date, end_date)
                    
                    # Create detailed charts for each commodity
                    oil_chart = plot_commodity_detail("Oil", detailed_data.get("Oil")) if "Oil" in detailed_data else plot_commodity_detail("Oil", None)
                    gold_chart = plot_commodity_detail("Gold", detailed_data.get("Gold")) if "Gold" in detailed_data else plot_commodity_detail("Gold", None)
                    silver_chart = plot_commodity_detail("Silver", detailed_data.get("Silver")) if "Silver" in detailed_data else plot_commodity_detail("Silver", None)
                    btc_chart = plot_commodity_detail("BTC", detailed_data.get("BTC")) if "BTC" in detailed_data else plot_commodity_detail("BTC", None)
                    eth_chart = plot_commodity_detail("ETH", detailed_data.get("ETH")) if "ETH" in detailed_data else plot_commodity_detail("ETH", None)
                    
                    return [comparison_chart, oil_chart, gold_chart, silver_chart, btc_chart, eth_chart]
                
                commodity_update_btn.click(
                    fn=update_commodities,
                    inputs=[commodity_start_date, commodity_end_date, commodity_selection],
                    outputs=[commodity_plot, oil_detail_plot, gold_detail_plot, silver_detail_plot, btc_detail_plot, eth_detail_plot]
                )
            
            with gr.TabItem("CPI Comparison"):
                with gr.Row():
                    cpi_country_selection = gr.CheckboxGroup(
                        label="Select Countries for CPI Comparison",
                        choices=["US", "China", "UK", "Canada", "Australia"],
                        value=["US", "China", "UK"]
                    )
                
                with gr.Row():
                    with gr.Column(scale=1):
                        cpi_start_date = gr.Dropdown(
                            label="Start Date",
                            choices=["5 Years Ago", "10 Years Ago", "15 Years Ago", "20 Years Ago"],
                            value="10 Years Ago"
                        )
                    
                    with gr.Column(scale=1):
                        cpi_end_date = gr.Dropdown(
                            label="End Date",
                            choices=["Today", "1 Year Ago", "5 Years Ago"],
                            value="Today"
                        )
                
                with gr.Row():
                    cpi_update_btn = gr.Button("Update CPI Comparison")
                
                cpi_plot = gr.Plot(label="CPI by Country Over Time")
                
                def update_cpi_comparison(countries, start_date_option, end_date_option):
                    start_date = convert_date_option(start_date_option)
                    end_date = convert_date_option(end_date_option)
                    
                    cpi_data = get_cpi_data(countries, start_date, end_date)
                    return plot_cpi_comparison(cpi_data)
                
                cpi_update_btn.click(
                    fn=update_cpi_comparison,
                    inputs=[cpi_country_selection, cpi_start_date, cpi_end_date],
                    outputs=cpi_plot
                )
                
            # Trading Consultant Tab
            with gr.TabItem("Trading Consultant"):
                with gr.Row():
                    with gr.Column(scale=1):
                        data_sources = gr.CheckboxGroup(
                            label="Select Data Sources to Consider",
                            choices=[
                                "Interest Rates", 
                                "GDP Trends", 
                                "Inflation Rates", 
                                "Commodity Prices", 
                                "Consumer Price Index"
                            ],
                            value=["Interest Rates", "Inflation Rates", "Commodity Prices"]
                        )
                
                with gr.Row():
                    user_query = gr.Textbox(
                        label="Ask about trading strategies or market insights",
                        placeholder="Example: What trading strategy would you recommend given the current interest rate trends?",
                        lines=3
                    )
                
                with gr.Row():
                    submit_btn = gr.Button("Get Trading Advice")
                
                with gr.Row():
                    advice_output = gr.Markdown(
                        label="Trading Advice",
                        value="*Submit a query to get personalized trading advice based on financial data.*"
                    )
                
                # Add some example queries for users to click
                with gr.Row():
                    gr.Examples(
                        examples=[
                            "What trading strategy would you recommend given the current interest rate trends?",
                            "How should I adjust my portfolio considering the inflation data for the US and China?",
                            "What commodities would be good hedges against current market conditions?",
                            "How might the GDP trends affect currency trading in the next quarter?",
                            "What's a good risk management approach for trading in the current economic climate?"
                        ],
                        inputs=user_query
                    )
                
                submit_btn.click(
                    fn=get_trading_advice,
                    inputs=[user_query, data_sources],
                    outputs=advice_output
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
            fn=update_inflation_comparison,
            inputs=[inflation_country_selection, start_year_slider, end_year_slider],
            outputs=inflation_plot
        )
        
        dashboard.load(
            fn=update_commodities,
            inputs=[commodity_start_date, commodity_end_date, commodity_selection],
            outputs=[commodity_plot, oil_detail_plot, gold_detail_plot, silver_detail_plot, btc_detail_plot, eth_detail_plot]
        )
        dashboard.load(
            fn=update_cpi_comparison,
            inputs=[cpi_country_selection, cpi_start_date, cpi_end_date],
            outputs=cpi_plot
        )
    
        dashboard.load(
            fn=get_trading_advice,
            inputs=[user_query, data_sources],
            outputs=advice_output
        )
    
    return dashboard

# Launch the dashboard
if __name__ == "__main__":
    dashboard = create_dashboard()
    dashboard.launch()