# International Finance Dashboard

## Overview
The International Finance Dashboard is an interactive web application that provides visualizations and analysis of key global economic indicators. Built with Gradio, this dashboard offers real-time access to financial data from multiple countries, allowing users to track and compare economic trends across different regions and time periods.

The dashboard consolidates data from various reliable sources including the Federal Reserve Economic Data (FRED), World Bank, and Yahoo Finance to present a comprehensive view of the global economic landscape.

## Value Proposition
- **Consolidated Data Access**: Brings together multiple economic indicators in one interface
- **Interactive Visualizations**: Dynamic charts that allow users to explore data trends
- **Cross-Country Comparison**: Easy comparison of economic performance across major economies
- **Flexible Time Ranges**: Ability to analyze data across different time periods
- **Real-time Updates**: Integration with live data sources for up-to-date information

## Data Sources and Methods

### Interest Rates
- Source: Federal Reserve Economic Data (FRED)
- Method: Using 10-year government bond yields as proxies for interest rates
- Implementation: `fredapi` library to fetch historical data
- Series IDs: DGS10 (USD), IRLTLT01EZM156N (EUR), etc.

### GDP Data
- Source: World Bank data (CSV file)
- Method: Reading from CSV file with annual GDP growth rates
- Implementation: `pandas` to read and process the data
- Data format: Annual percentage changes in GDP by country

### Inflation Data
- Source: World Bank data (same CSV file as GDP)
- Method: Using annual GDP growth rates as economic indicators
- Implementation: Reading from CSV and processing with `pandas`
- Data structure: Annual values converted to time series with July 1st dates

### Commodity Prices
- Source: Yahoo Finance
- Method: Fetching historical price data for commodities and cryptocurrencies
- Implementation: `yfinance` library for traditional commodities and crypto
- Tickers: CL=F (Oil), GC=F (Gold), SI=F (Silver), BTC-USD, ETH-USD

### CPI Data
- Source: World Bank data / Alternative sources when available
- Method: Reading from structured data files or APIs
- Implementation: Data processing with `pandas`

## Technical Implementation
- Frontend: Gradio for interactive dashboard
- Data Processing: Pandas for data manipulation
- Visualization: Plotly for interactive charts

## Planned Feature Additions

### CPI Data Enhancement
- Integration with dedicated CPI datasets from national statistical agencies
- More granular CPI breakdown by product categories
- Regional CPI comparisons within countries
- Historical CPI trend analysis with forecasting capabilities

### Inflation Analysis Improvements
- Addition of core inflation metrics (excluding food and energy)
- Comparison between official inflation rates and alternative measures
- Correlation analysis between inflation and other economic indicators
- Inflation expectation surveys and forward-looking indicators

### Future Roadmap
- Economic policy tracker to monitor central bank decisions
- Currency exchange rate analysis and forecasting
- Integration with additional economic indicators (unemployment, housing, etc.)
- Export functionality for data and visualizations
- Custom alert system for significant economic changes