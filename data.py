import pandas as pd
import yfinance as yf
import warnings
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
        #'EUR': 'IRLTLT01EZM156N',     # 10-Year Govt Bond Yields: Euro Area
        'EUR': 'IRLTLT01DEM156N',     # 10-Year Govt Bond Yields: Germany (instead of Euro Area)
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

def get_commodity_data_with_volume(commodities, start_date, end_date):
    """
    Get price and volume data for selected commodities

    Args:
        commodities (list): List of commodity names to retrieve data for
        start_date (str): Start date for the data range in YYYY-MM-DD format
        end_date (str): End date for the data range in YYYY-MM-DD format

    Returns:
        commodity_data_dict (dict): Dictionary of DataFrames for each commodity, with price and volume data
    """
    # Tickers for commodities
    tickers = {
        'Oil': 'CL=F',  # Crude Oil Futures
        'Gold': 'GC=F',  # Gold Futures
        'Silver': 'SI=F',  # Silver Futures
        'BTC': 'BTC-USD',  # Bitcoin
        'ETH': 'ETH-USD'   # Ethereum
    }
    
    # Dictionary to store DataFrames for each commodity
    commodity_data_dict = {}
    
    for commodity in commodities:
        if commodity in tickers:
            try:
                # Get full data including volume
                print(f"Downloading detailed data for {commodity} from {start_date} to {end_date}")
                data = yf.download(tickers[commodity], start=start_date, end=end_date)
                
                if not data.empty:
                    print(f"Retrieved {len(data)} data points for {commodity}")
                    commodity_data_dict[commodity] = data
                else:
                    print(f"No data returned for {commodity}")
            except Exception as e:
                print(f"Error getting data for {commodity}: {e}")
    
    return commodity_data_dict

def get_cpi_data(countries, start_date, end_date):
    """
    Retrieve Consumer Price Index (CPI) data from FRED for the specified countries.
    
    Parameters:
    -----------
    countries : list
        List of country codes to filter
    start_date : str or datetime-like
        Start date for data retrieval, in YYYY-MM-DD format or a datetime object.
    end_date : str or datetime-like
        End date for data retrieval, in YYYY-MM-DD format or a datetime object.
        
    Returns:
    --------
    cpi_data : pd.DataFrame
        DataFrame indexed by date, with one column per country representing the CPI time series.
    """
    # FRED series IDs for CPI data
    cpi_series_ids = {
        'US': 'CPIAUCSL',          # Consumer Price Index for All Urban Consumers: All Items in U.S. City Average
        'UK': 'GBRCPIALLMINMEI',   # Consumer Price Index: All Items for United Kingdom
        'China': 'CHNCPIALLMINMEI', # Consumer Price Index: All Items for China
        'Canada': 'CANCPIALLMINMEI', # Consumer Price Index: All Items for Canada
        'Australia': 'AUSCPIALLMINMEI' # Consumer Price Index: All Items for Australia
    }
    
    # Initialize FRED client
    fred = Fred(api_key=os.getenv('FRED_API_KEY'))
    
    # Map country codes to full names for filtering
    country_mapping = {
        'US': 'United States',
        'UK': 'United Kingdom',
        'China': 'China',
        'Canada': 'Canada',
        'Australia': 'Australia'
    }
    
    cpi_data = pd.DataFrame()
    
    # Loop over each requested country
    for country in countries:
        series_id = cpi_series_ids.get(country)
        if not series_id:
            print(f"No FRED CPI series found for {country}, skipping.")
            continue
        
        print(f"Downloading FRED CPI data for {country} (Series ID: {series_id})")
        
        try:
            # Get CPI data from FRED
            data_series = fred.get_series(series_id,
                                         observation_start=start_date,
                                         observation_end=end_date)
            
            if data_series is None or data_series.empty:
                print(f"No CPI data returned for {country}")
                continue
            
            # Add this series to the CPI data DataFrame
            country_name = country_mapping.get(country, country)
            cpi_data[country_name] = data_series
            print(f"Successfully retrieved CPI data for {country}")
            
        except Exception as e:
            print(f"Error retrieving CPI data for {country}: {e}")
    
    return cpi_data