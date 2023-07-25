import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime
from datetime import date, timedelta
import plotly.graph_objs as go

st.set_page_config(layout="wide")

# Function to get historical prices
def get_prices(tickers, start_date, end_date):
    prices = yf.download(tickers, start=start_date, end=end_date)
    return prices['Adj Close']

# Function to adjust weights
def adjust_weights(df, num_stocks, direction='top'):
    # Sort the dataframe by weights in descending order
    df = df.sort_values(by='weights', ascending=(direction=='bottom'))

    # Check if the number of stocks is more than available stocks
    if num_stocks > df.shape[0]:
        num_stocks = df.shape[0]
    
    # Select the top or bottom stocks
    df = df.iloc[:num_stocks]

    # Rescale the weights to sum to 1
    df['weights'] = df['weights'] / df['weights'].sum()

    return df



# Function to construct price series
def construct_price_series(df, prices):
    # Reset the index of df so that we have integer index instead of tickers
    df = df.reset_index()

    # Create a dictionary where key is the ticker and value is the weight
    weights_dict = pd.Series(df.weights.values, index=df.ticker).to_dict()

    # Copy the prices DataFrame to avoid modifying it in place
    weighted_prices = prices.copy()

    # Multiply each column in prices DataFrame by its corresponding weight
    for ticker in weighted_prices.columns:
        if ticker in weights_dict:
            weighted_prices[ticker] *= weights_dict[ticker]
        else:
            weighted_prices[ticker] *= 0

    # Calculate the price series by summing across columns (tickers)
    price_series = weighted_prices.sum(axis=1)

    # Normalize the price series to start at 1000
    price_series = price_series / price_series[0] * 1000

    return price_series




# Load the data
weights_df = pd.read_csv('weights.csv')

# User input for year
year = st.slider('Select Year', min_value=2018, max_value=date.today().year, value=2023)

# User inputs
col1, col2 = st.columns(2)
with col1:
    num_stocks_top = st.slider(f'Number of Top Stocks, based on weight', min_value=1, max_value=100, value=20)
with col2:
    num_stocks_bottom = st.slider('Number of Bottom Stocks, based on weight', min_value=0, max_value=99, value=20)


# Convert the year to datetime format for start and end dates
start_date = datetime(year, 1, 1)
end_date = date.today() - timedelta(days=1)
prices = get_prices(weights_df['ticker'].tolist(), start_date, end_date)

prices = prices[weights_df['ticker']]

# Adjust the weights for the top stocks
adjusted_weights_df_top = adjust_weights(weights_df.copy(), num_stocks_top, 'top')

# After adjusting the top stocks, drop them from the original dataframe
weights_df_bottom = weights_df.drop(adjusted_weights_df_top.index)

# Now adjust the weights for the bottom stocks from this updated dataframe
num_stocks_bottom = min(num_stocks_bottom, weights_df_bottom.shape[0])
adjusted_weights_df_bottom = adjust_weights(weights_df_bottom.copy(), num_stocks_bottom, 'bottom')

#handling possible missing tickers
missing_tickers_top = set(adjusted_weights_df_top['ticker']) - set(prices.columns)
missing_tickers_bottom = set(adjusted_weights_df_bottom['ticker']) - set(prices.columns)
# st.write('Missing tickers for top stocks:', missing_tickers_top)
# st.write('Missing tickers for bottom stocks:', missing_tickers_bottom)


# Construct the price series
# Construct the price series
nasdaq_price_series = construct_price_series(weights_df, prices)
custom_price_series_top = construct_price_series(adjusted_weights_df_top, prices)
if adjusted_weights_df_bottom is not None:
    custom_price_series_bottom = construct_price_series(adjusted_weights_df_bottom, prices)


# Plot the price series
with col1:
    st.write("Nasdaq 100 simulation based on weights and n top/bottom performers")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=prices.index, y=nasdaq_price_series, mode='lines', name='NASDAQ 100'))
    fig.add_trace(go.Scatter(x=prices.index, y=custom_price_series_top, mode='lines', name='Custom Index (Top)'))
    if adjusted_weights_df_bottom is not None:
        fig.add_trace(go.Scatter(x=prices.index, y=custom_price_series_bottom, mode='lines', name='Custom Index (Bottom)'))
    st.plotly_chart(fig)

with col2:
    if custom_price_series_bottom is not None:
        relative_price = custom_price_series_top / custom_price_series_bottom

        # Plot the relative price series
        bottom = 100 - num_stocks_bottom
        st.write(f'Top {num_stocks_top} weights / Bottom {bottom} weights ratio')
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=relative_price.index, y=relative_price, mode='lines', name='Relative Price Series (Top/Bottom)'))
        st.plotly_chart(fig)


# st.write("prices", prices)
# st.write("adjusted_weights_df_top", adjusted_weights_df_top)
# st.write("adjusted_weights_df_bottom", adjusted_weights_df_bottom)
# st.write("custom_price_series_bottom", custom_price_series_bottom)
# st.write("custom_price_series_top",custom_price_series_top)
# st.write("nasdaq_price_series", nasdaq_price_series)

# st.write('Bottom stocks dataframe')
# st.write(adjusted_weights_df_bottom)

# st.write('Bottom stocks price series')
# st.write(custom_price_series_bottom)

# if adjusted_weights_df_bottom is not None:
#     st.write('Bottom stocks adjusted weights dataframe')
#     st.write(adjusted_weights_df_bottom)

#     st.write('Bottom stocks custom price series')
#     st.write(custom_price_series_bottom)

# Add these lines after the weights are adjusted
st.write("Sum of weights for top stocks: ", adjusted_weights_df_top['weights'].sum())
st.write("Sum of weights for bottom stocks: ", adjusted_weights_df_bottom['weights'].sum())

