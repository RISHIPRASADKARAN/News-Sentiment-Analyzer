import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import yfinance as yf

# ---------------------------
# Function to Fetch Stock Data
# ---------------------------
def fetch_stock_data(symbols, start="2023-01-01", end="2024-01-01"):
    stock_data = {}
    
    for symbol in symbols:
        try:
            data = yf.download(symbol, start=start, end=end)
            stock_data[symbol] = data['Adj Close']
        except Exception as e:
            st.error(f"Error fetching data for {symbol}: {e}")
    
    return pd.DataFrame(stock_data)

# ---------------------------
# Function to Plot Heatmap
# ---------------------------
def plot_correlation_heatmap(df):
    if df.empty:
        st.error("Correlation matrix is empty! Ensure stock data is available.")
        return
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df, annot=True, cmap="coolwarm", linewidths=0.5, ax=ax)
    st.pyplot(fig)

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("📊 Stock Market Correlation Analysis")

# User input: Select stocks
default_stocks = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS"]
selected_stocks = st.multiselect("Select Stocks for Analysis:", default_stocks, default=default_stocks)

# Fetch stock data
if selected_stocks:
    st.subheader("📈 Fetching Stock Data...")
    stock_data = fetch_stock_data(selected_stocks)

    if not stock_data.empty:
        st.write("Stock Price Data (Last 5 Rows):")
        st.dataframe(stock_data.tail())

        # Calculate correlation matrix
        st.subheader("📊 Stock Price Correlation Heatmap")
        correlation_matrix = stock_data.corr()

        if correlation_matrix.isnull().values.any():
            st.warning("Some missing values found in correlation matrix. Cleaning data...")
            correlation_matrix = correlation_matrix.dropna(axis=0, how='all').dropna(axis=1, how='all')

        if correlation_matrix.shape[0] > 1:  # Ensure at least 2 stocks are present
            plot_correlation_heatmap(correlation_matrix)
        else:
            st.error("Not enough data for correlation analysis. Please select more stocks.")
    else:
        st.error("Failed to fetch stock data. Try different stock symbols.")

