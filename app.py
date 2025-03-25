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
            if not data.empty:
                stock_data[symbol] = data['Adj Close']
        except Exception as e:
            st.error(f"Error fetching data for {symbol}: {e}")

    return pd.DataFrame(stock_data)

# ---------------------------
# Function to Plot Heatmap
# ---------------------------
def plot_correlation_heatmap(df):
    if df.empty or df.shape[1] < 2:
        st.error("❌ Not enough data for correlation analysis. Select at least 2 stocks.")
        return

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df, annot=True, cmap="coolwarm", linewidths=0.5, ax=ax, vmin=-1, vmax=1, center=0)

    plt.close("all")  # Ensure Streamlit renders properly
    st.pyplot(fig)

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("📊 Stock Market Correlation Analysis")

# User input: Enter stock symbols
stock_input = st.text_input("Enter Stock Symbols (comma-separated)", "AAPL, GOOGL, TSLA, AMZN, MSFT")

# Process input into a list
selected_stocks = [stock.strip().upper() for stock in stock_input.split(",") if stock.strip()]

# Fetch stock data
if selected_stocks:
    st.subheader("📈 Fetching Stock Data...")
    stock_data = fetch_stock_data(selected_stocks)

    # Debugging: Show fetched stock data
    if not stock_data.empty:
        st.write("### Debug: Stock Price Data (Last 5 Rows)")
        st.dataframe(stock_data.tail())

        # Drop missing data
        stock_data = stock_data.dropna(how="all")  # Remove rows with all NaNs
        stock_data = stock_data.dropna(axis=1, how="all")  # Remove columns with all NaNs

        # Calculate correlation matrix
        correlation_matrix = stock_data.corr()

        # Debugging: Show correlation matrix
        st.write("### Debug: Correlation Matrix")
        st.dataframe(correlation_matrix)

        if correlation_matrix.isnull().values.any():
            st.warning("⚠️ Some missing values found in correlation matrix. Cleaning data...")
            correlation_matrix = correlation_matrix.dropna(axis=0, how='all').dropna(axis=1, how='all')

        # Ensure at least 2 stocks are available for correlation
        if correlation_matrix.shape[0] > 1:
            st.subheader("📊 Stock Price Correlation Heatmap")
            plot_correlation_heatmap(correlation_matrix)
        else:
            st.error("❌ Not enough data for correlation analysis. Please enter more valid stocks.")
    else:
        st.error("❌ Failed to fetch stock data. Try different stock symbols.")
