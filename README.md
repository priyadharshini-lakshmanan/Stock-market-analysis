Stock Market Analysis Dashboard

A comprehensive Streamlit dashboard for analyzing stock market data with interactive visualizations and insights.
________________________________________________________________________________________________________________
Features:

Market Overview: View overall market summary with green/red stock distribution

Top Performers: Analyze top gainers and losers by yearly returns

Volatility Analysis: Examine stock price fluctuations and risk metrics

Cumulative Returns: Track investment performance over time

Sector Performance: Compare returns across different sectors

Correlation Matrix: Identify relationships between stock prices

Monthly Analysis: Review monthly performance for specific periods
________________________________________________________________________________________________________________
Prerequisites

Python 3.7+

MySQL Server

Required Python packages (see Installation)
_________________________________________________________________________________________________________________
Installation

Install required packages:

bashpip install streamlit pandas numpy plotly mysql-connector-python

Set up MySQL database:

Create a database named stockmarket

Import your stock data tables


Update database credentials in the code:

pythonDB_CONFIG = {

'user': 'your_username',

'password': 'your_password',

'host': 'localhost',

'database': 'stockmarket'

}

Required Database Tables:

stockwhole - Main stock data with date, ticker, close, volume

concatenated_df - Stock yearly returns and sectors

top_10_sectors - Sector-wise performance

correlation - Stock correlation matrix

cumulative_return - Cumulative returns over time

volatility_per_ticker - Volatility metrics per stock

___________________________________________________________________________________________________________
Usage:

Run the dashboard

bashstreamlit run app.py
