Stock Market Analysis Dashboard
A comprehensive Streamlit dashboard for analyzing stock market data with interactive visualizations and insights.
Features

Market Overview: View overall market summary with green/red stock distribution
Top Performers: Analyze top gainers and losers by yearly returns
Volatility Analysis: Examine stock price fluctuations and risk metrics
Cumulative Returns: Track investment performance over time
Sector Performance: Compare returns across different sectors
Correlation Matrix: Identify relationships between stock prices
Monthly Analysis: Review monthly performance for specific periods

Prerequisites

Python 3.7+
MySQL Server
Required Python packages (see Installation)

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
Required Database Tables

stockwhole - Main stock data with date, ticker, close, volume
concatenated_df - Stock yearly returns and sectors
top_10_sectors - Sector-wise performance
correlation - Stock correlation matrix
cumulative_return - Cumulative returns over time
volatility_per_ticker - Volatility metrics per stock

Usage
Run the dashboard:
bashstreamlit run app.py
```

Navigate through different analyses using the sidebar menu.

## Project Structure
```
project/
├── app.py              # Main dashboard code
├── images/             # Volatility analysis images
│   ├── volatile1.png
│   └── volatile2.png
└── README.md
