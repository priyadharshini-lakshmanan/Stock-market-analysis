import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import mysql.connector
from pathlib import Path

st.set_page_config(
    page_title="Stock Market Analysis Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

DB_CONFIG = {
    'user': 'root',
    'password': 'Localhost@123',
    'host': 'localhost',
    'database': 'stockmarket'
}

# Tables to fetch from MySQL
TABLES = [
    "stockwhole",
    "concatenated_df",
    "top_10_sectors",
    "correlation",
    "cumulative_return",
    "volatility_per_ticker",
    "result"
]

# Function to query a table from MySQL using only mysql.connector
@st.cache_data(ttl=600)
def get_data(table_name):
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute(f"SELECT * FROM {table_name};")
        columns = [col[0] for col in cursor.description]
        rows = cursor.fetchall()
        df = pd.DataFrame(rows, columns=columns)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        return df
    except mysql.connector.Error as e:
        st.error(f"Database error on table {table_name}: {e}")
        return pd.DataFrame()
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()
# Sidebar 
st.sidebar.title("📊 Stock Analysis Dashboard")
st.sidebar.markdown("---")

# Navigation
page = st.sidebar.selectbox(
    "Select Analysis",
    ["🏠 Overview", 
     "🏆 Top Performers", 
     "📊 Volatility Analysis",
     "📈 Cumulative Returns",
     "🏭 Sector Performance",
     "🔗 Correlation Matrix",
     "📅 Monthly Analysis"]
)

df = get_data('stockwhole')

if not df.empty:
    if page == "🏠 Overview":
        st.title("📈 Market Overview Dashboard")
        st.markdown("### Overall Market Summary")

        df = df.sort_values(['Ticker', 'date'])
        start_prices = df.groupby('Ticker')['close'].first()
        end_prices = df.groupby('Ticker')['close'].last()
        ticker_returns = (end_prices / start_prices) - 1
        df['yearly_return'] = df['Ticker'].map(ticker_returns)
        unique_green_stocks = (ticker_returns > 0).sum()
        unique_red_stocks = (ticker_returns <= 0).sum()
        avg_price = df['close'].mean()
        avg_volume = df['volume'].mean()

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(label="💰 Average Price", value=f"₹{avg_price:,.2f}")
        with col2:
            st.metric(label="📊 Average Volume", value=f"{avg_volume/1e6:.2f}M")
        with col3:
            st.metric(label="📈 Green Stocks", value=f"{unique_green_stocks}")
        with col4:
            st.metric(label="📉 Red Stocks", value=f"{unique_red_stocks}")

        st.markdown("---")
        st.subheader("Green Stocks vs Red Stocks")
        pie_data = pd.DataFrame({
            'Category': ['Green Stocks', 'Red Stocks'],
            'Count': [unique_green_stocks, unique_red_stocks]
        })
        fig = px.pie(
            pie_data, 
            values='Count', 
            names='Category',
            color='Category',
            color_discrete_map={'Green Stocks': '#10b981', 'Red Stocks': '#ef4444'},
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Average Metrics Performance")
        green_df = df[df['yearly_return'] > 0]
        red_df = df[df['yearly_return'] <= 0]
        comparison = pd.DataFrame({
            'Metric': ['Average Price', 'Average Volume'],
            'Green Stocks': [green_df['close'].mean(), green_df['volume'].mean()],
            'Red Stocks': [red_df['close'].mean(), red_df['volume'].mean()]
        })
        st.dataframe(comparison, use_container_width=True, hide_index=True)

    elif page == "🏆 Top Performers":
        st.title("🏆 Top Performing Stocks")
        st.markdown('''
            <p style="
            font-size:20px; 
            line-height:1.6; 
            color:#2E4057;
            text-align:justify;
           ">
        Yearly return is the percentage gain or loss on an investment over one year.
    ''', unsafe_allow_html=True)
        st.markdown('''
           <p style="
           font-size:20px; 
           line-height:1.6; 
           color:#2E4057;
           text-align:justify;
           ">
        Yearly Return = (Ending Close Price / Starting ClosePrice)-1
        </p>
    ''', unsafe_allow_html=True)
        df_concat = get_data('concatenated_df')
        
        if not df_concat.empty:
            df = df_concat 
            df.columns = df.columns.str.strip() 

            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🟢 Top 10 Green Stocks")
                
                top_10 = df.nlargest(10, 'YearlyReturn')[['Ticker', 'YearlyReturn', 'sector']].reset_index(drop=True)
                top_10.index = top_10.index + 1
                
                # Display table
                st.dataframe(
                    top_10,
                    use_container_width=True
                )
                
                # Bar chart for Top 10
                fig_top = px.bar(
                    top_10, 
                    x='Ticker', 
                    y='YearlyReturn',
                    color='YearlyReturn',
                    color_continuous_scale='Greens',
                    title="Top 10 Gainers",
                    labels={'YearlyReturn': 'Yearly Return'}
                )
                fig_top.update_layout(showlegend=False)
                st.plotly_chart(fig_top, use_container_width=True) 
            
            with col2:
                st.markdown("### 🔴 Top 10 Red Stocks")
                
                bottom_10 = df.nsmallest(10, 'YearlyReturn')[['Ticker', 'YearlyReturn', 'sector']].reset_index(drop=True)
                bottom_10.index = bottom_10.index + 1
                
                st.dataframe(
                    bottom_10,
                    use_container_width=True
                )
                fig_bottom = px.bar( 
                    bottom_10, 
                    x='Ticker', 
                    y='YearlyReturn',
                    color='YearlyReturn',
                    color_continuous_scale='Reds',
                    title="Top 10 Losers",
                    labels={'YearlyReturn': 'Yearly Return'}
                )

                fig_bottom.update_layout(showlegend=False)
                st.plotly_chart(fig_bottom, use_container_width=True)
        
        else:
            st.error("Failed to load concatenated_df table from MySQL")

   
    elif page == "📊 Volatility Analysis":
        st.title("📊 Volatility Analysis")
        image_path1 = Path("images/volatile1.png")
        image_path2 = Path("images/volatile2.png")
        st.markdown('''
            <p style="
            font-size:20px; 
            line-height:1.6; 
            color:#2E4057;
            text-align:justify;
        ">
    * Volatility is a measure of the rate of fluctuations in the price of a security over time.
    ''', unsafe_allow_html=True)
        st.markdown('''
            <p style="
                    font-size:20px; 
            line-height:1.6; 
            color:#2E4057;
            text-align:justify;
        ">
    * Volatility gives insight into how much the price fluctuates, which is valuable for risk assessment. 
    ''', unsafe_allow_html=True)
        st.markdown('''
            <p style="
            font-size:20px; 
            line-height:1.6; 
            color:#2E4057;
            text-align:justify;
        "> 
    * Higher volatility often indicates more risk, while lower volatility indicates a more stable stock.
    ''', unsafe_allow_html=True)
        # Display the local images
        if image_path1.is_file():
            st.image(str(image_path1), caption='Volatility Analysis for ADANIENT', use_container_width=True)
        else:
            st.warning(f"Image not found at {image_path1}")

        if image_path2.is_file():
            st.image(str(image_path2), caption='Volatility Analysis for SUNPHARMA', use_container_width=True)
        else:
            st.warning(f"Image not found at {image_path2}")

        df_volatility = get_data('volatility_per_ticker')
        if not df_volatility.empty:
            # Top 10 most volatile stocks
            top_volatile = df_volatility.nlargest(50,'Standard_deviation')[['Ticker','Standard_deviation']].reset_index(drop=True)
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.subheader(" Volatile Stocks")
                fig = px.bar(
                    top_volatile,
                    x='Ticker',
                    y='Standard_deviation',
                    color='Standard_deviation',
                    color_continuous_scale='Reds',
                    title="Volatility (Standard Deviation of Daily Returns)",
                    labels={'Standard_deviation': 'Standard_deviation'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Volatility Data")
                st.dataframe(
                    top_volatile,
                    use_container_width=True,
                    height=400
                )
        else:
            st.warning("Volatility data not available in MySQL")

    elif page == "📈 Cumulative Returns":
        st.title("📈 Cumulative Returns")
        st.markdown('''
            <p style="
            font-size:20px; 
            line-height:1.6; 
            color:#2E4057;
            text-align:justify;
           "> 
       * Cumulative return is the total percentage change in an investment's value over a specific period, 
                    showing the overall gain or loss from the starting point to the ending point.
    ''', unsafe_allow_html=True)
        st.markdown('''
            <p style="
            font-size:20px; 
            line-height:1.6; 
            color:#2E4057;
            text-align:justify;
           "> 
       * This helps users compare how different stocks performed relative to each other.

    ''', unsafe_allow_html=True)
        st.markdown('''
            <p style="
            font-size:20px; 
            line-height:1.6; 
            color:#2E4057;
            text-align:justify;
           "> 
         * Cumulative Return = (Ending close Price / Starting close Price) - 1
    ''', unsafe_allow_html=True)    
        df_cumulative = get_data('cumulative_return')

        if not df_cumulative.empty:
            df_cumulative.columns = df_cumulative.columns.str.strip()
            df_cumulative['date'] = pd.to_datetime(df_cumulative['date'])
            df_cumulative['cumulative_return'] = pd.to_numeric(df_cumulative['cumulative_return'], errors='coerce')
            df_cumulative.dropna(subset=['cumulative_return'], inplace=True)

            final_returns = df_cumulative.groupby('Ticker')['cumulative_return'].max().reset_index()

            # Top 5 performing stocks
            top_5 = final_returns.nlargest(5, 'cumulative_return')
            top_5_tickers = top_5['Ticker'].tolist()

            fig = go.Figure()
            for ticker in top_5_tickers:
                stock_data = df_cumulative[df_cumulative['Ticker'] == ticker]
                fig.add_trace(go.Scatter(
                    x=stock_data['date'],
                    y=stock_data['cumulative_return'] * 100,
                    mode='lines',
                    name=ticker
                ))
            fig.update_layout(
                title="Top 5 Performing Stocks: Cumulative Return Over Time",
                xaxis_title="Date",
                yaxis_title="Cumulative Return (%)",
                hovermode='x unified',
                height=500
                )
            fig.add_hline(y=0, line_dash="dash", line_color="gray")

            st.plotly_chart(fig, use_container_width=True)

            # Show summary table
            st.subheader("Performance Summary")
            summary_df = top_5.copy()
            summary_df['cumulative_return'] = summary_df['cumulative_return'] * 100
            summary_df = summary_df.rename(columns={'cumulative_return': ' Cumulative Return (%)'})
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

        else:
            st.warning("Cumulative returns data not available in MySQL")

    elif page == "🏭 Sector Performance":
        st.title("🏭 Sector Performance")
        df_sectors = get_data('top_10_sectors')
        col1, col2 = st.columns([2, 1])
        with col1:
                # Bar chart
                fig = px.bar(
                    df_sectors,
                    x='sector',
                    y='AvgYearlyReturn',
                    color='AvgYearlyReturn',
                    color_continuous_scale='RdYlGn',
                    title="Average Yearly Return by Sector",
                    text='AvgYearlyReturn'
                )
                fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
                fig.add_hline(y=0, line_dash="dash", line_color="red")
                st.plotly_chart(fig, use_container_width=True)
            
        with col2:
                st.subheader("Sector Summary")
                st.dataframe(
                    df_sectors[['sector', 'AvgYearlyReturn']] ,
                    use_container_width=True,
                    hide_index=True
                )

    elif page == "🔗 Correlation Matrix":
        st.title("🔗 Stock Price Correlation")
        st.markdown('''
            <p style="
            font-size:20px; 
            line-height:1.6; 
            color:#2E4057;
            text-align:justify;
           "> 
         * Visualize the correlation between the stock prices of different companies
    ''', unsafe_allow_html=True) 
        df_corr = get_data('correlation')
         # Plot heatmap
        fig = px.imshow(
            df_corr,
            labels=dict(color="Correlation"),
            x=df_corr.columns,
            y=df_corr.columns,
            color_continuous_scale='RdBu_r',
            aspect="auto",
            title="Stock Price Correlation Heatmap",
            zmin=-1,
            zmax=1
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

        # correlation pairs
        corr_mat = df_corr.to_numpy()
        stocks = df_corr.columns.tolist()
        mask = np.eye(len(stocks), dtype=bool)  # mask diagonal

        pos_corr_pairs = []
        neg_corr_pairs = []
        for i in range(len(stocks)):
            for j in range(i + 1, len(stocks)):
                val = corr_mat[i, j]
                if val > 0.7:  # Highly positive correlation
                    pos_corr_pairs.append((stocks[i], stocks[j], val))
                elif val < -0.5:  # Highly negative correlation
                    neg_corr_pairs.append((stocks[i], stocks[j], val))

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🔴 Positively Correlated Pairs (> 0.7)")
            st.info("Stocks that tend to move together (correlation > 0.7)")
            if pos_corr_pairs:
                for a, b, v in pos_corr_pairs:
                    st.write(f"{a} ↔ {b}: {v:.2f}")
            else:
                st.write("No highly positively correlated pairs found.")
        with col2:
            st.subheader("🔵 Negatively Correlated Pairs (< -0.5)")
            st.info("Stocks that move in opposite direction (correlation < -0.5)")
            if neg_corr_pairs:
                for a, b, v in neg_corr_pairs:
                    st.write(f"{a} ↔ {b}: {v:.2f}")
            else:
                st.write("No negatively correlated pairs found.")
    elif page == "📅 Monthly Analysis":
        st.title("📅 Monthly Analysis")
        df_monthly = get_data('stockwhole')
        
        # Month selector
        months = ['January', 'February', 'March', 'April', 'May', 'June',
                'July', 'August', 'September', 'October', 'November', 'December']
        
        selected_month = st.selectbox("Select Month", months)
        # Convert date to datetime and extract month name
        df_monthly['date'] = pd.to_datetime(df_monthly['date'])
        df_monthly['month_name'] = df_monthly['date'].dt.strftime('%B')
            
            # Filter data for selected month
        df_filtered = df_monthly[df_monthly['month_name'] == selected_month].copy()
        # Sort by date
        df_filtered = df_filtered.sort_values('date')
                
                # Calculate returns for each ticker
        results = []
        for ticker in df_filtered['Ticker'].unique():
            ticker_data = df_filtered[df_filtered['Ticker'] == ticker]
            first_price = ticker_data['close'].iloc[0]
            last_price = ticker_data['close'].iloc[-1]
            return_pct = ((last_price - first_price) / first_price) * 100
            results.append({'Ticker': ticker, 'Return': return_pct})
                
                # Create dataframe
        returns_df = pd.DataFrame(results)
                
                # Get top 5 gainers and losers
        top_gainers = returns_df.nlargest(5, 'Return')
        top_losers = returns_df.nsmallest(5, 'Return')

        col1, col2 = st.columns(2)

        with col1:
             st.subheader(f"🟢 Top 5 Gainers - {selected_month}")
             fig = px.bar(top_gainers, x='Ticker', y='Return', 
                                color='Return', color_continuous_scale='Greens',
                                title=f"Top Gainers in {selected_month}")
             st.plotly_chart(fig, use_container_width=True)
                    
             display_gainers = top_gainers.copy()
             display_gainers['Return'] = display_gainers['Return'].apply(lambda x: f"{x:.2f}%")
             st.dataframe(display_gainers, use_container_width=True, hide_index=True)

        with col2:
                    st.subheader(f"🔴 Top 5 Losers - {selected_month}")
                    fig = px.bar(top_losers, x='Ticker', y='Return', 
                                color='Return', color_continuous_scale='Reds',
                                title=f"Top Losers in {selected_month}")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    display_losers = top_losers.copy()
                    display_losers['Return'] = display_losers['Return'].apply(lambda x: f"{x:.2f}%")
                    st.dataframe(display_losers, use_container_width=True, hide_index=True)

else:
    st.warning("⚠️ Dashboard data could not be loaded.")
    st.info("💡 Make sure the 'stockwhole' table exists in your MySQL database and contains data.")
