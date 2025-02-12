import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pickle
from datetime import datetime, timedelta
import plotly.graph_objects as go

# Load the saved model
@st.cache_resource
def load_model():
    try:
        with open('xgboost_model.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Fetch stock data with caching for efficiency
@st.cache_data
def fetch_stock_data(ticker, start_date, end_date):
    try:
        df = yf.download(ticker, start=start_date, end=end_date)
        if df.empty:
            st.warning(f"No data found for {ticker}. Try another ticker or adjust the date range.")
            return None
        return df
    except Exception as e:
        st.error(f"Error fetching stock data: {e}")
        return None

# Prepare features for the model
def prepare_features(df):
    features = pd.DataFrame()
    features['Close'] = df['Close']
    features['High'] = df['High']
    features['Low'] = df['Low']
    features['Open'] = df['Open']
    features['Volume'] = df['Volume']
    features['Return'] = df['Close'].pct_change()
    features = features.fillna(0)
    return features

# Make predictions using the model
def predict_stock_direction(model, features):
    try:
        predictions = model.predict(features)
        probabilities = model.predict_proba(features)
        return predictions.flatten(), probabilities
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None

# Main app function
def main():
    st.title("Stock Direction Prediction App")
    st.write("Predict whether a stock's price will go up or down using an XGBoost model.")

    # Sidebar Inputs
    st.sidebar.header("Input Parameters")
    ticker = st.sidebar.text_input("Stock Ticker", "PG").upper()
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)

    start_date = st.sidebar.date_input("Start Date", start_date)
    end_date = st.sidebar.date_input("End Date", end_date)

    if start_date >= end_date:
        st.error("Start date must be before end date.")
        return

    # Load model
    model = load_model()
    if model is None:
        return

    # Fetch Data and Predict
    if st.sidebar.button("Fetch Data and Predict"):
        df = fetch_stock_data(ticker, start_date, end_date)
        
        if df is not None and not df.empty:
            st.subheader("Stock Data")
            st.dataframe(df.tail())

            features = prepare_features(df)
            predictions, probabilities = predict_stock_direction(model, features)

            if predictions is not None and probabilities is not None:
                # Ensure all data is 1D before creating DataFrame
                results_df = pd.DataFrame({
                    'Date': df.index,
                    'Close': df['Close'].values.flatten(),
                    'Predicted Direction': ['Up' if p == 1 else 'Down' for p in predictions.flatten()],
                    'Probability Up': probabilities[:, 1].flatten(),
                    'Probability Down': probabilities[:, 0].flatten()
                })

                st.subheader("Predictions")
                st.dataframe(results_df)

                # Visualization
                st.subheader("Price Movement and Predictions")
                
                fig = go.Figure()
                
                # Price line
                fig.add_trace(go.Scatter(
                    x=results_df['Date'],
                    y=results_df['Close'],
                    name='Stock Price',
                    line=dict(color='blue')
                ))
                
                # Prediction markers
                colors = ['red' if pred == 'Down' else 'green' for pred in results_df['Predicted Direction']]
                fig.add_trace(go.Scatter(
                    x=results_df['Date'],
                    y=results_df['Close'],
                    mode='markers',
                    name='Predictions',
                    marker=dict(
                        color=colors,
                        size=10,
                        symbol='circle'
                    ),
                    showlegend=True
                ))
                
                fig.update_layout(
                    title=f'{ticker} Stock Price and Predictions',
                    xaxis_title='Date',
                    yaxis_title='Price',
                    hovermode='x unified'
                )

                st.plotly_chart(fig)

                # Prediction statistics
                st.subheader("Prediction Statistics")
                total_predictions = len(predictions)
                up_predictions = sum(predictions == 1)
                down_predictions = sum(predictions == 0)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Up Predictions", f"{up_predictions} ({up_predictions/total_predictions*100:.1f}%)")
                with col2:
                    st.metric("Down Predictions", f"{down_predictions} ({down_predictions/total_predictions*100:.1f}%)")

            else:
                st.error("Error making predictions.")
        else:
            st.error("No data available for the selected stock and date range.")

    # About the Model
    with st.expander("About the Model"):
        st.write("""
        This stock prediction model uses XGBoost to predict the direction of stock price movements.
        
        **Features Used:**
        - Closing Price
        - Opening Price
        - High Price
        - Low Price
        - Volume
        - Price Returns
        
        **Note:** These predictions are based on historical patterns and should not be used as the sole basis for investment decisions.
        """)

if __name__ == "__main__":
    main()
