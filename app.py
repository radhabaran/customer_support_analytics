import streamlit as st
import pandas as pd
import plotly.express as px

# Install required packages:
# pip install streamlit pandas plotly

def create_dashboard():
    st.set_page_config(page_title="Customer Sentiment Dashboard", layout="wide")
    
    # Read data
    df = pd.read_csv('chat_histories.csv')
    
    # Header
    st.title("Customer Sentiment Analysis Dashboard")
    
    # Create columns for metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Conversations", len(df))
    with col2:
        st.metric("Positive Sentiments", len(df[df['customer_sentiment'] == 'Positive']))
    with col3:
        st.metric("Negative Sentiments", len(df[df['customer_sentiment'] == 'Negative']))
    
    # Create charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Sentiment Distribution")
        fig_pie = px.pie(df, names='customer_sentiment')
        st.plotly_chart(fig_pie)
    
    with col2:
        st.subheader("Sentiment Counts")
        fig_bar = px.bar(df['customer_sentiment'].value_counts().reset_index(), 
                        x='index', y='customer_sentiment')
        st.plotly_chart(fig_bar)
    
    # Show recent messages
    st.subheader("Recent Customer Messages")
    st.dataframe(df[['chat_messages', 'customer_sentiment', 'chat_captured_date']].tail())

if __name__ == "__main__":
    create_dashboard()