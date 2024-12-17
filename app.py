import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def style_metric_cards():
    st.markdown("""
    <style>
    div[data-testid="metric-container"] {
        background-color: rgba(28, 131, 225, 0.1);
        border: 1px solid rgba(28, 131, 225, 0.1);
        padding: 5% 5% 5% 10%;
        border-radius: 5px;
        color: rgb(30, 103, 119);
        overflow-wrap: break-word;
    }
    div[data-testid="metric-container"] > label[data-testid="stMetricLabel"] > div {
        overflow-wrap: break-word;
        white-space: break-spaces;
        font-size: 12px;
    }
    </style>
    """, unsafe_allow_html=True)

def create_dashboard():
    st.set_page_config(
        page_title="Customer Service Analytics Dashboard",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    style_metric_cards()
    
    path = "data/chat_histories.csv"
    df = pd.read_csv(path)
    
    # Header with minimal space
    st.markdown("""
        <h2 style='text-align: center; color: #2c3e50; margin-bottom: 10px;'>
        Customer Service Analytics Dashboard
        </h2>
    """, unsafe_allow_html=True)
    
    # Top metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Conversations", f"{len(df):,}")
    with col2:
        positive_pct = (len(df[df['customer_sentiment'] == 'Positive']) / len(df)) * 100
        st.metric("Positive Rate", f"{positive_pct:.1f}%")
    with col3:
        unhappy_pct = (len(df[df['customer_classification'] == 'extremely unhappy']) / len(df)) * 100
        st.metric("Critical Cases", f"{unhappy_pct:.1f}%")
    with col4:
        refund_pct = (len(df[df['query_classification'] == 'refund']) / len(df)) * 100
        st.metric("Refund Requests", f"{refund_pct:.1f}%")
    with col5:
        replacement_pct = (len(df[df['query_classification'] == 'replacement']) / len(df)) * 100
        st.metric("Replacement Requests", f"{replacement_pct:.1f}%")

    # Create three columns for charts
    col1, col2, col3 = st.columns([1,1,1])
    
    with col1:
        # Sentiment Distribution
        sentiment_counts = df['customer_sentiment'].value_counts()
        fig_sentiment = go.Figure(data=[go.Pie(
            labels=sentiment_counts.index,
            values=sentiment_counts.values,
            hole=.4,
            marker_colors=['#2ecc71', '#e74c3c', '#f1c40f']
        )])
        fig_sentiment.update_layout(
            title="Sentiment Distribution",
            showlegend=True,
            height=300,
            margin=dict(l=0, r=0, t=30, b=0)
        )
        st.plotly_chart(fig_sentiment, use_container_width=True)

    with col2:
        # Query Classification
        query_counts = df['query_classification'].value_counts()
        fig_query = go.Figure(data=[go.Bar(
            x=query_counts.values,
            y=query_counts.index,
            orientation='h',
            marker_color='#3498db'
        )])
        fig_query.update_layout(
            title="Query Types",
            height=300,
            margin=dict(l=0, r=0, t=30, b=0),
            xaxis_title="Count",
            yaxis_title=None
        )
        st.plotly_chart(fig_query, use_container_width=True)

    with col3:
        # Customer Classification
        customer_counts = df['customer_classification'].value_counts()
        fig_customer = go.Figure(data=[go.Bar(
            x=customer_counts.index,
            y=customer_counts.values,
            marker_color=['#e74c3c', '#f39c12', '#2ecc71', '#95a5a6']
        )])
        fig_customer.update_layout(
            title="Customer State Distribution",
            height=300,
            margin=dict(l=0, r=0, t=30, b=0),
            xaxis_title=None,
            yaxis_title="Count"
        )
        st.plotly_chart(fig_customer, use_container_width=True)

    # Recent Critical Cases
    st.markdown("### Critical Cases (Extremely Unhappy Customers)")
    critical_cases = df[
        df['customer_classification'] == 'extremely unhappy'
    ][['chat_messages', 'query_classification', 'customer_sentiment', 'chat_captured_date']].tail(5)
    
    st.dataframe(
        critical_cases,
        use_container_width=True,
        height=200
    )

if __name__ == "__main__":
    create_dashboard()