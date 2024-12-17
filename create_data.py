import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

api_key = os.environ['OA_API']           
os.environ['OPENAI_API_KEY'] = api_key

# Initialize ChatOpenAI
llm = ChatOpenAI(model="gpt-3.5-turbo",temperature=0.3)


# Download required NLTK data
nltk.download('vader_lexicon')

def analyze_sentiment(text):
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = sia.polarity_scores(text)
    
    if sentiment_scores['compound'] >= 0.05:
        return 'Positive'
    elif sentiment_scores['compound'] <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

def classify_using_gpt(text, classification_type):
    try:
        if classification_type == 'query':
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a customer service analyst. Classify the given text into one of these categories: 'product related', 'order related', 'refund', 'replacement', 'generic', 'others'. Respond with only the category name."),
                ("user", "Message: {text}")
            ])
        else:
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a customer service analyst. Classify the customer's state into one of these categories: 'extremely unhappy', 'unhappy', 'happy', 'fence sitters'. Consider the tone, language, and content. Respond with only the category name."),
                ("user", "Message: {text}")
            ])

        chain = prompt | llm
        response = chain.invoke({"text": text})
        return response.content.strip().lower()

    except Exception as e:
        print(f"Error in classification: {e}")
        return 'others' if classification_type == 'query' else 'fence sitters'

def main():
    # Read the CSV file
    df = pd.read_csv('data/chat_histories.csv')
    
    # Apply sentiment analysis only to rows without existing sentiment
    df['customer_sentiment'] = df.apply(
        lambda row: row['customer_sentiment'] if pd.notna(row.get('customer_sentiment')) 
        else analyze_sentiment(row['chat_messages']), 
        axis=1
    )

    # Apply query classification
    if 'query_classification' not in df.columns:
        df['query_classification'] = ''
    
    # Only classify rows that don't have a classification
    mask = (df['query_classification'].isna()) | (df['query_classification'] == '')
    if mask.any():
        df.loc[mask, 'query_classification'] = df.loc[mask, 'chat_messages'].apply(
            lambda x: classify_using_gpt(x, 'query')
        )

    # Apply customer classification
    if 'customer_classification' not in df.columns:
        df['customer_classification'] = ''
    
    # Only classify rows that don't have a classification
    mask = (df['customer_classification'].isna()) | (df['customer_classification'] == '')
    if mask.any():
        df.loc[mask, 'customer_classification'] = df.loc[mask, 'chat_messages'].apply(
            lambda x: classify_using_gpt(x, 'customer')
        )

    # Save the updated dataframe
    df.to_csv('data/chat_histories.csv', index=False)

    # Display distributions
    print("\nSentiment Distribution:")
    print(df['customer_sentiment'].value_counts())
    print("\nQuery Classification Distribution:")
    print(df['query_classification'].value_counts())
    print("\nCustomer Classification Distribution:")
    print(df['customer_classification'].value_counts())

if __name__ == "__main__":
    main()