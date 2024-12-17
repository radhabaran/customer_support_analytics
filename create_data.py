import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Download required NLTK data
nltk.download('vader_lexicon')

def analyze_sentiment(text):
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = sia.polarity_scores(text)
    
    # Determine sentiment based on compound score
    if sentiment_scores['compound'] >= 0.05:
        return 'Positive'
    elif sentiment_scores['compound'] <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

# Read the CSV file
df = pd.read_csv('chat_histories.csv')

# Apply sentiment analysis
df['customer_sentiment'] = df['chat_messages'].apply(analyze_sentiment)

# Save the updated dataframe back to CSV
df.to_csv('chat_histories.csv', index=False)

# Display distribution of sentiments
sentiment_distribution = df['customer_sentiment'].value_counts()
print("\nSentiment Distribution:")
print(sentiment_distribution)