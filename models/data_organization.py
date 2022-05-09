
# Import external libraries
import pandas as pd
import nltk

# Import parameters from parameters file
import parameters

# Takes in a tweet and preprocesses it by doing the following:
# 1. Lowercase the tweet
# 2. Tokenize the tweet
# 3. Remove stopwords, numbers, and punctuation
# 4. Rejoin the tokens into a single string
# 5. Return tweet
def preprocess_tweet(tweet: str) -> str:
    tweet = tweet.lower()
    tweet = nltk.word_tokenize(tweet)
    tweet = [word for word in tweet if word.isalpha()]
    tweet = ' '.join(tweet)
    return tweet

# Given a filepath to a csv, reads the csv and returns a list of tweets and a list of labels
def read_twitter_dataset(file_path : str) -> (list, list):
    # Read CSV with pandas
    data = pd.read_csv(file_path, encoding='cp1252')

    # Save tweets and labels as list
    tweets = data.iloc[:, 0].values.tolist()
    labels = data.iloc[:, 1].values.tolist()
    # Preprocess every tweet
    tweets = [preprocess_tweet(tweet) for tweet in tweets]

    # Changes labels to binary depending on content
    labels = [1 if label == 'WillSmith' else 0 for label in labels]
    return tweets, labels
