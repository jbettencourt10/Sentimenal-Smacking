
import pandas as pd
import nltk

import parameters

def preprocess_tweet(tweet: str) -> str:
    tweet = tweet.lower()
    tweet = nltk.word_tokenize(tweet)
    tweet = [word for word in tweet if word.isalpha()]
    tweet = ' '.join(tweet)
    return tweet


def read_twitter_dataset(file_path : str) -> (list, list):
    data = pd.read_csv(file_path, encoding='cp1252')
    tweets = data.iloc[:, 0].values.tolist()
    labels = data.iloc[:, 1].values.tolist()
    tweets = [preprocess_tweet(tweet) for tweet in tweets]
    labels = [1 if label == 'WillSmith' else 0 for label in labels]
    return tweets, labels
