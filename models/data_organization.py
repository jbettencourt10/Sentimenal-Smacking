import pandas as pd
import tensorflow as tf
import nltk


def preprocess_tweet(tweet: str):
    tweet = tweet.lower()
    tweet = nltk.word_tokenize(tweet)
    tweet = [word for word in tweet if word.isalpha()]
    tweet = ' '.join(tweet)
    return tweet


def read_twitter_dataset(file: str):
    data = pd.read_csv(file, encoding='cp1252')
    tweets = data.iloc[:, 0].values.tolist()
    labels = data.iloc[:, 1].values.tolist()
    tweets = [preprocess_tweet(tweet) for tweet in tweets]
    labels = [1 if label == 'WillSmith' else 0 for label in labels]
    return tweets, labels









# We expect to use this file to tokenize tweets obtained from the Twitter API.
# Then, after tokenizing, we expect to use a sentiment analysis technique to determine the polarity of the tweet.
# Weights and biases, on the other hand, will utilize a GUI.
# So, the results from this file will be compared with the results from the weights and biases GUI.
# After comparing these results, we hope to answer the question:
# Can a non-programmer hope to feasibly use a GUI to create a sentiment analysis program?
