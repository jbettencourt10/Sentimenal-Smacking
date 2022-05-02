

import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
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



arrays = read_twitter_dataset("../data/sentimental_smacking_sample_twitter_dataset.csv")


BATCH_SIZE = 32


VOCAB_SIZE = 50
encoder = TextVectorization(max_tokens=VOCAB_SIZE)
encoder.adapt(arrays[0])



model = tf.keras.Sequential([
    encoder,
    tf.keras.layers.Embedding(
        input_dim=len(encoder.get_vocabulary()),
        output_dim=64,
        mask_zero=True),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])


model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])


# model.fit(arrays[0], arrays[1], batch_size=BATCH_SIZE, epochs=10)

history = model.fit(x=arrays[0], y=arrays[1],batch_size=BATCH_SIZE, epochs=150)






# Here we can see a toy example of a sentence tokenizer, more specifically nltk.
# sentence = """Hello! This is my CS520 project. I hope you enjoy it."""
# tokens = nltk.word_tokenize(sentence)
# tagged = nltk.pos_tag(tokens)
# print(tagged[0:6])


# We expect to use this file to tokenize tweets obtained from the Twitter API.
# Then, after tokenizing, we expect to use a sentiment analysis technique to determine the polarity of the tweet.
# Weights and biases, on the other hand, will utilize a GUI.
# So, the results from this file will be compared with the results from the weights and biases GUI.
# After comparing these results, we hope to answer the question:
# Can a non-programmer hope to feasibly use a GUI to create a sentiment analysis program?
