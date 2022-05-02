import pandas as pd
import tensorflow as tf
import nltk




def read_twitter_dataset(file: str):
    data = pd.read_csv(file, encoding='cp1252')
    tweets = data.iloc[:, 0].values.tolist()
    labels = data.iloc[:, 1].values.tolist()
    labels =
    return tweets, labels



arrays = read_twitter_dataset("../data/sentimental_smacking_sample_twitter_dataset.csv")



BATCH_SIZE = 32


VOCAB_SIZE = 1000
encoder = tf.keras.layers.TextVectorization(max_tokens=VOCAB_SIZE)
# encoder.adapt(arrays.map(lambda text, label: text))










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
