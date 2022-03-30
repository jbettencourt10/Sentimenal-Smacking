# These are the imports that will be necessary for sentiment analysis.
# NLTK lets us tokenize, clean, and stem words.
# Pandas lets us read in the data.
import nltk
import pandas as pd

# Eventually, this line will be uncommented. For now, there is an encoding issue with reading in tweets.
# The twitter bot will fix thsi encoding issue.
# df = pd.read_csv('../data/sentimental_smacking_sample_twitter_dataset.csv')

# Here we can see a toy example of a sentence tokenizer, more specifically nltk.
sentence = """Hello! This is my CS520 project. I hope you enjoy it."""
tokens = nltk.word_tokenize(sentence)
tagged = nltk.pos_tag(tokens)
print(tagged[0:6])


# We expect to use this file to tokenize tweets obtained from the Twitter API.
# Then, after tokenizing, we expect to use a sentiment analysis technique to determine the polarity of the tweet.
# Weights and biases, on the other hand, will utilize a GUI.
# So, the results from this file will be compared with the results from the weights and biases GUI.
# After comparing these results, we hope to answer the question:
# Can a non-programmer hope to feasibly use a GUI to create a sentiment analysis program?
