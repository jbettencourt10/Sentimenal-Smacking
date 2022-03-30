# import nltk
import pandas as pd

df = pd.read_csv('../data/sentimental_smacking_sample_twitter_dataset.csv')

print(df)

exit()

# Here we can see a toy example of a sentence tokenizer, more specifically nltk.
sentence = """"""
tokens = nltk.word_tokenize(sentence)
tagged = nltk.pos_tag(tokens)
print(tagged[0:6])


# We expect to use this file to tokenize tweets obtained from the Twitter API.
# Then, after tokenizing, we expect to use a sentiment analysis technique to determine the polarity of the tweet.
# Weights and biases, on the other hand, will utilize a GUI.
# So, the results from this file will be compared witht the results from the weights and biases GUI.
# After comparing these results, we hope to
