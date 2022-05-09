# Import files with helper functions
import data_organization
import neural_network


# Main method for program
def main():
    # Save tweets and labels after preprocessing tweets into lists
    (tweets, labels) = data_organization.read_twitter_dataset("../data/sentimental_smacking_sample_twitter_dataset.csv")
    # Create recurrent neural network based on tweets and labels
    rnn = neural_network.RNN(tweets, labels)
    # Train neural network and test on training set
    rnn.train()


if __name__ == '__main__':
    main()
