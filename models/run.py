import data_organization
import neural_network



def main():
    (tweets, labels) = data_organization.read_twitter_dataset("../data/sentimental_smacking_sample_twitter_dataset.csv")
    rnn = neural_network.RNN(tweets, labels)
    rnn.train()


if __name__ == '__main__':
    main()
