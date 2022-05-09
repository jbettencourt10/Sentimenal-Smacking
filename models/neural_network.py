# Import all necessary external libraries
import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import wandb
from wandb.keras import WandbCallback
import datetime

# Import file that contains parameters
import parameters

# Project is currently configured for WandB usage, so set up WandB
wandb.init(project="Sentimental Smacking", entity='jbettencourt')

# Commented settings for TensorBoard usage
# log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0)

# Initialize hyperparameters for WandB to log
wandb.config = {"epochs": parameters.EPOCHS, "batch_size": parameters.BATCH_SIZE}


# Class that represents the recurrent neural network
# Includes a constructor that initializes the network
# Also includes train method that trains the network and gets the accuracy on training set
class RNN:
    # Constructor that takes in list of tweets and list of labels and initializes the network
    def __init__(self, tweets: list, labels: list) -> None:

        # Save tweets and labels as object attributes
        self.tweets = tweets
        self.labels = labels

        # Create vocabulary from tweets with specified VOCAB_SIZE
        self.encoder = TextVectorization(max_tokens=parameters.VOCAB_SIZE)
        self.encoder.adapt(self.tweets)

        # Create recurrent neural network
        # Layer 1: Embedding layer
        # Layer 2: LSTM layer
        # Layer 3: ReLU layer
        # Layer 4: Dense Layer (output layer)
        self.model = tf.keras.Sequential([
        self.encoder,
        tf.keras.layers.Embedding(
            input_dim=len(self.encoder.get_vocabulary()),
            output_dim=64,
            mask_zero=True),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)])

    # Method that trains the network and gets the accuracy on training set
    def train(self) -> None:
        self.model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'])
        self.model.fit(self.tweets, self.labels, batch_size=parameters.BATCH_SIZE, epochs=parameters.EPOCHS, callbacks=[WandbCallback()])
