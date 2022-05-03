import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import wandb
from wandb.keras import WandbCallback
import datetime

import parameters

# wandb.init(project="Sentimental Smacking", entity='jbettencourt')

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0)

wandb.config = {"epochs": parameters.EPOCHS, "batch_size": parameters.BATCH_SIZE}


class RNN:
    def __init__(self, tweets: list, labels: list) -> None:

        self.tweets = tweets
        self.labels = labels

        self.encoder = TextVectorization(max_tokens=parameters.VOCAB_SIZE)
        self.encoder.adapt(self.tweets)


        self.model = tf.keras.Sequential([
        self.encoder,
        tf.keras.layers.Embedding(
            input_dim=len(self.encoder.get_vocabulary()),
            output_dim=64,
            mask_zero=True),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)])

    def train(self) -> None:
        self.model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'])
        self.model.fit(self.tweets, self.labels, batch_size=parameters.BATCH_SIZE, epochs=parameters.EPOCHS, callbacks=[tensorboard_callback])
