import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization


import parameters




class RNN():
    def __init__(self):
        self.encoder = 

        self.model = tf.keras.Sequential([
        encoder,
        tf.keras.layers.Embedding(
            input_dim=len(encoder.get_vocabulary()),
            output_dim=64,
            mask_zero=True),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)
])


def create_vocabulary(tweets, max_tokens=parameters.VOCAB_SIZE):
    encoder = TextVectorization(max_tokens)
    encoder.adapt(tweets[0])
    return encoder


def compile_model(optimizer, metric, loss):
    pass


# model = tf.keras.Sequential([
#     encoder,
#     tf.keras.layers.Embedding(
#         input_dim=len(encoder.get_vocabulary()),
#         output_dim=64,
#         mask_zero=True),
#     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
#     tf.keras.layers.Dense(64, activation='relu'),
#     tf.keras.layers.Dense(1)
# ])


# model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
#               optimizer=tf.keras.optimizers.Adam(1e-4),
#               metrics=['accuracy'])


# # model.fit(arrays[0], arrays[1], batch_size=BATCH_SIZE, epochs=10)

# history = model.fit(x=arrays[0], y=arrays[1],batch_size=BATCH_SIZE, epochs=150)
