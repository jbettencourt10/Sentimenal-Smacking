import numpy as np

# import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization



VOCAB_SIZE = 1000
encoder = TextVectorization(max_tokens=VOCAB_SIZE)
# encoder.adapt(train_dataset.map(lambda text, label: text))
