import numpy as np

# import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.keras import layers



VOCAB_SIZE = 1000
encoder = layers.TextVectorization(max_tokens=VOCAB_SIZE)
# encoder.adapt(train_dataset.map(lambda text, label: text))
