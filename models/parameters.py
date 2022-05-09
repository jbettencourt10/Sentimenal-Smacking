
# Parameters file that is used for changing parameters for the project

# Vocabulary size for the embedding layer
# High values will lead to overfitting
VOCAB_SIZE = 50

# Number of epochs used during training process
# High values will lead to overfitting
EPOCHS = 150

# Batch size used during training process
# Typically set as high as possible (power of 2) without overloading the GPU
# But for smaller datasets, value isn't so important
# 32 was chosen as a safe default
BATCH_SIZE = 32
