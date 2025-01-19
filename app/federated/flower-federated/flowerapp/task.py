import os
import keras
from sklearn.model_selection import train_test_split
from keras import layers
import pandas as pd
import numpy as np

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

def load_model(learning_rate: float = 0.001, input_dim: int = None):
    model = keras.Sequential(
        [
            keras.Input(shape=(input_dim,)),
            layers.Dense(64, activation="relu"),
            layers.Dense(32, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(1, activation="sigmoid"),  # For binary classification
        ]
    )
    optimizer = keras.optimizers.Adam(learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",  # Use an appropriate loss for your task
        metrics=["accuracy"],
    )
    return model


def load_data(partition_id, num_partitions):

    data = pd.read_csv('./../LLCP2022_filtered.csv')
    data = data.astype(np.float32)

    data['has_diabetes'] = data['has_diabetes'].replace({2.0: 0.0})
    data['has_diabetes'] = data['has_diabetes'].astype(int)

    y = data['has_diabetes']
    X = data.drop('has_diabetes', axis=1)

    # Partition the data manually based on the partition_id and num_partitions
    partition_size = len(X) // num_partitions
    start_idx = partition_id * partition_size
    end_idx = (partition_id + 1) * partition_size if partition_id < num_partitions - 1 else len(X)

        # Get the partitioned data
    x_partition = X[start_idx:end_idx]
    y_partition = y[start_idx:end_idx]

        # Split the partition into train and test (80% train, 20% test)
    x_train, x_test, y_train, y_test = train_test_split(x_partition, y_partition, test_size=0.2, random_state=42)
    return x_train, x_test, y_train, y_test