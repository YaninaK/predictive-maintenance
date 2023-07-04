import logging
import tensorflow as tf
from typing import Optional
import matplotlib.pyplot as plt


logger = logging.getLogger(__name__)

__all__ = ["train_LSTM_model"]


INPUT_SEQUENCE_LENGTH = 23
N_EPOCHS = 60
BATCH_SIZE = 64
N_VALID = 1024
PLOT_HISTORY = False


def train_LSTM(
    model,
    ethalon_dataset,
    input_sequence_length: Optional[int] = None,
    n_epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    n_valid: Optional[int] = None,
    plot_history: Optional[bool] = None,
):
    if input_sequence_length is None:
        input_sequence_length = INPUT_SEQUENCE_LENGTH
    if n_epochs is None:
        n_epochs = N_EPOCHS
    if batch_size is None:
        batch_size = BATCH_SIZE
    if n_valid is None:
        n_valid = N_VALID
    if plot_history is None:
        plot_history = PLOT_HISTORY

    reduce_lr = tf.keras.callbacks.LearningRateScheduler(
        lambda epoch: 3e-2 * 0.95**epoch
    )
    history = model.fit(
        ethalon_dataset[:-n_valid, :input_sequence_length, :],
        ethalon_dataset[:-n_valid, input_sequence_length:, :],
        epochs=n_epochs,
        validation_data=(
            ethalon_dataset[-n_valid:, :input_sequence_length, :],
            ethalon_dataset[-n_valid:, input_sequence_length:, :],
        ),
        batch_size=batch_size,
        verbose=1,
        callbacks=[reduce_lr],
        shuffle=True,
        workers=-1,
        use_multiprocessing=True,
    )
    if plot_history:
        plot_model_LSTM_training_history(history)

    return model


def plot_model_LSTM_training_history(history):
    plt.figure(figsize=(6, 4))
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("Model Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(["Train", "Valid"])
    plt.show()
