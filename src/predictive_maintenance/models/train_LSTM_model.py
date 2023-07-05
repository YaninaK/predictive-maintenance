import logging
import pickle
import tensorflow as tf
from typing import Optional
import matplotlib.pyplot as plt


logger = logging.getLogger(__name__)

__all__ = ["train_LSTM_model"]


INPUT_SEQUENCE_LENGTH = 23
N_EPOCHS = 60
BATCH_SIZE = 64
N_VALID = 1024

PATH = ""
FOLDER = "data/07_reporting/"
TRAINING_HISTORY_PATH = "LSTM_history.pkl"


def train_LSTM(
    model,
    etalon_dataset,
    input_sequence_length: Optional[int] = None,
    n_epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    n_valid: Optional[int] = None,
    path: Optional[str] = None,
    folder: Optional[str] = None,
    training_history_path: Optional[str] = None,
):
    if input_sequence_length is None:
        input_sequence_length = INPUT_SEQUENCE_LENGTH
    if n_epochs is None:
        n_epochs = N_EPOCHS
    if batch_size is None:
        batch_size = BATCH_SIZE
    if n_valid is None:
        n_valid = N_VALID
    if path is None:
        path = PATH
    if folder is None:
        folder = FOLDER
    if training_history_path is None:
        training_history_path = path + folder + TRAINING_HISTORY_PATH

    reduce_lr = tf.keras.callbacks.LearningRateScheduler(
        lambda epoch: 3e-2 * 0.95**epoch
    )

    history = model.fit(
        etalon_dataset[:-n_valid, :input_sequence_length, :],
        etalon_dataset[:-n_valid, input_sequence_length:, :],
        epochs=n_epochs,
        validation_data=(
            etalon_dataset[-n_valid:, :input_sequence_length, :],
            etalon_dataset[-n_valid:, input_sequence_length:, :],
        ),
        batch_size=batch_size,
        verbose=1,
        callbacks=[reduce_lr],
        shuffle=True,
        workers=-1,
        use_multiprocessing=True,
    )

    logging.info("Saving training history...")

    with open(training_history_path, "wb") as f:
        pickle.dump(history.history, f)

    return model


def plot_model_LSTM_training_history(
    path: Optional[str] = None,
    folder: Optional[str] = None,
    training_history_path: Optional[str] = None,
):
    if path is None:
        path = PATH
    if folder is None:
        folder = FOLDER
    if training_history_path is None:
        training_history_path = path + folder + TRAINING_HISTORY_PATH

    with open(training_history_path, "rb") as f:
        history = pickle.load(f)

    plt.figure(figsize=(6, 4))
    plt.plot(history["loss"])
    plt.plot(history["val_loss"])
    plt.title("Model Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(["Train", "Valid"])
    plt.show()
