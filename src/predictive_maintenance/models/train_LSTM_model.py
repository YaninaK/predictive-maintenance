import logging
import tensorflow as tf
from typing import Optional


logger = logging.getLogger(__name__)

__all__ = ["train_LSTM_model"]


INPUT_SEQUENCE_LENGTH = 23
N_EPOCHS = 60
BATCH_SIZE = 64
N_VALID = 1024


def train_LSTM(
    model,
    ethalon_dataset,
    input_sequence_length: Optional[int] = None,
    n_epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    n_valid: Optional[int] = None,
):
    if input_sequence_length is None:
        input_sequence_length = INPUT_SEQUENCE_LENGTH
    if n_epochs is None:
        n_epochs = N_EPOCHS
    if batch_size is None:
        batch_size = BATCH_SIZE
    if n_valid is None:
        n_valid = N_VALID

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
    return model, history
