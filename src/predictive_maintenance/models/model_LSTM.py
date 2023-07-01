import logging
import tensorflow as tf
from typing import Optional

__all__ = ["model_architecture"]

logger = logging.getLogger()


N_FEATURES = 5
N_UNITS = 150
INPUT_SEQUENCE_LENGTH = 23
OUTPUT_SEQUENCE_LENGTH = 27


def get_model_LSTM(
    n_features: Optional[int] = None,
    n_units: Optional[int] = None,
    input_sequence_length: Optional[int] = None,
    output_sequence_length: Optional[int] = None,
):
    if n_features is None:
        n_features = N_FEATURES
    if n_units is None:
        n_units = N_UNITS
    if input_sequence_length is None:
        input_sequence_length = INPUT_SEQUENCE_LENGTH
    if output_sequence_length is None:
        output_sequence_length = OUTPUT_SEQUENCE_LENGTH

    encoder_inputs = tf.keras.layers.Input(
        shape=(input_sequence_length, n_features), name="encoder_inputs"
    )
    encoder_output1, state_h1, state_c1 = tf.keras.layers.LSTM(
        n_units, return_sequences=True, return_state=True, name="encoder_output1"
    )(encoder_inputs)
    encoder_states1 = [state_h1, state_c1]

    encoder_output2, state_h2, state_c2 = tf.keras.layers.LSTM(
        n_units, return_state=True, name="encoder_output2"
    )(encoder_output1)
    encoder_states2 = [state_h2, state_c2]

    decoder_inputs = tf.keras.layers.RepeatVector(
        output_sequence_length, name="decoder_inputs"
    )(encoder_output2)

    decoder_l1 = tf.keras.layers.LSTM(
        n_units, return_sequences=True, name="decoder_l1"
    )(decoder_inputs, initial_state=encoder_states1)
    decoder_l2 = tf.keras.layers.LSTM(
        n_units, return_sequences=True, name="decoder_l2"
    )(decoder_l1, initial_state=encoder_states2)

    decoder_outputs2 = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(n_features, name="decoder_outputs2"),
        name="decoder_outputs",
    )(decoder_l2)

    model = tf.keras.models.Model(encoder_inputs, decoder_outputs2, name="lstm_model")
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.Huber())
    
    return model
