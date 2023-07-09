import sys
import os

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "src", "predictive_maintenance"))

import logging
import pandas as pd
import numpy as np
from sclearn.preprocessing import StandardScaler
from typing import Optional

from data.make_dataset import load_data
from data.resample_dataset import save_resampled_X
from features.etalon_periods import (
    select_etalon_periods,
    get_pca_components,
    generate_etalon_dataset,
)
from .save_artifacts import (
    save_etalon_periods,
    save_etalon_features,
    save_scaler_lstm,
    save_pca,
    save_etalon_dataset,
    save_scaler_bmb,
    save_T2_Q_from_LSTM_etalon,
)


logger = logging.getLogger(__name__)

__all__ = ["preprocess_data"]


PATH = ""
FREQUENCY = pd.Timedelta("1H")
INPUT_SEQUENCE_LENGTH = 23
OUTPUT_SEQUENCE_LENGTH = 27

SAVE_ARTIFACTS_LSTM = True
SAVE_ARTIFACTS_BMB = True


def lstm_data_preprocessing_pipeline(
    X_train,
    messages: pd.DataFrame,
    path: Optional[str] = None,
    freq: Optional[pd.Timedelta] = None,
    input_sequence_length: Optional[int] = None,
    output_sequence_length: Optional[int] = None,
    save: Optional[bool] = None,
):
    if path is None:
        path = PATH
    if freq is None:
        freq = FREQUENCY
    if input_sequence_length is None:
        input_sequence_length = INPUT_SEQUENCE_LENGTH
    if output_sequence_length is None:
        output_sequence_length = OUTPUT_SEQUENCE_LENGTH
    if save is None:
        save = SAVE_ARTIFACTS_LSTM

    logging.info("Resampling data...")

    period = freq // pd.Timedelta("1S")
    save_resampled_X(X_train, period, path)

    logging.info("Generating etalon dataset...")

    etalon_periods = select_etalon_periods(
        messages, path, freq, input_sequence_length, output_sequence_length
    )
    etalon_features, scaler_lstm, pca = get_pca_components(etalon_periods)
    etalon_dataset = generate_etalon_dataset(
        etalon_features, freq, input_sequence_length, output_sequence_length
    )

    if save:
        logging.info("Saving LSTM artifacts ...")

        save_etalon_periods(etalon_periods)
        save_etalon_features(etalon_features)
        save_scaler_lstm(scaler_lstm)
        save_pca(pca)
        save_etalon_dataset(etalon_dataset)

    return etalon_dataset


def bmb_data_preprocessing_pipeline(
    LSTM_model,
    etalon_dataset,
    input_sequence_length: Optional[int] = None,
    save: Optional[bool] = None,
):
    """
    Generates dataset for Bayesian model training.
    """
    if input_sequence_length is None:
        input_sequence_length = INPUT_SEQUENCE_LENGTH
    if save is None:
        save = SAVE_ARTIFACTS_BMB

    pred_etalon = LSTM_model.predict(etalon_dataset[:, :input_sequence_length, :])
    T2_Q_from_LSTM_etalon = pred_etalon[:, :, -2:].reshape(-1, 2)

    scaler_bmb = StandardScaler()
    T2_Q_from_LSTM_etalon = scaler_bmb.fit_transform(T2_Q_from_LSTM_etalon)
    T2_Q_from_LSTM_etalon = pd.DataFrame(T2_Q_from_LSTM_etalon, columns=["T2", "Q"])
    T2_Q_from_LSTM_etalon["good"] = 1

    if save:
        logging.info("Saving artifacts for Bayesian model ...")

        save_scaler_bmb(scaler_bmb)
        save_T2_Q_from_LSTM_etalon(T2_Q_from_LSTM_etalon)

    return T2_Q_from_LSTM_etalon
