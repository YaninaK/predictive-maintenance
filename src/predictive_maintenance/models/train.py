import sys
import os

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "src", "predictive_maintenance"))

import logging
import pandas as pd
import numpy as np
from typing import Optional

from data.make_dataset import load_data
from data.resample_dataset import save_resampled_X
from data.etalon_periods import (
    select_etalon_periods,
    get_pca_components,
    generate_etalon_dataset,
)
from models.model_LSTM import get_model_LSTM
from models.train_LSTM_model import train_LSTM
from features.build_dataset import get_T2_Q_from_LSTM_etalon
from .save_artifacts import (
    save_unified_tech_places,
    save_messages,
    save_etalon_dataset,
    save_scaler_lstm,
    save_pca,
    save_fitted_LSTM_model,
    save_scaler_bmb,
    save_T2_Q_from_LSTM_etalon,
    save_to_YC_s3,
)


logger = logging.getLogger(__name__)

__all__ = ["preprocess_data"]


PATH = ""
FREQUENCY = pd.Timedelta("1H")

N_FEATURES = 5
N_UNITS = 150
INPUT_SEQUENCE_LENGTH = 23
OUTPUT_SEQUENCE_LENGTH = 27

SAVE_ARTIFACTS = True

S3_PATH = "data/"
FOLDERS = ["02_intermediate/", "03_primary/", "04_feature/", "05_model_input/"]
FEATURE_STORE = "predictive-maintenance-feature-store"


def data_preprocessing_pipeline(
    path: Optional[str] = None,
    freq: Optional[pd.Timedelta] = None,
    n_features: Optional[int] = None,
    n_units: Optional[int] = None,
    input_sequence_length: Optional[int] = None,
    output_sequence_length: Optional[int] = None,
    save_artifacts: Optional[bool] = None,
) -> pd.DataFrame:

    if path is None:
        path = PATH
    if freq is None:
        freq = FREQUENCY
    if n_features is None:
        n_features = N_FEATURES
    if n_units is None:
        n_units = N_UNITS
    if input_sequence_length is None:
        input_sequence_length = INPUT_SEQUENCE_LENGTH
    if output_sequence_length is None:
        output_sequence_length = OUTPUT_SEQUENCE_LENGTH
    if save_artifacts is None:
        save_artifacts = SAVE_ARTIFACTS

    logging.info("Loading data...")

    X_train, _, messages, unified_tech_places = load_data()

    logging.info("Resampling data...")

    period = freq // pd.Timedelta("1S")
    save_resampled_X(X_train, period=period, path=path)

    logging.info("Generating etalon dataset...")

    time_to_stoppage = pd.Timedelta(input_sequence_length + output_sequence_length, "H")
    etalon_periods = select_etalon_periods(
        messages,
        path,
        freq,
        time_to_stoppage=time_to_stoppage,
    )
    df, scaler_lstm, pca = get_pca_components(etalon_periods)
    etalon_dataset = generate_etalon_dataset(df, freq, time_to_stoppage)

    logging.info("Training LSTM model...")

    model_LSTM = get_model_LSTM(
        n_features,
        n_units,
        input_sequence_length,
        output_sequence_length,
    )
    model_LSTM = train_LSTM(
        model_LSTM,
        etalon_dataset,
        input_sequence_length,
    )

    logging.info("Generating dataset for Bayesian model...")

    T2_Q_from_LSTM_etalon, scaler_bmb = get_T2_Q_from_LSTM_etalon(
        model_LSTM, etalon_dataset, input_sequence_length
    )

    if save_artifacts:
        logging.info("Saving artifacts ...")

        save_unified_tech_places(unified_tech_places),
        save_messages(messages),
        save_etalon_dataset(etalon_dataset)
        save_scaler_lstm(scaler_lstm)
        save_pca(pca)
        save_fitted_LSTM_model(model_LSTM)
        save_scaler_bmb(scaler_bmb)
        save_T2_Q_from_LSTM_etalon(T2_Q_from_LSTM_etalon)

        logging.info("Saving artifacts in Feature store in Yandex Object Storage...")

        save_to_YC_s3(FEATURE_STORE, S3_PATH, folders=FOLDERS)

    return T2_Q_from_LSTM_etalon
