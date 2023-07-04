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
from data.etalon_periods import generate_etalon_dataset
from models.model_LSTM import get_model_LSTM
from models.train_LSTM_model import train_LSTM
from features.build_dataset import get_T2_Q_from_LSTM_etalon
from .save_artifacts import (
    save_etalon_dataset,
    save_scaler_LSTM,
    save_pca,
    save_scaler_bmb,
    save_T2_Q_from_LSTM_etalon,
    save_to_YC_s3,
)


logger = logging.getLogger(__name__)

__all__ = ["preprocess_data"]


PATH = ""
RESAMPLE_PERIOD = 3600  # 1 hour
INPUT_SEQUENCE_LENGTH = 23

SAVE_ARTIFACTS = True

S3_PATH = "data/"
FOLDERS = ["02_intermediate/", "03_primary/", "04_feature/", "05_model_input/"]
FEATURE_STORE = "predictive-maintenance-feature-store"


def data_preprocessing_pipeline(
    path: Optional[str] = None,
    resample_period: Optional[int] = None,
    input_sequence_length: Optional[int] = None,
    save_artifacts: Optional[bool] = None,
) -> pd.DataFrame:

    if path is None:
        path = PATH
    if resample_period is None:
        resample_period = RESAMPLE_PERIOD
    if input_sequence_length is None:
        input_sequence_length = INPUT_SEQUENCE_LENGTH
    if save_artifacts is None:
        save_artifacts = SAVE_ARTIFACTS

    logging.info("Loading data...")

    X_train, _, messages, _ = load_data()

    logging.info("Resampling data...")

    save_resampled_X(X_train, period=resample_period, path=path)

    logging.info("Generating etalon dataset...")

    etalon_dataset, scaler_LSTM, pca = generate_etalon_dataset(messages)

    model_LSTM = get_model_LSTM()
    model_LSTM = train_LSTM(
        model_LSTM,
        etalon_dataset,
        input_sequence_length,
        path=path,
    )
    logging.info("Generating dataset for Bayesian model...")

    T2_Q_from_LSTM_etalon, scaler_bmb = get_T2_Q_from_LSTM_etalon(
        model_LSTM, etalon_dataset
    )

    if save_artifacts:
        logging.info("Saving artifacts ...")

        save_etalon_dataset(etalon_dataset)
        save_scaler_LSTM(scaler_LSTM)
        save_pca(pca)
        save_scaler_bmb(scaler_bmb)
        save_T2_Q_from_LSTM_etalon(T2_Q_from_LSTM_etalon)

        logging.info("Saving artifacts in Feature store in Yandex Object Storage...")

        save_to_YC_s3(FEATURE_STORE, S3_PATH, folders=FOLDERS)

    return T2_Q_from_LSTM_etalon
