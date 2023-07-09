#!/usr/bin/env python3
"""Train and save model for predictive-maintenance"""

import sys
import os

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), ".."))

import logging
import argparse
import pandas as pd
import tensorflow as tf
import bambi as bmb
from typing import Optional

from src.predictive_maintenance.data.make_dataset import load_data
from src.predictive_maintenance.models import train
from src.predictive_maintenance.models.model_LSTM import get_model_LSTM
from src.predictive_maintenance.models.serialize import store
from src.predictive_maintenance.models.save_artifacts import save_to_YC_s3


SEED = 25
FREQUENCY = pd.Timedelta("1H")
INPUT_SEQUENCE_LENGTH = 23
OUTPUT_SEQUENCE_LENGTH = 27

FEATURE_STORE = "predictive-maintenance-feature-store"
S3_PATH = "data/"
FOLDERS = ["02_intermediate/", "03_primary/", "04_feature/", "05_model_input/"]


logger = logging.getLogger()


def main():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        "-d1",
        "--data_path",
        required=False,
        default="data/01_raw/X_train.parquet",
        help="telemetry dataset store path",
    )
    argparser.add_argument(
        "-d2",
        "--labels_path",
        required=False,
        default="data/01_raw/y_train.parquet",
        help="labeled technical places store path",
    )
    argparser.add_argument(
        "-d3",
        "--messages_path",
        required=False,
        default="data/01_raw/messages.xlsx",
        help="messages store path",
    )
    argparser.add_argument(
        "-o1",
        "--lstm_output",
        required=True,
        help="filename to store LSTM model",
    )
    argparser.add_argument(
        "-o2",
        "--bmb_output",
        required=True,
        help="filename to store Bayesian model",
    )
    argparser.add_argument(
        "-s",
        "--save_to_YC_s3",
        help="save artifacts to Yandex Object Storage",
        action="store_true",
    )
    argparser.add_argument(
        "-v", "--verbose", help="increase output verbosity", action="store_true"
    )
    args = argparser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    logging.info("Reading data...")

    X_train, _, messages, unified_tech_places = load_data(
        args.data_path, args.labels_path, args.messages_path
    )
    logging.info("Preprocessing data for LSTM model...")
    etalon_dataset = train.lstm_data_preprocessing_pipeline(X_train, messages)
    logging.info("Training LSTM model...")
    model_LSTM = train_store_lstm(etalon_dataset, args.lstm_output)

    logging.info("Generating dataset for Bayesian model...")
    T2_Q_from_LSTM_etalon = train.bmb_data_preprocessing_pipeline(
        model_LSTM, etalon_dataset
    )
    logging.info("Training Bayesian model...")
    train_store_bmb(T2_Q_from_LSTM_etalon, args.bmb_output)

    if args.save_to_s3:
        logging.info("Saving artifacts in Feature store in Yandex Object Storage...")
        save_to_YC_s3(FEATURE_STORE, S3_PATH, folders=FOLDERS)


def train_store_lstm(
    etalon_dataset,
    filename: str,
    input_sequence_length: int = 23,
    n_valid: int = 1024,
    n_epochs: int = 60,
    batch_size: int = 64,
):
    """
    Trains and stores LSTM model.
    """
    model = get_model_LSTM()
    reduce_lr = tf.keras.callbacks.LearningRateScheduler(
        lambda epoch: 3e-2 * 0.95**epoch
    )
    n = n_valid
    t = input_sequence_length
    model.fit(
        etalon_dataset[:-n, :t, :],
        etalon_dataset[:-n, t:, :],
        epochs=n_epochs,
        validation_data=(etalon_dataset[-n:, :t, :], etalon_dataset[-n:, t:, :]),
        batch_size=batch_size,
        verbose=1,
        callbacks=[reduce_lr],
        shuffle=True,
        workers=-1,
        use_multiprocessing=True,
    )
    store(model, filename, model_type="tf")

    return model


def train_store_bmb(T2_Q_from_LSTM_etalon: pd.DataFrame, filename: str):
    """
    Trains and stores Bayesian model.
    """
    bmb_model = bmb.Model(
        "good ~ T2 + Q", data=T2_Q_from_LSTM_etalon, family="bernoulli"
    )
    model_fitted = bmb_model.fit(
        draws=2000,
        target_accept=0.85,
        random_seed=SEED,
        idata_kwargs={"log_likelihood": True},
    )
    store(model_fitted, filename, model_type="bmb")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.critical(e)
        sys.exit(1)
