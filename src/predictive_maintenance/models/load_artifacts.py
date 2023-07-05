import logging
import joblib
import pandas as pd
import tensorflow as tf
from typing import Optional, Tuple


logger = logging.getLogger(__name__)

__all__ = ["load_inference_artifacts"]


PATH = ""
FOLDER = "data/04_feature/"
SCALER_LSTM_PATH = "scaler_lstm.joblib"
PCA_PATH = "pca.joblib"
FITTED_LSTM_MODEL_PATH = "LSTM_model.h5"
SCALER_BMB_PATH = "scaler_bmb.joblib"


def load_inference_artifacts(
    path: Optional[str] = None,
    folder: Optional[str] = None,
    scaler_lstm_path: Optional[str] = None,
    pca_path: Optional[str] = None,
    LSTM_model_path: Optional[str] = None,
    scaler_bmb_path: Optional[str] = None,
) -> Tuple:
    """
    Loads fitted scaler_LSTM, pca, LSTM_model, scaler_bmb for inference.
    """
    if path is None:
        path = PATH
    if folder is None:
        folder = FOLDER
    if scaler_lstm_path is None:
        scaler_lstm_path = path + folder + SCALER_LSTM_PATH
    if pca_path is None:
        pca_path = path + folder + PCA_PATH
    if LSTM_model_path is None:
        LSTM_model_path = path + folder + FITTED_LSTM_MODEL_PATH
    if scaler_bmb_path is None:
        scaler_bmb_path = path + folder + SCALER_BMB_PATH

    logging.info("Loading inference artifacts...")

    scaler_lstm = joblib.load(scaler_lstm_path)
    pca = joblib.load(pca_path)
    LSTM_model = tf.keras.models.load_model(LSTM_model_path)
    scaler_bmb = joblib.load(scaler_bmb_path)

    return (scaler_lstm, pca, LSTM_model, scaler_bmb)
