import os
import logging
import tensorflow as tf
import arviz as az
import joblib


logger = logging.getLogger()

__all__ = ["store", "load"]


def store_lstm(model, filename: str, path: str = "default"):
    if path == "default":
        path = models_path()

    filepath = os.path.join(path, filename + ".h5")

    logger.info(f"Dumpung LSTM model into {filepath}")
    model.save(filepath)


def load_lstm(filename: str, path: str = "default"):
    if path == "default":
        path = models_path()
    filepath = os.path.join(path, filename + ".h5")

    logger.info(f"Loading LSTM model from {filepath}")

    return tf.keras.models.load_model(filepath)


def store_bmb(model, filename: str, path: str = "default"):
    if path == "default":
        path = models_path()

    filepath = os.path.join(path, filename + ".h5netcdf")

    logger.info(f"Dumpung Bayesian model into {filepath}")
    az.InferenceData.to_netcdf(model, filepath)


def load_bmb(filename: str, path: str = "default"):
    if path == "default":
        path = models_path()
    filepath = os.path.join(path, filename + ".h5netcdf")

    logger.info(f"Loading Bayesian model from {filepath}")

    return az.InferenceData.from_netcdf(filepath)


def store(model, filename: str, path: str = "default"):
    if path == "default":
        path = models_path()

    filepath = os.path.join(path, filename + ".joblib")

    logger.info(f"Dumpung model into {filepath}")
    joblib.dump(model, filepath)


def load(filename: str, path: str = "default"):
    if path == "default":
        path = models_path()
    filepath = os.path.join(path, filename + ".joblib")

    logger.info(f"Loading model from {filepath}")

    return joblib.load(filepath)


def models_path() -> str:
    script_path = os.path.abspath(__file__)
    script_dir_path = os.path.dirname(script_path)
    models_folder = os.path.join(script_dir_path, "..", "..", "..", "models")

    return models_folder
