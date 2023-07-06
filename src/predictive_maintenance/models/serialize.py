import os
import logging
import tensorflow as tf
import arviz as az
import joblib
from typing import Optional


logger = logging.getLogger()

__all__ = ["store", "load"]


def store(
    model,
    filename: str,
    path: str = "default",
    model_type: Optional[str] = None,
):
    if path == "default":
        path = models_path()
    filepath = os.path.join(path, filename)

    if model_type == "tf":
        filepath += ".h5"
        logger.info(f"Dumping model into {filepath}")
        model.save(filepath)
    elif model_type == "bmb":
        filepath += ".h5netcdf"
        logger.info(f"Dumping model into {filepath}")
        az.InferenceData.to_netcdf(model, filepath)
    else:
        filepath += ".joblib"
        logger.info(f"Dumping model into {filepath}")
        joblib.dump(model, filepath)


def load(
    filename: str,
    path: str = "default",
    model_type: Optional[str] = None,
):
    if path == "default":
        path = models_path()
    filepath = os.path.join(path, filename)

    if model_type == "tf":
        filepath += ".h5"
        logger.info(f"Loading model from {filepath}")
        return tf.keras.models.load_model(filepath)

    elif model_type == "bmb":
        filepath += ".h5netcdf"
        logger.info(f"Loading model from {filepath}")
        return az.InferenceData.from_netcdf(filepath)

    else:
        filepath += ".joblib"
        logger.info(f"Loading model from {filepath}")
        return joblib.load(filepath)


def models_path() -> str:
    script_path = os.path.abspath(__file__)
    script_dir_path = os.path.dirname(script_path)
    models_folder = os.path.join(script_dir_path, "..", "..", "..", "models")

    return models_folder
