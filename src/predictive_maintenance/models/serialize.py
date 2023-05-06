import os
import logging
import tensorflow as tf
import joblib


logger = logging.getLogger()

__all__ = ["store", "load"]


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