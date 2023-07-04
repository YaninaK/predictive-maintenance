import os
import logging
import pandas as pd
import numpy as np
import joblib
from typing import Optional
import boto3


logger = logging.getLogger(__name__)

__all__ = ["save_artifacts"]


PATH = ""

FOLDER_3 = "data/03_primary/"
ETALON_DATASET_PATH = "etalon_dataset.npy"
SCALER_LSTM_PATH = "scaler_LSTM.joblib"
PCA_PATH = "pca.joblib"

FOLDER_4 = "data/04_feature/"
SCALER_BMB_PATH = "scaler_bmb.joblib"

FOLDER_5 = "data/05_model_input/"
T2_Q_FROM_LSTM_ETALON_PATH = "T2_Q_from_LSTM_etalon.parquet"

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")


def save_etalon_dataset(
    etalon_dataset,
    path: Optional[str] = None,
    folder: Optional[str] = None,
    etalon_dataset_path: Optional[str] = None,
):
    if path is None:
        path = PATH
    if folder is None:
        folder = FOLDER_3
    if etalon_dataset_path is None:
        etalon_dataset_path = path + folder + ETALON_DATASET_PATH

    np.save(etalon_dataset_path, etalon_dataset)


def save_scaler_LSTM(
    scaler_LSTM,
    path: Optional[str] = None,
    folder: Optional[str] = None,
    scaler_LSTM_path: Optional[str] = None,
):
    if path is None:
        path = PATH
    if folder is None:
        folder = FOLDER_3
    if scaler_LSTM_path is None:
        scaler_LSTM_path = path + folder + SCALER_LSTM_PATH

    joblib.dump(scaler_LSTM, scaler_LSTM_path)


def save_pca(
    pca,
    path: Optional[str] = None,
    folder: Optional[str] = None,
    pca_path: Optional[str] = None,
):
    if path is None:
        path = PATH
    if folder is None:
        folder = FOLDER_3
    if pca_path is None:
        pca_path = path + folder + PCA_PATH

    joblib.dump(pca, pca_path)


def save_scaler_bmb(
    scaler_bmb,
    path: Optional[str] = None,
    folder: Optional[str] = None,
    scaler_bmb_path: Optional[str] = None,
):
    if path is None:
        path = PATH
    if folder is None:
        folder = FOLDER_4
    if scaler_bmb_path is None:
        scaler_bmb_path = path + folder + SCALER_BMB_PATH

    joblib.dump(scaler_bmb, scaler_bmb_path)


def save_T2_Q_from_LSTM_etalon(
    T2_Q_from_LSTM_etalon: pd.DataFrame,
    path: Optional[str] = None,
    folder: Optional[str] = None,
    T2_Q_from_LSTM_etalon_path: Optional[str] = None,
):
    if path is None:
        path = PATH
    if folder is None:
        folder = FOLDER_5
    if T2_Q_from_LSTM_etalon_path is None:
        T2_Q_from_LSTM_etalon_path = path + folder + T2_Q_FROM_LSTM_ETALON_PATH

    T2_Q_from_LSTM_etalon.to_parquet(T2_Q_from_LSTM_etalon_path, compression="gzip")


def save_to_YC_s3(
    bucket, path="", file_name=None, put_object=None, folders=None, s3_path=""
):
    session = boto3.session.Session()
    if (AWS_ACCESS_KEY_ID is None) | (AWS_SECRET_ACCESS_KEY is None):
        s3 = session.client(
            service_name="s3", endpoint_url="https://storage.yandexcloud.net"
        )
    else:
        s3 = session.client(
            service_name="s3",
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name="eu-west-1",
            endpoint_url="https://storage.yandexcloud.net",
        )
    if file_name:
        if put_object:
            s3.put_object(Body=put_object, Bucket=bucket, Key=file_name)
        else:
            s3.upload_file(path + file_name, bucket, s3_path + file_name)

    if folders:
        if type(folders) != list:
            folders = [folders]
        for folder in folders:
            files = os.listdir(path + folder)
            for f in files:
                if f != ".gitkeep":
                    s3.upload_file(path + folder + f, bucket, s3_path + folder + f)
