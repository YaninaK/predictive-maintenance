import logging
import pandas as pd
import numpy as np
import pyspark
from pyspark.sql import functions as F
from typing import Optional


logger = logging.getLogger(__name__)

__all__ = ["resample_and_save_data"]

RESAMPLE_PERIOD = 60 * 30
PATH = ""
FOLDER = "data/02_intermediate/"


app_name = "data_resampling"
spark_ui_port = 4041

spark = (
    pyspark.sql.SparkSession.builder.appName(app_name)
    .master("local[4]")
    .config("spark.executor.memory", "15g")
    .config("spark.driver.memory", "15g")
    .config("spark.ui.port", spark_ui_port)
    .getOrCreate()
)


def save_resampled_X(
    X,
    prefix="X_train",
    period: Optional[int] = None,
    path: Optional[str] = None,
    folder: Optional[str] = None,
):
    """
    Resamples X_train or X_test into larger periods, splits it into 6 parts
    by equipment and saves artifacts for further use.
    """
    if period is None:
        period = RESAMPLE_PERIOD
    if path is None:
        path = PATH
    if folder is None:
        folder = FOLDER

    X_cols = get_equipment_columns(X.schema.names)

    for i in range(4, 10):
        (
            resample(X.select(X_cols[i]), period)
            .groupBy("dt_resampled")
            .mean()
            .orderBy("dt_resampled")
            .write.parquet(
                path + folder + prefix + f"{i}_mean_resampled.parquet",
                compression="gzip",
            )
        )


def save_resampled_y_train(
    y_train,
    period: Optional[int] = None,
    path: Optional[str] = None,
    folder: Optional[str] = None,
):
    """
    Resamples y_train into larger periods, splits it into 6 parts by equipment
    and saves artifacts for further use.
    """
    if period is None:
        period = RESAMPLE_PERIOD
    if path is None:
        path = PATH
    if folder is None:
        folder = FOLDER

    y_cols = get_equipment_columns(y_train.schema.names)

    for i in range(4, 10):
        (
            resample(y_train.select(y_cols[i]), period)
            .groupBy("dt_resampled")
            .max()
            .orderBy("dt_resampled")
            .write.parquet(
                path + folder + f"y{i}_resampled.parquet",
                compression="gzip",
            )
        )


def get_equipment_columns(cols: list) -> dict:
    """
    Selects columns related to particular equipment.
    """
    cols_dict = {}
    for i in range(4, 10):
        cols_dict[i] = [cols[0]]
        cols_list = []
        for j in cols[1:]:
            if j[0] == str(i):
                cols_list.append(j)
        cols_dict[i] = cols_dict[i] + sorted(cols_list)

    return cols_dict


def resample(
    df,
    period: Optional[int] = None,
):
    """
    Resamples pyspark DataFrame into larger periods.
    """
    if period is None:
        period = RESAMPLE_PERIOD

    epoch = (F.col("DT").cast("bigint") / period).cast("bigint") * period
    with_epoch = df.withColumn("epoch", epoch)

    min_epoch, max_epoch = with_epoch.select(F.min("epoch"), F.max("epoch")).first()
    ref = spark.range(min_epoch, max_epoch + 1, period).toDF("epoch")

    return ref.join(with_epoch, "epoch", "left").withColumn(
        "dt_resampled", F.timestamp_seconds("epoch")
    )
