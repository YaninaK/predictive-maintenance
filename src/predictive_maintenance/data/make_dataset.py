import logging
import pandas as pd
import pyspark
import re
from functools import reduce
from typing import Optional


logger = logging.getLogger(__name__)

__all__ = ["load_train_dataset"]

PATH = "data/01_raw/"
X_TRAIN_PATH = PATH + "X_train.parquet"
Y_TRAIN_PATH = PATH + "y_train.parquet"
MESSAGES_PATH = PATH + "messages.xlsx"


app_name = "data_preprocessing"
spark_ui_port = 4041

spark = (
    pyspark.sql.SparkSession.builder.appName(app_name)
    .master("local[4]")
    .config("spark.executor.memory", "15g")
    .config("spark.driver.memory", "15g")
    .config("spark.ui.port", spark_ui_port)
    .getOrCreate()
)


def load_data(
    path: str,
    X_train_path: Optional[str] = None,
    y_train_path: Optional[str] = None,
    messages_path: Optional[str] = None,
):
    if X_train_path is None:
        X_train_path = path + X_TRAIN_PATH
    if y_train_path is None:
        y_train_path = path + Y_TRAIN_PATH
    if messages_path is None:
        messages_path = path + MESSAGES_PATH

    X_train = spark.read.parquet(X_train_path, header=True, inferSchema=True)
    y_train = spark.read.parquet(y_train_path, header=True, inferSchema=True)
    messages = pd.read_excel(messages_path, index_col=0)

    unified_tech_places = get_unified_tech_places(y_train)
    messages = add_unified_names_to_messages(messages, unified_tech_places)

    X_cols = get_new_X_column_names(X_train)
    X_train = rename_columns(X_train, X_cols)
    y_cols = get_new_y_column_names(y_train)
    y_train = rename_columns(y_train, y_cols)

    return X_train, y_train, messages, unified_tech_places


def get_unified_tech_places(y_train) -> pd.DataFrame:
    tech_places = y_train.schema.names[1:]
    eq = [i.split("_")[1][-1] for i in tech_places]
    desc = [i.split("_")[2] for i in tech_places]

    df = pd.DataFrame(zip(eq, desc), columns=["equipment", "description"])
    for i, name in enumerate(desc):
        for j in range(4, 10):
            if (name[7] == str(j)) & (name[-1] == str(j)):
                df.loc[i, "unified_name"] = name[:7] + name[8:-1]
            elif name[-5] == str(j):
                df.loc[i, "unified_name"] = name[:-5] + name[-4:]
            elif name[:6] == "САПФИР":
                df.loc[i, "unified_name"] = name
            elif name[:19] == "ПОДШИПНИК ОПОРНЫЙ №":
                df.loc[i, "unified_name"] = name[:20]
            elif name[:-1] in [
                "ТСМТ-101-010-50М-400 ТЕРМОПР.ПОДШ.Т.",
                "ТСМТ-101-010-50М-200 ТЕРМОПР.ПОДШ.Т.",
                "ТСМТ-101-010-50М-80 ТЕРМОПРЕОБР.МАСЛ",
                "ТИРИСТОРНЫЙ ВОЗБУДИТЕЛЬ СПВД-М10-400-",
            ]:
                df.loc[i, "unified_name"] = name
            elif name[:18] == "МАСЛОПРОВОДЫ ЭКСГ ":
                df.loc[i, "unified_name"] = "МАСЛОПРОВОДЫ ЭКСГАУСТЕРА №"
            elif name[-1] == str(j):
                df.loc[i, "unified_name"] = name[:-1]

    return df


def add_unified_names_to_messages(messages, unified_tech_places) -> pd.DataFrame:
    """
    Adds unified technical place names and equipment references to messages.
    """
    desc = unified_tech_places["description"].tolist()
    unified_desc = unified_tech_places["unified_name"].tolist()
    dict_ = {desc[i]: unified_desc[i] for i in range(len(unified_desc))}

    for i in messages.index:
        original_name = messages.loc[i, "НАЗВАНИЕ_ТЕХ_МЕСТА"]
        messages.loc[i, "equipment"] = messages.loc[i, "ИМЯ_МАШИНЫ"][-1]
        messages.loc[i, "unified_name"] = dict_[original_name]

    return messages


def get_new_X_column_names(X_train) -> list:
    """
    Generates unified column names.
    """
    cols = X_train.schema.names
    new_cols = [cols[0]]
    for col in cols[1:]:
        col = re.sub("\.", "", col[11:])
        col = re.sub("ТОК РОТОРА2", "ТОК РОТОРА 2", col)
        new_cols.append(col)

    return new_cols


def get_new_y_column_names(y_train) -> list:
    """
    Generates new y_column_names.
    """
    cols = y_train.schema.names
    new_cols = [cols[0]]
    for col in cols[1:]:
        col = re.sub("\(", "", col[18:])
        col = re.sub("\)", "", col)
        col = re.sub("\.", "_", col)
        new_cols.append(col)

    return new_cols


def rename_columns(df, new_colums: list):
    """
    Renames columns of pyspark DataFrame.
    """
    old_columns = df.schema.names

    return reduce(
        lambda data, idx: data.withColumnRenamed(old_columns[idx], new_colums[idx]),
        range(len(old_columns)),
        df,
    )
