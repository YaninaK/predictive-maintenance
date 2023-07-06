import logging
import pandas as pd
import pyspark
import pyspark.sql.functions as F
import re
from functools import reduce
from typing import Optional


logger = logging.getLogger(__name__)

__all__ = ["load_train_dataset"]


X_TRAIN_PATH = "data/01_raw/X_train.parquet"
Y_TRAIN_PATH = "data/01_raw/y_train.parquet"
MESSAGES_PATH = "data/01_raw/messages.xlsx"


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
    X_train_path: Optional[str] = None,
    y_train_path: Optional[str] = None,
    messages_path: Optional[str] = None,
):
    """
    Loads X_train, y_train and messages from raw data.
    Renames columns of X_train and y_train to be acceptable for pyspark.
    Cleans X_train from negative numbers, caps max oil temperature at 100
    and max bearings temperature at 800.
    Unifies names of tech places.
    Adds unified names and exhauser number ('equipment) to messages.

    """
    if X_train_path is None:
        X_train_path = X_TRAIN_PATH
    if y_train_path is None:
        y_train_path = Y_TRAIN_PATH
    if messages_path is None:
        messages_path = MESSAGES_PATH

    X_train = spark.read.parquet(X_train_path, header=True, inferSchema=True)
    y_train = spark.read.parquet(y_train_path, header=True, inferSchema=True)
    messages = pd.read_excel(messages_path, index_col=0)

    X_cols = get_new_X_column_names(X_train)
    X_train = rename_columns(X_train, X_cols)
    X_train = clean_data(X_train)

    y_cols = get_new_y_column_names(y_train)
    y_train = rename_columns(y_train, y_cols)

    unified_tech_places = get_unified_tech_places(y_cols)
    messages = add_unified_names_to_messages(messages, unified_tech_places)

    return X_train, y_train, messages, unified_tech_places


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


def clean_data(X):
    """
    Converts negative values to 0.
    Caps max oil temperature at 100.
    Caps max bearings temperature at 800.
    """
    for var in X.schema.names[1:]:
        X = X.withColumn(var, F.when(F.col(var) < 0, 0).otherwise(F.col(var)))
        if var[2:19] == "ТЕМПЕРАТУРА МАСЛА":
            X = X.withColumn(var, F.when(F.col(var) > 100, 100).otherwise(F.col(var)))
        if var[2:24] == "ТЕМПЕРАТУРА ПОДШИПНИКА":
            X = X.withColumn(var, F.when(F.col(var) > 800, 800).otherwise(F.col(var)))

    return X


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


def get_unified_tech_places(
    y_cols: list,
) -> pd.DataFrame:
    """
    Unifies technical places names to enable equipment comparison.
    """
    tech_places = y_cols[1:]
    eq = [i[0] for i in tech_places]
    desc = [i for i in tech_places]

    df = pd.DataFrame(zip(eq, desc), columns=["equipment", "description"])
    for i, name in enumerate(desc):
        for j in range(4, 10):
            if (name[9] == str(j)) & (name[-1] == str(j)):
                df.loc[i, "unified_name"] = name[2:9] + name[10:-1]
            elif name[-5] == str(j):
                df.loc[i, "unified_name"] = name[2:-5] + name[-4:]
            elif name[2:8] == "САПФИР":
                df.loc[i, "unified_name"] = name[2:]
            elif name[2:21] == "ПОДШИПНИК ОПОРНЫЙ №":
                df.loc[i, "unified_name"] = name[2:22]
            elif name[2:-1] in [
                "ТСМТ-101-010-50М-400 ТЕРМОПР_ПОДШ_Т_",
                "ТСМТ-101-010-50М-200 ТЕРМОПР_ПОДШ_Т_",
                "ТСМТ-101-010-50М-80 ТЕРМОПРЕОБР_МАСЛ",
                "ТИРИСТОРНЫЙ ВОЗБУДИТЕЛЬ СПВД-М10-400-",
            ]:
                df.loc[i, "unified_name"] = name[2:]
            elif name[2:20] == "МАСЛОПРОВОДЫ ЭКСГ ":
                df.loc[i, "unified_name"] = "МАСЛОПРОВОДЫ ЭКСГАУСТЕРА №"
            elif name[2:26] == "ЭЛЕКТРООБОРУДОВАНИЯ ЭКСГ":
                df.loc[i, "unified_name"] = "ЭЛЕКТРООБОРУДОВАНИЯ ЭКСГАУСТЕРА №"
            elif name[-1] == str(j):
                df.loc[i, "unified_name"] = name[2:-1]

    return df


def add_unified_names_to_messages(
    messages,
    unified_tech_places: pd.DataFrame,
) -> pd.DataFrame:
    """
    Adds unified technical place names and equipment references to messages
    to match messages to y_train data.
    """
    desc = [i[2:] for i in unified_tech_places["description"]]
    unified_desc = unified_tech_places["unified_name"].tolist()
    dict_ = {desc[i]: unified_desc[i] for i in range(len(unified_desc))}

    for i in messages.index:
        original_name = messages.loc[i, "НАЗВАНИЕ_ТЕХ_МЕСТА"]
        original_name = re.sub("\(", "", original_name)
        original_name = re.sub("\)", "", original_name)
        original_name = re.sub("\.", "_", original_name)
        messages.loc[i, "equipment"] = messages.loc[i, "ИМЯ_МАШИНЫ"][-1]
        messages.loc[i, "unified_name"] = dict_[original_name]

    return messages
