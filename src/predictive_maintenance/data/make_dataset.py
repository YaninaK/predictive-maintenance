import logging
import pandas as pd
import pyspark
from functools import reduce
from typing import Optional

logger = logging.getLogger(__name__)

__all__ = ["load_train_dataset"]

PATH = "data/01_raw/"

X_TRAIN_PATH = PATH + "X_train.parquet"
Y_TRAIN_PATH = PATH + "y_train.parquet"
MESSAGES_PATH = PATH + "messages.xlsx"

COLUMN_NAMES = [
    "rotor_current_1",
    "rotor_current_2",
    "stator_current",
    "oil_pressure_in_the_system",
    "bearing_temperature_on_support_1",
    "bearing_temperature_on_support_2",
    "bearing_temperature_on_support_3",
    "bearing_temperature_on_support_4",
    "oil_temperature_in_the_system",
    "oil_temperature_in_oil_block",
    "vibration_on_support_1",
    "vibration_on_support_2",
    "vibration_on_support_3",
    "vibration_on_support_3_longitudinal",
    "vibration_on_support_4",
    "vibration_on_support_4_longitudinal",
]

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

    unified_tech_places = get_y_unified_tech_places(y_train)

    X_cols = get_new_X_column_names()
    X_train = rename_columns(X_train, X_cols)
    y_cols = get_new_y_column_names(y_train)
    y_train = rename_columns(y_train, y_cols)

    unified_tech_places["new_names"] = y_cols[1:]

    return X_train, y_train, messages, unified_tech_places


def get_new_X_column_names(
    column_names: [list] = None,
) -> list:
    """
    Generates unified column names in latin transcription.
    """
    if column_names is None:
        column_names = COLUMN_NAMES

    new_names = ["dt"]
    for i in range(4, 10):
        name_list = [f"e{i}_" + name for name in column_names]
        new_names += name_list

    return new_names


def rename_columns(df, new_colums: str):
    """
    Renames columns of pyspark DataFrame.
    """
    old_columns = df.schema.names

    return reduce(
        lambda data, idx: data.withColumnRenamed(old_columns[idx], new_colums[idx]),
        range(len(old_columns)),
        df,
    )


def get_new_y_column_names(y_train) -> list:
    """
    Generates new y_column_names.
    """
    desc = [i.split("_")[2] for i in y_train.schema.names[1:]]
    y_cols = ["dt"] + [switch_to_latin_letters(col) for col in desc]

    return y_cols


def switch_to_latin_letters(string_in_cirill_transcription: str) -> str:
    """
    Transforms string in cirill transcription into latin transcription.
    """

    string_in_latin_transcription = (
        str(string_in_cirill_transcription)
        .translate(
            str.maketrans(
                "абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ .№",
                "abvgdeejzijklmnoprstufhzcss_y_euaABVGDEEJZIJKLMNOPRSTUFHZCSS_Y_EUA__N",
            )
        )
        .lower()
    )

    return string_in_latin_transcription


def get_y_unified_tech_places(y_train) -> pd.DataFrame:
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
