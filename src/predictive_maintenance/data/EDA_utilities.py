import logging
import pandas as pd
import numpy as np
from typing import Optional


logger = logging.getLogger(__name__)

__all__ = ["EDA_utilities"]

PATH = ""
FOLDER_1 = "data/02_intermediate/"
FOLDER_2 = "data/03_primary/"

POSTFIX = "resampled"


def load_y(
    i: int,
    path: Optional[str] = None,
    folder: Optional[str] = None,
    postfix: Optional[str] = None,
) -> pd.DataFrame:
    """
    Uploads resampled y_train.
    """
    if path is None:
        path = PATH
    if folder is None:
        folder = FOLDER_1
    if postfix is None:
        postfix = POSTFIX

    y = pd.read_parquet(path + folder + f"y{i}_{postfix}.parquet").drop(
        "max(epoch)", axis=1
    )
    y.columns = ["dt"] + [i[4:-1] for i in y.columns[1:].tolist()]
    y.set_index("dt", inplace=True)

    return y


def get_anomaly_dict(y: pd.DataFrame, t: pd.Timedelta) -> dict:
    """
    Generates dictionary of anomaly start and end time for each technical place of equipment.
    """
    a = y.sum(axis=0)
    tech_places = a[a > 0].T.index.tolist()
    all_anomalies = {}
    for tech_place in tech_places:
        anomalies = []
        anomaly_time_list = y.loc[y[tech_place] > 0, tech_place].index.tolist()
        t1 = anomaly_time_list[0]
        start_end_anomaly = [t1]
        for t2 in anomaly_time_list[1:]:
            if t2 - t1 > t:
                start_end_anomaly.append(t1)
                anomalies.append(start_end_anomaly)
                start_end_anomaly = [t2]
            t1 = t2
        if len(start_end_anomaly) == 1:
            start_end_anomaly.append(t1)
            anomalies.append(start_end_anomaly)
        all_anomalies[tech_place] = anomalies

    return all_anomalies


def get_description_dictionary(unified_tech_places: pd.DataFrame) -> dict:
    """
    Generates dictionary {description: unified_name}.
    """
    description = unified_tech_places["description"].tolist()
    unified_name = unified_tech_places["unified_name"].tolist()
    description_dictionary = {k: v for (k, v) in zip(description, unified_name)}

    return description_dictionary


def get_y_summary(
    unified_tech_places: pd.DataFrame,
    path: Optional[str] = None,
    folder: Optional[str] = None,
) -> pd.DataFrame:
    """
    Generates summary of anomaly labels in y_train.
    """
    if path is None:
        path = PATH
    if folder is None:
        folder = FOLDER_1

    description_dictionary = get_description_dictionary(unified_tech_places)
    t = pd.Timedelta("1T")
    n = 0
    df = pd.DataFrame()
    for i in range(4, 10):
        y = load_y(i, path, folder)
        all_anomalies = get_anomaly_dict(y, t)
        for e in all_anomalies.keys():
            for time in all_anomalies[e]:
                df.loc[n, "equipment"] = str(i)
                df.loc[n, "tech_place"] = e
                df.loc[n, "start_M"] = time[0]
                df.loc[n, "end_M"] = time[1]
                df.loc[n, "unified_name"] = description_dictionary[e]
                n += 1

    return df


def get_unified_name_dictionary(i: int, unified_tech_places: pd.DataFrame) -> dict:
    """
    Generates dictionary {unified_name: description}
    """
    df = unified_tech_places[unified_tech_places["equipment"] == str(i)]
    unified_name = df["unified_name"].tolist()
    description = df["description"].tolist()
    unified_name_dictionary = {k: v for (k, v) in zip(unified_name, description)}

    return unified_name_dictionary


def add_tech_place_description(
    i: int, unified_name_dictionary: dict, df: pd.DataFrame
) -> pd.DataFrame:
    """
    Adds technical place description based on unified_name and equipment number.
    """
    unified_names = df.loc[df["equipment"] == i, "unified_name"].unique().tolist()
    cond_1 = df["equipment"] == i
    for name in unified_names:
        description = unified_name_dictionary[name]
        cond_2 = df["unified_name"] == name
        df.loc[cond_1 & cond_2, "description"] = description

    return df


def get_missing_labels(
    y_summary: pd.DataFrame, messages: pd.DataFrame, unified_tech_places: pd.DataFrame
) -> pd.DataFrame:
    """
    Selects labels, which were not reflected in y_train, from messages.
    """
    df_M3 = pd.DataFrame()
    cols = [
        "start_M",
        "ДАТА_УСТРАНЕНИЯ_НЕИСПРАВНОСТИ",
        "ВИД_СООБЩЕНИЯ",
        "unified_name",
    ]
    cols_y = ["start_M", "tech_place"]
    for i in range(4, 10):
        a = y_summary.loc[y_summary["equipment"] == str(i), cols_y].set_index("start_M")
        b = messages.loc[messages["equipment"] == str(i), cols].set_index("start_M")
        df = pd.concat([a, b], axis=1)
        df = df.loc[
            df["tech_place"].isnull() & (df["ВИД_СООБЩЕНИЯ"] == "M3"), cols[1:]
        ].reset_index()
        df["equipment"] = i
        df_M3 = pd.concat([df_M3, df], axis=0)

        unified_name_dict = get_unified_name_dictionary(i, unified_tech_places)
        df_M3 = add_tech_place_description(i, unified_name_dict, df_M3)

    return df_M3.reset_index(drop=True)


def add_missing_labels(
    df_M3: pd.DataFrame,
    path: Optional[str] = None,
    folder: Optional[str] = None,
):
    """
    Adds labels reflected only in messages to y_train.
    """
    if path is None:
        path = PATH
    if folder is None:
        folder = FOLDER_2

    for i in range(4, 10):
        y = load_y(i, path)
        tech_places = df_M3.loc[df_M3["equipment"] == i, "description"].unique()
        for tech_place in tech_places:
            anomalies = df_M3.loc[df_M3["description"] == tech_place, :]
            for j in anomalies.index:
                t1 = anomalies.loc[j, "start_M"]
                t2 = anomalies.loc[j, "ДАТА_УСТРАНЕНИЯ_НЕИСПРАВНОСТИ"]
                y.loc[t1:t2, tech_place] = 2
        y.to_parquet(
            path + folder + f"y{i}_updated.parquet",
        )
