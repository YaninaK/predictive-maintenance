import logging
import pandas as pd
import numpy as np
from typing import Optional

from .make_dataset import load_X


logger = logging.getLogger(__name__)

__all__ = ["generate_anommalies_dataset"]


PATH = ""
FREQUENCY = "1H"

START = pd.Timestamp("2019-01-16 13:00:00")
END = pd.Timestamp("2021-12-31 23:59:00")

INPUT_PERIOD = pd.Timedelta(23, "H")
TIME_TO_STOPPAGE = pd.Timedelta(50, "H")
TIME_FROM_STOPPAGE = pd.Timedelta(1, "H")


def get_M3_dataset(
    equipment: int,
    messages: pd.DataFrame,
    path: Optional[str] = None,
    freq: Optional[str] = None,
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
    input_period: Optional[pd.Timedelta] = None,
    time_to_stoppage: Optional[pd.Timedelta] = None,
    time_from_stoppage: Optional[pd.Timedelta] = None,
):
    """
    Generates data set for anomaly detection.
    """
    if path is None:
        path = PATH
    if freq is None:
        freq = FREQUENCY
    if start is None:
        start = START
    if end is None:
        end = END
    if input_period is None:
        input_period = INPUT_PERIOD
    if time_to_stoppage is None:
        time_to_stoppage = TIME_TO_STOPPAGE
    if time_from_stoppage is None:
        time_from_stoppage = TIME_FROM_STOPPAGE

    X = load_X(equipment, path).resample(freq).median().bfill().ffill()
    y = generate_targets(
        equipment,
        messages,
        freq,
        start,
        end,
        time_to_stoppage,
        time_from_stoppage,
    )
    M3_dataset = []
    t = pd.Timedelta(freq)
    a = y.sum(axis=1)
    ind = y.loc[a[a > 0].index, :].index
    t0 = ind[0]
    t_start = t0
    for t1 in ind[1:]:
        if t1 - t0 > t:
            t_end = t0
            if t_end - t_start > time_to_stoppage:
                n = y[t_start + input_period : t_end].shape[0]
                for i in range(n):
                    t1_ = t_start + i * t
                    t2_ = t_start + i * t + time_to_stoppage - t
                    M3_dataset.append(X[t1_:t2_])
            t_start = t1
        t0 = t1

    return np.array(M3_dataset)


def generate_targets(
    equipment: int,
    messages: pd.DataFrame,
    freq: Optional[str] = None,
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
    time_to_stoppage: Optional[pd.Timedelta] = None,
    time_from_stoppage: Optional[pd.Timedelta] = None,
):
    """
    Generates targets for anomaly detection.
    """
    if freq is None:
        freq = FREQUENCY
    if start is None:
        start = START
    if end is None:
        end = END
    if time_to_stoppage is None:
        time_to_stoppage = TIME_TO_STOPPAGE
    if time_from_stoppage is None:
        time_from_stoppage = TIME_FROM_STOPPAGE

    t = pd.date_range(start, end, freq=freq)
    y_cols = ["equipment", "no_anomalies"] + sorted(messages["unified_name"].unique())
    y = pd.DataFrame(index=t, columns=y_cols)
    y["equipment"] = equipment
    df = messages[messages["equipment"] == str(equipment)]
    ind = df.index.tolist()
    for i in ind:
        t1 = df.loc[i, "ДАТА_НАЧАЛА_НЕИСПРАВНОСТИ"]
        t2 = df.loc[i, "ДАТА_УСТРАНЕНИЯ_НЕИСПРАВНОСТИ"]
        tech_place = df.loc[i, "unified_name"]

        if df.loc[i, "ВИД_СООБЩЕНИЯ"] == "M3":
            y.loc[t1:t2, tech_place] = 1
        else:
            t2 += time_from_stoppage
            y.loc[t1:t2, :] = np.nan

    a = y[y["equipment"].notnull()].iloc[:, 1:].sum(axis=1)
    y.loc[a[a == 0].index, "no_anomalies"] = 1

    return y.iloc[:, 1:]
