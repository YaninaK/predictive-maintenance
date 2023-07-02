import logging
import pandas as pd
import numpy as np
from scipy import linalg
from typing import Optional

from .make_dataset import load_X


logger = logging.getLogger(__name__)

__all__ = ["generate_stoppages_dataset"]


PATH = ""
STOPPAGES = ["ТЕХНИЧЕСКИЕ НЕПЛАНОВЫЕ", "ТЕХНОЛОГИЧЕСКИЕ НЕПЛАНОВЫЕ"]
FREQUENCY = "1H"

START = pd.Timestamp("2019-01-16 13:00:00")

INPUT_PERIOD = pd.Timedelta(23, "H")
TIME_TO_STOPPAGE = pd.Timedelta(50, "H")
TIME_FROM_STOPPAGE = pd.Timedelta(1, "H")


def get_M1_dataset(
    scaler,
    pca,
    messages,
    path: Optional[str] = None,
    freq: Optional[str] = None,
    stoppages: Optional[list] = None,
    start: Optional[pd.Timestamp] = None,
    input_period: Optional[pd.Timedelta] = None,
    time_to_stoppage: Optional[pd.Timedelta] = None,
    time_from_stoppage: Optional[pd.Timedelta] = None,
):
    """
    Generates stoppages dataset for all exhauster with M1_time_label.
    M1_time_label is a time period before the stoppage.

    Calculates Hotelling's T-squared and Q residuals to be used in Bayesian model.

    Hotelling’s T2 is the sum of the normalized squared scores.
    It measures the variation in each sample within the model indicating how far each sample
    is from the center (scores = 0) of the model.

    Q residuals represent the magnitude of the variation remaining in each sample
    after projection through the model.
    """
    if path is None:
        path = PATH
    if freq is None:
        freq = FREQUENCY
    if stoppages is None:
        stoppages = STOPPAGES
    if start is None:
        start = START
    if input_period is None:
        input_period = INPUT_PERIOD
    if time_to_stoppage is None:
        time_to_stoppage = TIME_TO_STOPPAGE
    if time_from_stoppage is None:
        time_from_stoppage = TIME_FROM_STOPPAGE

    t = pd.Timedelta(freq)
    M1_dataset = []
    M1_labels = []
    for equipment in range(4, 10):
        X = load_X(equipment, path).resample(freq).median().bfill().ffill()
        old_cols = X.columns.tolist()
        new_cols = [f"col_{i}" for i in range(X.shape[1])]
        X.rename(
            columns={old_cols[i]: new_cols[i] for i in range(len(old_cols))},
            inplace=True,
        )
        X_transformed = pca.transform(scaler.transform(X))
        df = pd.DataFrame(X_transformed, index=X.index)
        lambda_inv = linalg.inv(
            np.dot(X_transformed.T, X_transformed) / (X_transformed.shape[0] - 1)
        )
        df["Hotelling's T-squared"] = df.T.apply(
            lambda t_: np.dot(np.dot(t_, lambda_inv), t_.T)
        )
        errors = X - np.dot(X_transformed, pca.components_)
        df["Q residuals"] = errors.T.apply(lambda e: np.dot(e, e.T))

        M1_periods = select_M1_periods(
            equipment,
            messages,
            stoppages,
            start,
            input_period,
            time_to_stoppage,
            time_from_stoppage,
        )
        for t1, t2 in M1_periods:
            M1_dataset.append(df[t1 : t1 + input_period].values)
            label = (t2 - (t1 + input_period)) // pd.Timedelta(freq)
            M1_labels.append(label)

    M1_dataset = np.stack(M1_dataset, axis=0)

    return M1_dataset, M1_labels


def select_M1_periods(
    equipment: int,
    messages: pd.DataFrame,
    stoppages: Optional[list] = None,
    start: Optional[pd.Timestamp] = None,
    input_period: Optional[pd.Timedelta] = None,
    time_to_stoppage: Optional[pd.Timedelta] = None,
    time_from_stoppage: Optional[pd.Timedelta] = None,
):
    """
    For particular exhauster, selects periods of unplanned stoppages listed in messages
    starting from a given time before stoppage and ending by a given time after stoppage.
    """
    if stoppages is None:
        stoppages = STOPPAGES
    if start is None:
        start = START
    if input_period is None:
        input_period = INPUT_PERIOD
    if time_to_stoppage is None:
        time_to_stoppage = TIME_TO_STOPPAGE
    if time_from_stoppage is None:
        time_from_stoppage = TIME_FROM_STOPPAGE
    df = messages[
        (messages["equipment"] == str(equipment)) & (messages["ВИД_СООБЩЕНИЯ"] == "M1")
    ]
    t0 = start
    M1_periods = []
    ind = df.index.tolist()
    for i in ind:
        t1 = df.loc[i, "ДАТА_НАЧАЛА_НЕИСПРАВНОСТИ"]
        t2 = df.loc[i, "ДАТА_УСТРАНЕНИЯ_НЕИСПРАВНОСТИ"] + time_from_stoppage

        if df.loc[i, "ТЕКСТ_ГРУППЫ_КОДОВ"] in stoppages:
            if t1 - t0 >= input_period:
                if t1 - t0 >= time_to_stoppage:
                    t0 = t1 - time_to_stoppage
                M1_periods.append([t0, t1])
        t0 = t2

    return M1_periods
