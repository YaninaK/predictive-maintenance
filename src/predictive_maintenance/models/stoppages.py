import logging
import pandas as pd
import numpy as np
from scipy import linalg
from typing import Optional

from data.utilities import load_X


logger = logging.getLogger(__name__)

__all__ = ["generate_stoppages_dataset"]


PATH = ""

START = pd.Timestamp("2019-01-16 13:00:00")
FREQUENCY = pd.Timedelta("1H")

INPUT_SEQUENCE_LENGTH = 23
OUTPUT_SEQUENCE_LENGTH = 27

TIME_FROM_STOPPAGE = pd.Timedelta(1, "H")
STOPPAGES = ["ТЕХНИЧЕСКИЕ НЕПЛАНОВЫЕ", "ТЕХНОЛОГИЧЕСКИЕ НЕПЛАНОВЫЕ"]


def preprocess_stoppages(model_LSTM, M1_dataset, scaler_bmb) -> pd.DataFrame:
    """
    Makes predictions of model_LSTM fitted on etalon dataset to generate
    Hotelling's T-squared and Q residuals for forecasted period.
    Uses standard scaler fitted on etalon dataset to transform forecasted Hotelling's
    T-squared and Q residuals, obtaining results compareable to the ones of etalon dataset.
    Generates dataset for inference of Bayesian model.
    """
    pred_stoppages = model_LSTM.predict(M1_dataset)
    T2_Q_from_stoppages = pred_stoppages[:, :, -2:].reshape(-1, 2)
    T2_Q_from_stoppages = scaler_bmb.transform(T2_Q_from_stoppages)
    T2_Q_from_stoppages = pd.DataFrame(T2_Q_from_stoppages, columns=["T2", "Q"])

    return T2_Q_from_stoppages


def get_M1_dataset(
    scaler,
    pca,
    messages,
    path: Optional[str] = None,
    start: Optional[pd.Timestamp] = None,
    freq: Optional[pd.Timedelta] = None,
    input_sequence_length: Optional[int] = None,
    output_sequence_length: Optional[int] = None,
    time_from_stoppage: Optional[pd.Timedelta] = None,
    stoppages: Optional[list] = None,
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
    if start is None:
        start = START
    if freq is None:
        freq = FREQUENCY
    if input_sequence_length is None:
        input_sequence_length = INPUT_SEQUENCE_LENGTH
    if output_sequence_length is None:
        output_sequence_length = OUTPUT_SEQUENCE_LENGTH
    if time_from_stoppage is None:
        time_from_stoppage = TIME_FROM_STOPPAGE
    if stoppages is None:
        stoppages = STOPPAGES

    input_period = freq * input_sequence_length
    M1_dataset = []
    M1_labels = []
    for equipment in range(4, 10):
        X = load_X(equipment, path).bfill().ffill()
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
            start,
            freq,
            input_sequence_length,
            output_sequence_length,
            time_from_stoppage,
            stoppages,
        )
        for t1, t2 in M1_periods:
            M1_dataset.append(df[t1 : t1 + input_period].values)
            label = (t2 - (t1 + input_period)) // freq
            M1_labels.append(label)

    M1_dataset = np.stack(M1_dataset, axis=0)

    return M1_dataset, M1_labels


def select_M1_periods(
    equipment: int,
    messages: pd.DataFrame,
    start: Optional[pd.Timestamp] = None,
    freq: Optional[pd.Timedelta] = None,
    input_sequence_length: Optional[int] = None,
    output_sequence_length: Optional[int] = None,
    time_from_stoppage: Optional[pd.Timedelta] = None,
    stoppages: Optional[list] = None,
):
    """
    For particular exhauster, selects periods of unplanned stoppages listed in messages
    starting from a given time before stoppage and ending by a given time after stoppage.
    """
    if start is None:
        start = START
    if freq is None:
        freq = FREQUENCY
    if input_sequence_length is None:
        input_sequence_length = INPUT_SEQUENCE_LENGTH
    if output_sequence_length is None:
        output_sequence_length = OUTPUT_SEQUENCE_LENGTH
    if time_from_stoppage is None:
        time_from_stoppage = TIME_FROM_STOPPAGE
    if stoppages is None:
        stoppages = STOPPAGES

    input_period = freq * input_sequence_length
    time_to_stoppage = freq * (input_sequence_length + output_sequence_length)
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
