import logging
import pandas as pd
import numpy as np
from typing import Optional

from .make_dataset import load_X


logger = logging.getLogger(__name__)

__all__ = ["make_test_dataset"]


PATH = ""
FREQUENCY = "1H"
INPUT_PERIOD = pd.Timedelta(23, "H")


def generate_test_dataset(
    test_intervals: pd.DataFrame,
    path: Optional[str] = None,
    freq: Optional[str] = None,
    input_period: Optional[pd.Timedelta] = None,
) -> dict:
    """
    Generates test dataset for inference.
    """
    if path is None:
        path = PATH
    if freq is None:
        freq = FREQUENCY
    if input_period is None:
        input_period = INPUT_PERIOD

    all_test_periods = {}
    for equipment in range(4, 10):
        X_test = load_X(equipment, path, prefix="X_test").resample(freq).median()
        test_periods = []

        for t1 in test_intervals["start"]:
            t0 = t1 - input_period
            a = X_test.loc[t0:t1, :]
            if a.shape[0] == input_period // pd.Timedelta(freq):
                test_periods.append(X_test.loc[t0:t1, :].bfill().ffill())

        all_test_periods[equipment] = np.array(test_periods)

    return all_test_periods


def combine_test_intervals(test_intervals: pd.DataFrame) -> pd.DataFrame:
    """
    Combines test intervals when the current period ends earlier than the previous one,
    and when the current period starts before the end of the previous one, although ends after.
    """
    ind = []
    for i in range(1, len(test_intervals)):
        t_start_prev = test_intervals["start"][i - 1]
        t_end_prev = test_intervals["finish"][i - 1]
        t_start = test_intervals["start"][i]
        t_end = test_intervals["finish"][i]

        if t_end_prev >= t_end:
            ind.append(i)
        elif t_start < t_end_prev:
            test_intervals.loc[i - 1, "finish"] = t_end
            ind.append(i)

    test_intervals = test_intervals.drop(ind, axis=0)

    return test_intervals.reset_index(drop=True)
