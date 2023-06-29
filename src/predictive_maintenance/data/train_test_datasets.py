import logging
import pandas as pd


logger = logging.getLogger(__name__)

__all__ = ["make_test_dataset"]


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
