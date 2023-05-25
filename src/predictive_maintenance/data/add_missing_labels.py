import logging
import pandas as pd
import numpy as np
from typing import Optional


logger = logging.getLogger(__name__)

__all__ = ["add_missing_labels"]


COLUMNS_IN = ["equipment", "tech_place", "НАЗВАНИЕ_ТЕХ_МЕСТА", "ОПИСАНИЕ"]
COLUMNS_OUT = [
    "equipment",
    "НАЗВАНИЕ_ТЕХ_МЕСТА",
    "ДАТА_НАЧАЛА_НЕИСПРАВНОСТИ",
    "ДАТА_УСТРАНЕНИЯ_НЕИСПРАВНОСТИ",
]


def get_missing_labels_summary(
    M3_messages: pd.DataFrame,
    cols_in: Optional[list] = None,
    cols_out: Optional[list] = None,
) -> pd.DataFrame:

    if cols_in is None:
        cols_in = COLUMNS_IN
    if cols_out is None:
        cols_out = COLUMNS_OUT

    ind4 = get_missing_dates_4(M3_messages)
    ind5 = get_missing_dates_5(M3_messages)
    ind6 = get_missing_dates_6(M3_messages)
    ind7 = get_missing_dates_7(M3_messages)
    ind8 = get_missing_dates_8(M3_messages)
    ind9 = get_missing_dates_9(M3_messages)

    inds = {
        4: ind4,
        5: ind5,
        6: ind6,
        7: ind7,
        8: ind8,
        9: ind9,
    }
    df = pd.DataFrame()
    for i in range(4, 10):
        ind = inds[i]
        df = pd.concat([df, M3_messages.loc[ind, cols_out]], axis=0)

    return df


def get_missing_dates_4(M3_messages: pd.DataFrame, cols: Optional[list] = None):
    if cols is None:
        cols = COLUMNS_IN

    a = M3_messages.loc[
        (M3_messages["equipment"] == 4)
        & (
            (
                M3_messages["tech_place"].isnull()
                & M3_messages["НАЗВАНИЕ_ТЕХ_МЕСТА"].notnull()
            )
            | (
                M3_messages["tech_place"].notnull()
                & M3_messages["НАЗВАНИЕ_ТЕХ_МЕСТА"].isnull()
            )
        ),
        cols,
    ]
    ind = (
        a.index[:6].tolist()
        + a.index[7:18].tolist()
        + a.index[24:30].tolist()
        + a.index[32:34].tolist()
        + a.index[36:37].tolist()
        + a.index[39:40].tolist()
        + a.index[42:46].tolist()
        + a.index[48:50].tolist()
        + a.index[51:56].tolist()
        + a.index[58:60].tolist()
        + a.index[61:65].tolist()
        + a.index[66:71].tolist()
        + a.index[73:74].tolist()
        + a.index[75:81].tolist()
        + a.index[83:88].tolist()
        + a.index[91:134].tolist()
        + a.index[136:137].tolist()
    )

    return ind


def get_missing_dates_5(M3_messages: pd.DataFrame, cols: Optional[list] = None):
    if cols is None:
        cols = COLUMNS_IN

    a = M3_messages.loc[
        (M3_messages["equipment"] == 5)
        & (
            (
                M3_messages["tech_place"].isnull()
                & M3_messages["НАЗВАНИЕ_ТЕХ_МЕСТА"].notnull()
            )
            | (
                M3_messages["tech_place"].notnull()
                & M3_messages["НАЗВАНИЕ_ТЕХ_МЕСТА"].isnull()
            )
        ),
        cols,
    ]
    ind = (
        a.index[:3].tolist()
        + a.index[5:15].tolist()
        + a.index[16:42].tolist()
        + a.index[43:49].tolist()
        + a.index[51:61].tolist()
        + a.index[67:68].tolist()
        + a.index[70:72].tolist()
        + a.index[74:83].tolist()
        + a.index[84:85].tolist()
        + a.index[86:90].tolist()
        + a.index[93:96].tolist()
        + a.index[102:103].tolist()
        + a.index[107:115].tolist()
        + a.index[119:121].tolist()
        + a.index[123:125].tolist()
        + a.index[126:128].tolist()
        + a.index[130:139].tolist()
        + a.index[141:147].tolist()
    )

    return ind


def get_missing_dates_6(M3_messages: pd.DataFrame, cols: Optional[list] = None):
    if cols is None:
        cols = COLUMNS_IN

    a = M3_messages.loc[
        (M3_messages["equipment"] == 6)
        & (
            (
                M3_messages["tech_place"].isnull()
                & M3_messages["НАЗВАНИЕ_ТЕХ_МЕСТА"].notnull()
            )
            | (
                M3_messages["tech_place"].notnull()
                & M3_messages["НАЗВАНИЕ_ТЕХ_МЕСТА"].isnull()
            )
        ),
        cols,
    ]
    ind = (
        a.index[4:10].tolist()
        + a.index[11:36].tolist()
        + a.index[38:47].tolist()
        + a.index[49:53].tolist()
        + a.index[57:59].tolist()
        + a.index[63:97].tolist()
    )

    return ind


def get_missing_dates_7(M3_messages: pd.DataFrame, cols: Optional[list] = None):
    if cols is None:
        cols = COLUMNS_IN

    a = M3_messages.loc[
        (M3_messages["equipment"] == 7)
        & (
            (
                M3_messages["tech_place"].isnull()
                & M3_messages["НАЗВАНИЕ_ТЕХ_МЕСТА"].notnull()
            )
            | (
                M3_messages["tech_place"].notnull()
                & M3_messages["НАЗВАНИЕ_ТЕХ_МЕСТА"].isnull()
            )
        ),
        cols,
    ]
    ind = (
        a.index[:3].tolist()
        + a.index[7:11].tolist()
        + a.index[15:30].tolist()
        + a.index[32:33].tolist()
        + a.index[35:38].tolist()
        + a.index[40:54].tolist()
        + a.index[58:69].tolist()
        + a.index[71:86].tolist()
        + a.index[88:94].tolist()
        + a.index[95:99].tolist()
        + a.index[100:101].tolist()
        + a.index[103:104].tolist()
        + a.index[105:121].tolist()
        + a.index[123:132].tolist()
        + a.index[133:143].tolist()
        + a.index[144:145].tolist()
        + a.index[146:149].tolist()
        + a.index[151:153].tolist()
        + a.index[154:162].tolist()
    )

    return ind


def get_missing_dates_8(M3_messages: pd.DataFrame, cols: Optional[list] = None):
    if cols is None:
        cols = COLUMNS_IN

    a = M3_messages.loc[
        (M3_messages["equipment"] == 8)
        & (
            (
                M3_messages["tech_place"].isnull()
                & M3_messages["НАЗВАНИЕ_ТЕХ_МЕСТА"].notnull()
            )
            | (
                M3_messages["tech_place"].notnull()
                & M3_messages["НАЗВАНИЕ_ТЕХ_МЕСТА"].isnull()
            )
        ),
        cols,
    ]
    ind = (
        a.index[:17].tolist()
        + a.index[18:27].tolist()
        + a.index[28:41].tolist()
        + a.index[43:45].tolist()
        + a.index[47:48].tolist()
        + a.index[49:55].tolist()
        + a.index[58:66].tolist()
        + a.index[70:72].tolist()
        + a.index[74:76].tolist()
        + a.index[78:96].tolist()
        + a.index[98:103].tolist()
        + a.index[104:111].tolist()
    )

    return ind


def get_missing_dates_9(M3_messages: pd.DataFrame, cols: Optional[list] = None):
    if cols is None:
        cols = COLUMNS_IN

    a = M3_messages.loc[
        (M3_messages["equipment"] == 9)
        & (
            (
                M3_messages["tech_place"].isnull()
                & M3_messages["НАЗВАНИЕ_ТЕХ_МЕСТА"].notnull()
            )
            | (
                M3_messages["tech_place"].notnull()
                & M3_messages["НАЗВАНИЕ_ТЕХ_МЕСТА"].isnull()
            )
        ),
        cols,
    ]
    ind = (
        a.index[8:15].tolist()
        + a.index[17:26].tolist()
        + a.index[32:34].tolist()
        + a.index[36:37].tolist()
        + a.index[39:43].tolist()
        + a.index[45:48].tolist()
        + a.index[49:52].tolist()
        + a.index[54:56].tolist()
        + a.index[58:60].tolist()
        + a.index[64:67].tolist()
        + a.index[69:86].tolist()
    )

    return ind


def add_missing_M3_data(df2: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds missing in y_train data to M3 summary df2
    """

    def clean_y_names(col):
        col = re.sub("\(", "", col)
        col = re.sub("\)", "", col)
        col = re.sub("\.", "_", col)
        return col

    a = df["equipment"].tolist()
    b = df["НАЗВАНИЕ_ТЕХ_МЕСТА"].tolist()
    df["tech_place"] = [f"{i}_{j}" for (i, j) in zip(a, b)]
    df["tech_place"] = df["tech_place"].apply(clean_y_names)

    df["start_M"] = df.index
    df["end_M"] = df["ДАТА_УСТРАНЕНИЯ_НЕИСПРАВНОСТИ"].tolist()
    df["M_period"] = df["end_M"] - df["start_M"]

    cols = ["equipment", "tech_place", "start_M", "end_M", "M_period"]
    df2 = (
        pd.concat([df2, df[cols]], axis=0)
        .sort_values(by=["equipment", "start_M"])
        .reset_index(drop=True)
    )

    return df2
