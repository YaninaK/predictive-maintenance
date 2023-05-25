import logging
import pandas as pd
import numpy as np
import re
from typing import Optional


logger = logging.getLogger(__name__)

__all__ = ["add_missing_labels"]


PATH = ""
FOLDER = "data/02_intermediate/"
N_DAYS = 1.5


COLUMNS_IN = ["equipment", "tech_place", "НАЗВАНИЕ_ТЕХ_МЕСТА", "ОПИСАНИЕ"]
COLUMNS_OUT = [
    "equipment",
    "НАЗВАНИЕ_ТЕХ_МЕСТА",
    "ДАТА_НАЧАЛА_НЕИСПРАВНОСТИ",
    "ДАТА_УСТРАНЕНИЯ_НЕИСПРАВНОСТИ",
]


def get_M_start_finish_time(
    M_code=1,
    path: Optional[str] = None,
    folder: Optional[str] = None,
):
    if path is None:
        path = PATH
    if folder is None:
        folder = FOLDER

    m_dict = {}
    t = pd.to_timedelta(1, unit="T")
    for i in range(4, 10):
        y = pd.read_parquet(path + folder + f"y{i}_resampled.parquet").set_index(
            "dt_resampled"
        )

        m = np.where(y == M_code, 1, 0)
        m = pd.DataFrame(m, index=y.index, columns=y.columns)

        a = m[m.sum(axis=1) > 0].sum(axis=0)
        y_m_cols = a[a > 0].index.tolist()

        m_ = m.loc[m[y_m_cols].sum(axis=1) > 0, y_m_cols]
        t_lists = {}
        for col in y_m_cols:
            a = m_[m_[col] == 1]
            t_lists[col] = []
            s = 0
            t0 = a.index[0]
            for t1 in a.index:
                if s == 0:
                    start_fintsh = [t0]
                    s = 1
                if (s == 1) & (t1 - t0 > t):
                    start_fintsh.append(t0)
                    t_lists[col].append(start_fintsh)
                    s = 0
                t0 = t1
            if s == 1:
                start_fintsh.append(t1)
                t_lists[col].append(start_fintsh)

        m_dict[i] = t_lists

    return m_dict


def M_summary(m_dict: dict, n_days: Optional[int] = None) -> pd.DataFrame:
    """
    Performs M1 failure selection for model validation and training.
    """
    if n_days is None:
        n_days = N_DAYS

    df = pd.DataFrame(columns=["equipment", "tech_place", "start_M"])
    n = 0
    for e in m_dict:
        for col in m_dict[e]:
            s = 0
            for t in m_dict[e][col]:
                df.loc[n, "equipment"] = e
                df.loc[n, "tech_place"] = col[4:-1]
                df.loc[n, "start_M"] = t[0]
                df.loc[n, "end_M"] = t[1]
                df.loc[n, "M_period"] = t[1] - t[0]
                if s == 0:
                    df.loc[n, "delta_between_M"] = 0
                    df.loc[n, "accept"] = 1
                    s = 1
                else:
                    df.loc[n, "delta_between_M"] = (
                        df.loc[n, "start_M"] - df.loc[n - 1, "end_M"]
                    )
                    if (
                        df.loc[n, "delta_between_M"].total_seconds()
                        / pd.Timedelta(days=1).total_seconds()
                        > n_days
                    ):
                        df.loc[n, "accept"] = 1
                n += 1

    return df


def add_info_from_messages(df):
    """
    Adds missing in y_tain stoppage information from messages.
    """

    df = pd.DataFrame()
    df.loc[0, "equipment"] = str(9)
    df.loc[0, "tech_place"] = "9_ЭЛЕКТРООБОРУДОВАНИЯ ЭКСГАУСТЕРА №9"
    df.loc[0, "start_M"] = pd.Timestamp("2019-03-19 14:19:00")
    df.loc[0, "accept"] = 1

    df.loc[1, "equipment"] = str(6)
    df.loc[1, "tech_place"] = "6_ЗАДВИЖКА ЭКСГ_ №6"
    df.loc[1, "start_M"] = pd.Timestamp("2019-05-01 19:18:00")
    df.loc[1, "accept"] = 1

    df.loc[2, "equipment"] = str(4)
    df.loc[2, "tech_place"] = "4_ПОДШИПНИК ОПОРНЫЙ №2 ЭКСГ_ №4"
    df.loc[2, "start_M"] = pd.Timestamp("2019-07-30 19:21:00")
    df.loc[2, "accept"] = 1

    df = pd.concat([df1, df], axis=0).sort_values(by="start_M")

    return df.reset_index(drop=True)


def get_M_messages(df, messages, m_code="M1"):
    messages.rename(columns={"equipment": "equipment_"}, inplace=True)
    M_messages = pd.concat(
        [
            df.set_index("start_M"),
            messages[messages["ВИД_СООБЩЕНИЯ"] == m_code].set_index("start_M"),
        ],
        axis=1,
    )
    M_messages.loc[M_messages["equipment"].isnull(), "equipment"] = M_messages.loc[
        M_messages["equipment"].isnull(), "equipment_"
    ].astype(int)
    M_messages.drop("equipment_", axis=1, inplace=True)

    return M_messages


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
