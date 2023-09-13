import logging
import pandas as pd
import numpy as np
from typing import Optional

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import linalg

from data.utilities import load_X
import matplotlib.pyplot as plt


logger = logging.getLogger(__name__)

__all__ = ["generate_ethalon_dataset"]


PATH = ""

START = pd.Timestamp("2019-01-16 13:00:00")
END = pd.Timestamp("2021-12-31 23:59:00")
FREQUENCY = pd.Timedelta("1H")

INPUT_SEQUENCE_LENGTH = 23
OUTPUT_SEQUENCE_LENGTH = 27

TIME_FROM_STOPPAGE = pd.Timedelta("1H")

MAX_VIBRATION = 10
MAX_TEMPERATURE = 80

VIBRATION_COLUMNS = [
    "ВИБРАЦИЯ НА ОПОРЕ 1",
    "ВИБРАЦИЯ НА ОПОРЕ 2",
    "ВИБРАЦИЯ НА ОПОРЕ 3",
    "ВИБРАЦИЯ НА ОПОРЕ 3 ПРОДОЛЬНАЯ",
    "ВИБРАЦИЯ НА ОПОРЕ 4",
    "ВИБРАЦИЯ НА ОПОРЕ 4 ПРОДОЛЬНАЯ",
]
TEMPERATURE_COLUMNS = [
    "ТЕМПЕРАТУРА МАСЛА В МАСЛОБЛОКЕ",
    "ТЕМПЕРАТУРА МАСЛА В СИСТЕМЕ",
    "ТЕМПЕРАТУРА ПОДШИПНИКА НА ОПОРЕ 1",
    "ТЕМПЕРАТУРА ПОДШИПНИКА НА ОПОРЕ 2",
    "ТЕМПЕРАТУРА ПОДШИПНИКА НА ОПОРЕ 3",
    "ТЕМПЕРАТУРА ПОДШИПНИКА НА ОПОРЕ 4",
]
PLOT = False
SEED = 25


def generate_etalon_dataset(
    df: pd.DataFrame,
    freq: Optional[pd.Timedelta] = None,
    input_sequence_length: Optional[int] = None,
    output_sequence_length: Optional[int] = None,
    seed: Optional[int] = None,
):
    """
    Generates etalon dataset for training LSTM model.
    """
    if freq is None:
        freq = FREQUENCY
    if input_sequence_length is None:
        input_sequence_length = INPUT_SEQUENCE_LENGTH
    if output_sequence_length is None:
        output_sequence_length = OUTPUT_SEQUENCE_LENGTH
    if seed is None:
        seed = SEED

    time_to_stoppage = freq * (input_sequence_length + output_sequence_length)
    etalon_dataset = []
    for e in range(4, 10):
        df_ = df[df["equipment"] == e]
        ind = df_.index.tolist()
        t0 = ind[0]
        for t1 in ind[1:]:
            if t1 == t0 + time_to_stoppage:
                etalon_dataset.append(df_[t0 : t1 - freq].values)
                t0 += freq
            elif t1 > t0 + time_to_stoppage:
                t0 = t1
    etalon_dataset = np.stack(etalon_dataset, axis=0)[:, :, :-1]

    idx = np.arange(len(etalon_dataset))
    np.random.seed(seed)
    np.random.shuffle(idx)

    return etalon_dataset[idx]


def select_etalon_periods(
    messages: pd.DataFrame,
    path: Optional[str] = None,
    freq: Optional[pd.Timedelta] = None,
    input_sequence_length: Optional[int] = None,
    output_sequence_length: Optional[int] = None,
    time_from_stoppage: Optional[pd.Timedelta] = None,
    max_vibration: Optional[float] = None,
    max_temperature: Optional[float] = None,
    vibration_columns: Optional[list] = None,
    temperature_columns: Optional[list] = None,
):
    """
    Selects periods with vibration lower than max_vibration and temperature lower
    than max_temperature.
    All the stoppages and given periods before and after stoppages are excluded.
    """
    if path is None:
        path = PATH

    if freq is None:
        freq = FREQUENCY
    if input_sequence_length is None:
        input_sequence_length = INPUT_SEQUENCE_LENGTH
    if output_sequence_length is None:
        output_sequence_length = OUTPUT_SEQUENCE_LENGTH
    if time_from_stoppage is None:
        time_from_stoppage = TIME_FROM_STOPPAGE

    if max_vibration is None:
        max_vibration = MAX_VIBRATION
    if max_temperature is None:
        max_temperature = MAX_TEMPERATURE
    if vibration_columns is None:
        vibration_columns = VIBRATION_COLUMNS
    if temperature_columns is None:
        temperature_columns = TEMPERATURE_COLUMNS

    time_to_stoppage = freq * (input_sequence_length + output_sequence_length)
    messages = messages[messages["ДАТА_УСТРАНЕНИЯ_НЕИСПРАВНОСТИ"].notnull()]
    etalon_periods = pd.DataFrame()
    for equipment in range(4, 10):
        X = load_X(equipment, path).resample(freq).median().bfill().ffill()

        vibration_cols = [f"{str(equipment)} {i}" for i in vibration_columns]
        temperature_cols = [f"{str(equipment)} {i}" for i in temperature_columns]

        selected_periods = X[vibration_cols[0]] < max_vibration
        for col in vibration_cols[1:]:
            selected_periods &= X[col] < max_vibration
        for col in temperature_cols:
            selected_periods &= X[col] < max_temperature

        df = messages[messages["equipment"] == str(equipment)]
        ind = df.index.tolist()
        for i in ind:
            t1 = df.loc[i, "ДАТА_НАЧАЛА_НЕИСПРАВНОСТИ"]
            t2 = df.loc[i, "ДАТА_УСТРАНЕНИЯ_НЕИСПРАВНОСТИ"]

            if df.loc[i, "ВИД_СООБЩЕНИЯ"] == "M1":
                t1 -= time_to_stoppage
                t2 += time_from_stoppage
            selected_periods[t1:t2] = False

        ind = X[selected_periods].index.tolist()
        t0 = ind[0]
        t_start = t0
        t = freq

        etalon = pd.DataFrame()
        for t1 in ind[1:]:
            if t1 - t0 > t:
                t_end = t0
                if t_end - t_start >= time_to_stoppage:
                    etalon = pd.concat([etalon, X[t_start:t_end]])
                t_start = t1
            t0 = t1

        old_cols = etalon.columns.tolist()
        new_cols = [f"col_{i}" for i in range(etalon.shape[1])]
        etalon.rename(
            columns={old_cols[i]: new_cols[i] for i in range(len(old_cols))},
            inplace=True,
        )
        etalon["equipment"] = equipment

        etalon_periods = pd.concat([etalon_periods, etalon], axis=0)

    return etalon_periods


def get_pca_components(
    etalon_periods,
    plot: Optional[bool] = None,
):
    """
    Applies StandardScaler to scale features for PCA.
    Selects the number of components based on Kaiser's rule.
    Generates latent variables using PCA.

    Calculates Hotelling's T-squared and Q residuals to be used in Bayesian model.

    Hotelling’s T2 is the sum of the normalized squared scores.
    It measures the variation in each sample within the model indicating how far each sample
    is from the center (scores = 0) of the model.

    Q residuals represent the magnitude of the variation remaining in each sample
    after projection through the model.
    """
    if plot is None:
        plot = PLOT

    X = etalon_periods.iloc[:, :-1].copy()

    scaler = StandardScaler()
    X.iloc[:, :] = scaler.fit_transform(X.bfill().ffill())
    corr = X.corr()
    pca = PCA(n_components="mle")
    pca.fit(corr)
    n_components = (
        pca.explained_variance_ratio_ > 1 / len(pca.explained_variance_ratio_)
    ).sum()

    if plot:
        plot_explained_variance(pca, n_components)

    pca = PCA(n_components=n_components)
    X_transformed = pca.fit_transform(X)
    cols = [f"factor_{i}" for i in range(n_components)]
    df = pd.DataFrame(X_transformed, index=etalon_periods.index, columns=cols)

    lambda_inv = linalg.inv(
        np.dot(X_transformed.T, X_transformed) / (X_transformed.shape[0] - 1)
    )
    df["Hotelling's T-squared"] = df.T.apply(
        lambda t: np.dot(np.dot(t, lambda_inv), t.T)
    )
    errors = X - np.dot(X_transformed, pca.components_)
    df["Q residuals"] = errors.T.apply(lambda e: np.dot(e, e.T))
    df["equipment"] = etalon_periods["equipment"]

    return df, scaler, pca


def plot_explained_variance(pca, n_components: int):
    explained_variance = pca.explained_variance_ratio_[:n_components].sum().round(3)
    comps = pca.explained_variance_ratio_[:n_components].round(3).tolist()

    print(f"n_components = {n_components}")
    print(f"explained_variance = {explained_variance}")
    print(f"variance explained by {n_components} components: ", *comps, "\n")

    plt.plot(pca.explained_variance_ratio_)
    plt.title("Variance ratio explained by component")
    plt.xlabel("The number of component")
    plt.ylabel("Explained variance ratio")
