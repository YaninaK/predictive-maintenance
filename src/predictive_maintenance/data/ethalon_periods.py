import logging
import pandas as pd
import numpy as np
from typing import Optional

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import linalg

from .make_dataset import load_X
import matplotlib.pyplot as plt


logger = logging.getLogger(__name__)

__all__ = ["generate_ethalon_dataset"]


PATH = ""

START = pd.Timestamp("2019-01-16 13:00:00")
END = pd.Timestamp("2021-12-31 23:59:00")
TIME_TO_STOPPAGE = pd.Timedelta(50, "H")
TIME_FROM_STOPPAGE = pd.Timedelta(1, "H")
INPUT_PERIOD = pd.Timedelta(23, "H")

FREQUENCY = "1H"

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


def generate_ethalon_dataset(
    messages: pd.DataFrame,
    path: Optional[str] = None,
    freq: Optional[str] = None,
    max_vibration: Optional[float] = None,
    max_temperature: Optional[float] = None,
    vibration_columns: Optional[list] = None,
    temperature_columns: Optional[list] = None,
    time_to_stoppage: Optional[pd.Timedelta] = None,
    time_from_stoppage: Optional[pd.Timedelta] = None,
    plot: Optional[bool] = None,
    seed: Optional[int] = None,
):
    """
    Generates dataset for LSTM ethalon model.
    """

    if path is None:
        path = PATH
    if freq is None:
        freq = FREQUENCY
    if max_vibration is None:
        max_vibration = MAX_VIBRATION
    if max_temperature is None:
        max_temperature = MAX_TEMPERATURE
    if vibration_columns is None:
        vibration_columns = VIBRATION_COLUMNS
    if temperature_columns is None:
        temperature_columns = TEMPERATURE_COLUMNS
    if time_to_stoppage is None:
        time_to_stoppage = TIME_TO_STOPPAGE
    if time_from_stoppage is None:
        time_from_stoppage = TIME_FROM_STOPPAGE
    if plot is None:
        plot = PLOT
    if seed is None:
        seed = SEED

    ethalon_periods = select_ethalon_periods(
        messages,
        path,
        freq,
        max_vibration,
        max_temperature,
        vibration_columns,
        temperature_columns,
        time_to_stoppage,
        time_from_stoppage,
    )
    df, scaler, pca = get_pca_components(ethalon_periods, plot)

    t = pd.Timedelta(freq)
    ethalon_dataset = []
    for e in range(4, 10):
        df_ = df[df["equipment"] == e]
        t0 = df_.index[0]
        for t1 in df_.index[1:]:
            if t1 == t0 + time_to_stoppage:
                ethalon_dataset.append(df_[t0 : t1 - t])
                t0 += t
            elif t1 > t0 + time_to_stoppage:
                t0 = t1
    ethalon_dataset = np.stack(ethalon_dataset, axis=0)[:, :, :-1]

    ind = np.arange(len(ethalon_dataset))
    np.random.seed(seed)
    np.random.shuffle(ind)

    return ethalon_dataset[ind], scaler, pca


def select_ethalon_periods(
    messages: pd.DataFrame,
    path: Optional[str] = None,
    freq: Optional[str] = None,
    max_vibration: Optional[float] = None,
    max_temperature: Optional[float] = None,
    vibration_columns: Optional[list] = None,
    temperature_columns: Optional[list] = None,
    time_to_stoppage: Optional[pd.Timedelta] = None,
    time_from_stoppage: Optional[pd.Timedelta] = None,
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
    if max_vibration is None:
        max_vibration = MAX_VIBRATION
    if max_temperature is None:
        max_temperature = MAX_TEMPERATURE
    if vibration_columns is None:
        vibration_columns = VIBRATION_COLUMNS
    if temperature_columns is None:
        temperature_columns = TEMPERATURE_COLUMNS
    if time_to_stoppage is None:
        time_to_stoppage = TIME_TO_STOPPAGE
    if time_from_stoppage is None:
        time_from_stoppage = TIME_FROM_STOPPAGE

    messages = messages[messages["ДАТА_УСТРАНЕНИЯ_НЕИСПРАВНОСТИ"].notnull()]
    ethalon_periods = pd.DataFrame()
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
        t = pd.Timedelta(freq)

        ethalon = pd.DataFrame()
        for t1 in ind[1:]:
            if t1 - t0 > t:
                t_end = t0
                if t_end - t_start >= time_to_stoppage:
                    ethalon = pd.concat([ethalon, X[t_start:t_end]])
                t_start = t1
            t0 = t1

        old_cols = ethalon.columns.tolist()
        new_cols = [f"col_{i}" for i in range(ethalon.shape[1])]
        ethalon.rename(
            columns={old_cols[i]: new_cols[i] for i in range(len(old_cols))},
            inplace=True,
        )
        ethalon["equipment"] = equipment

        ethalon_periods = pd.concat([ethalon_periods, ethalon], axis=0)

    return ethalon_periods


def get_pca_components(
    ethalon_periods,
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

    X = ethalon_periods.iloc[:, :-1].copy()

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
    df = pd.DataFrame(X_transformed, index=ethalon_periods.index)

    lambda_inv = linalg.inv(
        np.dot(X_transformed.T, X_transformed) / (X_transformed.shape[0] - 1)
    )
    df["Hotelling's T-squared"] = df.T.apply(
        lambda t: np.dot(np.dot(t, lambda_inv), t.T)
    )
    errors = X - np.dot(X_transformed, pca.components_)
    df["Q residuals"] = errors.T.apply(lambda e: np.dot(e, e.T))
    df["equipment"] = ethalon_periods["equipment"]

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
