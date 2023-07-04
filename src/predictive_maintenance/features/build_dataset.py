import logging
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Optional


logger = logging.getLogger(__name__)

__all__ = ["generate_dataset_for_Bayesian_model"]


INPUT_SEQUENCE_LENGTH = 23


def get_T2_Q_from_LSTM_etalon(
    LSTM_model,
    etalon_dataset,
    input_sequence_length: Optional[int] = None,
):
    """
    Generates dataset for training of the Bayesian model.    
    """
    if input_sequence_length is None:
        input_sequence_length = INPUT_SEQUENCE_LENGTH

    pred_etalon = LSTM_model.predict(etalon_dataset[:, :input_sequence_length, :])
    T2_Q_from_LSTM_etalon = pred_etalon[:, :, -2:].reshape(-1, 2)

    scaler_bmb = StandardScaler()
    T2_Q_from_LSTM_etalon = scaler_bmb.fit_transform(T2_Q_from_LSTM_etalon)
    T2_Q_from_LSTM_etalon = pd.DataFrame(T2_Q_from_LSTM_etalon, columns=["T2", "Q"])
    T2_Q_from_LSTM_etalon["good"] = 1

    return T2_Q_from_LSTM_etalon, scaler_bmb
