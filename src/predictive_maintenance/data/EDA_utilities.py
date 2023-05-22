import logging
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from pymystem3 import Mystem
from typing import Optional

nltk.download("stopwords")


logger = logging.getLogger(__name__)

__all__ = ["group_messages_by_equipment"]

GROUPS = {
    "oil_system": [
        "ГСМ ЭКСГ. №",
        "ДВИГАТЕЛЬ ПУСКОВОГО МАСЛОНАСОСА ЭКСГ. №",
        "ДВИГАТЕЛЬ РЕЗЕРВНОГО МАСЛОНАСОСА ЭКСГ.№",
        "ЗАПОРНАЯ АРМАТУРА ЭКСГАУСТЕРА №",
        "КОЖУХ МУФТЫ ЭКСГ. №",
        "МАСЛОБАК ЭКСГ. №",
        "МАСЛОНАСОС РАБОЧИЙ ЭКСГ. №",
        "МАСЛОНАСОС ШЕСТЕРЕНЧАТЫЙ (ПУСК.) ЭКСГ.№",
        "МАСЛОНАСОС ШЕСТЕРЕНЧАТЫЙ (РЕЗ.) ЭКСГ. №",
        "МАСЛООХЛАДИТЕЛЬ  М-05-1 ЭКСГ. №",
        "МАСЛОПРОВОДЫ ЭКСГАУСТЕРА №",
        "МАСЛОСТАНЦИЯ ЖИДКОЙ СМАЗКИ ЭКСГ. №",
        "МАСЛЯНЫЙ ФИЛЬТР ЭКСГАУСТЕРА №",
        "МЕТРАН-100 ДАТЧИКИ ДАВЛЕНИЯ ЭКСГ.№",
        "ТСМТ-101-010-50М-80 ТЕРМОПРЕОБР.МАСЛО",
    ],
    "bearings": [
        "ВК 310С ВИБРОПРЕОБРАЗОВАТЕЛЬ ЭКСГ.№ Т.1",
        "ВК 310С ВИБРОПРЕОБРАЗОВАТЕЛЬ ЭКСГ.№ Т.2",
        "ВК 310С ВИБРОПРЕОБРАЗОВАТЕЛЬ ЭКСГ.№ Т.3",
        "ВК 310С ВИБРОПРЕОБРАЗОВАТЕЛЬ ЭКСГ.№ Т.4",
        "ПОДШИПНИК ОПОРНО-УПОРНЫЙ ЭКСГ. №",
        "ПОДШИПНИК ОПОРНЫЙ ЭКСГ. №",
        "ПОДШИПНИК ОПОРНЫЙ №1",
        "ПОДШИПНИК ОПОРНЫЙ №2",
        "РОТОР ЭКСГ. №",
        "ТСМТ-101-010-50М-400 ТЕРМОПР.ПОДШ.Т.1",
        "ТСМТ-101-010-50М-400 ТЕРМОПР.ПОДШ.Т.2",
        "ТСМТ-101-010-50М-200 ТЕРМОПР.ПОДШ.Т.3",
        "ТСМТ-101-010-50М-200 ТЕРМОПР.ПОДШ.Т.4",
    ],
    "electric_devices": [
        "КЛ1 ТР№ ДО ЭД ЭКСГАУСТЕРА №",
        "КЛ2 ТР№ ДО ЭД ЭКСГАУСТЕРА №",
        "ТИРИСТОРНЫЙ ВОЗБУДИТЕЛЬ СПВД-М10-400-5",
        "ТИРИСТОРНЫЙ ВОЗБУДИТЕЛЬ ТВ-400 ЭКСГ ВУ1",
        "ТИРИСТ. ВОЗБУДИТЕЛЬ ВТ-РЭМ-400 ЭКСГ ВУ1",
        "ТИРИСТ. ВОЗБУДИТЕЛЬ ВТ-РЭМ-400 ЭКСГ ВУ2",
        "ЭЛЕКТРОАППАРАТУРА ЭКСГ. №",
        "ЭЛЕКТРОДВИГАТЕЛЬ ДСПУ-140-84-4 ЭКСГ. №",
    ],
    "fittings": [
        "ГАЗОВАЯ ЗАДВИЖКА ЭКСГАУСТЕРА А/М №",
        "ЗАДВИЖКА ЭКСГ. №",
        "ЗАП. И РЕГ. АРМАТУРА ЭКСГ.№",
        "КОРПУС ЭКСГ. №",
        "РЕДУКТОР ГАЗ. ЗАДВИЖКИ ЭКСГ. №",
        "РЕГУЛИРУЮЩАЯ АППАРАТУРА ЭКСГАУСТЕРА №",
        "САПФИР 22 МДД ПЕРЕПАД ДАВЛ. НА ЦИКЛОНЕ",
        "САПФИР 22 МДД РАЗРЕЖЕНИЕ В КОЛЛЕКТОРЕ",
        "ТР-Р ТМ-4000-10/6 ЭКСГ. №",
        "ТР-Р ТМ-6300-10/6 ЭКСГ. №",
        "УЛИТА ЭКСГ. №",
        "ЭЛ/ДВИГАТЕЛЬ ГАЗ. ЗАДВИЖКИ ЭКСГ. №",
        "ЭЛЕКТРООБОРУДОВАНИЯ ЭКСГАУСТЕРА №",
    ],
    "other": ["ЭКСГАУСТЕР А/М №", "ЭКСГАУСТЕР Н-8000 А/М №"],
}
MAX_FEATURES = 150


def add_groups_to_messages(messages, groups: Optional[dict] = None):
    """
    Classifies equipment into larger groups.
    """
    if groups is None:
        groups = GROUPS

    for group in groups.keys():
        messages.loc[messages["unified_name"].isin(groups[group]), "groups"] = group

    return messages


def identify_lack_of_messages_in_y_train(
    messages: pd.DataFrame,
    unified_tech_places: pd.DataFrame,
):
    """
    Identifies equipment failures and anomalies in y_train without
    description in messages.
    """
    y_spec = pd.pivot_table(
        unified_tech_places,
        index="unified_name",
        columns="equipment",
        values="description",
        aggfunc="count",
        margins=True,
    )
    y_spec.sort_index(inplace=True)

    no_messages = list(
        set(unified_tech_places["unified_name"]) - set(messages["unified_name"])
    )
    messages_spec = pd.pivot_table(
        messages,
        index="unified_name",
        columns="equipment",
        values="ОПИСАНИЕ",
        aggfunc="count",
        margins=True,
    )
    for tech_place in no_messages:
        messages_spec.loc[tech_place, :] = np.nan
    messages_spec.sort_index(inplace=True)

    a = np.where(y_spec.values > 0, 1, 0)
    b = np.where(messages_spec.values > 0, 1, 0)
    missing_messages = pd.DataFrame(
        (a - b)[1:, :-1], index=y_spec.index[1:], columns=y_spec.columns[:-1]
    )
    missing_messages.loc[:, "All"] = (a - b)[1:, :-1].sum(axis=1)
    missing_messages = missing_messages[missing_messages["All"] > 0]

    return messages_spec, missing_messages


def get_vectorizer_and_messages_vectors(
    messages: pd.DataFrame, max_features: Optional[int] = None
):
    """
    Vectorizes messages and returns fitted vectorizer with
    sparse matrix representing descriptions in messages.
    """
    if max_features is None:
        max_features = MAX_FEATURES

    morph = Mystem()

    def preprocess_text(text):
        return " ".join(morph.lemmatize(text))

    text = messages["ОПИСАНИЕ"].apply(preprocess_text)

    stop_words = [
        "эксг.",
        "эк",
        "co",
        "6кв",
        "ввд",
        "№",
        "2ру",
        "22",
        "40",
        "\.",
    ]

    params = {
        "stop_words": stopwords.words("russian") + stop_words,
        "ngram_range": (1, 2),
        "max_df": 0.9,
        "max_features": max_features,
    }
    vectorizer = TfidfVectorizer(**params)
    X = vectorizer.fit_transform(text)

    return vectorizer, X
