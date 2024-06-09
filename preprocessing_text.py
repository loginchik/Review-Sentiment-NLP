"""
Набор функций для предварительной обработки текста.
"""

import re

import numpy as np
import pymorphy2


morph = pymorphy2.MorphAnalyzer()


def preprocess_text(raw_text) -> str:
    """
    Удаляет из текста все знаки, кроме букв русского алфавита, пробелов и дефисов (-).
    Заменяет множественные пробелы на одинарный, букву ё на букву е, переводит все знаки в нижний регистр.

    Args:
        raw_text (str): Сырой текст для обработки.

    Returns:
        str: Обработанный текст.
    """
    raw_text = re.sub(r'[^А-яËё\s\-]', '', raw_text)
    raw_text = re.sub(r'\s+', ' ', raw_text)
    return raw_text.lower()


def tokenize_text(raw_text, stop_words):
    """
    Делит слова на токены по пробелу и удаляет токены, которые представляют собой чистый дефис (-).
    Переводит все слова в начальную форму с помощью pymorphy2.MorphAnalyzer() и удаляет из токенов
    стоп-слова, указанные в stop_words.

    Args:
        raw_text (str): Сырой текст для токенизации.
        stop_words (tuple[str]): Список стоп-слов для исключения из токенов.

    Returns:
        list[str]: Список токенов из текста.
    """
    tokens = raw_text.split()
    tokens = [token for token in tokens if token != '-']
    tokens = list(map(lambda x: morph.parse(x)[0].normal_form, tokens))
    tokens = [t.replace('ё', 'е') for t in tokens]
    tokens = [t for t in tokens if t not in stop_words]

    return tokens


def vectorize_text(tokens, parts_dict, model, vector_size=300):
    """
    Производит векторизацию текста с помощью переданной модели Word2Vec.

    Среди токенов находит соответствующие в parts_dict, исходя из чего определяется вектор модели model.
    Если токен отсутствует в словаре векторов, то на его место помещается набор нулей, по длине = vector_size.
    Итоговый вектор текста - усреднённый вектор векторов каждого токена в тексте.

    Args:
        tokens (list[str] | tuple[str]): Список токенов, составляющих текст.
        parts_dict (dict[str, str]): Словарь соответствия токенов из текста токенам из модели.
        model: Модель Word2Vec для векторизации.
        vector_size (int): Размерность вектора. По умолчанию 300.

    Returns:
        numpy.ndarray: Итоговый вектор текста.
    """
    resulting_vectors = []
    for token in tokens:
        if token in parts_dict:
            target_model_key = parts_dict[token]
            vector = model[target_model_key]
            resulting_vectors.append(vector)
        else:
            resulting_vectors.append(np.zeros(vector_size))
    return np.mean(resulting_vectors, axis=0)
