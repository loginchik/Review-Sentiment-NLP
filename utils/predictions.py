import json

import joblib

import pandas as pd
import gensim.downloader as api
from sklearn.pipeline import Pipeline

try:
    import preprocessing_text
except ModuleNotFoundError:
    from utils import preprocessing_text


# Загрузка моделей, используемых далее
word2vec = api.load('word2vec-ruscorpora-300')
normalizer = joblib.load('pretrained_models/normalizer.joblib')
pca = joblib.load('pretrained_models/pca.joblib')
random_forest = joblib.load('pretrained_models/random_forest.joblib')
svc = joblib.load('pretrained_models/svc.joblib')
voting_classifier = joblib.load('pretrained_models/voting_classifier.joblib')

# Сценарии предсказания
forest_pipe = Pipeline([('normalizer', normalizer), ('pca', pca), ('classifier', random_forest)])
svc_pipe = Pipeline([('normalizer', normalizer), ('pca', pca), ('classifier', svc)])
voting_pipe = Pipeline([('normalizer', normalizer), ('pca', pca), ('classifier', svc)])


def make_prediction(text, model='all', full_report=True):
    """
    Предсказание вида комментария на основе сырого текста.

    Предварительно обрабатывает текст: удаляет недопустимые символы, разделяет на токены, проводит векторизацию
    с помощью модели Word2Vec. После этого проводит предсказания с помощью выбранной в model модели.

    Показатели precision доступных моделей:

    + Random Forest Classifier = 0.73
    + SVC = 0.79
    + Voting Classifier = 0.79

    Args:
        text (str): Изначальный текст для определения вида комментария.
        model (str): rf / svc / voting / all, где rf - Random Forest Classifier, svc - SVC, voting - Voting Classifier,
            включающий в себя Random Forest Classifier и SVC, all - все модели.
        full_report (bool): Нужно ли выводить вероятности попадания комментария в тот или иной класс.

    Returns:
        dict[str, dict[str, str | float]]: Словарь {``модель``: {``prediction``: вид комментария, ``negative_proba``:
            вероятность попадания комментария в класс негативных или нейтральных, ``positive_proba``: вероятность
            попадания комментария в класс позитивных}}

    """
    if model not in ['all', 'rf', 'svc', 'voting']:
        raise ValueError('Model must be "all", "rf", "svc" or "voting"')

    # Фиксируем названия итоговых классов
    classes = {0: 'негативный или нейтральный', 1: 'позитивный'}
    # Подгружаем словарь соответствий слов ключам модели Word2Vec
    with open('data/vocab_parts.json') as f:
        vocab_parts = json.load(f)

    # Подготовка текста: очистка от недопустимых знаков, токенизация и векторизация
    prepared_text = pd.DataFrame([preprocessing_text.vectorize_text(
        preprocessing_text.tokenize_text(preprocessing_text.preprocess_text(text), stop_words=tuple()),
        parts_dict=vocab_parts, model=word2vec, vector_size=300)])

    results = []
    numeric_results = dict()

    if model == 'rf' or model == 'all':
        numeric_results['rf'] = dict()
        model_report = ['>> Random Forest Classifier']
        forest_prediction = forest_pipe.predict(prepared_text)[0]
        numeric_results['rf']['prediction'] = classes[forest_prediction]
        if full_report:
            forest_prediction_proba = forest_pipe.predict_proba(prepared_text)[0]
            model_report.extend([f'Вероятность негативного или нейтрального: {round(forest_prediction_proba[0] * 100, 2)} %',
                                 f'Вероятность позитивного: {round(forest_prediction_proba[1] * 100, 2)} %'])
            numeric_results['rf']['negative_proba'] = forest_prediction_proba[0]
            numeric_results['rf']['positive_proba'] = forest_prediction_proba[1]
        model_report.append(f'>> Вердикт: {classes[forest_prediction].upper()}')
        results.append(model_report)

    if model == 'svc' or model == 'all':
        numeric_results['svc'] = dict()
        model_report = ['>> SVC']
        svc_prediction = svc_pipe.predict(prepared_text)[0]
        numeric_results['svc']['prediction'] = classes[svc_prediction]
        if full_report:
            svc_prediction_proba = svc_pipe.predict_proba(prepared_text)[0]
            model_report.extend(
                [f'Вероятность негативного или нейтрального: {round(svc_prediction_proba[0] * 100, 2)} %',
                 f'Вероятность позитивного: {round(svc_prediction_proba[1] * 100, 2)} %'])
            numeric_results['svc']['negative_proba'] = svc_prediction_proba[0]
            numeric_results['svc']['positive_proba'] = svc_prediction_proba[1]
        model_report.append(f'>> Вердикт: {classes[svc_prediction].upper()}')
        results.append(model_report)

    if model == 'voting' or model == 'all':
        numeric_results['voting'] = dict()
        model_report = ['>> Voting Classifier']
        voting_prediction = voting_pipe.predict(prepared_text)[0]
        numeric_results['voting']['prediction'] = classes[voting_prediction]
        if full_report:
            voting_prediction_proba = voting_pipe.predict_proba(prepared_text)[0]
            model_report.extend(
                [f'Вероятность негативного или нейтрального: {round(voting_prediction_proba[0] * 100, 2)} %',
                 f'Вероятность позитивного: {round(voting_prediction_proba[1] * 100, 2)} %'])
            numeric_results['voting']['negative_proba'] = voting_prediction_proba[0]
            numeric_results['voting']['positive_proba'] = voting_prediction_proba[1]
        model_report.append(f'>> Вердикт: {classes[voting_prediction].upper()}')
        results.append(model_report)

    for result in results:
        print(*result, sep='\n', end='\n\n')

    return numeric_results
