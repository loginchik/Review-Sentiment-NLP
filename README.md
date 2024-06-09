# Модель классификации отзывов на мультфильмы

Скрипт для автоматизированного сбора отзывов с Кинопоиска: [Kinopoisk Reviews](https://github.com/loginchik/Kinopoisk-Reviews)

Система отзывов (рецензий) на Кинопоиске устроена таким образом, что каждый зарегистрированный на сайте пользователь 
может написать рецензию на фильм и опубликовать её в публику. Перед публикацией пользователю необходимо определить 
сентимент рецензии: позитивная, нейтральная или негативная. В рамках проекта строится предсказательная модель, которая
на основе 5860 отзывов на различные мультфильмы предсказывает с точностью 70+% сентимент отзыва на основе его текста. 

В изначальных данных наблюдался значительный перекос в сторону позитивных отзывов, поэтому итоговая модель работает 
с двумя классами: позитивный и нейтрально-негативный. Кроме того, в процессе обучения дисбаланс классов компенсировался 
методом ADASYN. 

Практическое применение модели может быть таким: фильтрация рецензий для анализа плюсов и недостатков анимационных 
работ других режиссёров и совершенствования собственной другими авторами. 

## Содержимое проекта

+ Построение модели: [model_construction.ipynb](model_construction.ipynb)
+ Применение модели к реальным данным: [usage.ipynb](usage.ipynb)
+ Модель случайного леса (классификатор): [random_forest.joblib](pretrained_models/random_forest.joblib) (precision = 73%)
+ Модель ядерного метода опорных векторов: [svc.joblib](pretrained_models/svc.joblib) (precision = 79%)
+ Модель, объединяющая две других модели по принципу sort-voting: [voting_classifier.joblib](pretrained_models/voting_classifier.joblib) (precision = 79%)

