# CS:GO Match Data Analysis and ML Dashboard

# Обзор

Этот проект посвящен анализу данных и предсказаниям, основанным на статистической информации о матчах в популярной компьютерной игре Counter-Strike: Global Offensive (CS:GO). В работе используется датасет csgo.csv, содержащий детальные данные о различных аспектах игровых матчей, включая счет команд, здоровье, броню, финансовое состояние и другие важные показатели. Основная цель проекта - разработать и внедрить модели машинного обучения для предсказания исходов матчей и визуализировать данные через интерактивный дашборд, созданный с помощью Streamlit.

# Содержание Датасета
Файл csgo.csv включает в себя следующие ключевые столбцы:

- index: Индекс записи.
- time_left: Время до конца раунда.
- ct_score: Счёт команды контр-террористов.
- t_score: Счёт команды террористов.
- ct_health: Общее здоровье команды контр-террористов.
- t_health: Общее здоровье команды террористов.
- ct_armor: Уровень брони команды контр-террористов.
- t_armor: Уровень брони команды террористов.
- ct_money: Деньги команды контр-террористов.
- t_money: Деньги команды террористов.
- ct_helmets: Шлемы команды контр-террористов.
- t_helmets: Шлемы команды террористов.
- ct_defuse_kits: Комплекты для обезвреживания бомбы у CT.
- ct_players_alive: Живые игроки команды контр-террористов.
- t_players_alive: Живые игроки команды террористов.
- bomb_planted_True: Индикатор заложенной бомбы.

# Предобработка данных:

Удаление лишних столбцов: Например, столбец index.
Обработка пропущенных значений: Для обеспечения целостности данных.
Нормализация числовых данных: Для повышения эффективности моделей машинного обучения.
Кодирование категориальных переменных: Для преобразования номинальных данных в формат, пригодный для обработки моделями ML.

# Интерактивный Дашборд

Проект включает в себя разработку дашборда на Streamlit, который позволяет пользователям взаимодействовать с данными и моделями машинного обучения. Дашборд содержит следующие функции:

Визуализация данных: Графики и диаграммы для наглядного представления данных из датасета.
Инференс моделей ML: Возможность запускать предсказания на основе загруженных данных с использованием различных моделей машинного обучения.
Анализ результатов: Сравнение результатов различных моделей и оценка их эффективности.

# Использованные Технологии и Библиотеки:

- Streamlit
- Pandas
- Scikit-learn
- XGBoost
- TensorFlow
- Matplotlib
- Seaborn
