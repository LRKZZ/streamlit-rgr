import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import rand_score
from xgboost import XGBClassifier
from sklearn.ensemble import BaggingClassifier, StackingClassifier
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import Dense, Dropout
import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score

model_save_path = 'D:/vscode/rgr-streamlit/streamlit-models/'

csgo_data = pd.read_csv('csgo.csv')
X = csgo_data.drop('bomb_planted_True', axis=1)
y = csgo_data['bomb_planted_True']

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def load_models():
    model_ml1 = pickle.load(open(model_save_path + 'model_ml1.pkl', 'rb'))
    model_ml4 = pickle.load(open(model_save_path + 'model_ml4.pkl', 'rb'))
    model_ml5 = pickle.load(open(model_save_path + 'model_ml5.pkl', 'rb'))
    model_ml3 = XGBClassifier()
    model_ml3.load_model(model_save_path + 'model_ml3.json')
    model_ml6 = load_model(model_save_path + 'model_ml6.h5')
    model_ml2 = pickle.load(open(model_save_path + 'kmeans_model.pkl', 'rb'))
    return model_ml1, model_ml3, model_ml4, model_ml5, model_ml6, model_ml2

st.markdown("""
<style>
.sidebar .sidebar-content {
    background-color: #f1f3f6;
}
h1 {
    color: #0e1117;
}
</style>
""", unsafe_allow_html=True)

# Сайдбар для навигации
st.sidebar.image("kawasaki.png", width=100)
st.sidebar.title("Навигация")
page = st.sidebar.radio(
    "Выберите страницу:",
    ("Информация о разработчике", "Информация о наборе данных", "Визуализации данных", "Предсказание модели ML")
)

csgo_data = pd.read_csv('csgo.csv')

# Функции для каждой страницы
def page_developer_info():
    st.title("Информация о разработчике")
    
    col1, col2 = st.columns(2)
    with col1:
        st.header("Контактная информация")
        st.write("ФИО: Григор Александр Антонович")
        st.write("Номер учебной группы: ФИТ-222")
    
    with col2:
        st.header("Фотография")
        st.image("legenda.jpg", width=200)  # Укажите путь к вашей фотографии
    
    st.header("Тема РГР")
    st.write("Разработка Web-приложения для инференса моделей ML и анализа данных")

def page_dataset_info():
    st.title("Информация о наборе данных")

    # Добавляем красивое оформление описания датасета
    st.markdown("""
    ## Описание Датасета CS:GO
    **Файл датасета:** `csgo.csv`

    **Описание:**
    Данный датасет содержит статистическую информацию о матчах в популярной компьютерной игре Counter-Strike: Global Offensive (CS:GO). Включает следующие столбцы:

    - `index`: Индекс записи.
    - `time_left`: Время до конца раунда.
    - `ct_score`: Счёт команды контр-террористов.
    - `t_score`: Счёт команды террористов.
    - `ct_health`: Общее здоровье команды контр-террористов.
    - `t_health`: Общее здоровье команды террористов.
    - `ct_armor`: Уровень брони команды контр-террористов.
    - `t_armor`: Уровень брони команды террористов.
    - `ct_money`: Деньги команды контр-террористов.
    - `t_money`: Деньги команды террористов.
    - `ct_helmets`: Шлемы команды контр-террористов.
    - `t_helmets`: Шлемы команды террористов.
    - `ct_defuse_kits`: Комплекты для обезвреживания бомбы у CT.
    - `ct_players_alive`: Живые игроки команды контр-террористов.
    - `t_players_alive`: Живые игроки команды террористов.
    - `bomb_planted_True`: Индикатор заложенной бомбы.
                
    **Особенности предобработки данных:**
    - Удаление лишних столбцов, например, 'index'.
    - Обработка пропущенных значений.
    - Нормализация числовых данных для улучшения производительности моделей.
    - Кодирование категориальных переменных.
    """)

def page_data_visualization():
    st.title("Визуализации данных CS:GO")

    # Визуализация 1: Распределение здоровья команд
    fig, ax = plt.subplots()

    sns.histplot(csgo_data['ct_health'], kde=True, color="blue", label='CT Health', ax=ax)
    sns.histplot(csgo_data['t_health'], kde=True, color="red", label='T Health', ax=ax)
    ax.set_title('Распределение здоровья команд')
    ax.legend()
    st.pyplot(fig)

    # Визуализация 2: Сравнение счёта команд
    fig, ax = plt.subplots()
    sns.boxplot(data=csgo_data[['ct_score', 't_score']], ax=ax)
    ax.set_title('Сравнение счёта команд')
    st.pyplot(fig)

    # Визуализация 3: Общий уровень брони команд
    fig, ax = plt.subplots()
    sns.kdeplot(csgo_data['ct_armor'], shade=True, color="blue", label='CT Armor', ax=ax)
    sns.kdeplot(csgo_data['t_armor'], shade=True, color="red", label='T Armor', ax=ax)
    ax.set_title('Общий уровень брони команд')
    ax.legend()
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=csgo_data, x='index', y='ct_money', label='CT Money', ax=ax)
    sns.lineplot(data=csgo_data, x='index', y='t_money', label='T Money', ax=ax)
    ax.set_title('Количество денег у команд в разных раундах')
    ax.set_xlabel('Индекс раунда')
    ax.set_ylabel('Деньги')
    ax.legend()
    st.pyplot(fig)
    

# Функция для загрузки моделей
def load_models():
    model_ml1 = pickle.load(open(model_save_path + 'model_ml1.pkl', 'rb'))
    model_ml4 = pickle.load(open(model_save_path + 'model_ml4.pkl', 'rb'))
    model_ml5 = pickle.load(open(model_save_path + 'model_ml5.pkl', 'rb'))
    model_ml3 = XGBClassifier()
    model_ml3.load_model(model_save_path + 'model_ml3.json')
    model_ml6 = load_model(model_save_path + 'model_ml6.h5')
    model_ml2 = pickle.load(open(model_save_path + 'kmeans_model.pkl', 'rb'))
    return model_ml1, model_ml3, model_ml4, model_ml5, model_ml6, model_ml2

def page_ml_prediction():
    st.title("Предсказания моделей машинного обучения")

    # Виджет для загрузки файла
    uploaded_file = st.file_uploader("Загрузите ваш CSV файл", type="csv")

    # Интерактивный ввод данных, если файл не загружен
    if uploaded_file is None:
        st.subheader("Введите данные для предсказания:")

        # Интерактивные поля для ввода данных
        input_data = {}
        feature_names = ['index','time_left','ct_score','t_score','ct_health','t_health','ct_armor','t_armor','ct_money','t_money','ct_helmets','t_helmets','ct_defuse_kits','ct_players_alive','t_players_alive']
        for feature in feature_names:
            input_data[feature] = st.number_input(f"{feature}", min_value=0.0, max_value=100000.0, value=50.0)

        if st.button('Сделать предсказание'):
            # Загрузка моделей
            model_ml1, model_ml3, model_ml4, model_ml5, model_ml6, model_ml2 = load_models()

            input_df = pd.DataFrame([input_data])
            
            # Проверяем, что данные изменились
            st.write("Входные данные:", input_df)

            # Используем масштабировщик, обученный на обучающих данных
            scaler = StandardScaler().fit(X_train)
            scaled_input = scaler.transform(input_df)

            

            # Делаем предсказания
            prediction_ml1 = model_ml1.predict(scaled_input)
            prediction_ml3 = model_ml3.predict(scaled_input)
            prediction_ml4 = model_ml4.predict(scaled_input)
            prediction_ml5 = model_ml5.predict(scaled_input)
            prediction_ml6 = (model_ml6.predict(scaled_input) > 0.5).astype(int) # Для нейронной сети

            # Вывод результатов
            st.success(f"Результат предсказания LogisticRegression: {prediction_ml1[0]}")
            st.success(f"Результат предсказания XGBClassifier: {prediction_ml3[0]}")
            st.success(f"Результат предсказания BaggingClassifier: {prediction_ml4[0]}")
            st.success(f"Результат предсказания StackingClassifier: {prediction_ml5[0]}")
            st.success(f"Результат предсказания нейронной сети Tensorflow: {prediction_ml6[0]}")
    else:
        try:
            model_ml2 = pickle.load(open(model_save_path + 'kmeans_model.pkl', 'rb'))
            model_ml1 = pickle.load(open(model_save_path + 'model_ml1.pkl', 'rb'))
            model_ml4 = pickle.load(open(model_save_path + 'model_ml4.pkl', 'rb'))
            model_ml5 = pickle.load(open(model_save_path + 'model_ml5.pkl', 'rb'))
            model_ml3 = XGBClassifier()
            model_ml3.load_model(model_save_path + 'model_ml3.json')
            model_ml6 = load_model(model_save_path + 'model_ml6.h5')

            # Сделать предсказания на тестовых данных
            cluster_labels = model_ml2.predict(X_test)
            predictions_ml1 = model_ml1.predict(X_test)
            predictions_ml4 = model_ml4.predict(X_test)
            predictions_ml5 = model_ml5.predict(X_test)
            predictions_ml3 = model_ml3.predict(X_test)
            predictions_ml6 = model_ml6.predict(X_test).round() # Округление для нейронной сети

            # Оценить результаты
            rand_score_ml2 = rand_score(y_test, cluster_labels)
            accuracy_ml1 = accuracy_score(y_test, predictions_ml1)
            accuracy_ml4 = accuracy_score(y_test, predictions_ml4)
            accuracy_ml5 = accuracy_score(y_test, predictions_ml5)
            accuracy_ml3 = accuracy_score(y_test, predictions_ml3)
            accuracy_ml6 = accuracy_score(y_test, predictions_ml6)

            st.success(f"rand_score KMeans: {rand_score_ml2}")
            st.success(f"Точность LogisticRegression: {accuracy_ml1}")
            st.success(f"Точность XGBClassifier: {accuracy_ml4}")
            st.success(f"Точность BaggingClassifier: {accuracy_ml5}")
            st.success(f"Точность StackingClassifier: {accuracy_ml3}")
            st.success(f"Точность нейронной сети Tensorflow: {accuracy_ml6}")
        except Exception as e:
            st.error(f"Произошла ошибка при чтении файла: {e}")


if page == "Информация о разработчике":
    page_developer_info()
elif page == "Информация о наборе данных":
    page_dataset_info()
elif page == "Визуализации данных":
    page_data_visualization()
elif page == "Предсказание модели ML":
    page_ml_prediction()
