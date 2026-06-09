# индивидульное задание 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

class KMeansMicroservice:
    """
    Микросервис для кластеризации данных методом K-средних
    Данные содержат 3 параметра:
    - Параметр 1: [0, 1]
    - Параметр 2: [-2, 2]
    - Параметр 3: 'да' или 'нет'
    """

    def __init__(self, n_samples=300, random_state=42):
        """
        Инициализация микросервиса

        Parameters:
        -----------
        n_samples : int - количество записей
        random_state : int - seed для воспроизводимости
        """
        self.n_samples = n_samples
        self.random_state = random_state
        self.data = None          # исходные данные (с категориями)
        self.data_encoded = None  # закодированные данные
        self.kmeans = None
        self.optimal_k = None

    def generate_data(self):
        """
        Генерация данных согласно варианту:
        - Параметр 1: равномерное распределение [0, 1]
        - Параметр 2: равномерное распределение [-2, 2]
        - Параметр 3: 'да' или 'нет' (случайно)
        """
        print("\n" + "="*60)
        print("ГЕНЕРАЦИЯ ДАННЫХ ПО ВАРИАНТУ")
        print("="*60)

        np.random.seed(self.random_state)

        # Параметр 1: диапазон [0, 1]
        param1 = np.random.uniform(0, 1, self.n_samples)

        # Параметр 2: диапазон [-2, 2]
        param2 = np.random.uniform(-2, 2, self.n_samples)

        # Параметр 3: 'да' или 'нет' (вероятность 50/50)
        param3 = np.random.choice(['да', 'нет'], size=self.n_samples, p=[0.5, 0.5])

        # Создание DataFrame
        self.data = pd.DataFrame({
            'param1': param1,
            'param2': param2,
            'param3': param3
        })

        print(f"Сгенерировано {self.n_samples} записей")
        print("\nПервые 10 строк исходных данных:")
        print(self.data.head(10))

        print("\nСтатистика по параметрам:")
        print(f"Параметр 1: min={param1.min():.3f}, max={param1.max():.3f}, mean={param1.mean():.3f}")
        print(f"Параметр 2: min={param2.min():.3f}, max={param2.max():.3f}, mean={param2.mean():.3f}")
        print(f"Параметр 3: 'да' - {(param3 == 'да').sum()}, 'нет' - {(param3 == 'нет').sum()}")

        return self.data

    def preprocess_data(self):
        """
        Предобработка данных для KMeans:
        - Кодирование категориального признака ('да'/'нет' -> 0/1)
        - Масштабирование числовых признаков
        """
        print("\n" + "="*60)
        print("ПРЕДОБРАБОТКА ДАННЫХ")
        print("="*60)

        # Копируем данные
        self.data_encoded = self.data.copy()

        # Кодирование категориального признака ('да' -> 1, 'нет' -> 0)
        le = LabelEncoder()
        self.data_encoded['param3_encoded'] = le.fit_transform(self.data['param3'])

        print(f"Кодирование параметра 3: {dict(zip(le.classes_, le.transform(le.classes_)))}")

        # Выбираем признаки для кластеризации (все 3 параметра)
        features_for_clustering = ['param1', 'param2', 'param3_encoded']
        X = self.data_encoded[features_for_clustering]

        # Масштабирование (важно для KMeans, т.к. признаки в разных диапазонах)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        self.data_scaled = X_scaled

        print("\nПризнаки после предобработки:")
        print(f"- param1: масштабирован")
        print(f"- param2: масштабирован")
        print(f"- param3: закодирован как 0/1 и масштабирован")
        print(f"\nПервые 5 строк после масштабирования:")
        print(pd.DataFrame(X_scaled[:5], columns=features_for_clustering))

        return X_scaled

    def find_optimal_clusters(self, max_k=10):
        """
        Поиск оптимального количества кластеров методом локтя и силуэта

        Parameters:
        -----------
        max_k : int - максимальное количество кластеров для проверки
        """
        print("\n" + "="*60)
        print("ПОИСК ОПТИМАЛЬНОГО КОЛИЧЕСТВА КЛАСТЕРОВ")
        print("="*60)

        inertias = []
        silhouette_scores = []
        K_range = range(2, min(max_k + 1, self.n_samples))

        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            kmeans.fit(self.data_scaled)
            inertias.append(kmeans.inertia_)

            # Расчет silhouette score
            if k > 1:
                score = silhouette_score(self.data_scaled, kmeans.labels_)
                silhouette_scores.append(score)

        # Определение оптимального k методом локтя (поиск "колена")
        if len(inertias) > 2:
            # Находим точку максимального изменения
            deltas = np.diff(inertias)
            deltas2 = np.diff(deltas)
            if len(deltas2) > 0:
                self.optimal_k = np.argmin(deltas2) + 2
            else:
                self.optimal_k = 2
        else:
            self.optimal_k = 2

        # Построение графика метода локтя
        plt.figure(figsize=(12, 5))

        # График метода локтя
        plt.subplot(1, 2, 1)
        plt.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
        plt.axvline(x=self.optimal_k, color='red', linestyle='--', 
                   label=f'Оптимальное k = {self.optimal_k}')
        plt.xlabel('Количество кластеров (k)')
        plt.ylabel('Внутрикластерная сумма квадратов (Inertia)')
        plt.title('Метод локтя (Elbow Method)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # График силуэта
        plt.subplot(1, 2, 2)
        plt.plot(K_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
        best_k = K_range[np.argmax(silhouette_scores)]
        plt.axvline(x=best_k, color='red', linestyle='--', 
                   label=f'Лучший k = {best_k}')
        plt.xlabel('Количество кластеров (k)')
        plt.ylabel('Silhouette Score')
        plt.title('Метод силуэта (Silhouette Method)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('elbow_silhouette_plots.png', dpi=150, bbox_inches='tight')
        print(f"\n✓ Графики сохранены в 'elbow_silhouette_plots.png'")
        plt.close()

        print(f"\nМетод локтя рекомендует k = {self.optimal_k}")
        print(f"Лучший silhouette score = {max(silhouette_scores):.4f} при k = {best_k}")

        # Если данные плохо кластеризуются
        if max(silhouette_scores) < 0.5:
            print("\n⚠️ ВНИМАНИЕ: Данные плохо кластеризуются (Silhouette Score < 0.5)")
            print("Рекомендации: можно попробовать использовать другой метод кластеризации")

        # Используем лучшее k по силуэту, если оно отличается от метода локтя
        if best_k != self.optimal_k and max(silhouette_scores) > 0.5:
            self.optimal_k = best_k
            print(f"\nИспользуем k = {self.optimal_k} (по методу силуэта)")

        return self.optimal_k

    def perform_clustering(self, n_clusters=None):
        """
        Выполнение кластеризации методом K-средних

        Parameters:
        -----------
        n_clusters : int - количество кластеров (если None, используется оптимальное)
        """
        print("\n" + "="*60)
        print("ВЫПОЛНЕНИЕ КЛАСТЕРИЗАЦИИ K-MEANS")
        print("="*60)

        if n_clusters is None:
            n_clusters = self.optimal_k

        self.kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=self.random_state,
            n_init=10,
            max_iter=300
        )

        self.kmeans.fit(self.data_scaled)

        # Добавляем метки кластеров в исходные данные
        self.data['cluster'] = self.kmeans.labels_
        self.data_encoded['cluster'] = self.kmeans.labels_

        # Вывод результатов в консоль
        print(f"\n{'='*60}")
        print(f"РЕЗУЛЬТАТЫ КЛАСТЕРИЗАЦИИ (k={n_clusters})")
        print(f"{'='*60}")

        print("\n1. ПРОГНОЗИРУЕМЫЕ КЛАСТЕРЫ ДЛЯ КАЖДОЙ ТОЧКИ ДАННЫХ (первые 20):")
        print("-" * 60)
        print("Запись | Парам1 | Парам2 | Парам3 | Кластер")
        print("-" * 60)
        for i in range(min(20, self.n_samples)):
            print(f"  {i:3d}  |  {self.data.iloc[i]['param1']:.3f} | "
                  f"{self.data.iloc[i]['param2']:.3f} | "
                  f"{self.data.iloc[i]['param3']:^3}   |   {self.data.iloc[i]['cluster']}")

        print("\n2. КООРДИНАТЫ ЦЕНТРОИДОВ ДЛЯ КАЖДОГО КЛАСТЕРА:")
        print("-" * 60)
        for i, centroid in enumerate(self.kmeans.cluster_centers_):
            print(f"Кластер {i}: [{centroid[0]:.3f}, {centroid[1]:.3f}, {centroid[2]:.3f}]")

        print(f"\n3. ВНУТРИКЛАСТЕРНАЯ СУММА КВАДРАТОВ (Inertia): {self.kmeans.inertia_:.4f}")

        print(f"\n4. КОЛИЧЕСТВО ИТЕРАЦИЙ: {self.kmeans.n_iter_}")

        print("\n5. РАЗМЕР КАЖДОГО КЛАСТЕРА:")
        print("-" * 60)
        cluster_sizes = self.data['cluster'].value_counts().sort_index()
        for cluster_id, size in cluster_sizes.items():
            percentage = (size / self.n_samples) * 100
            print(f"Кластер {cluster_id}: {size} записей ({percentage:.1f}%)")

        # Анализ распределения параметра 3 по кластерам
        print("\n6. РАСПРЕДЕЛЕНИЕ ПАРАМЕТРА 'да/нет' ПО КЛАСТЕРАМ:")
        print("-" * 60)
        for cluster_id in sorted(self.data['cluster'].unique()):
            cluster_data = self.data[self.data['cluster'] == cluster_id]
            yes_count = (cluster_data['param3'] == 'да').sum()
            no_count = (cluster_data['param3'] == 'нет').sum()
            print(f"Кластер {cluster_id}: 'да' = {yes_count}, 'нет' = {no_count}")

        return self.kmeans

    def visualize_clusters_3d(self):
        """
        Визуализация кластеров в 3D (для трех параметров)
        """
        print("\n" + "="*60)
        print("ВИЗУАЛИЗАЦИЯ КЛАСТЕРОВ")
        print("="*60)

        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure(figsize=(14, 6))

        # 3D визуализация
        ax1 = fig.add_subplot(121, projection='3d')

        colors = plt.cm.Set3(np.linspace(0, 1, self.kmeans.n_clusters))

        for i in range(self.kmeans.n_clusters):
            cluster_data = self.data[self.data['cluster'] == i]
            # Преобразуем 'да'/'нет' в числовые значения для 3D графика
            param3_num = cluster_data['param3'].map({'да': 1, 'нет': 0})
            ax1.scatter(cluster_data['param1'], cluster_data['param2'], param3_num,
                       c=[colors[i]], label=f'Кластер {i}', alpha=0.6, s=50)

        # Центроиды в 3D
        ax1.scatter(self.kmeans.cluster_centers_[:, 0], 
                   self.kmeans.cluster_centers_[:, 1],
                   self.kmeans.cluster_centers_[:, 2],
                   c='red', marker='X', s=200, label='Центроиды')

        ax1.set_xlabel('Параметр 1 [0,1]')
        ax1.set_ylabel('Параметр 2 [-2,2]')
        ax1.set_zlabel('Параметр 3 (0=нет, 1=да)')
        ax1.set_title('3D визуализация кластеров')
        ax1.legend()

        # 2D визуализация (param1 vs param2, цвет = кластер)
        ax2 = fig.add_subplot(122)

        scatter = ax2.scatter(self.data['param1'], self.data['param2'], 
                             c=self.data['cluster'], cmap='Set3', 
                             alpha=0.6, s=50)

        # Центроиды в 2D (первые два параметра)
        ax2.scatter(self.kmeans.cluster_centers_[:, 0], 
                   self.kmeans.cluster_centers_[:, 1],
                   c='red', marker='X', s=200, label='Центроиды')

        ax2.set_xlabel('Параметр 1 [0,1]')
        ax2.set_ylabel('Параметр 2 [-2,2]')
        ax2.set_title('2D визуализация (параметры 1 и 2)')
        ax2.legend()

        plt.colorbar(scatter, ax=ax2, label='Кластер')
        plt.tight_layout()
        plt.savefig('clustering_visualization.png', dpi=150, bbox_inches='tight')
        print("✓ Визуализация сохранена в 'clustering_visualization.png'")
        plt.show()

    def run(self):
        """
        Запуск полного пайплайна микросервиса
        """
        
        print("МИКРОСЕРВИС K-MEANS КЛАСТЕРИЗАЦИИ")
        print("Данные: 3 параметра ([0,1], [-2,2], да/нет)")
        

        # Шаг 1: Генерация данных
        self.generate_data()

        # Шаг 2: Предобработка
        self.preprocess_data()

        # Шаг 3: Поиск оптимального k
        self.find_optimal_clusters(max_k=8)

        # Шаг 4: Кластеризация
        self.perform_clustering()

        # Шаг 5: Визуализация
        self.visualize_clusters_3d()

      
        print("РАБОТА МИКРОСЕРВИСА ЗАВЕРШЕНА")
       

        return self.kmeans


# Запуск микросервиса
if __name__ == "__main__":
    # Создание и запуск микросервиса
    microservice = KMeansMicroservice(
        n_samples=300,      # 300 записей
        random_state=42     # для воспроизводимости
    )

    # Запуск полного пайплайна
    kmeans_model = microservice.run()