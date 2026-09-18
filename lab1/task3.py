import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


glass_df = pd.read_csv('glass.csv')

X = glass_df.drop(columns=['Id', 'Type']).values
y = glass_df['Type'].values

scaler = StandardScaler()  # масштабирование
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

k_neighbors = range(1, 30)

# p = 1 => manhattan
# p = 2 => euclidean
# p = 3 => minkowski
for p in range(1, 4):
    train_err = []
    test_err = []

    for k in k_neighbors:
        clf = KNeighborsClassifier(n_neighbors=k, metric='minkowski', p=p)
        clf.fit(X_train, y_train)

        pred_train = clf.predict(X_train)
        pred_test = clf.predict(X_test)

        train_err.append(1 - accuracy_score(y_train, pred_train))
        test_err.append(1 - accuracy_score(y_test, pred_test))

    # =========== ГРАФИКИ ERROR(K) ПО КАЖДОЙ МЕТРИКЕ ==========
    plt.plot(list(k_neighbors), test_err, marker='o', label=f'p={p} (test)')
    plt.xlabel('k (число соседей)')
    plt.ylabel('Ошибка (1 - accuracy)')
    plt.title('Ошибка k-NN на glass')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

