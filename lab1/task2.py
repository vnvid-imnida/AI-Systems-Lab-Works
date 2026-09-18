import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    roc_auc_score,
    average_precision_score,
)


# ============= ПАРАМЕТРЫ ВАРИАНТА ============

# Класс -1
mean_X1_minus1 = 18
mean_X2_minus1 = 22
std_minus1 = 3
n_samples_minus1 = 90

# Класс 1
mean_X1_plus1 = 17
mean_X2_plus1 = 0
std_plus1 = 3
n_samples_plus1 = 10

# =========== ГЕНЕРАЦИЯ ТОЧЕК С ПРИЗНАКАМИ Х1, Х2 ===========
np.random.seed(42)

X1_minus1 = np.random.normal(mean_X1_minus1, std_minus1, n_samples_minus1)
X2_minus1 = np.random.normal(mean_X2_minus1, std_minus1, n_samples_minus1)

X1_plus1 = np.random.normal(mean_X1_plus1, std_plus1, n_samples_plus1)
X2_plus1 = np.random.normal(mean_X2_plus1, std_plus1, n_samples_plus1)

# Собираем датасет и объединяем всё в одну таблицу
df_minus1 = pd.DataFrame({'X1': X1_minus1, 'X2': X2_minus1, 'Class': -1})
df_plus1 = pd.DataFrame({'X1': X1_plus1, 'X2': X2_plus1, 'Class': 1})

dataset = pd.concat([df_minus1, df_plus1]).reset_index(drop=True)

# =========== ДИАГРАММА ДАННЫХ ===========
plt.figure(figsize=(8, 6))
for label, color, marker in [(-1, 'tab:blue', 'o'), (1, 'tab:orange', 's')]:
    part = dataset[dataset['Class'] == label]
    plt.scatter(
        part['X1'], part['X2'],
        c=color, marker=marker, label=f'Класс {label}',
        edgecolors='black', alpha=0.8
    )

plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Сгенерированные данные (вариант 18)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# =========== БАЙЕСОВСКИЙ КЛАССИФИКАТОР ===========
X = dataset[['X1', 'X2']].values
y = dataset['Class'].values

# Для ROC/PR удобнее метки 0/1 (положительный класс = 1)
y_bin = (y == 1).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_bin,
    test_size=0.3,
    random_state=42,
    stratify=y_bin
)

clf = GaussianNB()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]  # вероятность класса 1

# ----- точность -----
acc = accuracy_score(y_test, y_pred)
print(f'Accuracy (test): {acc:.4f}')

# ----- матрица ошибок -----
cm = confusion_matrix(y_test, y_pred)
print('Confusion matrix:')
print(cm)

fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay(cm, display_labels=['класс -1', 'класс 1']).plot(ax=ax_cm)
ax_cm.set_title('Матрица ошибок (test)')
plt.tight_layout()
plt.show()

# ----- ROC -----
roc_auc = roc_auc_score(y_test, y_proba)
print(f'ROC AUC: {roc_auc:.4f}')

fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax_roc)
ax_roc.set_title(f'ROC-кривая (AUC = {roc_auc:.3f})')
plt.tight_layout()
plt.show()

# ----- PR -----
pr_auc = average_precision_score(y_test, y_proba)
print(f'PR AUC (Average Precision): {pr_auc:.4f}')

fig_pr, ax_pr = plt.subplots(figsize=(6, 5))
PrecisionRecallDisplay.from_predictions(y_test, y_proba, ax=ax_pr)
ax_pr.set_title(f'PR-кривая (AP = {pr_auc:.3f})')
plt.tight_layout()
plt.show()

# ----- краткий вывод в консоль -----
print('\nВывод:')
print('Классы хорошо разделены по X2 (центры 22 и 0 при СКО=3),')
print('поэтому GaussianNB обычно даёт высокую ROC AUC.')
print('Но классы несбалансированы (90 vs 10), поэтому смотри не только accuracy,')
print('а ещё матрицу ошибок и PR-кривую: важно, находит ли модель редкий класс 1.')
if roc_auc >= 0.9 and pr_auc >= 0.7:
    print('Итог: классификатор скорее ХОРОШИЙ.')
elif roc_auc >= 0.8:
    print('Итог: классификатор приемлемый / скорее хороший.')
else:
    print('Итог: классификатор слабый.')
