import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB, GaussianNB
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.metrics import accuracy_score


def measure_accuracies(model, X, y, ratios, random_state=42):
    """Считает accuracy на train, test при разных их долях"""
    train_accs = []
    test_accs = []

    for p in ratios:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            train_size=p,
            random_state=random_state,
            stratify=y
        )
        clf = model.__class__(**model.get_params())

        # Обучение
        clf.fit(X_train, y_train)

        # Проверка качества обученной модели
        train_accs.append(accuracy_score(y_train, clf.predict(X_train)))
        test_accs.append(accuracy_score(y_test, clf.predict(X_test)))

    return train_accs, test_accs

def plot_accuracies(ratios, train_accs, test_accs, title):
    """Рисует график зависимости accuracy от соотношения обучающей выборки и
    тестовых данных"""
    plt.figure(figsize=(8, 5))
    plt.plot(ratios, train_accs, 'o-', label='train accuracy')
    plt.plot(ratios, test_accs, 'o-', label='test accuracy')
    plt.xlabel('Доля train')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.ylim(0, 1.05)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


ratios = np.round(np.arange(0.1, 1.0, 0.1), 2)  # 0.1 ... 0.9

# =========== TIC-TAC-TOE ==========
ttt = pd.read_csv('tic_tac_toe.txt', header=None)

X_ttt = ttt.iloc[:, :-1].copy()
y_ttt = ttt.iloc[:, -1].copy()

X_ttt = OrdinalEncoder().fit_transform(X_ttt)  # x/o/b => числа
y_ttt = LabelEncoder().fit_transform(y_ttt)  # positive/negative => 0/1

ttt_train_accs, ttt_test_accs = measure_accuracies(
    CategoricalNB(), X_ttt, y_ttt, ratios
)

print('=== Tic-Tac-Toe (CategoricalNB) ===')
print(f'{"Доля train":>12} | {"Accuracy train":>14} | {"Accuracy test":>13}')
for p, tr, te in zip(ratios, ttt_train_accs, ttt_test_accs):
    print(f'{p:12.1f} | {tr:14.4f} | {te:13.4f}')
print()


# ============ SPAM ============
spam = pd.read_csv('spam.csv')

X_spam = spam.iloc[:, 1:-1].values
y_spam = LabelEncoder().fit_transform(spam.iloc[:, -1].values)

spam_train_accs, spam_test_accs = measure_accuracies(
    GaussianNB(), X_spam, y_spam, ratios
)

print('=== Spam (GaussianNB) ===')
print(f'{"Доля train":>12} | {"Accuracy train":>14} | {"Accuracy test":>13}')
for p, tr, te in zip(ratios, spam_train_accs, spam_test_accs):
    print(f'{p:12.1f} | {tr:14.4f} | {te:13.4f}')
print()


# ============ ГРАФИКИ ============
plot_accuracies(
    ratios, ttt_train_accs, ttt_test_accs,
    'Tic-Tac_Toe: accuracy vs доля train (CategoricalNB)'
)

plot_accuracies(
    ratios, spam_train_accs, spam_test_accs,
    'Spam: accuracy vs доля train (GaussianNB)'
)
