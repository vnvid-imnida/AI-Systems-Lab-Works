import pandas as pd
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

svm_data = pd.read_csv('svmdata_a.txt', sep='\t')
svm_data_test = pd.read_csv('svmdata_a_test.txt', sep='\t')

X_train = svm_data[['X1', 'X2']].values
y_train = svm_data['Color'].values

X_test = svm_data_test[['X1', 'X2']].values
y_test = svm_data_test['Color'].values

clf = SVC(kernel='linear')
clf.fit(X_train, y_train)

# как в примере sklearn plot_iris_svc: области классов через DecisionBoundaryDisplay
# https://scikit-learn.org/stable/auto_examples/svm/plot_iris_svc.html
# точки в том же порядке классов, что заливка DecisionBoundaryDisplay (clf.classes_)
color_to_num = {name: i for i, name in enumerate(clf.classes_)}
n_sv = clf.support_vectors_.shape[0]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, X, y, title in [
    (axes[0], X_train, y_train, 'обучающая выборка'),
    (axes[1], X_test, y_test, 'тестовая выборка'),
]:
    DecisionBoundaryDisplay.from_estimator(
        clf,
        X_train,  # сетка по диапазону train — границы самой модели
        response_method='predict',
        cmap=plt.cm.coolwarm,
        alpha=0.8,
        ax=ax,
        xlabel='X1',
        ylabel='X2',
    )
    y_num = [color_to_num[c] for c in y]
    ax.scatter(
        X[:, 0], X[:, 1],
        c=y_num, cmap=plt.cm.coolwarm, s=30, edgecolors='k',
        label='объекты',
    )
    ax.scatter(
        clf.support_vectors_[:, 0],
        clf.support_vectors_[:, 1],
        s=120, facecolors='none', edgecolors='k', linewidths=1.5,
        label='опорные векторы',
    )
    ax.set_title(f'SVC, linear kernel ({title})')
    ax.legend()

fig.suptitle(f'Число опорных векторов: {n_sv}')
plt.tight_layout()
plt.show()

# ----- точность и матрицы ошибок (train / test) -----
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)

acc_train = accuracy_score(y_train, y_pred_train)
acc_test = accuracy_score(y_test, y_pred_test)
cm_train = confusion_matrix(y_train, y_pred_train, labels=clf.classes_)
cm_test = confusion_matrix(y_test, y_pred_test, labels=clf.classes_)

print('Число опорных векторов:', n_sv)
print(f'Accuracy (train): {acc_train:.4f}')
print(f'Accuracy (test):  {acc_test:.4f}')
print('Confusion matrix (train):')
print(cm_train)
print('Confusion matrix (test):')
print(cm_test)

fig_cm, axes_cm = plt.subplots(1, 2, figsize=(10, 4))
ConfusionMatrixDisplay(cm_train, display_labels=clf.classes_).plot(ax=axes_cm[0])
axes_cm[0].set_title('Матрица ошибок (train)')
ConfusionMatrixDisplay(cm_test, display_labels=clf.classes_).plot(ax=axes_cm[1])
axes_cm[1].set_title('Матрица ошибок (test)')
plt.tight_layout()
plt.show()
