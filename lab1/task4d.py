import pandas as pd
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import accuracy_score


svm_data = pd.read_csv('svmdata_d.txt', sep='\t')
svm_data_test = pd.read_csv('svmdata_d_test.txt', sep='\t')

X_train = svm_data[['X1', 'X2']].values
y_train = svm_data['Colors'].values

X_test = svm_data_test[['X1', 'X2']].values
y_test = svm_data_test['Colors'].values

models = (
    ('poly degree=1', SVC(kernel='poly', degree=1)),
    ('poly degree=2', SVC(kernel='poly', degree=2)),
    ('poly degree=3', SVC(kernel='poly', degree=3)),
    ('poly degree=4', SVC(kernel='poly', degree=4)),
    ('poly degree=5', SVC(kernel='poly', degree=5)),
    ('sigmoid', SVC(kernel='sigmoid')),
    ('rbf (гауссово)', SVC(kernel='rbf'))
)

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes.flatten()[7].axis('off')
for (title, clf), ax in zip(models, axes.flatten()):
    clf.fit(X_train, y_train)
    color_to_num = {name: i for i, name in enumerate(clf.classes_)}

    acc_tr = accuracy_score(y_train, clf.predict(X_train))
    acc_te = accuracy_score(y_test, clf.predict(X_test))
    n_sv = clf.support_vectors_.shape[0]
    print(f'{title:20s}  train={acc_tr:.4f}  test={acc_te:.4f}  sv={n_sv}')

    DecisionBoundaryDisplay.from_estimator(
        clf,
        X_train,
        response_method='predict',
        cmap=plt.cm.coolwarm,
        alpha=0.8,
        ax=ax,
        xlabel='X1',
        ylabel='X2'
    )
    y_num = [color_to_num[c] for c in y_train]
    ax.scatter(
        X_train[:, 0], X_train[:, 1],
        c=y_num, cmap=plt.cm.coolwarm, s=20, edgecolors='k'
    )
    ax.scatter(
        clf.support_vectors_[:, 0],
        clf.support_vectors_[:, 1],
        s=90, facecolors='none', edgecolors='k', linewidths=1.2,
    )
    ax.set_title(f'{title}\ntrain={acc_tr:.2f}, test={acc_te:.2f}')

plt.tight_layout()
plt.show()
