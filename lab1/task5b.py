import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
)

spam = pd.read_csv('spam7.csv')
feature_names = spam.drop(columns='yesno').columns

X = spam.drop(columns='yesno').values
y = spam['yesno'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y,
)

print('Письма: всего', len(y), 'train', len(y_train), 'test', len(y_test))
print('Классы:', dict(pd.Series(y).value_counts()))

# ----- дерево без ограничений (как в 5a: сначала смотрим, потом режем) -----
full = DecisionTreeClassifier(random_state=42)
full.fit(X_train, y_train)
print('\nДерево по умолчанию (без ограничений)')
print('  глубина:', full.get_depth())
print('  листьев:', full.get_n_leaves())
print('  узлов:  ', full.tree_.node_count)
print(f'  accuracy train: {accuracy_score(y_train, full.predict(X_train)):.4f}')
print(f'  accuracy test:  {accuracy_score(y_test, full.predict(X_test)):.4f}')

plt.figure(figsize=(22, 10))
plot_tree(
    full,
    feature_names=feature_names,
    filled=True,
    rounded=True,
    fontsize=6,
    max_depth=3,
    class_names=full.classes_.astype(str),
)
plt.title('Письма: неограниченное дерево, первые 3 уровня')
plt.tight_layout()
plt.show()

# ----- подбор параметров по test accuracy -----
print('\nПодбор параметров (лучшее test, при равенстве — меньше листьев):')
grid = []
for crit in ['gini', 'entropy']:
    for d in [3, 4, 5, 6, 7, 8, 10, None]:
        for m in [1, 5, 10, 20, 30, 50]:
            clf = DecisionTreeClassifier(
                criterion=crit,
                max_depth=d,
                min_samples_leaf=m,
                random_state=42,
            )
            clf.fit(X_train, y_train)
            tr = accuracy_score(y_train, clf.predict(X_train))
            te = accuracy_score(y_test, clf.predict(X_test))
            grid.append((te, -clf.get_n_leaves(), crit, d, m, tr, clf.get_depth(), clf.get_n_leaves()))

grid.sort(reverse=True)
print('  топ-8 по test:')
print('  test    train   crit     depth  leaf  n_leaves')
for te, _, crit, d, m, tr, dep, nleaf in grid[:8]:
    print(f'  {te:.4f}  {tr:.4f}  {crit:8s}  {str(d):>4s}  {m:4d}  {nleaf:4d}  (факт.глубина {dep})')

best_te, _, best_crit, best_d, best_m, best_tr, best_dep, best_nleaf = grid[0]
print(
    f'\nВыбрано: criterion={best_crit}, max_depth={best_d}, '
    f'min_samples_leaf={best_m}'
)

# ----- выбранное дерево -----
opt = DecisionTreeClassifier(
    criterion=best_crit,
    max_depth=best_d,
    min_samples_leaf=best_m,
    random_state=42,
)
opt.fit(X_train, y_train)
y_tr_pred = opt.predict(X_train)
y_te_pred = opt.predict(X_test)
print('\nВыбранное дерево')
print('  глубина:', opt.get_depth())
print('  листьев:', opt.get_n_leaves())
print(f'  accuracy train: {accuracy_score(y_train, y_tr_pred):.4f}')
print(f'  accuracy test:  {accuracy_score(y_test, y_te_pred):.4f}')
print('\nОтчёт на тесте:')
print(classification_report(y_test, y_te_pred, digits=3))

plt.figure(figsize=(24, 12))
plot_tree(
    opt,
    feature_names=feature_names,
    filled=True,
    rounded=True,
    fontsize=8,
    class_names=opt.classes_.astype(str),
)
plt.title(
    f'Выбранное дерево: {best_crit}, max_depth={best_d}, '
    f'min_samples_leaf={best_m}'
)
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_te_pred, ax=ax, colorbar=False
)
ax.set_title('Матрица ошибок на тесте')
plt.tight_layout()
plt.show()

# ----- важность признаков -----
imp = pd.Series(opt.feature_importances_, index=feature_names).sort_values(ascending=False)
print('\nВажность признаков (feature_importances_):')
print(imp.to_string())

plt.figure(figsize=(8, 4))
imp.sort_values().plot(kind='barh')
plt.xlabel('Важность')
plt.title('Важность признаков выбранного дерева')
plt.tight_layout()
plt.show()

# для справки: как accuracy зависит от max_depth при выбранном leaf
print('\nmax_depth при criterion и min_samples_leaf выбранной модели:')
for d in range(1, 13):
    clf = DecisionTreeClassifier(
        criterion=best_crit,
        max_depth=d,
        min_samples_leaf=best_m,
        random_state=42,
    )
    clf.fit(X_train, y_train)
    print(
        f'  {d:2d}  train={accuracy_score(y_train, clf.predict(X_train)):.4f}  '
        f'test={accuracy_score(y_test, clf.predict(X_test)):.4f}  '
        f'leaves={clf.get_n_leaves()}'
    )
