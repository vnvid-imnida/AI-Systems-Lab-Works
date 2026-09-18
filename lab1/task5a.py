import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score


glass_df = pd.read_csv('glass.csv')
feature_names = glass_df.drop(columns=['Id', 'Type']).columns

X = glass_df.drop(columns=['Id', 'Type']).values
y = glass_df['Type'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y,
)

# дерево без ограничений — как в задании «построить и посмотреть»
model_tree = DecisionTreeClassifier(random_state=42)
model_tree.fit(X_train, y_train)

acc_tr = accuracy_score(y_train, model_tree.predict(X_train))
acc_te = accuracy_score(y_test, model_tree.predict(X_test))
print('Дерево по умолчанию (без ограничений)')
print('  глубина:', model_tree.get_depth())
print('  листьев:', model_tree.get_n_leaves())
print('  узлов:  ', model_tree.tree_.node_count)
print(f'  accuracy train: {acc_tr:.4f}')
print(f'  accuracy test:  {acc_te:.4f}')

# полное дерево мелким шрифтом + «обрезка» картинки до 3 уровней, чтобы читать развилки
plt.figure(figsize=(24, 12))
plot_tree(
    model_tree,
    feature_names=feature_names,
    filled=True,
    rounded=True,
    fontsize=6,
    max_depth=3,
)
plt.title('Дерево по умолчанию: первые 3 уровня (полное глубже и нечитаемо)')
plt.tight_layout()
plt.show()

# ----- критерий разбиения -----
print('\nКритерий:')
for crit in ['gini', 'entropy']:
    clf = DecisionTreeClassifier(criterion=crit, random_state=42)
    clf.fit(X_train, y_train)
    print(
        f'  {crit:8s}  depth={clf.get_depth()}  '
        f'train={accuracy_score(y_train, clf.predict(X_train)):.4f}  '
        f'test={accuracy_score(y_test, clf.predict(X_test)):.4f}'
    )

# ----- max_depth -----
depths = list(range(1, 13))
train_accs, test_accs = [], []
print('\nmax_depth:')
for d in depths:
    clf = DecisionTreeClassifier(max_depth=d, random_state=42)
    clf.fit(X_train, y_train)
    tr = accuracy_score(y_train, clf.predict(X_train))
    te = accuracy_score(y_test, clf.predict(X_test))
    train_accs.append(tr)
    test_accs.append(te)
    print(f'  {d:2d}  train={tr:.4f}  test={te:.4f}  leaves={clf.get_n_leaves()}')

plt.figure(figsize=(8, 5))
plt.plot(depths, train_accs, 'o-', label='train')
plt.plot(depths, test_accs, 'o-', label='test')
plt.xlabel('max_depth')
plt.ylabel('Accuracy')
plt.title('glass: accuracy дерева от max_depth')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# ----- min_samples_leaf -----
print('\nmin_samples_leaf:')
for m in [1, 2, 3, 5, 8, 10]:
    clf = DecisionTreeClassifier(min_samples_leaf=m, random_state=42)
    clf.fit(X_train, y_train)
    print(
        f'  {m:2d}  depth={clf.get_depth()}  '
        f'train={accuracy_score(y_train, clf.predict(X_train)):.4f}  '
        f'test={accuracy_score(y_test, clf.predict(X_test)):.4f}'
    )

# более простое дерево для интерпретации (лучшая зона test ≈ depth 7)
simple_tree = DecisionTreeClassifier(max_depth=7, random_state=42)
simple_tree.fit(X_train, y_train)
print('\nДерево max_depth=7')
print(f'  train={accuracy_score(y_train, simple_tree.predict(X_train)):.4f}')
print(f'  test= {accuracy_score(y_test, simple_tree.predict(X_test)):.4f}')

plt.figure(figsize=(22, 10))
plot_tree(
    simple_tree,
    feature_names=feature_names,
    filled=True,
    rounded=True,
    fontsize=7,
)
plt.title('Интерпретируемое дерево, max_depth=7')
plt.tight_layout()
plt.show()
