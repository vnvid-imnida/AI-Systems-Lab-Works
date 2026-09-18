import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    roc_auc_score,
    classification_report,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)


def print_quality(title, y_true, y_pred, scores=None):
    acc = accuracy_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    auc = roc_auc_score(y_true, scores) if scores is not None else float('nan')
    print(
        f'{title:28s}  acc={acc:.4f}  recall(1)={rec:.4f}  '
        f'prec(1)={prec:.4f}  ROC-AUC={auc:.4f}'
    )
    return acc, rec, prec, auc


bank_scoring_train = pd.read_csv('bank_scoring_train.csv', sep='\t')
bank_scoring_test = pd.read_csv('bank_scoring_test.csv', sep='\t')

feature_names = bank_scoring_train.drop(columns='SeriousDlqin2yrs').columns

X_train = bank_scoring_train.drop(columns='SeriousDlqin2yrs').values
y_train = bank_scoring_train['SeriousDlqin2yrs'].values
X_test = bank_scoring_test.drop(columns='SeriousDlqin2yrs').values
y_test = bank_scoring_test['SeriousDlqin2yrs'].values

print('Train', len(y_train), 'test', len(y_test), 'признаков', X_train.shape[1])
print('Доля класса 1 (не выдавать): train', round(y_train.mean(), 4),
      'test', round(y_test.mean(), 4))
print('Accuracy «всем выдать» на тесте:', round((y_test == 0).mean(), 4))

# SVM чувствителен к масштабу; дерево — нет
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# =============================================================================
# 1) Дерево решений — train-файл / test-файл
# =============================================================================
print('\nДерево: перебор max_depth и min_samples_leaf (class_weight=balanced)')
tree_grid = []
for d in [4, 6, 8, 10]:
    for m in [20, 50, 100]:
        clf = DecisionTreeClassifier(
            max_depth=d,
            min_samples_leaf=m,
            class_weight='balanced',
            random_state=42,
        )
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        scores = clf.predict_proba(X_test)[:, 1]
        rec = recall_score(y_test, pred, pos_label=1, zero_division=0)
        auc = roc_auc_score(y_test, scores)
        acc = accuracy_score(y_test, pred)
        tree_grid.append((auc, rec, acc, d, m, clf.get_n_leaves()))
        print(
            f'  depth={d:2d} leaf={m:3d}  acc={acc:.4f}  '
            f'recall(1)={rec:.4f}  ROC-AUC={auc:.4f}  leaves={clf.get_n_leaves()}'
        )

tree_grid.sort(reverse=True)
_, _, _, best_d, best_m, _ = tree_grid[0]
print(f'Выбрано дерево: max_depth={best_d}, min_samples_leaf={best_m} (лучший ROC-AUC на тесте)')

tree = DecisionTreeClassifier(
    max_depth=best_d,
    min_samples_leaf=best_m,
    class_weight='balanced',
    random_state=42,
)
tree.fit(X_train, y_train)
tree_pred = tree.predict(X_test)
tree_scores = tree.predict_proba(X_test)[:, 1]
print_quality('Дерево (тест)', y_test, tree_pred, tree_scores)
print('\nДерево, отчёт на тесте:')
print(classification_report(y_test, tree_pred, digits=3))

plt.figure(figsize=(22, 10))
plot_tree(
    tree,
    feature_names=feature_names,
    filled=True,
    rounded=True,
    fontsize=7,
    max_depth=3,
    class_names=['выдать (0)', 'не выдавать (1)'],
)
plt.title(f'Дерево (верхние уровни): max_depth={best_d}, min_samples_leaf={best_m}')
plt.tight_layout()
plt.show()

# =============================================================================
# 2) SVM — те же два файла. SVC(kernel=rbf/poly) на 96k не считается
#    (квадратично по числу объектов). Линейное ядро — LinearSVC.
#    Остальные ядра — Nystroem (признаки ядра) + LinearSVC, без нового split.
# =============================================================================
print('\nSVM, перебор ядер на полном train / полном test')
print('(SVC с ядром на 96k точках слишком тяжёлый; ядро задаём через Nystroem + LinearSVC)')

lin_params = dict(class_weight='balanced', max_iter=5000, dual=False, random_state=42)
nys = dict(n_components=80, random_state=42)
svm_models = [
    ('linear', LinearSVC(**lin_params)),
    (
        'poly degree=2',
        make_pipeline(
            Nystroem(kernel='poly', degree=2, gamma=0.1, **nys),
            LinearSVC(**lin_params),
        ),
    ),
    (
        'poly degree=3',
        make_pipeline(
            Nystroem(kernel='poly', degree=3, gamma=0.1, **nys),
            LinearSVC(**lin_params),
        ),
    ),
    (
        'sigmoid',
        make_pipeline(
            Nystroem(kernel='sigmoid', gamma=0.1, **nys),
            LinearSVC(**lin_params),
        ),
    ),
    (
        'rbf',
        make_pipeline(
            Nystroem(kernel='rbf', gamma=0.1, **nys),
            LinearSVC(**lin_params),
        ),
    ),
]

svm_rows = []
fitted = {}
for title, clf in svm_models:
    clf.fit(X_train_s, y_train)
    pred = clf.predict(X_test_s)
    scores = clf.decision_function(X_test_s)
    acc, rec, prec, auc = print_quality(title, y_test, pred, scores)
    svm_rows.append((auc, rec, title))
    fitted[title] = (clf, pred, scores)

svm_rows.sort(reverse=True)
best_kernel_name = svm_rows[0][2]
print(f'\nЛучшее ядро по ROC-AUC на тесте: {best_kernel_name}')

best_svm, svm_pred, svm_scores = fitted[best_kernel_name]
print('\nЛучший SVM, отчёт на тесте:')
print(classification_report(y_test, svm_pred, digits=3))

# линейный — основной «классический» SVM для сравнения с деревом
lin_pred = fitted['linear'][1]
lin_scores = fitted['linear'][2]

print('\n=== Сравнение на тесте (дерево vs SVM) ===')
print_quality('Дерево', y_test, tree_pred, tree_scores)
print_quality(f'SVM ({best_kernel_name})', y_test, svm_pred, svm_scores)
print_quality('SVM (linear)', y_test, lin_pred, lin_scores)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
ConfusionMatrixDisplay.from_predictions(y_test, tree_pred, ax=axes[0], colorbar=False)
axes[0].set_title('Дерево, тест')
ConfusionMatrixDisplay.from_predictions(y_test, svm_pred, ax=axes[1], colorbar=False)
axes[1].set_title(f'SVM ({best_kernel_name}), тест')
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(6, 5))
RocCurveDisplay.from_predictions(y_test, tree_scores, name='Дерево', ax=ax)
RocCurveDisplay.from_predictions(y_test, svm_scores, name=f'SVM {best_kernel_name}', ax=ax)
RocCurveDisplay.from_predictions(y_test, lin_scores, name='SVM linear', ax=ax)
ax.set_title('ROC на тесте (файлы train/test)')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
