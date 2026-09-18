import pandas as pd
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import accuracy_score


svm_data = pd.read_csv('svmdata_e.txt', sep='\t')
svm_data_test = pd.read_csv('svmdata_e_test.txt', sep='\t')

X_train = svm_data[['X1', 'X2']].values
y_train = svm_data['Colors'].values

X_test = svm_data_test[['X1', 'X2']].values
y_test = svm_data_test['Colors'].values

# gamma есть у poly / sigmoid / rbf; у linear нет
# poly по ТЗ — все степени 1–5
gammas = [0.1, 1, 50]
poly_degrees = range(1, 6)


def make_svc(kernel, gamma, degree=3):
    # max_iter: poly высокой степени иначе SMO может «висеть» минутами
    return SVC(kernel=kernel, gamma=gamma, degree=degree, max_iter=8000)


def plot_one(ax, clf, title):
    print(f'  считаю {title} ...', flush=True)
    clf.fit(X_train, y_train)
    color_to_num = {name: i for i, name in enumerate(clf.classes_)}
    acc_tr = accuracy_score(y_train, clf.predict(X_train))
    acc_te = accuracy_score(y_test, clf.predict(X_test))
    n_sv = clf.support_vectors_.shape[0]
    print(f'{title:22s}  train={acc_tr:.4f}  test={acc_te:.4f}  sv={n_sv}', flush=True)

    DecisionBoundaryDisplay.from_estimator(
        clf, X_train, response_method='predict',
        cmap=plt.cm.coolwarm, alpha=0.8, ax=ax, xlabel='X1', ylabel='X2',
        grid_resolution=40,
    )
    y_num = [color_to_num[c] for c in y_train]
    ax.scatter(
        X_train[:, 0], X_train[:, 1],
        c=y_num, cmap=plt.cm.coolwarm, s=16, edgecolors='k',
    )
    ax.scatter(
        clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
        s=70, facecolors='none', edgecolors='k', linewidths=1.1,
    )
    ax.set_title(f'{title}\ntrain={acc_tr:.2f}, test={acc_te:.2f}')
    return 1 - acc_tr, 1 - acc_te


print('=== poly, степени 1–5 ===', flush=True)
fig_poly, axes_poly = plt.subplots(5, 3, figsize=(12, 16))
err_poly = {d: {'train': [], 'test': []} for d in poly_degrees}
for i, degree in enumerate(poly_degrees):
    for j, gamma in enumerate(gammas):
        clf = make_svc('poly', gamma, degree=degree)
        title = f'poly d={degree}, g={gamma}'
        tr_e, te_e = plot_one(axes_poly[i, j], clf, title)
        err_poly[degree]['train'].append(tr_e)
        err_poly[degree]['test'].append(te_e)
plt.tight_layout()
plt.show()

print('=== sigmoid и rbf ===', flush=True)
other_kernels = ('sigmoid', 'rbf')
fig_oth, axes_oth = plt.subplots(2, 3, figsize=(12, 7))
err_oth = {k: {'train': [], 'test': []} for k in other_kernels}
for i, kernel in enumerate(other_kernels):
    for j, gamma in enumerate(gammas):
        clf = make_svc(kernel, gamma)
        title = f'{kernel}, g={gamma}'
        tr_e, te_e = plot_one(axes_oth[i, j], clf, title)
        err_oth[kernel]['train'].append(tr_e)
        err_oth[kernel]['test'].append(te_e)
plt.tight_layout()
plt.show()

fig_err, axes_err = plt.subplots(2, 4, figsize=(16, 8))
axes_flat = axes_err.flatten()
for ax, degree in zip(axes_flat[:5], poly_degrees):
    ax.plot(gammas, err_poly[degree]['train'], 'o-', label='train')
    ax.plot(gammas, err_poly[degree]['test'], 'o-', label='test')
    ax.set_xscale('log')
    ax.set_title(f'poly degree={degree}')
    ax.set_xlabel('gamma')
    ax.set_ylabel('ошибка')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend()
for ax, kernel in zip(axes_flat[5:7], other_kernels):
    ax.plot(gammas, err_oth[kernel]['train'], 'o-', label='train')
    ax.plot(gammas, err_oth[kernel]['test'], 'o-', label='test')
    ax.set_xscale('log')
    ax.set_title(kernel)
    ax.set_xlabel('gamma')
    ax.set_ylabel('ошибка')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend()
axes_flat[7].axis('off')
fig_err.suptitle('svmdata_e: ошибка от gamma (poly 1–5, sigmoid, rbf)')
plt.tight_layout()
plt.show()
