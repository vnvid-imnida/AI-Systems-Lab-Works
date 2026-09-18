import pandas as pd
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

svm_data = pd.read_csv('svmdata_b.txt', sep='\t')
svm_data_test = pd.read_csv('svmdata_b_test.txt', sep='\t')

X_train = svm_data[['X1', 'X2']].values
y_train = svm_data['Colors'].values

X_test = svm_data_test[['X1', 'X2']].values
y_test = svm_data_test['Colors'].values

# C по умолчанию = 1
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)
print('C=1 (default)')
print('Accuracy train:', accuracy_score(y_train, clf.predict(X_train)))
print('Accuracy test: ', accuracy_score(y_test, clf.predict(X_test)))

# маленький C — широкий зазор, можно ошибиться на train
# большой C — почти запрещает ошибки на train (риск переобучения)
C_values = [0.01, 0.1, 0.5, 1, 2, 5, 10, 50, 100, 200, 500, 1000]
train_err = []
test_err = []

print('\nC\tacc_train\tacc_test\tn_sv')
for C in C_values:
    clf_c = SVC(kernel='linear', C=C)
    clf_c.fit(X_train, y_train)
    acc_tr = accuracy_score(y_train, clf_c.predict(X_train))
    acc_te = accuracy_score(y_test, clf_c.predict(X_test))
    train_err.append(1 - acc_tr)
    test_err.append(1 - acc_te)
    print(f'{C}\t{acc_tr:.4f}\t\t{acc_te:.4f}\t\t{clf_c.support_vectors_.shape[0]}')

plt.plot(C_values, train_err, 'o-', label='ошибка train')
plt.plot(C_values, test_err, 'o-', label='ошибка test')
plt.xscale('log')
plt.xlabel('C (штраф)')
plt.ylabel('Ошибка (1 - accuracy)')
plt.title('SVM linear, svmdata_b: ошибка от C')
plt.grid(True, which='both', alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
