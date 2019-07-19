from sklearn.linear_model import LogisticRegression
import sklearn.metrics
from random import shuffle, seed
import numpy as np
import pickle as pkl

positive_data = np.load("positive_data.npy")
positive_ground_truth = np.load("positive_ground_truth.npy")
negative_data = np.load("negative_data.npy")
X = np.concatenate((positive_data, negative_data))
y = np.concatenate((positive_ground_truth, np.zeros(negative_data.shape[0]))) 
for f in range(10):
    with open("folds/fold%d.pkl"%f, "rb") as fold:
        fold_data = pkl.load(fold)
    print("Fold", f)
    train_index = np.array(list(map(lambda x:x[0], fold_data[0])))
    test_index = np.array(list(map(lambda x:x[0], fold_data[1])))

    X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
    
    model = LogisticRegression(solver='lbfgs').fit(X_train,y_train)
    y_pred = np.array(model.predict_proba(X_test)[:,1])
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_test, y_pred)
    print(sklearn.metrics.auc(fpr, tpr))
    print(sklearn.metrics.average_precision_score(y_test, y_pred))
