from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold #train_test_split
import sklearn.metrics
from random import shuffle, seed
import numpy as np
import pickle as pkl


positive_data = np.load("positive_data.npy")
positive_ground_truth = np.load("positive_ground_truth.npy")
negative_data = np.load("negative_data.npy")
with open("positive_id.txt") as pos:
    pos_ids = pos.read().split("\n")[:-1]
with open("negative_id.txt") as neg:
    neg_ids = neg.read().split("\n")[:-1]

all_ids = pos_ids+neg_ids
X = np.concatenate((positive_data, negative_data))
y = np.concatenate((positive_ground_truth, np.zeros(negative_data.shape[0])))
idnums = [(i,all_ids[i]) for i in range(len(all_ids))]
seed(42)
shuffle(idnums)
kf = KFold(n_splits=10)
i = 0 
for train_index, test_index in kf.split(idnums):
    with open("folds/fold%d"%i,"wb+") as fold:
        pkl.dump(([idnums[j] for j in train_index], [idnums[j] for j in test_index]),fold)
    i += 1


"""
k_folds = 10

for i in range(k_folds):
	with open('folds/fold{}.pkl'.format(i), 'rb') as pickle_file:
		data = pkl.load(pickle_file)
		print(data)
"""