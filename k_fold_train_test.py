import config
import models
import numpy as np
import os
import time
import datetime
import json
from sklearn.metrics import average_precision_score
import sys
import os
import argparse
import statistics

import ctypes

# START
""" UNCOMMENT
sb = ctypes.create_string_buffer
r = b"/home/jakob/experiment/OpenCRF/HardcodedPotentials/bar/rawdata/"
ll = ctypes.cdll.LoadLibrary
lib = ll('../../experiment/OpenCRF/HardcodedPotentials/crflib.so')
lib.InitializeCRF.argtypes = [ctypes.c_char_p for i in range(5)]
lib.InitializeCRF.restype = ctypes.c_void_p
lib.Gradient.argtypes = [ctypes.c_void_p] + [ctypes.POINTER(ctypes.c_double) for i in range(6)] + [ctypes.c_int]
lib.Gradient.restype = None
"""
# END

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='pcnn_att', help='name of the model')
parser.add_argument('--max_epoch', type=int, default=30, help='number of epochs to run for')
parser.add_argument('--k_folds', type=int, default=10, help='number of folds to split the data into')
args = parser.parse_args()
model = {
	'pcnn_att': models.PCNN_ATT,
	'pcnn_one': models.PCNN_ONE,
	'pcnn_ave': models.PCNN_AVE,
	'cnn_att': models.CNN_ATT,
	'cnn_one': models.CNN_ONE,
	'cnn_ave': models.CNN_AVE,
	'lstm_att': models.LSTM_ATT
}
con = config.Config()
con.set_max_epoch(args.max_epoch)
con.set_k_folds(args.k_folds)

# Variables
pr_auc_all = []
roc_auc_all = []

fpr_x_avg = []
tpr_y_avg = []
roc_x_avg = []
roc_y_avg = []

k_folds = args.k_folds

for k in range(k_folds):
	#model = lib.InitializeCRF(sb(b"0.001"), sb(r+b"fold-{}-edges-ALT.txt".format(k)), sb(r+b"CRFmodel-{}.txt".format(k)), sb(b"model-fold-{}.txt".format(k)),sb(r+b"maps/map{}.txt".format(k)))
	#modelcrf = lib.InitializeCRF(sb(b"0.001"), sb(r+b"fold-{}-edges-ALT.txt".format(k)), sb(r+b"CRFmodel-{}.txt".format(k)), sb(b"model-fold-{}.txt".format(k)),sb(r+b"maps/map{}.txt".format(k)))

	con = config.Config()
	con.set_max_epoch(args.max_epoch)
	con.set_k_folds(args.k_folds)
	con.load_k_fold_train_data(k)
	con.load_k_fold_test_data(k)
	con.set_train_model(model[args.model_name])
	#con.set_test_model(model[args.model_name])
	roc_auc, pr_auc, pr_x, pr_y, fpr, tpr, scores, ks = con.train_each_fold(k)#, lib) UNCOMMENT
	roc_auc_all.append(roc_auc)
	pr_auc_all.append(pr_auc)
	#print(scores)
	#print("^^ SCORES ^^")

	np.save('./raw_data/jake/fold{}/scores'.format(k), scores)
	print("Probability scores for fold {} saved to 'scores.npy' in fold {} folder.".format(k, k))

	"""
	if k == 0:
		pr_x_avg = pr_x
		pr_y_avg = pr_y
		roc_x_avg = fpr
		roc_y_avg = tpr
		continue
	"""

pr_auc_avg = statistics.mean(pr_auc_all)
roc_auc_avg = statistics.mean(roc_auc_all)


#for i in range(len(p_avg)):
	#p_avg[i] /= k_folds

#p_avg /= k_folds

#for j in range(len(r_avg)):
	#r_avg[j] /= k_folds

#r_avg /= k_folds
#con.set_train_model(model[args.model_name])

print('Finish training')
print('ROC-AUC average for {}-fold cross-validation training and testing: {}'.format(k_folds, roc_auc_avg))
print('PR-AUC average for {}-fold cross-validation training and testing: {}'.format(k_folds, pr_auc_avg))
"""
print('Storing result...')
if not os.path.isdir('./data/folds'):
	os.mkdir('./data/folds')

np.save(os.path.join('./data/folds', args.model_name + '_pr_x.npy'), pr_x_avg)
np.save(os.path.join('./data/folds', args.model_name + '_pr_y.npy'), pr_y_avg)
np.save(os.path.join('./data/folds', args.model_name + '_roc_x.npy'), roc_x_avg)
np.save(os.path.join('./data/folds', args.model_name + '_roc_y.npy'), roc_y_avg)
print('Finish storing...')
"""

#con.train_k_folds()
