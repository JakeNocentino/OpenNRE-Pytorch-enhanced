#coding:utf-8
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd import grad
import torch.optim as optim
import numpy as np
import os
import time
import datetime
import json
import sys
import sklearn.metrics
from tqdm import tqdm
import statistics
import ctypes

from networks.selector import *
from networks.encoder import *
from networks.embedding import *
from networks.classifier import *

# FILES FOR USE OF JACOB WHEN CRF IS NOT CONNECTED
wgrad = np.load('wgrad3.npy')

""" UNCOMMENT
as_p = lambda a:ctypes.cast(a.__array_interface__['data'][0],ctypes.POINTER(ctypes.c_double))
sb = ctypes.create_string_buffer 
r = b"/home/jakob/experiment/rawdata/"
"""
def to_var(x):
	#return Variable(torch.from_numpy(x).cuda()) JN EDIT. CHANGE LATER
	return Variable(torch.from_numpy(x))


class Accuracy(object):
	def __init__(self):
		self.correct = 0
		self.total = 0
	def add(self, is_correct):
		self.total += 1
		if is_correct:
			self.correct += 1
	def get(self):
		if self.total == 0:
			return 0.0
		else:
			return float(self.correct) / self.total
	def clear(self):
		self.correct = 0
		self.total = 0 

class Config(object):
	def __init__(self):
		self.acc_NA = Accuracy()
		self.acc_not_NA = Accuracy()
		self.acc_total = Accuracy()
		self.data_path = './data'
		self.data_folds_path = './raw_data/jake/' # path for folds
		self.data_word_vec = torch.zeros # Not initialized until load_train_data()
		self.use_bag = False # usually 'True' when using attention and others
		#self.use_gpu = True JN EDIT. CHANGE LATER
		self.use_gpu = False
		self.is_training = True
		self.max_length = 120
		self.pos_num = 2 * self.max_length
		self.num_classes = 2 # negative: 0, positive: 1
		self.k_folds = 10 # For 10-fold cross-validation
		self.hidden_size = 256 #230
		self.pos_size = 5
		self.max_epoch =50
		self.opt_method = 'SGD'
		self.optimizer = None
		self.learning_rate = 0.9 # normally .5 or .9
		self.weight_decay = 1e-5
		self.drop_prob = 0.25 # DEFAULT: 0.5
		self.checkpoint_dir = './checkpoint'
		self.test_result_dir = './test_result'
		self.k_folds_test_result_dir = './test_result/k_folds_test_result'
		self.save_epoch = 1
		self.save = False
		self.test_epoch = 1
		self.test_k_fold = 1 # test and store auc of all k-folds at an iteration of 1
		self.pretrain_model = self.checkpoint_dir + '/PCNN_AVE-28'
		self.trainModel = None
		self.testModel = None
		self.batch_size = 7402 # try lower batch size for higher AUC; normally 160
		self.word_size = 50
		self.window_size = 3
		self.epoch_range = None
		self.num_lstm_layers = 1 # number of LSTMS stacked on top of each other
	def set_data_path(self, data_path):
		self.data_path = data_path
	def set_max_length(self, max_length):
		self.max_length = max_length
		self.pos_num = 2 * self.max_length
	def set_num_classes(self, num_classes):
		self.num_classes = num_classes
	def set_hidden_size(self, hidden_size):
		self.hidden_size = hidden_size
	def set_window_size(self, window_size):
		self.window_size = window_size
	def set_pos_size(self, pos_size):
		self.pos_size = pos_size
	def set_word_size(self, word_size):
		self.word_size = word_size
	def set_max_epoch(self, max_epoch):
		self.max_epoch = max_epoch
	def set_batch_size(self, batch_size):
		self.batch_size = batch_size
	def set_opt_method(self, opt_method):
		self.opt_method = opt_method
	def set_learning_rate(self, learning_rate):
		self.learning_rate = learning_rate
	def set_weight_decay(self, weight_decay):
		self.weight_decay = weight_decay
	def set_drop_prob(self, drop_prob):
		self.drop_prob = drop_prob
	def set_checkpoint_dir(self, checkpoint_dir):
		self.checkpoint_dir = checkpoint_dir
	def set_test_epoch(self, test_epoch):
		self.test_epoch = test_epoch
	def set_save_epoch(self, save_epoch):
		self.save_epoch = save_epoch
	def set_pretrain_model(self, pretrain_model):
		self.pretrain_model = pretrain_model
	def set_is_training(self, is_training):
		self.is_training = is_training
	def set_use_bag(self, use_bag):
		self.use_bag = use_bag
	def set_use_gpu(self, use_gpu):
		self.use_gpu = use_gpu
	def set_epoch_range(self, epoch_range):
		self.epoch_range = epoch_range
	def set_k_folds(self, k_folds):
		self.k_folds = k_folds
	
	def load_train_data(self):
		print("Reading training data...")
		self.data_word_vec = np.load(os.path.join(self.data_path, 'vec.npy'))
		self.data_train_word = np.load(os.path.join(self.data_path, 'train_word.npy'))
		self.data_train_pos1 = np.load(os.path.join(self.data_path, 'train_pos1.npy'))
		self.data_train_pos2 = np.load(os.path.join(self.data_path, 'train_pos2.npy'))
		self.data_train_mask = np.load(os.path.join(self.data_path, 'train_mask.npy'))
		if self.use_bag:
			self.data_query_label = np.load(os.path.join(self.data_path, 'train_ins_label.npy'))
			self.data_train_label = np.load(os.path.join(self.data_path, 'train_bag_label.npy'))
			self.data_train_scope = np.load(os.path.join(self.data_path, 'train_bag_scope.npy'))
		else:
			self.data_train_label = np.load(os.path.join(self.data_path, 'train_ins_label.npy'))
			self.data_train_scope = np.load(os.path.join(self.data_path, 'train_ins_scope.npy'))
		print("Finish reading")
		self.train_order = list(range(len(self.data_train_label)))
		self.train_batches = len(self.data_train_label) // self.batch_size
		if len(self.data_train_label) % self.batch_size != 0:
			self.train_batches += 1

	def load_test_data(self):
		print("Reading testing data...")
		self.data_word_vec = np.load(os.path.join(self.data_path, 'vec.npy'))
		self.data_test_word = np.load(os.path.join(self.data_path, 'test_word.npy'))
		self.data_test_pos1 = np.load(os.path.join(self.data_path, 'test_pos1.npy'))
		self.data_test_pos2 = np.load(os.path.join(self.data_path, 'test_pos2.npy'))
		self.data_test_mask = np.load(os.path.join(self.data_path, 'test_mask.npy'))
		if self.use_bag:
			self.data_test_label = np.load(os.path.join(self.data_path, 'test_bag_label.npy'))
			self.data_test_scope = np.load(os.path.join(self.data_path, 'test_bag_scope.npy'))
		else:
			self.data_test_label = np.load(os.path.join(self.data_path, 'test_ins_label.npy'))
			self.data_test_scope = np.load(os.path.join(self.data_path, 'test_ins_scope.npy'))
		print("Finish reading")
		self.test_batches = len(self.data_test_label) // self.batch_size
		if len(self.data_test_label) % self.batch_size != 0:
			self.test_batches += 1

		self.total_recall = self.data_test_label[:, 1:].sum()

	def set_train_model(self, model):
		print("Initializing training model...")
		self.model = model
		self.trainModel = self.model(config = self)
		if self.pretrain_model != None:
			self.trainModel.load_state_dict(torch.load(self.pretrain_model))
		#self.trainModel.cuda() JN EDIT. CHANGE LATER
		self.trainModel
		if self.optimizer != None:
			pass
		elif self.opt_method == "Adagrad" or self.opt_method == "adagrad":
			self.optimizer = optim.Adagrad(self.trainModel.parameters(), lr = self.learning_rate, lr_decay = self.lr_decay, weight_decay = self.weight_decay)
		elif self.opt_method == "Adadelta" or self.opt_method == "adadelta":
			self.optimizer = optim.Adadelta(self.trainModel.parameters(), lr = self.learning_rate, weight_decay = self.weight_decay)
		elif self.opt_method == "Adam" or self.opt_method == "adam":
			self.optimizer = optim.Adam(self.trainModel.parameters(), lr = self.learning_rate, weight_decay = self.weight_decay)
		else:
			self.optimizer = optim.SGD(self.trainModel.parameters(), lr = self.learning_rate, weight_decay = self.weight_decay)
		print("Finish initializing")
 			 
	def set_test_model(self, model):
		print("Initializing test model...")
		self.model = model
		self.testModel = self.model(config = self)
		#self.testModel.cuda() JN EDIT. CHAGE LATER
		self.testModel.eval()
		print("Finish initializing")

	def get_train_batch(self, batch):
		input_scope = np.take(self.data_train_scope, self.train_order[batch * self.batch_size : (batch + 1) * self.batch_size], axis = 0)
		index = []
		scope = [0]
		for num in input_scope:
			index = index + list(range(num[0], num[1] + 1))
			scope.append(scope[len(scope) - 1] + num[1] - num[0] + 1)
		self.batch_word = self.data_train_word[index, :]
		self.batch_pos1 = self.data_train_pos1[index, :]
		self.batch_pos2 = self.data_train_pos2[index, :]
		self.batch_mask = self.data_train_mask[index, :]	
		self.batch_label = np.take(self.data_train_label, self.train_order[batch * self.batch_size : (batch + 1) * self.batch_size], axis = 0)
		if self.use_bag:
			self.batch_attention_query = self.data_query_label[index]
		self.batch_scope = scope
	
	def get_test_batch(self, batch):
		input_scope = self.data_test_scope[batch * self.batch_size : (batch + 1) * self.batch_size]
		index = []
		scope = [0]
		for num in input_scope:
			index = index + list(range(num[0], num[1] + 1))
			scope.append(scope[len(scope) - 1] + num[1] - num[0] + 1)
		self.batch_word = self.data_test_word[index, :]
		self.batch_pos1 = self.data_test_pos1[index, :]
		self.batch_pos2 = self.data_test_pos2[index, :]
		self.batch_mask = self.data_test_mask[index, :]
		self.batch_scope = scope

	"""
	This function actually trains the model for one iteration of the batch data.
	This function does not backpropagate; that is handled by train_one_step_part2.
	"""

	def train_one_step_part1(self):
		self.trainModel.embedding.word = to_var(self.batch_word)#.retain_grad()
		self.trainModel.embedding.pos1 = to_var(self.batch_pos1)#.retain_grad()
		self.trainModel.embedding.pos2 = to_var(self.batch_pos2)#.retain_grad()
		self.trainModel.encoder.mask = to_var(self.batch_mask)#.retain_grad()
		self.trainModel.selector.scope = self.batch_scope
		if self.use_bag:
			self.trainModel.selector.attention_query = to_var(self.batch_attention_query)
		self.trainModel.selector.label = to_var(self.batch_label)#.retain_grad()
		self.trainModel.classifier.label = to_var(self.batch_label)#.retain_grad()
		self.optimizer.zero_grad()
		(loss, _output), training_scores, h_w_logits = self.trainModel()

		for i, prediction in enumerate(_output):
			if self.batch_label[i] == 0:
				self.acc_NA.add(prediction == self.batch_label[i])
			else:
				self.acc_not_NA.add(prediction == self.batch_label[i])
			self.acc_total.add(prediction == self.batch_label[i])

		return (loss, _output), training_scores, h_w_logits

	"""
	This function takes care of backpropagation using the w and h gradients
	calculated by the conditional random field.
	"""
	def train_one_step_part2(self, w_grad, loss):
		# format w properly into 2 x K numpy array
		print(w_grad)
		w_grad_new = []
		for i in range(2):
			l = []
			for j in range(i, len(w_grad), 2):
				l.append(w_grad[j])
			w_grad_new.insert(0, l)
			#w_grad_new.append(l)

		# replace w gradient with new gradient calculated
		w_grad_numpy = np.array(w_grad_new)
		w_grad_tensor = torch.from_numpy(w_grad_numpy).float()
		w_grad_var = Variable(w_grad_tensor)

		# Displaying
		#for name, param in self.trainModel.named_parameters():
		#	print(f"{name}\ndata: {param.data}\nrequires_grad: {param.requires_grad}\ngrad: {param.grad}\ngrad_fn: {param.grad_fn}\nis_leaf: {param.is_leaf}\n")

		loss.backward()

		# params = self.trainModel.state_dict()

		for name, param in self.trainModel.named_parameters():
			if name == 'selector.relation_matrix.weight':
				param.grad = w_grad_var

		self.optimizer.step()

	def train_one_step_original(self):
		self.trainModel.embedding.word = to_var(self.batch_word)
		self.trainModel.embedding.pos1 = to_var(self.batch_pos1)
		self.trainModel.embedding.pos2 = to_var(self.batch_pos2)
		self.trainModel.encoder.mask = to_var(self.batch_mask)
		self.trainModel.selector.scope = self.batch_scope
		if self.use_bag:
			self.trainModel.selector.attention_query = to_var(self.batch_attention_query)
		self.trainModel.selector.label = to_var(self.batch_label)
		self.trainModel.classifier.label = to_var(self.batch_label)
		self.optimizer.zero_grad()
		(loss, _output), training_scores, h_w_logits = self.trainModel()

		loss.backward()

		self.optimizer.step()
		#print(self.batch_label)
		for i, prediction in enumerate(_output):
			if self.batch_label[i] == 0:
				self.acc_NA.add(prediction == self.batch_label[i])
			else:
				self.acc_not_NA.add(prediction == self.batch_label[i])
			self.acc_total.add(prediction == self.batch_label[i])
		return loss.data.item(), training_scores


	def test_one_step(self):
		self.testModel.embedding.word = to_var(self.batch_word)
		self.testModel.embedding.pos1 = to_var(self.batch_pos1)
		self.testModel.embedding.pos2 = to_var(self.batch_pos2)
		self.testModel.encoder.mask = to_var(self.batch_mask)
		self.testModel.selector.scope = self.batch_scope
		return self.testModel.test()


	def train(self):
		if not os.path.exists(self.checkpoint_dir):
			os.mkdir(self.checkpoint_dir)
		best_pr_auc = 0.0
		best_roc_auc = 0.0
		best_p = None
		best_r = None
		best_fpr = None
		best_tpr = None
		best_epoch = 0
		best_scores = []
		for epoch in range(self.max_epoch):
			print('Epoch ' + str(epoch) + ' starts...')
			self.acc_NA.clear()
			self.acc_not_NA.clear()
			self.acc_total.clear()
			np.random.shuffle(self.train_order)
			print(self.train_batches)
			for batch in range(self.train_batches):
				self.get_train_batch(batch)
				loss, train_score = self.train_one_step()#NEW
				time_str = datetime.datetime.now().isoformat()
				sys.stdout.write("epoch %d step %d time %s | loss: %f, negative (NA) accuracy: %f, positive (NOT NA) accuracy: %f, total accuracy: %f\r" % (epoch, batch, time_str, loss, self.acc_NA.get(), self.acc_not_NA.get(), self.acc_total.get()))	
				sys.stdout.flush()
			if (epoch + 1) % self.save_epoch == 0:
				print('Epoch ' + str(epoch) + ' has finished')
				print('Saving model...')
				path = os.path.join(self.checkpoint_dir, self.model.__name__ + '-' + str(epoch))
				torch.save(self.trainModel.state_dict(), path)
				print('Have saved model to ' + path)
			if (epoch + 1) % self.test_epoch == 0:
				self.testModel = self.trainModel
				roc_auc, pr_auc, pr_x, pr_y, fpr, tpr, _ = self.test_one_epoch()
				if roc_auc > best_roc_auc:
					best_roc_auc = roc_auc
					best_fpr = fpr
					best_tpr = tpr
					best_pr_auc = pr_auc
					best_p = pr_x
					best_r = pr_y
					# best_scores = scores
					best_epoch = epoch

		# Do one last step of Batch Gradient Descent to gather scores

		print("Finish training")
		print("Best epoch = %d | pr_auc = %f | roc_auc = %f" % (best_epoch, best_pr_auc, roc_auc))
		print("Storing best result...")
		if not os.path.isdir(self.test_result_dir):
			os.mkdir(self.test_result_dir)
		np.save(os.path.join(self.test_result_dir + '/pr_auc', self.model.__name__ + '_pr_x.npy'), best_p)
		np.save(os.path.join(self.test_result_dir + '/pr_auc', self.model.__name__ + '_pr_y.npy'), best_r)
		np.save(os.path.join(self.test_result_dir + '/roc_auc', self.model.__name__ + '_roc_x.npy'), best_fpr)
		np.save(os.path.join(self.test_result_dir + '/roc_auc', self.model.__name__ + '_roc_y.npy'), best_tpr)

		print("Finish storing")

	def test_one_epoch(self):
		test_score = []
		test_h = []
		test_w = []
		test_logits = []
		for batch in tqdm(range(int(self.test_batches))):#JN EDIT HERE; REMOVE 'floor()'
			self.get_test_batch(batch)
			batch_score, h_w_logits = self.test_one_step()
			test_score = test_score + batch_score
			test_h += h_w_logits[0]
			test_w += h_w_logits[1]
			test_logits += h_w_logits[2]
		test_result = []
		for i in range(len(test_score)):
			for j in range(1, len(test_score[i])):
				test_result.append([self.data_test_label[i][j], test_score[i][j]])

		test_result = sorted(test_result, key = lambda x: x[1])
		test_result = test_result[::-1]
		pr_x = []
		pr_y = []
		correct = 0
		for i, item in enumerate(test_result):
			correct += item[0]
			pr_y.append(float(correct) / (i + 1))
			pr_x.append(float(correct) / self.total_recall)
		pr_auc = sklearn.metrics.auc(x = pr_x, y = pr_y)
		print("PR-AUC: ", pr_auc)

		# Calculate ROC-AUC
		ground_truths = []
		marginals = []
		# Separate ground truth and marginal probabilities into own lists
		for result in test_result:
			ground_truths.append(result[0])
			marginals.append(result[1])
		fpr, tpr, _ = sklearn.metrics.roc_curve(ground_truths, marginals)
		roc_auc = sklearn.metrics.auc(fpr, tpr)
		print("ROC-AUC: ", roc_auc)

		return roc_auc, pr_auc, pr_x, pr_y, fpr, tpr, test_score, (test_h, test_w, test_logits)

	def test(self):
		best_epoch = None
		best_pr_auc = 0.0
		best_roc_auc = 0.0
		best_p = None
		best_r = None
		best_fpr = None
		best_tpr = None
		best_scores = []
		for epoch in self.epoch_range:
			path = os.path.join(self.checkpoint_dir, self.model.__name__ + '-' + str(epoch))
			if not os.path.exists(path):
				continue
			print("Start testing epoch %d" % (epoch))
			self.testModel.load_state_dict(torch.load(path))
			roc_auc, pr_auc, p, r, fpr, tpr, scores = self.test_one_epoch()
			if roc_auc > best_roc_auc:
				best_roc_auc = roc_auc
				best_fpr = fpr
				best_tpr = tpr
				best_pr_auc = pr_auc
				best_p = p
				best_r = r
				best_scores = scores
				best_epoch = epoch
			print("Finish testing epoch %d" % (epoch))
		print("Best epoch = %d | pr_auc = %f | roc_auc = %f" % (best_epoch, best_pr_auc, best_roc_auc))
		print("Storing best result...")
		if not os.path.isdir(self.test_result_dir):
			os.mkdir(self.test_result_dir)
		np.save(os.path.join(self.test_result_dir + '/pr_auc', self.model.__name__ + '_pr_x.npy'), best_p)
		np.save(os.path.join(self.test_result_dir + '/pr_auc', self.model.__name__ + '_pr_y.npy'), best_r)
		np.save(os.path.join(self.test_result_dir + '/roc_auc', self.model.__name__ + '_roc_x.npy'), best_fpr)
		np.save(os.path.join(self.test_result_dir + '/roc_auc', self.model.__name__ + '_roc_y.npy'), best_tpr)
		# SHOULD DO SOMETHING WITH SCORES HERE
		print("Finish storing")


	"""
	All functions below are for the case of k-fold cross-validation. They are added by
	Jake Nocentino.
	"""

	def load_k_fold_train_data(self, k):
		print("Reading training data...")
		self.data_word_vec = np.load(os.path.join(self.data_folds_path + 'fold{}/'.format(k), 'vec.npy'))
		self.data_train_word = np.load(os.path.join(self.data_folds_path + 'fold{}/'.format(k), 'train_word.npy'))
		self.data_train_pos1 = np.load(os.path.join(self.data_folds_path + 'fold{}/'.format(k), 'train_pos1.npy'))
		self.data_train_pos2 = np.load(os.path.join(self.data_folds_path + 'fold{}/'.format(k), 'train_pos2.npy'))
		self.data_train_mask = np.load(os.path.join(self.data_folds_path + 'fold{}/'.format(k), 'train_mask.npy'))
		if self.use_bag:
			self.data_query_label = np.load(os.path.join(self.data_folds_path + 'fold{}/'.format(k), 'train_ins_label.npy'))
			self.data_train_label = np.load(os.path.join(self.data_folds_path + 'fold{}/'.format(k), 'train_bag_label.npy'))
			self.data_train_scope = np.load(os.path.join(self.data_folds_path + 'fold{}/'.format(k), 'train_bag_scope.npy'))
		else:
			self.data_train_label = np.load(os.path.join(self.data_folds_path + 'fold{}/'.format(k), 'train_ins_label.npy'))
			self.data_train_scope = np.load(os.path.join(self.data_folds_path + 'fold{}/'.format(k), 'train_ins_scope.npy'))
		print("Finish reading")
		self.train_order = list(range(len(self.data_train_label)))
		self.train_batches = len(self.data_train_label) // self.batch_size
		if len(self.data_train_label) % self.batch_size != 0:
			self.train_batches += 1

	def load_k_fold_test_data(self, k):
		print("Reading testing data...")
		self.data_word_vec = np.load(os.path.join(self.data_folds_path + 'fold{}/'.format(k), 'vec.npy'))
		self.data_test_word = np.load(os.path.join(self.data_folds_path + 'fold{}/'.format(k), 'test_word.npy'))
		self.data_test_pos1 = np.load(os.path.join(self.data_folds_path + 'fold{}/'.format(k), 'test_pos1.npy'))
		self.data_test_pos2 = np.load(os.path.join(self.data_folds_path + 'fold{}/'.format(k), 'test_pos2.npy'))
		self.data_test_mask = np.load(os.path.join(self.data_folds_path + 'fold{}/'.format(k), 'test_mask.npy'))
		if self.use_bag:
			self.data_test_label = np.load(os.path.join(self.data_folds_path + 'fold{}/'.format(k), 'test_bag_label.npy'))
			self.data_test_scope = np.load(os.path.join(self.data_folds_path + 'fold{}/'.format(k), 'test_bag_scope.npy'))
		else:
			self.data_test_label = np.load(os.path.join(self.data_folds_path + 'fold{}/'.format(k), 'test_ins_label.npy'))
			self.data_test_scope = np.load(os.path.join(self.data_folds_path + 'fold{}/'.format(k), 'test_ins_scope.npy'))
		print("Finish reading")
		self.test_batches = len(self.data_test_label) / self.batch_size
		if len(self.data_test_label) % self.batch_size != 0:
			self.test_batches += 1

		self.total_recall = self.data_test_label[:, 1:].sum()


	# The method used to train the model for k-fold cross validation, as opposed to the regular train or test methods.
	def train_each_fold(self, k):#, lib):
		# CRF MODEL
		#unspeakable evil

		""" UNCOMMENT
		nums = [b"0",b"1",b"2",b"3",b"4",b"5",b"6",b"7",b"8",b"9"]
		param1 = sb(b"0.001")
		param2 = sb(r+b"fold-"+nums[k]+b"-edges-ALT.txt")
		param3 = sb(r+b"CRFmodel-"+nums[k]+b".txt")
		param4 = sb(b"model-fold-"+nums[k]+b".txt")
		param5 = sb(r+b"maps/map"+nums[k]+b".txt")
		model_crf = lib.InitializeCRF(param1,param2,param3,param4,param5)
		"""
		

		best_pr_auc = 0.0
		best_roc_auc = 0.0
		best_p = None
		best_r = None
		best_fpr = None
		best_tpr = None

		train_epoch_score = []
		best_train_score = []
		best_test_score = []
		best_epoch = 0

		train_epoch_h= []
		train_epoch_w = []
		train_epoch_logits = []
		print('Fold ' + str(k) + ' starts...\n')
		for epoch in range(self.max_epoch):
			print(epoch, "epoch")

			self.acc_NA.clear()
			self.acc_not_NA.clear()
			self.acc_total.clear()
			train_epoch_score.clear()
			train_epoch_h.clear()
			train_epoch_w.clear()
			train_epoch_logits.clear()

			"""
			for name, param in self.trainModel.named_parameters():
				if param.requires_grad:
					print(name, param.data)
					print('{} SHAPE: {}'.format(name.upper(), param.shape))
					print('{} GRAD: {}'.format(name.upper(), param.grad))
			"""

			print(self.train_batches)
			for batch in range(self.train_batches):
				# TRAINING PORTION

				self.get_train_batch(batch)
				(loss, _output), train_batch_score, train_h_w_logits = self.train_one_step_part1()
				#loss, train_batch_score= self.train_one_step_original()
				train_epoch_h += train_h_w_logits[0]
				train_epoch_w += train_h_w_logits[1]
				train_epoch_logits += train_h_w_logits[2]
				train_epoch_score += train_batch_score
				time_str = datetime.datetime.now().isoformat()
				sys.stdout.write('Fold %d Epoch %d Step %d Time %s | Loss: %f, Neg Accuracy: %f, Pos Accuracy: %f, Total Accuracy: %f\r' % (k, epoch, batch, time_str, loss.data.item(), self.acc_NA.get(), self.acc_not_NA.get(), self.acc_total.get()))
				sys.stdout.flush()

				# use gradients to backpropagate
				self.train_one_step_part2(wgrad.astype('float32'), loss)

				# TESTING PORTING
				#self.testModel = self.trainModel
				#roc_auc, pr_auc, pr_x, pr_y, fpr, tpr, test_score, test_h_w_logits = self.test_one_epoch()
				#print(scores)
				#scores = np.concatenate(scores,axis=0)
				#total_score = train_epoch_score + test_score
				#total_h = train_epoch_h + test_h_w_logits[0]
				#total_w = train_epoch_w + test_h_w_logits[1]
				#total_logits = train_epoch_logits + test_h_w_logits[2]

				# If you want to test your model OLD! IGNORE for now
			if (epoch + 1) % self.test_epoch == 0:
				self.testModel = self.trainModel
				roc_auc, pr_auc, pr_x, pr_y, fpr, tpr, test_score, test_h_w_logits = self.test_one_epoch()
				#print(scores);scores = np.concatenate(scores,axis=0)
				total_score = train_epoch_score + test_score
				total_h = train_epoch_h + test_h_w_logits[0]
				total_w = train_epoch_w + test_h_w_logits[1]
				total_logits = train_epoch_logits + test_h_w_logits[2]
				# CRF CODE
				""" UNCOMMENT
				as_p = lambda a:ctypes.cast(a.__array_interface__['data'][0],ctypes.POINTER(ctypes.c_double))
				h = np.concatenate(total_h,axis=0).astype("float64");
				w = np.concatenate(train_epoch_w,axis=0).astype("float64");
				l = np.concatenate(total_logits,axis=0).astype("float64");
				l2 = np.concatenate(test_h_w_logits[1],axis=0).astype("float64")
				hgradient = np.zeros(h.shape)
				wgradient = np.zeros(w.shape)
				lib.Gradient(model_crf, as_p(hgradient), as_p(h), as_p(wgradient), as_p(w), as_p(l), as_p(l2), 768)
				"""
				#print(hgradient.tolist())
				#print(wgradient.tolist());exit(0)
				# END CRF CODE AND USE GRADIENTS TO BACKPROPAGATE

				if roc_auc > best_roc_auc:
					best_pr_auc = pr_auc
					best_roc_auc = roc_auc
					best_p = pr_x
					best_r = pr_y
					best_fpr = fpr
					best_tpr = tpr
					best_train_score = train_epoch_score
					best_test_score = test_score
					best_epoch = epoch

			# If you want to save your models for later use:
			if (epoch + 1) % self.save_epoch == 0 and self.save:
				print('Epoch ' + str(epoch) + ' has finished')
				print('Saving model...')
				path = os.path.join(self.checkpoint_dir, self.model.__name__ + '-' + str(epoch))
				torch.save(self.trainModel.state_dict(), path)
				print('Have saved model to ' + path)

			# If you want to test your model OLD! IGNORE for now
			#if (epoch + 1) % self.test_epoch == 0:
				#self.testModel = self.trainModel
				#roc_auc, pr_auc, pr_x, pr_y, fpr, tpr, test_score, test_h_w_logits = self.test_one_epoch()
				#print(scores);scores = np.concatenate(scores,axis=0)
				#total_score = train_epoch_score + test_score
				#total_h = train_epoch_h + test_h_w_logits[0]
				#total_w = train_epoch_w + test_h_w_logits[1]
				#total_logits = train_epoch_logits + test_h_w_logits[2]
				# CRF CODE
				#as_p = lambda a:ctypes.cast(a.__array_interface__['data'][0],ctypes.POINTER(ctypes.c_double))
				#h = np.concatenate(total_h,axis=0).astype("float64");
				#w = np.concatenate(train_epoch_w,axis=0).astype("float64");
				#l = np.concatenate(total_logits,axis=0).astype("float64");
				#l2 = np.concatenate(test_h_w_logits[1],axis=0).astype("float64")
				#hgradient = np.zeros(h.shape)
				#wgradient = np.zeros(w.shape)
				#lib.Gradient(model_crf, as_p(hgradient), as_p(h), as_p(wgradient), as_p(w), as_p(l), as_p(l2), 768) UNCOMMENT
				#print(hgradient.tolist())
				#print(wgradient.tolist());exit(0)

				# END CRF CODE AND USE GRADIENTS TO BACKPROPAGATE
				# self.train_one_step_part2(hgrad.astype('float32'), wgrad.astype('float32'), loss_output[0])

				#if roc_auc > best_roc_auc:
					#best_pr_auc = pr_auc
					#best_roc_auc = roc_auc
					#best_p = pr_x
					#best_r = pr_y
					#best_fpr = fpr
					#best_tpr = tpr
					#best_train_score = train_epoch_score
					#best_test_score = test_score
					#best_epoch = epoch
		best_total_score = best_train_score.extend(best_test_score)
		print("Finish training")
		print("Best epoch = {} | pr_auc = {} | roc_auc = {}".format(best_epoch, best_pr_auc, roc_auc))

		self.cur_epoch += 1

		return best_roc_auc, best_pr_auc, best_p, best_r, best_fpr, best_tpr, best_total_score



		"""for epoch in range(self.max_epoch):
			print('Epoch ' + str(epoch) + ' starts...')
			self.acc_NA.clear()
			self.acc_not_NA.clear()
			self.acc_total.clear()
			np.random.shuffle(self.train_order)
			print(self.train_batches)
			for batch in range(self.train_batches):
				self.get_train_batch(batch)
				loss = self.train_one_step()
				time_str = datetime.datetime.now().isoformat()
				sys.stdout.write("epoch %d step %d time %s | loss: %f, NA accuracy: %f, not NA accuracy: %f, total accuracy: %f\r" % (epoch, batch, time_str, loss, self.acc_NA.get(), self.acc_not_NA.get(), self.acc_total.get()))	
				sys.stdout.flush()
			if (epoch + 1) % self.save_epoch == 0:
				print('Epoch ' + str(epoch) + ' has finished')
				print('Saving model...')
				path = os.path.join(self.checkpoint_dir, self.model.__name__ + '-' + str(epoch))
				torch.save(self.trainModel.state_dict(), path)
				print('Have saved model to ' + path)
			if (epoch + 1) % self.test_epoch == 0:
				self.testModel = self.trainModel
				auc, pr_x, pr_y = self.test_one_epoch()
				if auc > best_auc:
					best_auc = auc
					best_p = pr_x
					best_r = pr_y
					best_epoch = epoch
		print("Finish training")
		print("Best epoch = %d | auc = %f" % (best_epoch, best_auc))
		print("Storing best result...")
		if not os.path.isdir(self.test_result_dir):
			os.mkdir(self.test_result_dir)
		np.save(os.path.join(self.test_result_dir, self.model.__name__ + '_x.npy'), best_p)
		np.save(os.path.join(self.test_result_dir, self.model.__name__ + '_y.npy'), best_r)
		print("Finish storing")"""
