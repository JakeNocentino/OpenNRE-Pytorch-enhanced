from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold #train_test_split
import sklearn.metrics
from random import shuffle, seed
import numpy as np
import pickle as pkl
import json
import os

k_folds = 10 # NUM FOLDS

with open('../formatted_data_newest.json', 'r') as infile:
	data_complete = json.load(infile)
"""
for k in range(k_folds):
	objs = []
	with open('folds/fold{}.pkl'.format(k), 'rb') as infile:
		while True:
			try:
				objs.append(pkl.load(infile))
			except EOFError:
				break
	objs = objs[0]
	print('train len: ', len(objs[0]), ' ', 'test len: ', len(objs[1]))
"""


for k in range(k_folds):
	objs = []
	with open('val_folds/val-fold-{}.pkl'.format(k), 'rb') as infile:
		while True:
			try:
				objs.append(pkl.load(infile))
			except EOFError:
				break
	objs = objs[0]
	print('train len: ', len(objs[0]), ' ', 'test len: ', len(objs[1]))


	new_train_obj = [] # object used to store train data
	new_test_obj = [] # object used to store test data
	new_complete_obj = []  # object used to store both the train and test data for new json file
	i = 0 # train counter
	j = 0 # test counter

	# Generating train data
	for obj in objs[0]:
		# 1.) Extract sent_id, adj_idx, and noun_idx
		dot_idx = obj[1].find('.')
		dot_idx2 = obj[1].rfind('.')
		sent_id = obj[1][:dot_idx]
		adj_noun_idxs = obj[1][dot_idx+2:dot_idx2-1]
		adj_idx = int(adj_noun_idxs[:adj_noun_idxs.find(',')])
		noun_idx = int(adj_noun_idxs[adj_noun_idxs.find(' ')+1:])

		adj_noun = obj[2]
		adj_word = adj_noun[:adj_noun.find('_')]
		#print(adj_word)
		noun_word = adj_noun[adj_noun.find('_')+1:]
		#print(noun_word)

		data_position = objs[0]

		# Scan data_complete for sent_id and adj_noun match with a 
		# +- 1 idx range.

		for entry in data_complete:
			if entry['id'] == sent_id and adj_idx == entry['adjective_position'] and noun_idx == entry['noun_position'] and entry['head']['word'] == adj_word and entry['tail']['word'] == noun_word:
				new_train_obj.append(entry)
				i += 1
				#print("Train {}".format(i))
				break
			#tokens = entry['sentence'].split(' ')
			#if entry['id'] == sent_id and tokens[adj_idx] == entry['head']['word'] and tokens[noun_idx] == entry['tail']['word']:
				#new_train_obj.append(entry)
				#i += 1
				#break

			# if last entry and nothing found yet, try again with +- 1 range
			#if data_complete[len(data_complete) - 1] == entry:
				#for entry in data_complete:
					#if entry['id'] == sent_id and (entry['adjective_position'] == adj_idx or entry['adjective_position'] == adj_idx - 1 or entry['adjective_position'] == adj_idx + 1) and (entry['noun_position'] == noun_idx or entry['noun_position'] == noun_idx - 1 or entry['noun_position'] == noun_idx + 1):
						#new_train_obj.append(entry)
						#print('TRAIN HIT!: {}'.format(i))
						#print(entry)
						#i += 1
						#break
	#print('Fold {}: Train has {} entries.'.format(k, i))

	for obj in objs[1]:
		# 1.) Extract sent_id, adj_idx, and noun_idx
		dot_idx = obj[1].find('.')
		dot_idx2 = obj[1].rfind('.')
		sent_id = obj[1][:dot_idx]
		adj_noun_idxs = obj[1][dot_idx+2:dot_idx2-1]
		adj_idx = int(adj_noun_idxs[:adj_noun_idxs.find(',')])
		noun_idx = int(adj_noun_idxs[adj_noun_idxs.find(' ')+1:])

		adj_noun = obj[2]
		adj_word = adj_noun[:adj_noun.find('_')]
		noun_word = adj_noun[adj_noun.find('_')+1:]

		data_position = objs[0]

		# 2.) Scan data_complete.json for sent_id and adj_noun match with a 
		# +- 1 idx range.
		
		for entry in data_complete:
			if entry['id'] == sent_id and adj_idx == entry['adjective_position'] and noun_idx == entry['noun_position']  and entry['head']['word'] == adj_word and entry['tail']['word'] == noun_word:
				new_test_obj.append(entry)
				j += 1
				#print("Test {}".format(j))
				break
			#tokens = entry['sentence'].split(' ')
			#if entry['id'] == sent_id and tokens[adj_idx] == entry['head']['word'] and tokens[noun_idx] == entry['tail']['word']:
				#new_test_obj.append(entry)
				#print('HIT!: {}'.format(i))
				#print(entry)
				#j += 1
				#break

			# if last entry and nothing found yet, try again with +- 1 range
			#if data_complete[len(data_complete) - 1] == entry:
				#for entry in data_complete:
					#if entry['id'] == sent_id and (entry['adjective_position'] == adj_idx or entry['adjective_position'] == adj_idx - 1 or entry['adjective_position'] == adj_idx + 1) and (entry['noun_position'] == noun_idx or entry['noun_position'] == noun_idx - 1 or entry['noun_position'] == noun_idx + 1):
						#new_test_obj.append(entry)
						#print('TEST HIT!: {}'.format(i))
						#print(entry)
						#j += 1
						#break
	#print('Fold {}: Test has {} entries.'.format(k, j))

	# make sure new_complete_obj is correct length
	new_complete_obj.append(new_train_obj)
	new_complete_obj.append(new_test_obj)
	print("Train length: {}".format(len(new_complete_obj[0])))
	print("Test length: {}".format(len(new_complete_obj[1])))
	print("Total length: {}".format(len(new_complete_obj[0]) + len(new_complete_obj[1])))

	# 3.) Store into one json file in foler 'fold{k}'
	dir_name = 'fold{}'.format(k)
	if not os.path.exists(dir_name):
		os.mkdir(dir_name)

	with open('fold{}/data_complete.json'.format(k), 'w') as outfile:
		json.dump(new_complete_obj, outfile, indent=4)

	print('Fold {} dumped and ready to go!'.format(k))

