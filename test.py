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

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type = str, default = 'pcnn_att', help = 'name of the model')
parser.add_argument('--max_epoch', type = int, default = 20, help = 'max number of epochs to run.')
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
con.load_test_data()
con.set_test_model(model[args.model_name])
con.set_epoch_range([i for i in range(args.max_epoch)])
con.test()
