import sklearn.metrics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sys
import argparse
import os

result_dir = './test_result'

parser = argparse.ArgumentParser()
parser.add_argument('--metric', type = str, help = 'Metric to plot.')
parser.add_argument('--model', type = str, help = 'Model to plot.')
args = parser.parse_args()

def main():
	#models = sys.argv[1]
	model = args.model

	if args.metric =='pr':
		result_dir += '/pr_auc'
		for model in models:
			x = np.load(os.path.join(result_dir, model + '_pr_x.npy'))
			y = np.load(os.path.join(result_dir, model + '_pr_y.npy'))
			f1 = (2 * x * y / (x + y + 1e-20)).max()
			auc = sklearn.metrics.auc(x = x, y = y)
			plt.plot(x, y, lw = 2, label = model)
			print(model + ' : ' + 'auc = ' + str(auc) + ' | ' + 'max F1 = ' + str(f1) + '    P@100: {} | P@200: {} | P@300: {} | Mean: {}'.format(y[100], y[200], y[300], (y[100] + y[200] + y[300]) / 3))
	
		plt.xlabel('Recall')
		plt.ylabel('Precision')
		plt.ylim(0.0, 1.0)
		plt.xlim(0.0, 1.0)
		plt.title('Precision-Recall')
		plt.legend(loc = "upper right")
		plt.grid(True)
		plt.savefig(os.path.join(result_dir, model + '_pr_curve'))
	elif args.metric == 'roc':
		result_dir = './test_result/roc_auc'
		x = np.load(os.path.join(result_dir, model + '_roc_x.npy'))
		y = np.load(os.path.join(result_dir, model + '_roc_y.npy'))
		auc = sklearn.metrics.auc(x = x, y = y)
		plt.plot(x, y, lw = 2, label = model)
		print(model + ' : ' + 'auc = ' + str(auc))

		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')
		plt.ylim(0.0, 1.0)
		plt.xlim(0.0, 1.0)
		plt.title('ROC Curve')
		plt.legend(loc = 'upper right')
		plt.grid(True)
		plt.savefig(os.path.join(result_dir, model + '_roc_curve'))

if __name__ == "__main__":
	main()
