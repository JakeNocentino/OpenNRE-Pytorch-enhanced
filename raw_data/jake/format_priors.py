"""
Script to format priots into a format acceptable for Jakob's CRF.
"""
import numpy as np

k_folds = 10

for k in range(k_folds):
	prior_scores = np.load('fold{}/scores.npy'.format(k))
	print(prior_scores)
	print(len(prior_scores))
	print('\n')
	"""
	new_priors = []
	for priors in prior_scores:
		#print(priors)
		new_priors.append(priors[0])
		new_priors.append(priors[1])

	print("New priors array len:")
	print(len(new_priors))
	print('\n')

	np.save('fold{}/formatted_scores'.format(k), new_priors)
	"""
	new_priors = np.load('fold{}/formatted_scores.npy'.format(k))
	print(new_priors)
	print(len(new_priors))
	print('\n')