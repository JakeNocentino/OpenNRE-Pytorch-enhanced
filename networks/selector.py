import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy
class Selector(nn.Module):
	def __init__(self, config, relation_dim):
		super(Selector, self).__init__()
		self.config = config
		self.relation_matrix = nn.Embedding(self.config.num_classes, relation_dim)
		self.bias = nn.Parameter(torch.Tensor(self.config.num_classes), requires_grad=True)
		# comment the below out unless using attention
		#self.attention_matrix = nn.Embedding(self.config.num_classes, relation_dim)
		self.current_batch = self.config.current_batch
		self.init_weights()
		self.scope = None
		self.attention_query = None
		self.label = None
		self.dropout = nn.Dropout(self.config.drop_prob)

	def init_weights(self):	
		nn.init.xavier_uniform_(self.relation_matrix.weight.data)
		nn.init.normal_(self.bias)
		# comment the below out unless using attention
		#nn.init.xavier_uniform(self.attention_matrix.weight.data)

	def get_logits(self, x):
		w = torch.transpose(self.relation_matrix.weight, 0, 1)
		logits = torch.matmul(x, w) + self.bias

		return x, w, logits

	def forward(self, x):
		raise NotImplementedError

	def test(self, x):
		raise NotImplementedError

class Attention(Selector):
	def _attention_train_logit(self, x):
		relation_query = self.relation_matrix(self.attention_query)
		attention = self.attention_matrix(self.attention_query)
		#attention_logit = torch.sum(x * attention * relation_query, 1, True)
		attention_logit = torch.sum(x * attention * relation_query, 1, True)
		return attention_logit

	def _attention_test_logit(self, x):
		attention_logit = torch.matmul(x, torch.transpose(self.attention_matrix.weight * self.relation_matrix.weight, 0, 1))
		return attention_logit

	def forward(self, x):
		attention_logit = self._attention_train_logit(x)
		tower_repre = []
		for i in range(len(self.scope) - 1):
			sen_matrix = x[self.scope[i] : self.scope[i + 1]]
			attention_score = F.softmax(torch.transpose(attention_logit[self.scope[i] : self.scope[i + 1]], 0, 1), 1)
			final_repre = torch.squeeze(torch.matmul(attention_score, sen_matrix))
			tower_repre.append(final_repre)
		stack_repre = torch.stack(tower_repre)
		stack_repre = self.dropout(stack_repre)
		logits = self.get_logits(stack_repre)
		return logits

	def test(self, x):
		attention_logit = self._attention_test_logit(x)	
		tower_output = []
		for i in range(len(self.scope) - 1):
			sen_matrix = x[self.scope[i] : self.scope[i + 1]]
			attention_score = F.softmax(torch.transpose(attention_logit[self.scope[i] : self.scope[i + 1]], 0, 1), 1)
			final_repre = torch.matmul(attention_score, sen_matrix)
			logits = self.get_logits(final_repre)
			tower_output.append(torch.diag(F.softmax(logits, 1)))
		stack_output = torch.stack(tower_output)
		# print('We out here getting tested >:)')
		return list(stack_output.data.cpu().numpy())

class One(Selector):
	def forward(self, x):
		tower_logits = []
		for i in range(len(self.scope) - 1):
			sen_matrix = x[self.scope[i] : self.scope[i + 1]]
			sen_matrix = self.dropout(sen_matrix)
			logits = self.get_logits(sen_matrix)
			score = F.softmax(logits, 1)
			_, k = torch.max(score, dim = 0)
			k = k[self.label[i]]
			tower_logits.append(logits[k])
		return torch.cat(tower_logits, 0)

	def test(self, x):
		tower_score = []
		for i in range(len(self.scope) - 1):
			sen_matrix = x[self.scope[i] : self.scope[i + 1]]
			logits = self.get_logits(sen_matrix)
			score = F.softmax(logits, 1)
			score, _ = torch.max(score, 0)
			tower_score.append(score)
		tower_score = torch.stack(tower_score)
		return list(tower_score.data.cpu().numpy())

class Average(Selector):

	def forward(self, x):
		# hgrad and wgrad formatting
		hgrad = numpy.load('hgrad3.npy')
		wgrad = numpy.load('wgrad3.npy')
		#print(hgrad.tolist())
		K_CONST = 768
		N_CONST = 7402
		N_TRAIN_CONST = 6661

		hgrad_new = []
		for n in range(N_CONST):
			h_row = []
			for k in range(K_CONST):
				h_row.append(hgrad[(n * K_CONST) + k])
			hgrad_new.append(h_row)
			#hgrad_new.insert(0, h_row)
		#h_row = []
		#for k in range(K_CONST):
		#	h_row.append(hgrad[(self.current_batch * K_CONST) + k])
		#h_grad_numpy = numpy.array(h_row)
		#h_grad_tensor = torch.from_numpy(h_grad_numpy).float()
		#h_grad_tensor.unsqueeze_(0)

		hgrad_train = hgrad_new[:N_TRAIN_CONST]
		h_grad_numpy = numpy.array(hgrad_train)
		h_grad_tensor = torch.from_numpy(h_grad_numpy).float()
		#print(h_grad_tensor)
		#print(h_grad_tensor.shape)
		# print(h_grad_tensor)
		# print(h_grad_tensor.shape)

		w_grad_new = []
		for i in range(2):
			l = []
			for j in range(i, len(wgrad), 2):
				l.append(wgrad[j])
			#w_grad_new.insert(0, l)
			w_grad_new.append(l)

		# replace w gradient with new gradient calculated
		w_grad_numpy = numpy.array(w_grad_new)
		w_grad_tensor = torch.from_numpy(w_grad_numpy).float()
		# end hgrad and wgrad formatting

		tower_repre = []
		for i in range(len(self.scope) - 1):
			sen_matrix = x[self.scope[i] : self.scope[i+ 1]]
			final_repre = torch.mean(sen_matrix, 0)
			tower_repre.append(final_repre)
		stack_repre = torch.stack(tower_repre)
		stack_repre = self.dropout(stack_repre)

		#print(stack_repre.shape)
		#print('H SHAPE!!')

		# experimental backward()
		# stack_repre.backward(gradient=h_grad_tensor)
		# autograd.backward(stack_repre, grad_tensors=h_grad_tensor)

		h, w, logits = self.get_logits(stack_repre)

		#autograd.set_detect_anomaly(True)

		# ADDITIONS FOR CROWDSOURCE PROJECT BELOW!
		#score = F.softmax(logits, 1)
		return h, w, logits#, list(score.data.cpu().numpy())

	def test(self, x):
		tower_repre = []
		for i in range(len(self.scope) - 1):
			sen_matrix = x[self.scope[i] : self.scope[i + 1]]
			final_repre = torch.mean(sen_matrix, 0)
			tower_repre.append(final_repre)
		stack_repre = torch.stack(tower_repre)

		h, w, logits = self.get_logits(stack_repre)



		score = F.softmax(logits,1)
		a =lambda b: list(b.data.cpu().numpy())
		return list(score.data.cpu().numpy()), (a(h),a(w),a(logits))
