import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from networks.embedding import *
from networks.encoder import *
from networks.selector import *
from networks.classifier import *
import numpy
class Model(nn.Module):
	def __init__(self, config):	
		super(Model, self).__init__()
		self.config = config
		self.embedding = Embedding(config)
		self.encoder = None
		self.selector = None
		self.classifier = Classifier(config)

	def forward(self):
		embedding = self.embedding()
		sen_embedding = self.encoder(embedding)
		h, w, logits = self.selector(sen_embedding)
		score = F.softmax(logits,1)
		a = lambda b:list(b.data.cpu().numpy())
		return self.classifier(logits), list(score.data.cpu().numpy()), (a(h),a(w),a(logits))
		
	def test(self):
		embedding = self.embedding()
		sen_embedding = self.encoder(embedding)
		return self.selector.test(sen_embedding)
