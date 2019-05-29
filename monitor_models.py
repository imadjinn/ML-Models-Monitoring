import numpy as np

from sklearn.covariance import EllipticEnvelope
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor

import torch
from torch import nn
from torch import optim
from torch.autograd import Variable


class DenseAE(nn.Module):
    def __init__(self):
        super(DenseAE, self).__init__()
        
        # Training Parameters
        self.FEATURES_DIM = None
        self.INNER_DIM = None
        self.N_EPOCHES = 100
        self.BATCH_SIZE = 1000
        self.LEARNING_RATE = 0.01
        
        # AE Parameters
        self.intermediate_dim = None
        self.input = None
        self.h1 = None
        self.h1_inv = None
        self.output = None
        
        # Auxiliary data
        self.train_losses = []
        self.learned = False
        
    def _islearned(foo):
        def wrap(self, xx):
            if self.learned == False:
                raise RuntimeError('The Autoencoder has not been fitted yet.')
            else:
                return foo(self, xx)
        return wrap
        
    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x
    
    def encode(self, x):
        x = torch.relu(self.input(x))
        x = torch.relu(self.h1(x))
        return x
    
    def decode(self, x):
        x = torch.relu(self.h1_inv(x))
        x = self.output(x)
        return x
    
    @_islearned
    def getloss(self, data):
        data = Variable(torch.from_numpy(np.matrix(data).astype(np.float32)))
        x = self.forward(data)
        lf = nn.MSELoss()
        return lf(x, data).detach()
    
    def learn(self, features):
        
        # Initialization AE Parameters :: data dependent architecture
        self.FEATURES_DIM = features.shape[1]
        self.INNER_DIM = self.FEATURES_DIM // 2
        self.intermediate_dim = (self.FEATURES_DIM + self.INNER_DIM) // 2
        self.input = nn.Linear(self.FEATURES_DIM, self.intermediate_dim)
        self.h1 = nn.Linear(self.intermediate_dim, self.INNER_DIM)
        self.h1_inv = nn.Linear(self.INNER_DIM, self.intermediate_dim)
        self.output = nn.Linear(self.intermediate_dim, self.FEATURES_DIM)
        
        optimizer = optim.Adam(self.parameters(), lr=self.LEARNING_RATE)
        criterion = nn.MSELoss()
        self._autoencoder_training(optimizer, criterion, features, 
                                   self.N_EPOCHES, self.BATCH_SIZE)
        self.learned = True
        
    def _fetch_batch(self, data, batch_size):
        n_batches = int(np.ceil(len(data)/batch_size))
        for j in range(n_batches):
            start = j*batch_size
            end = start + batch_size
            yield data[start:end]

    def _autoencoder_training(self, optimizer, criterion, data, n_epoches, batch_size):
        train_losses = []
        for i in range(1, n_epoches+1):
            for bi, batch in enumerate(self._fetch_batch(data, batch_size)):
                batch = Variable(torch.from_numpy(batch))
                optimizer.zero_grad()
                outputs = self(batch)
                loss = criterion(outputs, batch)
                loss.backward()
                optimizer.step()
            self.train_losses.append(loss.item()/batch_size)
            if i%20 == 0 or i == n_epoches:
                print('\rEpoch: %2d train loss: %.5f'
                                 %(i, loss.item()))

class EllipticEnvelopeExtended(EllipticEnvelope):
    def getloss(self, data):
        loss = self.mahalanobis(data)
        return np.sum(loss)
    def learn(self, data):
        self.fit(data)
        
class OneClassSVMExtended(OneClassSVM):
    def getloss(self, data):
        loss = self.decision_function(data)
        return np.sum(loss)
    def learn(self, data):
        self.fit(data)
        
class LocalOutlierFactorExtended(LocalOutlierFactor):
    def getloss(self, data):
        loss = self.decision_function(data)
        return np.sum(loss)
    def learn(self, data):
        self.fit(data)