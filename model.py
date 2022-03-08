import torch
from torch.nn import Module

from create_datasets.Initialize_Datasets import create_datasets
class Model(Module):
  def __init__(self, ignore_titles=True):
    super(Model, self).__init__()

    self._ignore_titles = ignore_titles

    ###########
    # t5 stuff
    ###########
  
  #def forward(self, input):

    #if self._ignore_titles:



# dataloader example (we need shuffle to true only on train)
from torch.utils.data import DataLoader
import numpy as np
myModel = Model()
t, tt = create_datasets()
td = DataLoader(t, 69, False)
for X, Y in td:

  print("#\n\n\n\n\n\n\n#")
  X = np.array(X)
  print(np.shape(Y))
  print(np.shape(X))

