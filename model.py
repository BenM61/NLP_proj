import torch
import torch.nn as nn
from torch.nn import Module
import torch.optim as optim
import numpy as np
from nltk.tokenize import word_tokenize


from create_datasets.Initialize_Datasets import create_datasets

class Config():
  def __init__(self):
    self.loss = nn.BCELoss()
    self.lr = 1e-3
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Model(Module):
  def __init__(self, config, ignore_titles=True):
    super(Model, self).__init__()

    self._ignore_titles = ignore_titles
    self.embeddings = nn.Embedding.from_pretrained(torch.FloatTensor(np.load("glove.npy")))

  def forward(self, lyrics_ids, mask, title=None):  
    masked = lyrics_ids * mask
    # encode
    embs = self.embeddings(masked)
    # squeeze? .long().to(config.device)
    
    #if not self._ignore_titles:
    # do something else

# dataloader example (we need shuffle to true only on train)
from torch.utils.data import DataLoader



def train():
  config = Config()

  myModel = Model(config).to(config.device)

  loss = config.loss

  params = list(myModel.parameters())
  optimizer = optim.AdamW(params, lr=config.lr)

  tr_ds, te_ds = create_datasets(ignore_titles=False)
  
  
  tr_dl = DataLoader(tr_ds, 69, False)

  train_loss = 0

  for batch in tr_dl:
    lyrics = batch['raw_lyrics']
    labels = batch['raw_labels']
    titles = batch['raw_titles']

    outputs = myModel(lyrics, titles)
    
    # clear accumulated gradients
    optimizer.zero_grad()
    # backward pass
    loss.backward()
    # update weights
    optimizer.step()
	
    train_loss += loss.item()

  avg_train_loss = train_loss / len(tr_ds)
  print('Training loss:', avg_train_loss)

def tokenize_sentence(sent):
  word_tokenize
train()