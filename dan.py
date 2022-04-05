from create_datasets.Initialize_Datasets import create_datasets
from torch.utils.data import DataLoader
#from datasets import load_dataset

import pyarrow as pa
import pyarrow.dataset as ds
import pandas as pd
from datasets import Dataset

from nltk.tokenize import word_tokenize
import json

from datasets import load_metric
import numpy as np
import torch
import torch.nn as nn

from dataclasses import dataclass

from transformers import Trainer
from transformers import TrainingArguments

################## create ds ######################################3
tr_ds, te_ds = create_datasets(ignore_titles=False)
tr_dl = DataLoader(tr_ds, 69, False)

#raw_datasets = load_dataset("imdb")

################################ tokenize #######################################33
with open("vocab.json") as f:
  vocab = json.load(f)

def tokenize_function(example):
  sentences = [x.lower() for x in example['text']]
  tokenized_sentences = [word_tokenize(x) for x in sentences]
  tokenized_idx = [[vocab[word] if word in vocab else vocab["unk"] for word in x] for x in tokenized_sentences]
  max_size = max([len(x) for x in tokenized_idx])
  final_tokenized_idx = tokenized_idx
  d = {"POP":0, "ROCK":1, "ELECTRONIC":2, "JAZZ":3, "FUNK":4, "HIP_HOP":5, "BLUES":6}
 
  for i in range(len(example['label'])): 
    example['label'][i] = d[example['label'][i][0]]
  return {"labels":example['label'],'input_ids':final_tokenized_idx}

######################## train ###########################3
b = {"text":[tr_ds[i]["raw_lyrics"] for i in range(len(tr_ds))] , 
      "label":[tr_ds[i]["raw_labels"] for i in range(len(tr_ds))]}
tokens = tokenize_function(b)

df = pd.DataFrame({'text': [tr_ds[i]["raw_lyrics"] for i in range(len(tr_ds))], 
                              "label":b["label"],
                              "labels":tokens["labels"],
                              "input_ids":tokens["input_ids"]
                            }
                  )
dataset = ds.dataset(pa.Table.from_pandas(df).to_batches())
### convert to Huggingface dataset
hg_dataset1 = Dataset(pa.Table.from_pandas(df))

############################# eval #####################################
b = {"text":[te_ds[i]["raw_lyrics"] for i in range(len(te_ds))] , 
      "label":[te_ds[i]["raw_labels"] for i in range(len(te_ds))]}
tokens = tokenize_function(b)

df = pd.DataFrame({'text': [te_ds[i]["raw_lyrics"] for i in range(len(te_ds))], 
                              "label":tokens["labels"],
                              "labels":tokens["labels"],
                              "input_ids":tokens["input_ids"]
                            }
                  )
dataset = ds.dataset(pa.Table.from_pandas(df).to_batches())
### convert to Huggingface dataset
hg_dataset2 = Dataset(pa.Table.from_pandas(df))

rd = {"train":{"text":[tr_ds[i]["raw_lyrics"] for i in range(len(tr_ds))],
                "label":[tr_ds[i]["raw_labels"] for i in range(len(tr_ds))],
                "labels":tokens["labels"],
                "input_ids":tokens["input_ids"]
                }
      }

################################################################################################

#small_train_dataset = raw_datasets['train'].shuffle(seed=42).map(tokenize_function,batched=True)
#small_eval_dataset = raw_datasets['test'].shuffle(seed=42).map(tokenize_function,batched=True)
small_train_dataset = hg_dataset2
small_eval_dataset = hg_dataset1

def pad_sequence_to_length(
    sequence,
    desired_length: int,
    default_value = lambda: 0,
    padding_on_right: bool = True,
):
    sequence = list(sequence)
    # Truncates the sequence to the desired length.
    if padding_on_right:
        padded_sequence = sequence[:desired_length]
    else:
        padded_sequence = sequence[-desired_length:]
    # Continues to pad with default_value() until we reach the desired length.
    pad_length = desired_length - len(padded_sequence)
    # This just creates the default value once, so if it's a list, and if it gets mutated
    # later, it could cause subtle bugs. But the risk there is low, and this is much faster.
    values_to_pad = [default_value()] * pad_length
    if padding_on_right:
        padded_sequence = padded_sequence + values_to_pad
    else:
        padded_sequence = values_to_pad + padded_sequence
    return padded_sequence

metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    logits = eval_pred.predictions
    labels = eval_pred.label_ids
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

@dataclass
class DataCollatorWithPadding:
  
  def __call__(self, features):
    features_dict={}
    if "labels" in features[0]:
        
      features_dict["labels"] = torch.tensor([x.pop("labels") for x in features]).long()

    input_ids = [x.pop("input_ids") for x in features]
    max_len = max(len(x) for x in input_ids)
    masks = [[1]*len(x) for x in input_ids]
    
    features_dict["input_ids"] = torch.tensor([pad_sequence_to_length(x,max_len) for x in input_ids]).long()
    features_dict["attention_masks"] = torch.tensor([pad_sequence_to_length(x,max_len) for x in masks]).long()

    return features_dict

class DAN(nn.Module):
  def __init__(self):
          super().__init__()
          self.num_labels = 7
          self.embeddings = nn.Embedding.from_pretrained(torch.FloatTensor(np.load("glove.npy")))
          self.hidden_size = torch.FloatTensor(np.load("glove.npy")).size()[1]
          self.classifier = nn.Sequential(
              nn.Linear(self.hidden_size, self.hidden_size),
              nn.ReLU(), 
              nn.Linear(self.hidden_size, self.hidden_size),
              nn.ReLU(),
              nn.Linear(self.hidden_size, self.num_labels)
          )

          self.loss = nn.CrossEntropyLoss()

  def forward(self,input_ids,attention_masks,labels=None,**kwargs):
      masked = input_ids * attention_masks
      embs = self.embeddings(masked)

      # word dropout - same vector for each sample
      input_size = input_ids.size()
      p = 0.3
      apply_dropout = torch.nn.Dropout(p)

      m = torch.ones(input_size[0], input_size[1])
      m = apply_dropout(m).bool().int()

      nonzeros_arr = [torch.count_nonzero(m[i]) for i in range(input_size[0])]

      for i in range(input_size[0]):
          for j in range(input_size[1]):
              torch.mul(embs[i][j], m[i][j])
          
      avg = torch.sum(embs, 1)

      for i in range(input_size[0]):
        avg[i] = torch.mul(avg[i], 1 / nonzeros_arr[i])

      res = self.classifier(avg)
      loss = self.loss(res,labels)
      return {"loss":loss,"logits":res}

#Hint: You may want to look at https://huggingface.co/transformers/main_classes/callback.html

co = DataCollatorWithPadding()
training_args = TrainingArguments("DAN",
                                  num_train_epochs= 20, #must be at least 10.
                                  per_device_train_batch_size=32,
                                  per_device_eval_batch_size=4,
                                  learning_rate= 0.01,
                                  save_total_limit=2,
                                  log_level="error",
                                  evaluation_strategy="epoch")
model = DAN()  
trainer = Trainer(
    model=model,
    data_collator=co,
    args=training_args,
    callbacks = [],
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()