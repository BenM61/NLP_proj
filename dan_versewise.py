from create_datasets.Initialize_Datasets import create_datasets
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

def untokenize_song(token_ids_tensor):  
	token_ids_list = token_ids_tensor.tolist()
	word_list = [inv_vocab[str(x)] for x in token_ids_list]
	untokenized = ' '.join(word_list).replace("#", "\n").replace("PADDING_TOKEN","")
	print(untokenized)
	return untokenized

def tokenize_function(example):
	sentences = [x.lower().replace("\n", "#") for x in example['text']]
	tokenized_sentences = [word_tokenize(x) for x in sentences]
	tokenized_idx = [[vocab[word] if word in vocab else vocab["unk"] for word in x] for x in tokenized_sentences]
	max_size = max([len(x) for x in tokenized_idx])
	final_tokenized_idx = tokenized_idx

	d = {"POP":0, "ROCK":1, "ELECTRONIC":2, "JAZZ":3, "FUNK":4, "HIP_HOP":5, "BLUES":6}
 
	for i in range(len(example['label'])): 
		example['label'][i] = d[example['label'][i]]
	return {"labels":example['label'],'input_ids':final_tokenized_idx}

def make_ds(ds):
	b = {"text":[ds[i]["lyrics"] for i in range(len(ds))] , 
			"label":[ds[i]["label"] for i in range(len(ds))]}
	tokens = tokenize_function(b)

	df = pd.DataFrame({'text': b["text"], 
																"label":tokens["labels"],
																"labels":tokens["labels"],
																"input_ids":tokens["input_ids"]
															}
										)
	### convert to Huggingface dataset
	hg_dataset = Dataset(pa.Table.from_pandas(df))
	return hg_dataset

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

def compute_metrics(eval_pred):
		metric = load_metric("accuracy")
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

		# converting the token ids tensors to list of strings, so we can
		# divide to phrases and lines
		input_ids = [x.pop("input_ids") for x in features]
		hashtag = vocab["#"]
		versed_input_ids = []
		
		# break batch to list of lists of verse tokenizations
		for tokenized_song in input_ids:
			tokenized_song = [str(x) for x in tokenized_song if x != 0]
			verses = " ".join(tokenized_song).split(f"{hashtag} {hashtag}")
			verses_as_lists = [v.replace(str(hashtag), "").split(" ") for v in verses]
			# remove unwanted spaces and cast back to int
			verses_as_lists = [[int(x) for x in v if x != ''] for v in verses_as_lists]
			versed_input_ids.append(verses_as_lists)

		input_ids = versed_input_ids
		attention_masks = [[[1]*len(v) for v in s] for s in input_ids]

		# verse level pad      
		input_ids = [[torch.tensor(																										\
									pad_sequence_to_length(v, VERSE_PAD_LEN, VERSE_PAD_FUNC))				\
														.long() for v in s] for s in input_ids]
		attention_masks = [[torch.tensor(																										\
									pad_sequence_to_length(v, VERSE_PAD_LEN, VERSE_PAD_FUNC))				\
														.long() for v in s] for s in attention_masks]    
		
		# song level pad      
		input_ids = [torch.stack(                                                    \
								tuple(pad_sequence_to_length(s, SONG_PAD_LEN, SONG_PAD_FUNC))    \
													, 0) for s in input_ids]
		attention_masks = [torch.stack(                                              \
								tuple(pad_sequence_to_length(s, SONG_PAD_LEN, SONG_PAD_FUNC))    \
													, 0) for s in attention_masks]

		# one tensor for the batch
		input_ids = torch.stack(input_ids, 0)
		attention_masks = torch.stack(attention_masks, 0)

		features_dict["input_ids"] = input_ids
		features_dict["attention_masks"] = attention_masks
		return features_dict

class DAN(nn.Module):
	def __init__(self):
					super().__init__()
					self.num_labels = 7
					self.embeddings = nn.Embedding.from_pretrained(torch.FloatTensor(np.load("glove.npy")))
					self.input_size = torch.FloatTensor(np.load("glove.npy")).size()[1]
					self.hidden_size = self.input_size

					self.gru = torch.nn.GRU(self.input_size, self.hidden_size, batch_first=True)
					self.h0 = torch.randn(self.hidden_size)
					self.classifier = nn.Sequential(
							nn.Linear(self.hidden_size, self.hidden_size),
							nn.ReLU(), 
							nn.Linear(self.hidden_size, self.hidden_size),
							nn.ReLU(),
							nn.Linear(self.hidden_size, self.num_labels)
					)

					self.loss = nn.CrossEntropyLoss()

	def forward(self,input_ids,attention_masks,labels=None,**kwargs):
			# word dropout - same vector for each sample
			input_size = input_ids.size()
			p = 0.3
			apply_dropout = torch.nn.Dropout(p)

			dropout_mask = torch.ones(*input_size)
			dropout_mask = apply_dropout(dropout_mask).bool().int()

			attention_masks = attention_masks * dropout_mask
			masked = input_ids * attention_masks
			embs = self.embeddings(masked)

			nonzeros_arr = torch.count_nonzero(attention_masks, -1)
			# avoid division by zero
			nonzeros_arr[nonzeros_arr==0] = 1
			avg = torch.sum(embs, -2)

			avg = torch.mul(avg, 1 / nonzeros_arr.unsqueeze(-1))

		
			h0s = self.h0.repeat(1, input_size[0], 1)
			_, output = self.gru(avg, h0s)
			output = output.squeeze(0)

			res = self.classifier(output)
			loss = self.loss(res,labels)
			return {"loss":loss,"logits":res}



tr_ds, te_ds = create_datasets(ignore_titles=True)
with open("vocab.json") as f:
	vocab = json.load(f)
with open("vocab_inverted.json") as f:
	inv_vocab = json.load(f)

small_train_dataset = make_ds(tr_ds)
small_eval_dataset = make_ds(te_ds)

# constants
VERSE_PAD_LEN = 36
SONG_PAD_LEN = 6

VERSE_PAD_FUNC = lambda: 0
SONG_PAD_FUNC = lambda: torch.zeros(VERSE_PAD_LEN).long()


co = DataCollatorWithPadding()
training_args = TrainingArguments("DAN",
																	num_train_epochs= 2, #must be at least 10.
																	per_device_train_batch_size=32,
																	per_device_eval_batch_size=4,
																	learning_rate= 0.001,
																	save_total_limit=1,
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

preds = trainer.predict(small_eval_dataset)
print(compute_metrics(preds))