from torch.utils.data import Dataset
import torch
import os
from transformers import T5Tokenizer, T5Model, T5ForConditionalGeneration, get_linear_schedule_with_warmup
from torch import nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from torch.optim import AdamW
import copy

from torch.utils.data import DataLoader
import numpy as np

from create_datasets.Initialize_Datasets import create_datasets

class Config:
    def __init__(self):
        super(Config, self).__init__()

        self.SEED = 42

        # data
        self.TOKENIZER = T5Tokenizer.from_pretrained("t5-base")
        self.SRC_MAX_LENGTH = 1500
        self.TGT_MAX_LENGTH = 30
        self.BATCH_SIZE = 16

        # model
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.FULL_FINETUNING = True
        self.LR = 3e-5
        self.OPTIMIZER = 'AdamW'
        self.CRITERION = 'BCEWithLogitsLoss'
        self.SAVE_BEST_ONLY = True
        self.N_VALIDATE_DUR_TRAIN = 3
        self.EPOCHS = 1

class T5Model(nn.Module): # *********************************************************************
    def __init__(self):
        super(T5Model, self).__init__()

        self.t5_model = T5ForConditionalGeneration.from_pretrained("t5-base")

    def forward(
        self,
        input_ids, 
        attention_mask=None, 
        decoder_input_ids=None, 
        decoder_attention_mask=None, 
        lm_labels=None
        ):

        return self.t5_model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=lm_labels
        )

# textual predictions to one hot
def get_ohe(x):
    labels_li = ["POP", "HIP_HOP", "ROCK", "ELECTRONIC", "BLUES", "JAZZ", "FUNK"]
    labels_li_indices = dict()
    for idx, label in enumerate(labels_li):
        labels_li_indices[label] = idx
        
    y = [labels.split(', ') for labels in x]
    ohe = []
    for labels in y:
        temp = [0] * 6
        for label in labels:
            idx = labels_li_indices.get(label, -1)
            if idx != -1:
                temp[idx] = 1
        ohe.append(temp)
    ohe = np.array(ohe)
    return ohe

def val(model, val_dataloader, criterion):
    
    config=Config()

    val_loss = 0
    true, pred = [], []
    
    # set model.eval() every time during evaluation
    model.eval()
    
    for step, batch in enumerate(val_dataloader):
        # unpack the batch contents and push them to the device (cuda or cpu).
        b_src_input_ids = batch['src_input_ids'].to(config.DEVICE)
        b_src_attention_mask = batch['src_attention_mask'].to(config.DEVICE)
    
        b_tgt_input_ids = batch['tgt_input_ids']
        lm_labels = b_tgt_input_ids.to(config.DEVICE)
        lm_labels[lm_labels[:, :] == config.TOKENIZER.pad_token_id] = -100

        b_tgt_attention_mask = batch['tgt_attention_mask'].to(config.DEVICE)

        # using torch.no_grad() during validation/inference is faster -
        # - since it does not update gradients.
        with torch.no_grad():
            # forward pass
            outputs = model(
                input_ids=b_src_input_ids, 
                attention_mask=b_src_attention_mask,
                lm_labels=lm_labels,
                decoder_attention_mask=b_tgt_attention_mask)
            loss = outputs[0]

            val_loss += loss.item()

            # get true 
            for true_id in b_tgt_input_ids:
                true_decoded = config.TOKENIZER.decode(true_id)
                true.append(true_decoded)

            # get pred (decoder generated textual label ids)
            pred_ids = model.t5_model.generate(
                input_ids=b_src_input_ids, 
                attention_mask=b_src_attention_mask
            )
            pred_ids = pred_ids.cpu().numpy()
            for pred_id in pred_ids:
                pred_decoded = config.TOKENIZER.decode(pred_id)
                pred.append(pred_decoded)

    print(pred)
    true_ohe = get_ohe(true)
    pred_ohe = get_ohe(pred)

    avg_val_loss = val_loss / len(val_dataloader)
    print('Val loss:', avg_val_loss)
    print('Val accuracy:', accuracy_score(true_ohe, pred_ohe))

    val_micro_f1_score = f1_score(true_ohe, pred_ohe, average='micro')
    print('Val micro f1 score:', val_micro_f1_score)
    return val_micro_f1_score

def train(model, train_dataloader, val_dataloader, criterion, optimizer, scheduler, epoch):
    config=Config()
    # we validate config.N_VALIDATE_DUR_TRAIN times during the training loop
    nv = 1
    temp = len(train_dataloader) // nv
    temp = temp - (temp % 100)
    validate_at_steps = [temp * x for x in range(1, nv + 1)]
    
    train_loss = 0
    for step, batch in enumerate(tqdm(train_dataloader, 
                                      desc='Epoch ' + str(epoch))):
        # set model.train() every time during training
        model.train()

        # unpack the batch contents and push them to the device (cuda or cpu).
        b_src_input_ids = batch['lyrics']['input_ids'].long().to(config.DEVICE)
        b_src_attention_mask = batch['lyrics']['attention_mask'].long().to(config.DEVICE)
    
        lm_labels = batch['labels']['input_ids'].long().to(config.DEVICE)
				# replace pad tokens with -100
        lm_labels[lm_labels[:, :] == config.TOKENIZER.pad_token_id] = -100

        b_tgt_attention_mask = batch['labels']['attention_mask'].long().to(config.DEVICE)

        b_src_input_ids = b_src_input_ids.squeeze(1)
        b_src_attention_mask = b_src_input_ids.squeeze(1)
        lm_labels = lm_labels.squeeze(1)
        b_tgt_attention_mask = b_tgt_attention_mask.squeeze(1)

        # forward pass
        outputs = model(input_ids=b_src_input_ids, 
                        attention_mask=b_src_attention_mask,
                        lm_labels=lm_labels,
                        decoder_attention_mask=b_tgt_attention_mask)
        loss = outputs[0]
        train_loss += loss.item()

        # clear accumulated gradients
        optimizer.zero_grad()

        # backward pass
        loss.backward()

        # update weights
        optimizer.step()
        
        # update scheduler
        scheduler.step()

        if step in validate_at_steps:
            print(f'-- Step: {step}')
            _ = val(model, val_dataloader, criterion)
    
    avg_train_loss = train_loss / len(train_dataloader)
    print('Training loss:', avg_train_loss)

def run():
	config = Config()
	model = T5Model()

	model.to(config.DEVICE)

	train_ds, test_ds = create_datasets(False)

	train_dataloader = DataLoader(train_ds, 16, True)
	val_dataloader = DataLoader(test_ds, 16, False)

	# setting a seed ensures reproducible results.
	# seed may affect the performance too.
	torch.manual_seed(config.SEED)

	criterion = nn.BCEWithLogitsLoss()

	# define the parameters to be optmized -
	# - and add regularization
	if config.FULL_FINETUNING:
			param_optimizer = list(model.named_parameters())
			no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
			optimizer_parameters = [
					{
							"params": [
									p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
							],
							"weight_decay": 0.001,
					},
					{
							"params": [
									p for n, p in param_optimizer if any(nd in n for nd in no_decay)
							],
							"weight_decay": 0.0,
					},
			]
			optimizer = AdamW(optimizer_parameters, lr=config.LR)

	num_training_steps = len(train_dataloader) * config.EPOCHS
	scheduler = get_linear_schedule_with_warmup(
		optimizer,
		num_warmup_steps=0,
		num_training_steps=num_training_steps
	)

	max_val_micro_f1_score = float('-inf')
	for epoch in range(config.EPOCHS):
		train(model, train_dataloader, val_dataloader, criterion, optimizer, scheduler, epoch)
		val_micro_f1_score = val(model, val_dataloader, criterion)

		if config.SAVE_BEST_ONLY:
			if val_micro_f1_score > max_val_micro_f1_score:
				best_model = copy.deepcopy(model)
				best_val_micro_f1_score = val_micro_f1_score

				model_name = 't5_best_model'
				torch.save(best_model.state_dict(), model_name + '.pt')

				print(f'--- Best Model. Val loss: {max_val_micro_f1_score} -> {val_micro_f1_score}')
				max_val_micro_f1_score = val_micro_f1_score

	return best_model, best_val_micro_f1_score

best_model, best_val_micro_f1_score = run()
