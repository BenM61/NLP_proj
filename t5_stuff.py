from torch.utils.data import Dataset
import torch
import os
from transformers import T5Tokenizer, T5ForConditionalGeneration, get_linear_schedule_with_warmup
from torch import nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from torch.optim import AdamW
import copy
from torch import cuda

from torch.utils.data import DataLoader
import numpy as np

from create_datasets.Initialize_Datasets import create_datasets

def get_gpu_allocated_size():
		"""Inspect cached/reserved and allocated memory on specified gpus and return the id of the less used device"""
		gpus = range(cuda.device_count())
		# check gpus arg VS available gpus
		sys_gpus = list(range(cuda.device_count()))

		cur_allocated_mem = {}

		for i in gpus:
				cur_allocated_mem[i] = cuda.memory_allocated(i)

		print('Current allocated memory:', {f'cuda:{k}': v for k, v in cur_allocated_mem.items()})

class Config:
	def __init__(self):
		super(Config, self).__init__()

		self.SEED = 42

		# data
		self.TOKENIZER = T5Tokenizer.from_pretrained("t5-small")
		self.SRC_MAX_LENGTH = 400
		self.TGT_MAX_LENGTH = 30
		self.BATCH_SIZE = 8

		# model
		self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.FULL_FINETUNING = True
		self.LR = 3e-5
		self.SAVE_BEST_ONLY = True
		self.N_VALIDATE_DUR_TRAIN = 3
		self.EPOCHS = 3

class T5Model(nn.Module) : # *********************************************************************
	def __init__(self):
		super(T5Model, self).__init__()
		self.t5_model = T5ForConditionalGeneration.from_pretrained("t5-small")

	def forward(
		self,
		input_ids, 
		attention_mask=None, 
		decoder_input_ids=None, 
		decoder_attention_mask=None, 
		lm_labels=None
		):

		res = self.t5_model(
			input_ids,
			attention_mask=attention_mask,
			decoder_input_ids=decoder_input_ids,
			decoder_attention_mask=decoder_attention_mask,
			labels=lm_labels
		)

		return res

# textual predictions to one hot
def get_ohe(x):
	labels_li = ["POP","HIP_HOP", "ROCK", "ELECTRONIC", "BLUES", "JAZZ", "FUNK"]
	labels_li_indices = dict()
	for idx, label in enumerate(labels_li):
		labels_li_indices[label] = idx
		
	y = [labels.replace(" ","").split(',') for labels in x]
	ohe = []
	for labels in y:
		temp = [0] * 7
		for label in labels:
			idx = labels_li_indices.get(label, -1)
			if idx != -1:
				temp[idx] = 1
		ohe.append(temp)
	ohe = np.array(ohe)
	return ohe

def val(model, val_dataloader):
	
	config=Config()

	val_loss = 0
	true, pred = [], []
	
	# set model.eval() every time during evaluation
	model.eval()
	
	for step, batch in enumerate(val_dataloader):
		# unpack the batch contents and push them to the device (cuda or cpu).
		ly_ids = batch["lyrics"]["input_ids"].squeeze(1).long().to(config.DEVICE)
		ly_mask = batch["lyrics"]["attention_mask"].squeeze(1).long().to(config.DEVICE)
		la_ids = batch["labels"]['input_ids'].squeeze(1).long().to(config.DEVICE)
		la_mask = batch["labels"]["attention_mask"].squeeze(1).long().to(config.DEVICE)
	
		la_ids_after_replace = la_mask
		# replace pad tokens with -100
		la_ids_after_replace[la_ids_after_replace[:, :] == config.TOKENIZER.pad_token_id] = -100

		# using torch.no_grad() during validation/inference is faster -
		# - since it does not update gradients.
		with torch.no_grad():
			# forward pass
			outputs = model(input_ids=ly_ids, 
							attention_mask=ly_mask,
							lm_labels=la_ids_after_replace,
							decoder_attention_mask=la_mask)
			loss = outputs[0].mean()
			val_loss += loss.item()

			# get true 
			for true_id in la_ids:
				true_decoded = config.TOKENIZER.decode(true_id)
				true.append(true_decoded)

			# get pred (decoder generated textual label ids)
			pred_ids = model.module.t5_model.generate(
				input_ids=ly_ids, 
				attention_mask=ly_mask
			)
			pred_ids = pred_ids.cpu().numpy()
			for pred_id in pred_ids:
				pred_decoded = config.TOKENIZER.decode(pred_id)
				pred.append(pred_decoded)

	true_ohe = get_ohe(true)
	pred_ohe = get_ohe(pred)

	avg_val_loss = val_loss / len(val_dataloader)
	print('Val loss:', avg_val_loss)
	print('Val accuracy:', accuracy_score(true_ohe, pred_ohe))

	val_micro_f1_score = f1_score(true_ohe, pred_ohe, average='micro')
	print('Val micro f1 score:', val_micro_f1_score)
	return val_micro_f1_score

def train(model, train_dataloader, val_dataloader, optimizer, scheduler, epoch):
	config=Config()
	train_loss = 0
	for step, batch in enumerate(tqdm(train_dataloader, 
										desc='Epoch ' + str(epoch))):
		# set model.train() every time during training
		model.train()

		# unpack the batch contents and push them to the device (cuda or cpu).
		ly_ids = batch["lyrics"]["input_ids"].squeeze(1).long().to(config.DEVICE)
		ly_mask = batch["lyrics"]["attention_mask"].squeeze(1).long().to(config.DEVICE)
		la_ids = batch["labels"]['input_ids'].squeeze(1).long().to(config.DEVICE)
		la_mask = batch["labels"]["attention_mask"].squeeze(1).long().to(config.DEVICE)
	
		get_gpu_allocated_size()

		# replace pad tokens with -100
		la_ids[la_ids[:, :] == config.TOKENIZER.pad_token_id] = -100

		#forward pass
		outputs = model(input_ids=ly_ids, 
						attention_mask=ly_mask,
						lm_labels=la_ids,
						decoder_attention_mask=la_mask)
		loss = outputs[0].mean()
		train_loss += loss.item()

		# clear accumulated gradients
		optimizer.zero_grad()

		# backward pass
		loss.backward()

		# update weights
		optimizer.step()
		
		# update scheduler
		scheduler.step()
	
	avg_train_loss = train_loss / len(train_dataloader)
	print('Training loss:', avg_train_loss)

def run():
	config = Config()
	model = T5Model()
	model = torch.nn.DataParallel(model)

	model.to(config.DEVICE)

	train_ds, test_ds = create_datasets(False)

	train_dataloader = DataLoader(train_ds, config.BATCH_SIZE, True)
	val_dataloader = DataLoader(test_ds, config.BATCH_SIZE, False)

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
		train(model, train_dataloader, val_dataloader, optimizer, scheduler, epoch)
		val_micro_f1_score = val(model, val_dataloader)

		if config.SAVE_BEST_ONLY:
			if val_micro_f1_score > max_val_micro_f1_score:
				best_model = copy.deepcopy(model)
				best_val_micro_f1_score = val_micro_f1_score

				model_name = 't5_best_model'
				torch.save(best_model.module.state_dict(), model_name + '.pt')

				print(f'--- Best Model. Val loss: {max_val_micro_f1_score} -> {val_micro_f1_score}')
				max_val_micro_f1_score = val_micro_f1_score

	return best_model, best_val_micro_f1_score

best_model, best_val_micro_f1_score = run()



#########
from ...modeling_outputs import (
    BaseModelOutput,
    Seq2SeqModelOutput,
)

def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Returns:
        Example:
        ```python
        >>> from transformers import T5Tokenizer, T5Model
        >>> tokenizer = T5Tokenizer.from_pretrained("t5-small")
        >>> model = T5Model.from_pretrained("t5-small")
        >>> input_ids = tokenizer(
        ...     "Studies have been shown that owning a dog is good for you", return_tensors="pt"
        >>> ).input_ids  # Batch size 1
        >>> decoder_input_ids = tokenizer("Studies show that", return_tensors="pt").input_ids  # Batch size 1
        >>> # forward pass
        >>> outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
        >>> last_hidden_states = outputs.last_hidden_state
        ```"""
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
