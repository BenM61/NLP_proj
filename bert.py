import torch
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from torch.utils.data import TensorDataset
from transformers import BertForSequenceClassification
from transformers import AutoTokenizer,TFBertModel
import pandas as pd
from create_datasets.Initialize_Datasets import create_datasets, delete_dataset_files
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score
import numpy as np
import random
import torch.nn as nn
import matplotlib.pyplot as plt


def f1_score_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_flat, average='weighted')

def accuracy_per_class(preds, labels):
    #label_dict_inverse = {v: k for k, v in label_dict.items()}
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    total_acc = 0
    sum = 0
    for label in np.unique(labels_flat):
        y_preds = preds_flat[labels_flat==label]
        y_true = labels_flat[labels_flat==label]
        print(f'Class: {label}')
        print(f'Accuracy: {len(y_preds[y_preds==label])}/{len(y_true)}\n')
        total_acc += len(y_preds[y_preds==label])
        sum += len(y_true)
    print(f'Total Accuracy: {total_acc}/{sum}\n')

def evaluate(dataloader_val):

    model.eval()
    loss_val_total = 0
    predictions, true_vals = [], []
    
    for batch in dataloader_val:   
        batch = tuple(b.to(device) for b in batch)  
        inputs = {'input_ids':      batch[0].to(device),
                  'attention_mask': batch[1].to(device),
                  'labels':         batch[2].to(device),
                 }

        with torch.no_grad():
            outputs = model(**inputs)
            
        loss = outputs[0]
     
        logits = outputs[1]
        loss_val_total += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)
    
    loss_val_avg = loss_val_total/len(dataloader_val)   
    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)         
    return loss_val_avg, predictions, true_vals

def create_df(ds):
    n = len(ds)
    a = [[ds[i]["lyrics"],ds[i]["label"]] for i in range(n)]
    df = pd.DataFrame(a, columns=["lyrics","genre"])
    #dict = {"POP":0, "HIP_HOP":0, "ROCK":0, "JAZZ":0, "FUNK":0, "ELECTRONIC":0, "BLUES":0}
    #print(df.genre)
    #exit()
    #possible_labels = df.genre.unique() #genres to numbers!
    #label_dict = {}
    #for index, possible_label in enumerate(possible_labels):
    #    label_dict[possible_label] = index
    df['label'] = df["genre"]
    return df


#getting df
#delete_dataset_files()
tr_ds, te_ds = create_datasets(ignore_titles=True)
df_eval = create_df(te_ds)
df_train = create_df(tr_ds)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#BERT part
bert_ver = "bert-base-uncased"
#prajjwal1/bert-tiny
#bert-base-uncased

tokenizer = BertTokenizer.from_pretrained(bert_ver, do_lower_case=True)

encoded_data_train = tokenizer.batch_encode_plus(
    df_train.lyrics.values, 
    add_special_tokens=True, 
    return_attention_mask=True, 
    pad_to_max_length=True, 
    return_tensors='pt',
    max_length=256,
    truncation=True
).to(device)

#to converts lyrics to encoded form
encoded_data_val = tokenizer.batch_encode_plus(
    df_eval.lyrics.values, 
    add_special_tokens=True, 
    return_attention_mask=True, 
    pad_to_max_length=True,  
    return_tensors='pt',
    max_length=256,
    truncation=True
).to(device)

input_ids_train = encoded_data_train['input_ids']
attention_masks_train = encoded_data_train['attention_mask']
labels_train = torch.tensor(df_train.label.values).to(device)

input_ids_val = encoded_data_val['input_ids']
attention_masks_val = encoded_data_val['attention_mask']
labels_val = torch.tensor(df_eval.label.values).to(device)

dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)
dataset_val = TensorDataset(input_ids_val, attention_masks_val, labels_val)

#model
model = BertForSequenceClassification.from_pretrained(bert_ver,
                                                      num_labels=7,
                                                      output_attentions=False,
                                                      output_hidden_states=False).to(device)


#dataloader
dataloader_train = DataLoader(dataset_train, 
                              sampler=RandomSampler(dataset_train), 
                              batch_size=8)
dataloader_validation = DataLoader(dataset_val, 
                                   sampler=SequentialSampler(dataset_val), 
                                   batch_size=8)

#optimizer
optimizer = AdamW(model.parameters(),lr=1e-5, eps=1e-8)                  
epochs = 5
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps=0,
                                            num_training_steps=len(dataloader_train)*epochs)

#training
seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

from sklearn.utils.class_weight import compute_class_weight

all_losses = []
val_loss_for_epoch=0
all_vals = []
loss_avg = 0
f1_avg = 0
all_f1=[]
plot_every = 1

for epoch in tqdm(range(1, epochs+1)):
    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch, epochs))
    print('Training...')
    model.train()
    loss_train_total = 0
    progress_bar = tqdm(dataloader_train, desc='Epoch {:1d}'.format(epoch), leave=False, disable=False)
    for batch in progress_bar:
        model.zero_grad()
        batch = tuple(b.to(device) for b in batch)
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[2],
                 }

        outputs = model(**inputs) 
        loss = outputs[0]
        loss_train_total += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item()/len(batch))})
    
    torch.save(model.state_dict(), f'BERT/finetuned_BERT_epoch_{epoch}.model')   
    tqdm.write(f'\nEpoch {epoch}')
    
    loss_train_avg = loss_train_total/len(dataloader_train)            
    tqdm.write(f'Training loss: {loss_train_avg}')

    val_loss, predictions, true_vals = evaluate(dataloader_validation)
    val_f1 = f1_score_func(predictions, true_vals)
    tqdm.write(f'Validation loss: {val_loss}')
    tqdm.write(f'F1 Score (Weighted): {val_f1}')

    loss_avg += loss_train_avg
    val_loss_for_epoch += val_loss
    f1_avg+=val_f1
    if epoch % plot_every == 0:
        all_losses.append(loss_avg / plot_every)
        loss_avg = 0 
        all_vals.append(val_loss_for_epoch / plot_every)
        val_loss_for_epoch = 0 
        all_f1.append(f1_avg / plot_every)
        f1_avg = 0 

model = BertForSequenceClassification.from_pretrained(bert_ver,
                                                      num_labels=7,
                                                      output_attentions=False,
                                                      output_hidden_states=False)

model.to(device)
model.load_state_dict(torch.load('BERT/finetuned_BERT_epoch_3.model', map_location=torch.device(device)))
_, predictions, true_vals = evaluate(dataloader_validation)
accuracy_per_class(predictions, true_vals)

plt.grid(True)
plt.xlabel('# of epochs (divided by plot_every)')
plt.ylabel('average loss')
plt.plot(all_losses, color="r", label="train_loss")
plt.plot(all_vals, color="b", label="val_loss")
#plt.plot(all_f1, color="g", label="f1")
plt.savefig('test.png')
