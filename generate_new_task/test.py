import pandas as pd
import gzip
from sklearn.model_selection import train_test_split
import os
from tqdm.auto import tqdm
from os import listdir
from simpletransformers.t5 import T5Model
from pprint import pprint
#copy files
#pip freeze > filename.txt
#connect to the server from vc
# /home/joberant/NLP_2122/buchnik

genre = []
#for label in listdir("multi-tut/train"):
for title in listdir("train/BLUES"):
    f = open("train/BLUES/"+title, "r")
    genre += [["MOR",f.read(),"1"]] 
    f.close()

#for label in listdir("multi-tut/train"):
for title in listdir("train/ELECTRONIC"):
    f = open("train/ELECTRONIC/"+title, "r")
    genre += [["MOR",f.read(),"1"]]  
    f.close() 

i = 0
#for title in listdir("train/FUNK"):
 #   i+=1
  #  if i >100:
   #     break
    #f = open("train/FUNK/"+title, "r")
    #genre += [["MOR",f.read(),"1"]]  
    #f.close()             

df = pd.DataFrame(genre)
df.columns = ["prefix", "input_text", "target_text"]

df.to_csv(f"data_all.tsv", "\t")

train_df, eval_df = train_test_split(df, test_size=0.05)
train_df, eval_df = train_test_split(df, test_size=0.5)

train_df.to_csv("train_df.tsv", "\t")
eval_df.to_csv("eval_df.tsv", "\t")

model_args = {
    "reprocess_input_data": True,
    "overwrite_output_dir": True,
    "max_seq_length": 128,
    "train_batch_size": 100,
    "num_train_epochs": 1,
    "save_eval_checkpoints": True,
    "save_steps": -1,
    "use_multiprocessing": False,
    "evaluate_generated_text": True,
    "evaluate_during_training": True,
    "evaluate_during_training_steps": 15000,
    "evaluate_during_training_verbose": True,
    "fp16": False,

    "wandb_project": "MOR with T5",
}

def count_matches(labels, preds):
    print(labels)
    print(preds)
    return sum([1 if label == pred else 0 for label, pred in zip(labels, preds)])

model = T5Model("t5", "t5-small", args=model_args, use_cuda=False)
model.train_model(train_df, eval_data=eval_df, matches=count_matches)
#model = T5Model("t5", "outputs/best_model", args=model_args, use_cuda=False)
print(model.eval_model(eval_df, matches=count_matches))

###############################3