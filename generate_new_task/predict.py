from pprint import pprint

import pandas as pd

from simpletransformers.t5 import T5Model

from transformers import T5Tokenizer

model_args = {
    "reprocess_input_data": True,
    "overwrite_output_dir": True,
    "max_seq_length": 128,
    "eval_batch_size": 100,
    "num_train_epochs": 1,
    "save_eval_checkpoints": False,
    "use_multiprocessing": False,
    "num_beams": None,
    "do_sample": True,
    "max_length": 50,
    "top_k": 50,
    "top_p": 0.95,
    "num_return_sequences": 1, #how many times to "predict"
}

model = T5Model("t5", "outputs/best_model", args=model_args, use_cuda=False)

df = pd.read_csv("train_df.tsv", sep="\t").astype(str)

#outputs = model.generate(song for song in df["input_text"].tolist()) ####

preds = model.predict(["MOR: " + song for song in df["input_text"].tolist()])

genre = df["target_text"].tolist()

with open("outputs/generated_genres.txt", "w") as f:
    for i, song in enumerate(df["input_text"].tolist()):
        pprint(song)
        pprint(preds[i])
        print()

        f.write(str(song) + "\n\n")

        f.write("Real genre:\n")
        f.write(genre[i] + "\n\n")

        f.write("Generated genre:\n")
        f.write(str(preds[i]) + "\n")
        f.write(
            "________________________________________________________________________________\n"
        )