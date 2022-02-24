from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

input_ids = tokenizer("translate English to German: The house is wonderful.", return_tensors="pt").input_ids
labels = tokenizer("Das Haus ist wunderbar.", return_tensors="pt").input_ids
# the forward function automatically creates the correct decoder_input_ids
print(model(input_ids=input_ids, labels=labels).loss)
labels = tokenizer("mew mew", return_tensors="pt").input_ids
# the forward function automatically creates the correct decoder_input_ids
print(model(input_ids=input_ids, labels=labels).loss)
