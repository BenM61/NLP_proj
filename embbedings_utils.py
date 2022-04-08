import gdown
import nltk
import json 

def download():
  # downloads
  gdown.download('https://drive.google.com/uc?export=download&id=1YvXC05yykEwSF8TPgAkJo-OjAJz9GGdk', 'glove.npy', quiet=False)
  gdown.download('https://drive.google.com/uc?export=download&id=1-3SxpirQjmX-RCRyRjKdP2L7G_tNgp00', 'vocab.json', quiet=False)
  nltk.download('punkt')

def inverse_vocab_dict():
  with open("vocab.json") as f:
    read_data = f.read()
    j = json.loads(read_data)
    inverted_dict = {v: k for k,v in j.items()}
    inverted_file = open("vocab_inverted.json", "w")

    json.dump(inverted_dict, inverted_file)

inverse_vocab_dict()