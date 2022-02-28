import os
import config
from torch.utils.data import Dataset


class TokenizedLyricsDataset(Dataset):
    def __init__(self, songs_paths, is_train):
        super(TokenizedLyricsDataset, self).__init__()

        self.songs_paths = songs_paths
        self.is_train = is_train

        self.tokenizer = config.TOKENIZER
        self.src_max_length = config.SRC_MAX_LENGTH
        self.tgt_max_length = config.TGT_MAX_LENGTH

    def __len__(self):
            return len(self.path_list)
    
    def __getitem__(self, index):
        # list of:
        #   index 0 - title
        # all others are verses/chorus

        f = open(self.songs_paths[index],"r")
        f = f.read()
        title = self.songs_paths[index].split(os.path.sep())[-1].replace(".txt","")
        result = [title]
        f = f.split("\n\n").replace("\n","")
        result += f
        return result

        src_tokenized = self.tokenizer.encode_plus(
            self.texts[index], 
            max_length=self.src_max_length,
            pad_to_max_length=True,
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors='pt'
        )
        src_input_ids = src_tokenized['input_ids'].squeeze()
        src_attention_mask = src_tokenized['attention_mask'].squeeze()

        if self.set_type != 'test':
            tgt_tokenized = self.tokenizer.encode_plus(
                self.labels[index], 
                max_length=self.tgt_max_length,
                pad_to_max_length=True,
                truncation=True,
                return_attention_mask=True,
                return_token_type_ids=False,
                return_tensors='pt'
            )
            tgt_input_ids = tgt_tokenized['input_ids'].squeeze()
            tgt_attention_mask = tgt_tokenized['attention_mask'].squeeze()

            return {
                'src_input_ids': src_input_ids.long(),
                'src_attention_mask': src_attention_mask.long(),
                'tgt_input_ids': tgt_input_ids.long(),
                'tgt_attention_mask': tgt_attention_mask.long()
            }

        return {
            'src_input_ids': src_input_ids.long(),
            'src_attention_mask': src_attention_mask.long()
        }