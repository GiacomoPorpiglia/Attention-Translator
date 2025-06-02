import torch
import torch.nn as nn
from torch.utils.data import Dataset
from my_tokenizer import loaded_tokenizer

class PhrasesDataset(Dataset):
    
    def __init__(self, df, tokenizer):
        pattern = r'^[\x20-\x7E]+$'

        # Apply the pattern to both columns
        mask = df['en'].str.match(pattern) & df['fr'].str.match(pattern)
        clean_df = df[mask].copy()
        clean_df.reset_index(drop=True, inplace=True) ### Reset index after filtering
        self.df = clean_df
        self.tokenizer = tokenizer
        
        
    def __len__(self):
        return self.df.shape[0]


    def __getitem__(self, idx):

        en_encoding = torch.tensor(self.tokenizer.encode(self.df.iloc[idx]['en']).ids, dtype=torch.long)
        fr_encoding  = torch.tensor(self.tokenizer.encode(self.df.iloc[idx]['fr']).ids, dtype=torch.long)

        return en_encoding, fr_encoding
        