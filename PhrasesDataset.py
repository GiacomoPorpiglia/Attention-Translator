import torch
import torch.nn as nn
from torch.utils.data import Dataset
from my_tokenizer import loaded_tokenizer

class PhrasesDataset(Dataset):
    
    def __init__(self, df, tokenizer):
        df_clean = df[df[['en', 'fr']].applymap(lambda x: isinstance(x, str)).all(axis=1)]
        self.df = df_clean
        self.tokenizer = tokenizer
        
        
    def __len__(self):
        return self.df.shape[0]//20 # use 1/10 of the dataset


    def __getitem__(self, idx):

        en_encoding = torch.tensor(self.tokenizer.encode(self.df.iloc[idx]['en']).ids, dtype=torch.long)
        fr_encoding  = torch.tensor(self.tokenizer.encode(self.df.iloc[idx]['fr']).ids, dtype=torch.long)

        return en_encoding, fr_encoding
        