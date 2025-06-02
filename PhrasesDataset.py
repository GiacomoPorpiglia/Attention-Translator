import torch
import torch.nn as nn
from torch.utils.data import Dataset
from my_tokenizer import loaded_tokenizer
import unicodedata
import regex

pattern = regex.compile(r'^[\p{Latin}\p{N}\p{P}\p{Zs}]*$', regex.UNICODE)

def is_latin(text):
    return bool(pattern.fullmatch(text))

    
class PhrasesDataset(Dataset):
    
    def __init__(self, df, tokenizer):

        # Apply the pattern to both columns
        df = df.dropna(subset=['en', 'fr'])
        df = df[(df['en'].apply(lambda x: isinstance(x, str))) & (df['fr'].apply(lambda x: isinstance(x, str)))]
    
        # Apply the pattern to both columns
        mask = df['en'].apply(is_latin) & df['fr'].apply(is_latin)
        clean_df = df[mask].reset_index(drop=True)
        
        self.df = clean_df
        self.tokenizer = tokenizer
        
        
    def __len__(self):
        return self.df.shape[0]


    def __getitem__(self, idx):

        en_encoding  = torch.tensor(self.tokenizer.encode(self.df.iloc[idx]['en']).ids, dtype=torch.long)
        fr_encoding  = torch.tensor(self.tokenizer.encode(self.df.iloc[idx]['fr']).ids, dtype=torch.long)

        return en_encoding, fr_encoding
        