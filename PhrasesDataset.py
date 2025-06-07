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
    
    def __init__(self, df, tokenizer, max_length):

        self.max_length = max_length

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

        en_encoded = self.tokenizer.encode(self.df.iloc[idx]['en']).ids
        fr_encoded = self.tokenizer.encode(self.df.iloc[idx]['fr']).ids

        # Truncate if longer than max_length
        if len(en_encoded) > self.max_length-2:
            en_encoded = en_encoded[:self.max_length-2] # -2 is to give space for BOS and EOS tokens
        if len(fr_encoded) > self.max_length-2:
            fr_encoded = fr_encoded[:self.max_length-2]

        en_encoding = torch.tensor(en_encoded, dtype=torch.long)
        fr_encoding = torch.tensor(fr_encoded, dtype=torch.long)

        return en_encoding, fr_encoding
        