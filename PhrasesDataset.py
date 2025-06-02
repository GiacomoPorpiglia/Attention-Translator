import torch
import torch.nn as nn
from torch.utils.data import Dataset
from my_tokenizer import loaded_tokenizer
import unicodedata

class PhrasesDataset(Dataset):
    
    def __init__(self, df, tokenizer):
        def is_latin_text(text):
            try:
                # Only check alphabetic characters for Latin-ness
                for char in text:
                    if char.isalpha():
                        if 'LATIN' not in unicodedata.name(char):
                            return False
                return True
            except ValueError:
                # Handles characters with no name in Unicode
                return False

        # Apply the pattern to both columns
        mask = df['en'].apply(is_latin_text) & df['fr'].apply(is_latin_text)
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
        