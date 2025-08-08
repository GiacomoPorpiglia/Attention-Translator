import torch
from torch.utils.data import Dataset
import random


    
class PhrasesDataset(Dataset):
    
    def __init__(self, df, tokenizer, max_seq_len):

        self.max_seq_len = max_seq_len

        self.data = []
        self.tokenizer = tokenizer

        # Apply the pattern to both columns
        df = df.dropna(subset=['en', 'fr'])
        df = df[(df['en'].apply(lambda x: isinstance(x, str))) & (df['fr'].apply(lambda x: isinstance(x, str)))]
    
        # Apply the pattern to both columns
        df = df.reset_index(drop=True)
        
        for _, row in df.iterrows():
            en_tokens = tokenizer.encode(row['en']).ids
            fr_tokens = tokenizer.encode(row['fr']).ids
            
            ### -2 is because here there aren't the BOS and EOS tokens, for which we want to keep space.
            if len(en_tokens) <= max_seq_len-2 and len(fr_tokens) <= max_seq_len-2:
                self.data.append((en_tokens, fr_tokens))

        
    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        
        en_encoded, fr_encoded = self.data[idx]

        en_tensor = torch.tensor(en_encoded, dtype=torch.long)
        fr_tensor = torch.tensor(fr_encoded, dtype=torch.long)

        return en_tensor, fr_tensor
        