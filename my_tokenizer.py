from tokenizers import Tokenizer, models, trainers, pre_tokenizers, normalizers
from tokenizers.processors import TemplateProcessing
import os
from datasets import load_dataset
import pandas as pd
import kagglehub
import unicodedata

tokenizer_path = "tokenizer.json"



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

def train_tokenizer(df):

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer =pre_tokenizers.Whitespace()

    special_tokens = ["<pad>", "<bos>", "<eos>", "<unk>"]
    tokenizer.add_special_tokens(special_tokens)
    tokenizer.normalizer = normalizers.Lowercase()


    trainer = trainers.BpeTrainer(
        special_tokens=special_tokens, # Re-iterate special tokens here
        vocab_size=10000, # Desired vocabulary size
        min_frequency=10, # Minimum frequency for a token to be included
    )


    # Apply the pattern to both columns
    mask = df['en'].apply(is_latin_text) & df['fr'].apply(is_latin_text)
    clean_df = df[mask].copy()
    clean_df.reset_index(drop=True, inplace=True)

    texts_en = clean_df['en'].astype(str).tolist()
    texts_fr = clean_df['fr'].astype(str).tolist()

    texts = texts_en + texts_fr
    tokenizer.train_from_iterator(texts, trainer=trainer)
    tokenizer.save(tokenizer_path)
    
if not os.path.exists(tokenizer_path):
    
    # Download latest version
    path = kagglehub.dataset_download("dhruvildave/en-fr-translation-dataset")
    df = pd.read_csv(os.path.join(path, "en-fr.csv"))
    
    print(df.head(10))
    print("Path to dataset files:", path)

    train_tokenizer(df)

loaded_tokenizer = Tokenizer.from_file(tokenizer_path)

