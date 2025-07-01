from tokenizers import Tokenizer, models, trainers, pre_tokenizers, normalizers
import os
import pandas as pd
import kagglehub
import regex

pattern = regex.compile(r'^[\p{Latin}\p{N}\p{P}\p{Zs}]*$', regex.UNICODE)

def is_latin(text):
    return bool(pattern.fullmatch(text))

tokenizer_path = "tokenizer.json"

def train_tokenizer(df):

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer =pre_tokenizers.ByteLevel()

    special_tokens = ["<pad>", "<bos>", "<eos>", "<unk>"]
    tokenizer.add_special_tokens(special_tokens)
    tokenizer.normalizer = normalizers.Lowercase()


    trainer = trainers.BpeTrainer(
        special_tokens=special_tokens, # Re-iterate special tokens here
        vocab_size=20000, # Desired vocabulary size
        min_frequency=10, # Minimum frequency for a token to be included
    )

    df = df.dropna(subset=['en', 'fr'])
    df = df[(df['en'].apply(lambda x: isinstance(x, str))) & (df['fr'].apply(lambda x: isinstance(x, str)))]

    # Apply the pattern to both columns
    mask = df['en'].apply(is_latin) & df['fr'].apply(is_latin)
    clean_df = df[mask].reset_index(drop=True)

    texts_en = clean_df['en'].astype(str).tolist()
    texts_fr = clean_df['fr'].astype(str).tolist()

    texts = texts_en + texts_fr
    tokenizer.train_from_iterator(texts, trainer=trainer)
    tokenizer.save(tokenizer_path)
    
if not os.path.exists(tokenizer_path):
    
    # Download the dataset
    path = kagglehub.dataset_download("dhruvildave/en-fr-translation-dataset")
    df = pd.read_csv(os.path.join(path, "en-fr.csv"))
    
    print("Path to dataset files:", path)

    train_tokenizer(df)

loaded_tokenizer = Tokenizer.from_file(tokenizer_path)

