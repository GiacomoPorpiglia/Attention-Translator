import torch
import torch.nn as nn
from PhrasesDataset import PhrasesDataset
from my_tokenizer import loaded_tokenizer
from torch.utils.data import DataLoader, random_split
from model import Encoder, Decoder
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
from tqdm import tqdm
from BatchSampler import BucketBatchSampler
import pandas as pd
import math
import config
import kagglehub
import argparse


def collate_fn(batch, pad_token_id, bos_token_id, eos_token_id, max_length=config.max_seq_len):
    encoder_inputs, decoder_inputs = zip(*batch)

    encoder_inputs = [x[:max_length-2] if len(x) > max_length-2 else x for x in encoder_inputs]
    decoder_inputs = [x[:max_length-2] if len(x) > max_length-2 else x for x in decoder_inputs]

    # [BOS,..., ..., EOS, (PAD, PAD)]
    encoder_inputs  = [torch.cat([torch.tensor([bos_token_id]), x, torch.tensor([eos_token_id])]) for x in encoder_inputs]
    encoder_inputs  = pad_sequence(encoder_inputs, batch_first=True, padding_value=pad_token_id)

    # [BOS,..., ..., EOS, (PAD, PAD)]
    decoder_inputs  = [torch.cat([torch.tensor([bos_token_id]), y, torch.tensor([eos_token_id])]) for y in decoder_inputs]
    decoder_inputs  = pad_sequence(decoder_inputs, batch_first=True, padding_value=pad_token_id)

    # encoder_len = encoder_inputs.size(1)
    # decoder_len = decoder_inputs.size(1)

    # decoder_targets = torch.stack([torch.cat([x[1:], torch.tensor([pad_token_id])]) for x in decoder_inputs], dim=0)
    decoder_targets = decoder_inputs[:, 1:].clone()
    decoder_targets = torch.cat((decoder_targets, torch.full(size=(decoder_targets.size(0), 1),fill_value=pad_token_id)), dim=1)
    decoder_targets[decoder_targets == pad_token_id] = -100 # set to -100 where the value to predict is pad. These tokens will be ignored in loss calculation

    

    # Build attention masks (1 where token ≠ pad, 0 where token == pad)
    ### true for tokens that are not pad --> true for tokens we want to attend
    encoder_attention_mask  = (encoder_inputs != pad_token_id).bool() 
    decoder_attention_mask  = (decoder_inputs != pad_token_id).bool()


    return {
      'encoder_input_ids':           encoder_inputs,
      'encoder_attention_mask':      encoder_attention_mask,
      'decoder_input_ids':   decoder_inputs,    # your decoder uses teacher forcing
      'decoder_attention_mask': decoder_attention_mask,
      'labels':              decoder_targets,
      'encoder_lengths': (encoder_inputs != pad_token_id).sum(dim=1),
      'decoder_lengths': (decoder_inputs != pad_token_id).sum(dim=1),
    }





# input = next(iter(dataloader_train))
# print(input['encoder_input_ids'].shape, input['encoder_attention_mask'].shape, input['decoder_input_ids'].shape, input['decoder_attention_mask'].shape, input['labels'].shape)

# encoding=encoder(input['encoder_input_ids'], input['encoder_attention_mask'])
# output = decoder(input['decoder_input_ids'], input['decoder_attention_mask'], encoding)

# targets = input['labels']
# logits = output



def save_checkpoint(checkpoint, filename):
    print("=> saving checkpoint...")
    try:
        torch.save(checkpoint, filename)
        print("\t=> checkpoint saved!")
    except:
        print("\tX => Something went wrong in saving the network")



def load_checkpoint(encoder, decoder, optimizer, checkpoint):
    print("=> loading checkpoint...")
    try:
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("\t=> checkpoint loaded!")
    except Exception as e:
        print("\tX => Something went wrong in loading the checkpoint")
        raise RuntimeError




def sample_from_top_k(logits, k=3):
    # Beam search, sample from top-k probabilities.
    
    topk_logits, topk_indices = torch.topk(logits, k)
    probs = torch.nn.functional.softmax(topk_logits, dim=-1)
    sampled_index = torch.multinomial(probs, 1).squeeze(-1)
    return topk_indices[sampled_index]



def test(input, encoder, decoder, tokenizer, device="cpu", pad_token_id=0, bos_token_id=1, eos_token_id=2):
    encoder.to(device)
    decoder.to(device)
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        # Tokenize the reaction

        en_encoding = torch.tensor(tokenizer.encode(input).ids, dtype=torch.long)
        fr_encoding = torch.tensor([0], dtype=torch.long) ### not important

        batch = [(en_encoding, fr_encoding)]
        batch = collate_fn(batch, pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id)
        
        encoder_input_ids = batch['encoder_input_ids'].to(device)
        encoder_attention_mask = batch['encoder_attention_mask'].to(device)

        # Encode the input
        encoding = encoder(encoder_input_ids, encoder_attention_mask)

        
        print("")
        print(f"Input english text: {input}")
        ### output 3 versions of the translation
        for j in range(3):
            output_tokens = [bos_token_id]
            
            for i in range(config.max_seq_len - 1):
                # Create decoder input with current tokens
                decoder_input = torch.tensor([output_tokens], dtype=torch.long, device=device)
                decoder_mask  = torch.tensor([[True] * len(output_tokens)], dtype=torch.bool, device=device)

                output_logits = decoder(decoder_input, decoder_mask, encoder_attention_mask, encoding)
                
                ### sample next token between the most probable ones
                last_logits = output_logits[0, len(output_tokens)-1, :]
                next_token = sample_from_top_k(last_logits, k=3)

                ### Always get token with highest probability 
                ### Get the next token's logits (at the current position)
                # next_token_logits = output_logits[0, len(output_tokens)-1, :]
                # next_token = next_token_logits.argmax().item()
                
                output_tokens.append(next_token)
                
                if next_token == eos_token_id:
                    break

            fr_decoded = tokenizer.decode(output_tokens, skip_special_tokens=True)

            print(f"\n(Version {j+1}) -->  Output french text: {fr_decoded}")




def get_lr(num_iters):

    if(num_iters < config.warmup_iters):
        return config.start_lr * num_iters / config.warmup_iters

    if(num_iters < config.lr_decay_iters):
        coeff = math.cos(num_iters / config.lr_decay_iters * math.pi/2)
        return coeff * config.start_lr + (1-coeff) * config.min_lr
    else:
        return config.min_lr


def train(encoder, decoder, optimizer, dataloader_train, dataloader_val, criterion, test_text, device="cpu", num_epochs=100):
    encoder.to(device)
    decoder.to(device)

    num_iters = 0

    ### set initial learning rate according to scheduler
    for param_group in optimizer.param_groups:
        lr = get_lr(1)
        param_group['lr'] = lr

    for epoch in range(num_epochs):

        encoder.train()
        decoder.train()
        optimizer.zero_grad()
        total_loss = 0
        loss_for_examination = 0

        for batch_idx, batch in tqdm(enumerate(dataloader_train), total=len(dataloader_train), desc=f"Epoch {epoch+1}/{num_epochs}"):

            ### print current translation of the test phrase
            if batch_idx%1000 == 1:
                test(test_text, encoder, decoder, loaded_tokenizer, device)
                encoder.train()
                decoder.train()
                print(f"Temp loss: {(loss_for_examination/1000):.4f}")
                loss_for_examination = 0

            encoder_input_ids = batch['encoder_input_ids'].to(device)
            encoder_attention_mask = batch['encoder_attention_mask'].to(device)
            decoder_input_ids = batch['decoder_input_ids'].to(device)
            decoder_attention_mask = batch['decoder_attention_mask'].to(device)
            labels = batch['labels'].to(device)

            encoding = encoder(encoder_input_ids, encoder_attention_mask)
            output_logits =  decoder(decoder_input_ids, decoder_attention_mask, encoder_attention_mask, encoding)
               

            loss = criterion(output_logits.view(-1, output_logits.size(-1)), labels.view(-1))
            loss = loss / config.grad_acc_steps
            loss.backward()

            ### grad accumulation
            if (batch_idx+1)%config.grad_acc_steps==0:

                ### gradient clipping
                torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(decoder.parameters()), 1.0)

                optimizer.step()      

                optimizer.zero_grad()

                # Update learning rate
                lr = get_lr(num_iters)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                num_iters += 1

            total_loss += loss.item() * config.grad_acc_steps
            loss_for_examination += loss.item() * config.grad_acc_steps

        avg_loss = total_loss / len(dataloader_train)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

        # Validation
        encoder.eval()
        decoder.eval()
        total_val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in dataloader_val:
                encoder_input_ids = batch['encoder_input_ids'].to(device)
                encoder_attention_mask = batch['encoder_attention_mask'].to(device)
                decoder_input_ids = batch['decoder_input_ids'].to(device)
                decoder_attention_mask = batch['decoder_attention_mask'].to(device)
                labels = batch['labels'].to(device)

                encoding = encoder(encoder_input_ids, encoder_attention_mask)
                output = decoder(decoder_input_ids, decoder_attention_mask, encoder_attention_mask, encoding)

                loss = criterion(output.view(-1, output.size(-1)), labels.view(-1))

                total_val_loss += loss.item()

                # Calculate accuracy
                predicted = torch.argmax(output, dim=-1)
                mask = labels != -100
                correct += ((predicted == labels) & mask).sum().item()
                total += mask.sum().item()
            accuracy = correct / total if total > 0 else 0

        avg_val_loss = total_val_loss / len(dataloader_val)
        print(f"Validation Loss: {avg_val_loss:.4f}")
        print(f"Validation Accuracy: {accuracy:.4f}")
        # Save the model checkpoint
        checkpoint = {
            'encoder_state_dict': encoder.state_dict(),
            'decoder_state_dict': decoder.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
        }
        save_checkpoint(checkpoint, f"checkpoint_epoch_{epoch+1}.pth")



# Load the model checkpoint if needed
# checkpoint = torch.load("checkpoint_epoch_13.pth")
# load_checkpoint(encoder, decoder, optimizer, checkpoint)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='Attention Translator')
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='train', help='Mode of execution (train/test).')
    parser.add_argument('--model_path', type=str, help="Relative or absolute checkpoint path to load for testing.")

    args = parser.parse_args()

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device)

    # Download latest version
    if args.mode == 'train':

        if args.model_path != None:
            print("Warning: --model_path is not a valid parameter for mode=train. Its value will be ignored.")

        path = kagglehub.dataset_download("dhruvildave/en-fr-translation-dataset")
        print("Path to dataset files:", path)
        df = pd.read_csv(path + "/en-fr.csv").head(3000000)

        dataset = PhrasesDataset(df, loaded_tokenizer, config.max_seq_len)
        train_len = int(len(dataset) * 0.9)
        val_len = len(dataset) - train_len


        train_dataset, val_dataset = random_split(dataset, [train_len, val_len])

        train_sampler = BucketBatchSampler(train_dataset, config.mini_batch_size, config.bucket_size)
        val_sampler   = BucketBatchSampler(val_dataset,   config.mini_batch_size, config.bucket_size)


        dataloader_train = DataLoader(train_dataset, batch_sampler=train_sampler, collate_fn=lambda batch: collate_fn(batch, pad_token_id=0, bos_token_id=1,  eos_token_id=2), num_workers=4, persistent_workers=True, pin_memory=True)
        dataloader_val = DataLoader(val_dataset, batch_sampler=val_sampler, collate_fn=lambda batch: collate_fn(batch, pad_token_id=0, bos_token_id=1,  eos_token_id=2), num_workers=4, persistent_workers=True, pin_memory=True)


        encoder = Encoder(num_embeddings=20000, num_heads_per_block=8, num_blocks=5, sequence_length_max=config.max_seq_len, dim=config.embd_dim).to(device)
        print("Encoder parameters:", sum(p.numel() for p in encoder.parameters() if p.requires_grad))
        decoder = Decoder(num_embeddings=20000, num_heads_per_block=8, num_blocks=5, sequence_length_max=config.max_seq_len, dim=config.embd_dim).to(device)
        print("Decoder parameters:", sum(p.numel() for p in decoder.parameters() if p.requires_grad))


        criterion = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)
        optimizer = optim.Adam(params=list(encoder.parameters())+list(decoder.parameters()), lr=config.start_lr, weight_decay=config.weight_decay)

        test_text = "One of the most widely recognised animal symbols in human culture, the lion has been extensively depicted in sculptures and paintings."
        
        train(encoder, decoder, optimizer, dataloader_train, dataloader_val, criterion, test_text, device=device, num_epochs=100, )
    
    elif args.mode == 'test':

        if args.model_path == None:
            print("Error: you must specify the --model_path option in order to load the model to test!")
            exit(1)

        encoder = Encoder(num_embeddings=20000, num_heads_per_block=8, num_blocks=5, sequence_length_max=config.max_seq_len, dim=config.embd_dim).to(device)
        print("Encoder parameters:", sum(p.numel() for p in encoder.parameters() if p.requires_grad))
        decoder = Decoder(num_embeddings=20000, num_heads_per_block=8, num_blocks=5, sequence_length_max=config.max_seq_len, dim=config.embd_dim).to(device)
        print("Decoder parameters:", sum(p.numel() for p in decoder.parameters() if p.requires_grad))
        optimizer = optim.Adam(params=list(encoder.parameters())+list(decoder.parameters()), lr=config.start_lr, weight_decay=config.weight_decay)
        try:
            checkpoint = torch.load(args.model_path)
            load_checkpoint(encoder, decoder, optimizer=optimizer, checkpoint=checkpoint)
        except:
            print("There was an error in loading the checkpoint. Please make sure that the specified path is correct")
        
        while(1):
            print("\n\n")
            test_text = input("Please enter the english phrase you would like to translate: ")

            test(test_text, encoder, decoder, loaded_tokenizer, device=device)