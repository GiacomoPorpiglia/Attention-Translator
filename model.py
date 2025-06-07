import torch
import torch.nn as nn
import torch.nn.functional as F
from positional_encoder import PositionalEncoder

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, model_dim, num_heads, sequence_length_max, is_causal):
        super(MultiHeadAttentionBlock, self).__init__()

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.sequence_length_max = sequence_length_max
        self.is_causal = is_causal

        self.head_dim = model_dim // num_heads

        self.query = nn.Linear(model_dim, model_dim, bias=False)
        self.key   = nn.Linear(model_dim, model_dim, bias=False)
        self.value = nn.Linear(model_dim, model_dim, bias=False)

        self.dropout = nn.Dropout(0.25)
        
        self.c_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.resid_dropout = nn.Dropout(0.25)

        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

        self.register_buffer("tril", torch.tril(torch.ones(sequence_length_max, sequence_length_max)).view(1, 1, sequence_length_max, sequence_length_max))

        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
                
    
    def forward(self, x, attention_mask, encoder_output=None):
        
        B, T, C = x.shape

        ### adjust attention mask, initially [B, T]

        pad_mask = (attention_mask == 1)     # [B, T], True for tokens we want to take part into attention
        pad_mask = pad_mask.view(B, 1, 1, T)

        
        q = self.query(x) ### [B, T, H] (it comes from [B, T, C] dot [C, head_size] --> [B, T, H])
        if encoder_output is not None:
            k = self.key(encoder_output)
            v = self.value(encoder_output)
        else:
            k = self.key(x)   ### [B, T, H]
            v = self.value(x) ### [B, T, H]
        
        # Split into multiple heads
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, T, head_dim]
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, T, head_dim]
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, T, head_dim]

        if self.flash:

            pad_mask4 = pad_mask.expand(-1, self.num_heads, T, -1)  # [B, num_heads, T, T]
            # if causal, also exclude future positions
            if self.is_causal:
                causal = self.tril[:, :, :T, :T].bool() # True for tokens we want to attention according to causal mask rules
                attn_mask = pad_mask4 & causal 
            else:
                attn_mask = pad_mask4

            out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.25 if self.training else 0)
        else:

            correlation = (q @ k.transpose(-2, -1)) * self.head_dim**(-0.5) ### [B, num_heads, T, T]
            
            pad_mask4 = pad_mask.expand(-1, self.num_heads, T, -1)  # [B, num_heads, T, T]
            if self.is_causal:
                correlation = correlation.masked_fill((self.tril[:, :, :T, :T]==0) | ~pad_mask4, float('-inf')) # the mask we pass is where we want to fill with -inf, so the ones we don't want to attend to
            else:
                correlation = correlation.masked_fill(~pad_mask4, float('-inf'))
            correlation = F.softmax(correlation, dim=-1) ### [B, num_heads, T, T]
            correlation = self.dropout(correlation)

            ### output of self attention
            ### [B, num_heads, T, T] dot [B, num_heads, T, heads_dim] =  [B, num_heads, T, head_dim]
            out = correlation @ v 

        out = out.transpose(1, 2).contiguous().view(B, T, C) ### [B, T, num_heads, head_dim] --> [B, T, C]
        out = self.resid_dropout(self.c_proj(out))

        return out
    



class MLPLayer(nn.Module):
    def __init__(self, model_dim):
        super(MLPLayer, self).__init__()
        self.fc1 = nn.Linear(model_dim, 4*model_dim, bias=False)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4*model_dim, model_dim, bias=False)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, model_dim, num_heads, sequence_length, is_causal):
        super(TransformerBlock, self).__init__()

        self.norm1 = nn.LayerNorm(model_dim)

        self.attention = MultiHeadAttentionBlock(model_dim, num_heads, sequence_length, is_causal)  

        self.norm2 = nn.LayerNorm(model_dim)
        self.mlp = MLPLayer(model_dim)


    def forward(self, x, attention_mask, encoder_output=None):
        x = x + self.attention(self.norm1(x), attention_mask, encoder_output=encoder_output)
        x = x + self.mlp(self.norm2(x))
        return x





class Encoder(nn.Module):

    def __init__(self, num_embeddings, num_heads_per_block, num_blocks, sequence_length_max=192, dim=512):
        super(Encoder, self).__init__()

        self.encoder = nn.Embedding(num_embeddings, dim)
        self.positional_encoder = PositionalEncoder(dim, sequence_length_max)
        self.dropout1 = nn.Dropout(0.25)

        assert (dim % num_heads_per_block) == 0, "Embedding size is not divisible by number of heads"

        self.layers = nn.ModuleList([
            TransformerBlock(dim, num_heads_per_block, sequence_length_max, is_causal=False) for _ in range(num_blocks)
        ])

        self.layer_norm = nn.LayerNorm(dim)
        self.fc1_out = nn.Linear(dim, dim, bias=False)
        self.dropout2 = nn.Dropout(0.25)

        self.apply(self._init_weights_)



    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=0.02)

    def forward(self, x, attention_mask):
        x = self.encoder(x)
        x = self.dropout1(self.positional_encoder(x))
        for layer in self.layers:
            x = layer(x, attention_mask, encoder_output=None)
        x = self.layer_norm(x)
        x = self.fc1_out(self.dropout2(x))
        return x ### [B, T, C]


class Decoder(nn.Module):

    
    def __init__(self, num_embeddings, num_heads_per_block, num_blocks, sequence_length_max=192, dim=512):
        super(Decoder, self).__init__()

        self.encoder = nn.Embedding(num_embeddings, dim)
        self.positional_encoder = PositionalEncoder(dim, sequence_length_max)
        self.dropout1 = nn.Dropout(0.25)

        assert (dim % num_heads_per_block) == 0, "Embedding size is not divisible by number of heads"

        self.layers = nn.ModuleList([
            TransformerBlock(dim, num_heads_per_block, sequence_length_max, is_causal=True) for _ in range(num_blocks)
        ])

        self.layer_norm = nn.LayerNorm(dim)
        self.dropout2 = nn.Dropout(0.25)
        self.fc1_out = nn.Linear(dim, num_embeddings, bias=False)

        self.apply(self._init_weights)


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=0.02)

    def forward(self, x, attention_mask, encoder_output):
        x = self.encoder(x)
        x = self.dropout1(self.positional_encoder(x))
        for layer in self.layers:
            x = layer(x, attention_mask, encoder_output=encoder_output)
        x = self.layer_norm(x)
        x = self.fc1_out(self.dropout2(x))
        return x ### [B, T, C]
