import torch
import torch.nn as nn
import torch.nn.functional as F
from positional_encoder import PositionalEncoder

class NonCausalSelfAttentionBlock(nn.Module):
    def __init__(self, model_dim, num_heads, sequence_length_max):
        super(NonCausalSelfAttentionBlock, self).__init__()

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.sequence_length_max = sequence_length_max

        self.head_dim = model_dim // num_heads

        self.query = nn.Linear(model_dim, model_dim, bias=False)
        self.key   = nn.Linear(model_dim, model_dim, bias=False)
        self.value = nn.Linear(model_dim, model_dim, bias=False)

        self.dropout = nn.Dropout(0.25)
        
        self.c_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.resid_dropout = nn.Dropout(0.25)

        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
                
    
    def forward(self, x, attention_mask):
        
        B, T, C = x.shape

        ### adjust attention mask, initially [B, T]

        attn_mask = (attention_mask == 1)     # [B, T], True for tokens we want to take part into attention
        attn_mask = attn_mask.view(B, 1, 1, T)

        
        q = self.query(x) ### [B, T, H] (it comes from [B, T, C] dot [C, head_size] --> [B, T, H])
        k = self.key(x)   ### [B, T, H]
        v = self.value(x) ### [B, T, H]
        
        # Split into multiple heads
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, T, head_dim]
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, T, head_dim]
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, T, head_dim]

        if self.flash:

            attn_mask = attn_mask.expand(-1, self.num_heads, T, -1)  # [B, num_heads, T, T]
            out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.25 if self.training else 0)
        else:

            correlation = (q @ k.transpose(-2, -1)) * self.head_dim**(-0.5) ### [B, num_heads, T, T]
            
            attn_mask = attn_mask.expand(-1, self.num_heads, T, -1)  # [B, num_heads, T, T]
            correlation = correlation.masked_fill(~attn_mask, float('-inf'))
            correlation = F.softmax(correlation, dim=-1) ### [B, num_heads, T, T]
            correlation = self.dropout(correlation)

            ### output of self attention
            ### [B, num_heads, T, T] dot [B, num_heads, T, heads_dim] =  [B, num_heads, T, head_dim]
            out = correlation @ v 

        out = out.transpose(1, 2).contiguous().view(B, T, C) ### [B, T, num_heads, head_dim] --> [B, T, C]
        out = self.resid_dropout(self.c_proj(out))

        return out
    



class CausalAttentionBlock(nn.Module):
    def __init__(self, model_dim, num_heads, sequence_length_max):
        super(CausalAttentionBlock, self).__init__()

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.sequence_length_max = sequence_length_max

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
                
    
    def forward(self, x, attention_mask):
        
        B, T, C = x.shape

        ### adjust attention mask, initially [B, T]

        attn_mask = (attention_mask == 1)     # [B, T], True for tokens we want to take part into attention
        attn_mask = attn_mask.view(B, 1, 1, T)

        
        q = self.query(x) ### [B, T, H] (it comes from [B, T, C] dot [C, head_size] --> [B, T, H])
        k = self.key(x)   ### [B, T, H]
        v = self.value(x) ### [B, T, H]
        
        # Split into multiple heads
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, T, head_dim]
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, T, head_dim]
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, T, head_dim]

        if self.flash:

            attn_mask4 = attn_mask.expand(-1, self.num_heads, T, -1)  # [B, num_heads, T, T]
            # if causal, also exclude future positions
            causal = self.tril[:, :, :T, :T].bool() # True for tokens we want to attention according to causal mask rules
            attn_mask = attn_mask4 & causal

            out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.25 if self.training else 0)
        else:

            correlation = (q @ k.transpose(-2, -1)) * self.head_dim**(-0.5) ### [B, num_heads, T, T]
            
            attn_mask4 = attn_mask.expand(-1, self.num_heads, T, -1)  # [B, num_heads, T, T]
        
            correlation = correlation.masked_fill((self.tril[:, :, :T, :T]==0) | ~attn_mask4, float('-inf')) # the mask we pass is where we want to fill with -inf, so the ones we don't want to attend to
            
            correlation = F.softmax(correlation, dim=-1) ### [B, num_heads, T, T]
            correlation = self.dropout(correlation)

            ### output of self attention
            ### [B, num_heads, T, T] dot [B, num_heads, T, heads_dim] =  [B, num_heads, T, head_dim]
            out = correlation @ v 

        out = out.transpose(1, 2).contiguous().view(B, T, C) ### [B, T, num_heads, head_dim] --> [B, T, C]
        out = self.resid_dropout(self.c_proj(out))

        return out
    


class CrossAttentionBlock(nn.Module):
    def __init__(self, model_dim, num_heads, sequence_length_max):
        super(CrossAttentionBlock, self).__init__()

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.sequence_length_max = sequence_length_max

        self.head_dim = model_dim // num_heads

        self.query = nn.Linear(model_dim, model_dim, bias=False)
        self.key   = nn.Linear(model_dim, model_dim, bias=False)
        self.value = nn.Linear(model_dim, model_dim, bias=False)

        self.dropout = nn.Dropout(0.25)
        
        self.c_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.resid_dropout = nn.Dropout(0.25)

        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
                
    
    def forward(self, x, encoder_attention_mask, encoder_output):
        
        B, T, C = x.shape

        ### adjust attention mask, initially [B, T]

        attn_mask = (encoder_attention_mask == 1)     # [B, T], True for tokens we want to take part into attention
        attn_mask = attn_mask.view(B, 1, 1, T)

        
        q = self.query(x) ### [B, T, H] (it comes from [B, T, C] dot [C, head_size] --> [B, T, H])
        k = self.key(encoder_output)
        v = self.value(encoder_output)
        
        # Split into multiple heads
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, T, head_dim]
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, T, head_dim]
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, T, head_dim]

        if self.flash:

            attn_mask4 = attn_mask.expand(-1, self.num_heads, T, -1)  # [B, num_heads, T, T]
            
            attn_mask = attn_mask4

            out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.25 if self.training else 0)
        else:

            correlation = (q @ k.transpose(-2, -1)) * self.head_dim**(-0.5) ### [B, num_heads, T, T]
            
            attn_mask4 = attn_mask.expand(-1, self.num_heads, T, -1)  # [B, num_heads, T, T]
            
            correlation = correlation.masked_fill(~attn_mask4, float('-inf'))
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


class EncoderTransformerBlock(nn.Module):
    def __init__(self, model_dim, num_heads, sequence_length):
        super(EncoderTransformerBlock, self).__init__()

        self.norm1 = nn.LayerNorm(model_dim)

        self.attention = NonCausalSelfAttentionBlock(model_dim, num_heads, sequence_length)  

        self.norm2 = nn.LayerNorm(model_dim)
        self.mlp = MLPLayer(model_dim)


    def forward(self, x, attention_mask):
        x = x + self.attention(self.norm1(x), attention_mask)
        x = x + self.mlp(self.norm2(x))
        return x

class DecoderTransformerBlock(nn.Module):
    def __init__(self, model_dim, num_heads, sequence_length):
        super(DecoderTransformerBlock, self).__init__()

        self.norm1 = nn.LayerNorm(model_dim)

        self.self_attention = CausalAttentionBlock(model_dim, num_heads, sequence_length)  
        self.cross_attention = CrossAttentionBlock(model_dim, num_heads, sequence_length)  

        self.norm2 = nn.LayerNorm(model_dim)

        
        self.norm3 = nn.LayerNorm(model_dim)
        self.mlp = MLPLayer(model_dim)


    def forward(self, x, decoder_attention_mask, encoder_attention_mask, encoder_output):
        x = x + self.self_attention(self.norm1(x), decoder_attention_mask)
        x = x + self.cross_attention(self.norm2(x), encoder_attention_mask, encoder_output)
        x = x + self.mlp(self.norm3(x))
        return x




class Encoder(nn.Module):

    def __init__(self, num_embeddings, num_heads_per_block, num_blocks, sequence_length_max=192, dim=512):
        super(Encoder, self).__init__()

        self.encoder = nn.Embedding(num_embeddings, dim)
        self.positional_encoder = PositionalEncoder(dim, sequence_length_max)
        self.dropout1 = nn.Dropout(0.25)

        assert (dim % num_heads_per_block) == 0, "Embedding size is not divisible by number of heads"

        self.layers = nn.ModuleList([
            EncoderTransformerBlock(dim, num_heads_per_block, sequence_length_max) for _ in range(num_blocks)
        ])

        self.layer_norm = nn.LayerNorm(dim)
        self.fc1_out = nn.Linear(dim, dim, bias=False)
        self.dropout2 = nn.Dropout(0.25)

        self.apply(self._init_weights)



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
            x = layer(x, attention_mask)
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
            DecoderTransformerBlock(dim, num_heads_per_block, sequence_length_max) for _ in range(num_blocks)
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

    def forward(self, x, decoder_attention_mask, encoder_attention_mask, encoder_output):
        x = self.encoder(x)
        x = self.dropout1(self.positional_encoder(x))
        for layer in self.layers:
            x = layer(x, decoder_attention_mask, encoder_attention_mask, encoder_output)
        x = self.layer_norm(x)
        x = self.fc1_out(self.dropout2(x))
        return x ### [B, T, C]
