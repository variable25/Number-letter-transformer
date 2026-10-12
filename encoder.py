'''
Encoder
1. Reads the input tokens and the embeddings
2. Performs self-attention across multiple heads
3. Adds it with the input tokens and normalizes the value to not let them wander too high
4. Feeds the resultant of the prev step to a Feed Forward Neural Network
5. Again the input to the NN is added to the resultant from prev step.
6. Resultant is in the shape [B,S,128]

'''
from torch import nn

from attention import pad_mask
from embeddings import D_model, Embedder
from multi_head import MultiHeadAttention, N_heads

D_ff = 512 # 128 numbers embedded will be stretched out to 512 numbers
N_layers = 3 # 3 stacked encoder layers (attention + Add & Norm + feed-forward + Add & Norm each)

# Class to define the neural network
class FeedForward(nn.Module):
    def __init__(self, d_model=D_model, d_ff=D_ff):
        super().__init__()
        # stretches from 128 numbers to 512
        self.stretch = nn.Linear(d_model, d_ff)
        # relu is a max(0, .) filter: negatives become 0 (not a loss function)
        self.relu = nn.ReLU()
        # resultant is squeezed back again to 128 
        self.squeeze = nn.Linear(d_ff,d_model)

    def forward(self,x):
        return self.squeeze(self.relu(self.stretch(x)))
    
# Class to define a layer in the encoding operation
class EncoderLayer(nn.Module):
    def __init__(self, d_model=D_model, n_heads=N_heads, d_ff=D_ff, dropout=0.1):
        super().__init__()
        # Performs multi head attention step
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.ffn = FeedForward(d_model,d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x,mask=None):
        x = self.norm1(x + self.dropout(self.self_attn(x,x,x,mask)))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


# The whole encoding process takes place in this class
class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model=D_model, n_heads=N_heads, d_ff=D_ff,
                 n_layers=N_layers, dropout=0.1, pad_id=0):
        super().__init__()
        self.pad_id = pad_id
        self.embed = Embedder(vocab_size, d_model, dropout=dropout, pad_id=pad_id)   # Stage 2
        # ModuleList: a list PyTorch can see, so all 3 layers' weights get trained and moved to the GPU
        # the loop builds 3 SEPARATE layers, each with its own random weights
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])

    def forward(self, src):                                  # src: digit IDs [B, S]
        mask = pad_mask(src, self.pad_id)                    # [B, 1, S], True = real digit (Stage 3)
        x = self.embed(src)                                  # [B, S, 128]
        for layer in self.layers:                            # layer 1's output is layer 2's input, and so on
            x = layer(x, mask)
        return x                                             # memory: [B, S, 128]