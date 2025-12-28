import math
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

# Parameters
max_len = 20
n_embed = 64
vocab_size = 21400

dropout = 0.1
n_heads = 2
n_blocks = 4
class LayerNorm(nn.Module):
    def __init__(self,eps=1e-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(n_embed))
        self.beta = nn.Parameter(torch.zeros(n_embed))
    
    def forward(self,x):
        mean = x.mean(dim=-1,keepdim=True)
        std = x.std(dim=-1,keepdim=True)

        return self.alpha*(x-mean)/(std+self.eps)+self.beta
    
class TransformerEmbeddings(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, n_embed)
        pe = torch.zeros(max_len, n_embed)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, n_embed, 2).float() * (-math.log(10000.0) / n_embed)
        )
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        self.register_buffer("pe", pe)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, idx):  # idx: Batch, Seq_len
        x = self.embedding(idx)*math.sqrt(n_embed)  # Batch, Seq_len, n_embed
        x = x + self.pe[: x.size(1), :].requires_grad_(False)
        return self.dropout(x)


class MultiheadAttention(nn.Module):
    def __init__(self, masked=False):
        super().__init__()
        self.masked = masked
        self.key = nn.Linear(n_embed,n_embed,bias=False)
        self.query = nn.Linear(n_embed,n_embed,bias=False)
        self.value = nn.Linear(n_embed,n_embed,bias=False)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(n_embed,n_embed)
        self.register_buffer('tril',torch.tril(torch.ones(max_len,max_len)).unsqueeze(0).unsqueeze(0))

    def forward(self, idx,input_mask,enc_data=None):
        # print(idx.shape)
        B1,T1,C1 = idx.shape
        
        h_dim= int(C1//n_heads)
        if enc_data is not None:
            B2,T2,C2 = enc_data.shape
            key = self.key(enc_data).view(B2,T2,n_heads,h_dim).transpose(1,2) # B,n_heads,T,h_dim
            query = self.query(idx).view(B1,T1,n_heads,h_dim).transpose(1,2) # B,n_heads,T,h_dim
            value = self.value(enc_data).view(B2,T2,n_heads,h_dim).transpose(1,2) # B,n_heads,T,h_dim
        else:
            key = self.key(idx).view(B1,T1,n_heads,h_dim).transpose(1,2) # B,n_heads,T,h_dim
            query = self.query(idx).view(B1,T1,n_heads,h_dim).transpose(1,2) # B,n_heads,T,h_dim
            value = self.value(idx).view(B1,T1,n_heads,h_dim).transpose(1,2) # B,n_heads,T,h_dim
        # runtime checks to help debug CUDA/cuBLAS failures
        try:
            assert query.device == key.device == value.device, 'Mismatched devices between query/key/value'
            assert query.dtype == key.dtype == value.dtype, 'Mismatched dtypes between query/key/value'
            # ensure mask is on same device
            if input_mask is not None:
                assert input_mask.device == query.device, 'input_mask must be on same device as tensors'
        except AssertionError as e:
            print(f"[MultiheadAttention] Runtime check failed: {e}")
            print(f" query: shape={tuple(query.shape)}, dtype={query.dtype}, device={query.device}, contiguous={query.is_contiguous()}")
            print(f" key:   shape={tuple(key.shape)}, dtype={key.dtype}, device={key.device}, contiguous={key.is_contiguous()}")
            print(f" value: shape={tuple(value.shape)}, dtype={value.dtype}, device={value.device}, contiguous={value.is_contiguous()}")
            raise

        try:
            wei = query @ key.transpose(-1,-2)/math.sqrt(h_dim) # B,n_heads,T,T
        except RuntimeError as e:
            print('[MultiheadAttention] RuntimeError during query@key matmul:', e)
            print(f" query: shape={tuple(query.shape)}, dtype={query.dtype}, device={query.device}")
            print(f" key:   shape={tuple(key.shape)}, dtype={key.dtype}, device={key.device}")
            raise
        # print('shape of wei:',wei.shape)
        # print(wei)
        if self.masked:
            wei = wei.masked_fill(input_mask*self.tril[0,0,:T1,:T1] == 0,float('-inf'))
        else:
            wei = wei.masked_fill(input_mask == 0,float('-inf'))

        wei = F.softmax(wei,dim=-1)
        wei = self.dropout(wei)
        try:
            value = (wei @ value).transpose(1,2).contiguous().view(B1,T1,C1)
        except RuntimeError as e:
            print('[MultiheadAttention] RuntimeError during wei@value matmul:', e)
            print(f" wei:   shape={tuple(wei.shape)}, dtype={wei.dtype}, device={wei.device}")
            print(f" value: shape={tuple(value.shape)}, dtype={value.dtype}, device={value.device}")
            raise

        return self.proj(value) 


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.L1 = nn.Linear(n_embed, n_embed * 4)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.L2 = nn.Linear(n_embed * 4, n_embed)

    def forward(self, idx):
        x = self.L1(idx)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.L2(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.multiheadattention = MultiheadAttention()
        self.feedforward = FeedForward()
        self.norm = nn.ModuleList([LayerNorm() for _ in range(2) ])
        self.dropout = nn.ModuleList([nn.Dropout() for _ in range(2) ])

    def forward(self, idx,input_mask):
        x = self.norm[0](idx + self.dropout[0](self.multiheadattention(idx,input_mask)))
        x = self.norm[1](x + self.dropout[0](self.feedforward(x)))
        return x


class DecoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.maskedmultiheadattention = MultiheadAttention(masked=True)
        self.multiheadattention = MultiheadAttention()
        self.feedforward = FeedForward()
        self.norm = nn.ModuleList([LayerNorm() for _ in range(3) ])
        self.dropout = nn.ModuleList([nn.Dropout() for _ in range(3) ])

    def forward(self, idx,input_mask, enc_idx,enc_mask):
        x = self.norm[0](idx + self.dropout[0](self.maskedmultiheadattention(idx,input_mask)))
        x = self.norm[1](x + self.dropout[1](self.multiheadattention(idx,enc_mask, enc_idx)))
        x = self.norm[2](x + self.dropout[2](self.feedforward(x)))
        return x

class TransformerPipeline(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = TransformerEmbeddings()
        self.encoder = nn.ModuleList([EncoderBlock() for _ in range(n_blocks)])
        self.decoder = nn.ModuleList([DecoderBlock() for _ in range(n_blocks)])
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def encode(self, enc_x,enc_mask):
        x = self.embedding(enc_x)
        for blocks in self.encoder:
            x = blocks(x,enc_mask)
        return x
    
    def decode(self, dec_x,dec_mask,enc_output,enc_mask):
        x = self.embedding(dec_x)
        for blocks in self.decoder:
            x = blocks(x,dec_mask, enc_output,enc_mask)
        x = self.lm_head(x)
        return x