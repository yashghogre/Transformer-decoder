# Importing Libraries
import torch
import torch.nn as nn
from torch.nn import functional as F


# Setting hyperparameters
BLOCK_SIZE = 128         # Context-window
BATCH_SIZE = 64
n_embd = 64
dropout = 0.2

# Setting device cpu/cuda
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
print(device)
# Loading dataset
with open("/kaggle/input/dataset-for-gpt-at-home/t.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Tokenization and encode/decode functions
chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[ch] for ch in s]
decode = lambda s: [itos[i] for i in s]

# Data Tensor preparation
data = torch.tensor(encode(text), dtype=torch.long, device=device)

# Spliting train and val set
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

# Getting batches
def get_batch(split):
    data = train_data if split == 'train' else val_data
    idx = torch.randint(len(data)-BLOCK_SIZE, (BATCH_SIZE, ))
    x = torch.stack([data[i: i+BLOCK_SIZE] for i in idx])
    y = torch.stack([data[i+1: i+BLOCK_SIZE+1] for i in idx])
    return x, y


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones([BLOCK_SIZE, BLOCK_SIZE])))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)

        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)

        wei = self.dropout(wei)

        v = self.value(x)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.ffwd = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
            )

    def forward(self, x):
        return self.ffwd(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_heads):
        super().__init__()
        head_size = n_embd // n_heads
        self.sa_heads = MultiHeadAttention(n_heads, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa_heads(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))

        return x


class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, n_embd)
        self.blocks = nn.Sequential(
            Block(n_embd, n_heads=4),
            Block(n_embd, n_heads=4),
            Block(n_embd, n_heads=4),
            nn.LayerNorm(n_embd)
        )
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        token_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = token_emb + pos_emb
        x = self.blocks(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_tokens):
        for _ in range(max_tokens):
            idx_cond = idx[:, -BLOCK_SIZE: ]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            # print(f"idx: {idx}")
        return idx


x1, y1 = get_batch('train')
x1, y1 = x1.to(device), y1.to(device)

torch.manual_seed(1111)
model = BigramLanguageModel()
model = model.to(device)
logits, loss = model(x1, y1)


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

sum(p.numel() for p in model.parameters())

print("Training started succesfully!")

eval_iters = 1000
learning_rate = 3e-8
eval_interval = eval_iters/10
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(eval_iters):

    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"Epoch: {iter}   |   Train Loss: {losses['train']}    |   Val Loss: {losses['val']}")

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(loss.item())

print("".join(decode(model.generate(torch.zeros((1, 1), dtype=torch.long, device=device), max_tokens=1111)[0].tolist())))