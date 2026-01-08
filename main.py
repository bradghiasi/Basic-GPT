import torch
import torch.nn as nn
from torch.nn import functional as F


# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256  # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-3
device = 'cuda'
# if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd =384
n_head = 6
n_layer = 6
dropout = 0.2
# -------------

torch.manual_seed(1337)

"""there is something called residual path imagine paper: Deep Residual Learning for Image Recognition
 the residual path has some characteristics:
 1) right after initialization the amount residual path contribution to the loss is small
 y = x + F(x) in this equation F(x) is the residual path imagine we have a loss function L
 L(y,t) = 1/2||y-t||^2 --> delta L/delta y = y-t the gradient of loss based on y is y-t
 on the other hand: delta y/ delta x = (1 + delta(F(x))/delta delta (x)) --> when the gradient of F(x) in the beginning of the training is ~0 --> delta(L)/delta(x) = y-t which means the gradient of x is transfered to the upper layer and it's not vanished.
 2) The gradient is delivered to the identity(main-stream) branch and the residual branch is equal in case of calculating the gradient based on x 
 receives exactly that same gradient, modulated further by ∂𝐹/∂𝑥 as it flows through its weights. further information is in the residual path file
 3) the change in the weights in the future occurs based on the ∂L/∂w.
 """

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]
# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

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

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

class MultiHeadAttention(nn.Module):
    """multiple heads of self-attention in parallel"""
    def __init__(self,num_heads,head_size):
        super().__init__()
        self.heads = nn.ModuleList(Head(head_size) for _ in range (num_heads))
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) # because the stuff from all the heads are stacked up and they have their own feature vectors, we need to project hem on a single unit feature vector that mixes up all the information gathered from all the heads, otherwise that information just would be head1's feature vector/ head2's feature vector/ and so on
        out = self.proj(out) # this linear projection mixes the information from all sources (heads) to make a single feature space
        out = self.dropout(out)
        return out


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) # we are using buffer for two reasons:1) if we defined tril as a regular parameter the system might have updated the values, which is something we don't want2)by defining it as a buffer we make sure that the variable is in the right device (cpu or gpu) and the system won't raise an error
        self.dropout = nn.Dropout(dropout)

        """some explanation about the normalization below:
        When we say “each entry 
        1) 𝑀(𝑡,𝑖) has mean 0,” we’re really talking about the random variable that generates that entry, not the one number you observe after sampling."""

    def forward(self,x):
        B,T,C = x.shape
        k = self.key(x) # (B,T,C) this is a Gaussian input meaning MEAN = 0 and VARIANCE = 1
        q = self.query(x)# (B,T,C) this is a Gaussian input meaning MEAN = 0 and VARIANCE = 1
        #compute attention scores ('affinities')
        wei = q @ k.transpose(-2,-1) * C ** -0.5 # (B,T,C) @ (B,C, T) -> (B,T,T) consider that here in this code n_embd = head_size, so C is head_size, 2) if we don't multiply the result by * C ** -0.5 (head_seize square root), it would be on the order of the last dimension (head_size), otherwise it would be 1
        wei = wei.masked_fill(self.tril[:T,:T]==0, float('-inf')) # (B,T,T) we used [:T,:T] in case T is smaller than block_size and we just want to cut a slice of the self.tril lower triangular matrix
        wei = F.softmax(wei, dim=-1)# (B,T,T) in the generation process we only need the last row of the wei matrix because it includes all the dot products between the last token and the keys from previous tokens. However, we've designed this for both training and generation so during the generation although we only need the last token and its dot product with previous tokens in the sequence, we have to generate all the possible situations due to a dual-functional design of this decoder which works both for training and generation
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B,T,C)
        return out

class Block(nn.Module):
    """Transformer Block: communication followed by computation"""
    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimmension, h_head: the number of head we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head,head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x)) # when you add numbers to feature space the patterns you've learned and the patterns from newer vectores are all gathered into one feature space
        x = x + self.ffwd(self.ln2(x))
        return x

# super simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) # random initialization, nn.Embedding always makes a lookup table
        self.position_embedding_table = nn.Embedding(block_size, n_embd) # random initialization
        self.blocks = nn.Sequential(*[Block(n_embd, n_head = n_head)for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) #nn.LayerNorm is a predefined function that we use, so when we call it later in the forward function we must send the x as input because it's designed to receive that.
        self.lm_head = nn.Linear(n_embd, vocab_size)
        """every thing we create here are just the initializations of the objects that we create out of the classes, later on in the forward function of each class we call them which goes directly to the forward function of the class that we made an object out of it"""

    def forward(self, idx, targets=None):
        B , T = idx.shape
        # idx and targets are both (B=Batch_size=32,T=block_size=8) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T,device=device)) #(T,n_embd) torch.arange will make [0,1,...,7] so it will return all the values for all the requested indices
        x = tok_emb + pos_emb # (B,T,n_embd)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x) # (B,T, Vocab size), which includes both the content and the positional information,

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx  to the last block_size tokens
            idx_cond = idx[:,-block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx

model = BigramLanguageModel()
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')
    # print(xb.shape, yb.shape)

    # e valuate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
