# import torch
# import torch.nn as nn
#
# # loss_fn = nn.CrossEntropyLoss()
# #
# # # Model output logits (not softmax!) → shape [batch_size, num_classes]
# # logits = torch.tensor([[1.5, 2.5, 0.3]])  # predicted scores
# # labels = torch.tensor([1])  # correct class index
# #
# # loss = loss_fn(logits, labels)
# # # print(loss)
# # A=[]
# # A = [i for i in range(1,11)]
# # A = list(range(1,11))
# #
# # print(A[-1])
# a={}
# a['behrad']= 'ghiasi'
# a['behrad']= 'hojattolah'
# a['koj'] = 'windsor'
# print(a)
# eval_iters = 200
# @torch.no_gra()
# def estimate_loss():
#     out = {}
#     model.eval()
#     for split in ['train', 'val']:
#         losses = torch.zeros(eval_iters)
#         for k in range(eval_iters):
#             X, Y = get_batch(split)
#             logits, loss = model.(X, Y)
#             losses[k] = loss.item()
#         out[split] = losses.mean()
#         model.train()
#     return out
#
# import torch
# print(torch.cuda.is_available())
#
# xbow = torch.zeros((B,T,C))
# for b in range(B):
#     for t in range(T):
#         xprev = [b,:t+1]
#         xbow[b,t] = torch.mean(xprev, 0)
#
#         torch.tril(torch.ones(T,T))
#
# import torch
#
# wei = torch.tensor([[1., 2., 3.],
#                     [4., 5., 6.],
#                     [7., 8., 9.]])
#
# row_sum = wei.sum(0, keepdim=True)
#
# print("wei shape:", wei.shape)
# print("row_sum shape:", row_sum.shape)
# print("row_sum:")
# print(row_sum)

import torch
import torch.nn as nn
#
# # Define dimensions
# n_embd = 4
# vocab_size = 6
#
# # Create the lm_head linear layer
# lm_head = nn.Linear(n_embd, vocab_size)
#
# # Print its weights and bias
# print("Weight shape:", lm_head.weight.shape)  # [6, 4]
# print("Bias shape:", lm_head.bias.shape)      # [6]
#

# context = torch.zeros((1, 1), dtype=torch.long)
# B,T = context.shape
# print(B,T)
# print(context)

import torch, torch.nn as nn

x    = torch.tensor([ 2.0, -3.5, 0.0, 1.2 ])
relu = nn.ReLU()
y    = relu(x)
print(y)   # tensor([2.0000, 0.0000, 0.0000, 1.2000])
class BatchNorm1d:
    def __init__(self,dim, eps=1e-5):
        self.dim = dim
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)

    def __call__(self, x):
        xmean = x.mean(1, keepdim=True)
        xvar = x.var(1, keepdim=True)
        xhat = (x - xmean) / torch.sqrt(xvar + 1e-5)
        self.out = self.gamma * xhat + self.beta
        return self.out

    def parameters(self):
        return [self.gamma, self.beta]

