"""
Character-level Vanilla RNN model implementation in Tinygrad. Inspired by Andrej Karpathy
https://gist.github.com/karpathy/d4dee566867f8291f086
"""


import sys
from typing import Optional, List
from tqdm import tqdm
import numpy as np
from tinygrad import Tensor, TinyJit
import tinygrad.nn as nn

from ntm.models import RNNet

# Loading data
data = open("el_quijote.txt", "r").read()
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print(f"data has {data_size} characters, {vocab_size} unique")
char_to_idx = {ch:i for i,ch in enumerate(chars)}
idx_to_char = {i:ch for i,ch in enumerate(chars)}

# hyperparameters
hidden_size = 100
seq_len = 25 
lr = 1e-1
num_layers = 1

model = RNNet(input_size=vocab_size, hidden_size=100, num_layers=num_layers, dropout=0.5, output_size=vocab_size)
opt = nn.optim.Adam(nn.state.get_parameters(model), lr=lr)

@Tensor.test()
def sample(h, seed_x, n) -> List[int]:
  """
  sample a sequence of integers from the model
  h is memory state, seed_idx is seed letter for the first timestep 
  """ 
  x = Tensor.zeros(n, 1, vocab_size, requires_grad=False).contiguous()
  x[0] = seed_x
  y, _ = model(x, h)
  p = y.softmax() 
  idxs =  [np.random.choice(range(vocab_size), p=p[i].numpy().ravel()).item() for i in range(p.shape[0])]
  return idxs

@Tensor.train()
def training_step(inputs, targets, hprev):
  # inputs (seq_len, batch_size, vocab_size)
  # targets (seq_len, batch_size)
  # hprev (n_layers, batch_size, hidden_size)
  y_pred, h = model(inputs, hprev)
  loss = y_pred.sparse_categorical_crossentropy(targets)
  opt.zero_grad()
  loss.backward()
  for param in nn.state.get_parameters(model):
    param.grad = param.grad.clip(-5, 5)
  opt.step()
  return loss, h

n, p = 0, 0
smooth_loss = -np.log(1.0/vocab_size)*seq_len
print(f"Expected initial loss: {smooth_loss:.4f}")
while True:
  if p+seq_len+1 >= len(data) or n == 0:
    hprev = Tensor.zeros(num_layers, 1, hidden_size, requires_grad=False) # reset RNN memory
    p = 0 # go from start of data
  inputs = Tensor.zeros(seq_len, 1, vocab_size).contiguous()
  for i, ch in enumerate(data[p:p+seq_len]): 
    inputs[i, 0, char_to_idx[ch]] = 1
  targets = Tensor([char_to_idx[ch] for ch in data][p+1:p+seq_len+1])[None, :]

  # sample from the model now and then
  if n % 100 == 0:
    sample_idxs = sample(hprev, inputs[0], 200)
    txt = "".join(idx_to_char[idx] for idx in sample_idxs)
    print(f"----\n {txt} \n----")
  
  loss, hprev = training_step(inputs, targets, hprev)

  if n % 100 == 0:
    print(f"iter {n}, loss: {loss.numpy():.4f}")
  
  p += seq_len
  n += 1