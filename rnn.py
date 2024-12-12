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
seq_len = 100
lr = 3e-4
bs = 100
steps = 70


X, Y = [], []
data_encoded = np.array([char_to_idx[char] for char in data])
for p in tqdm(range(0, data_size-seq_len-1, seq_len)):
  inputs = np.zeros((seq_len, vocab_size))
  for i, idx in enumerate(data_encoded[p:p+seq_len]):
    inputs[i, idx] = 1
  targets = data_encoded[p+1:p+seq_len+1]
  X.append(inputs)
  Y.append(targets)
X = np.stack(X, axis=1)
Y = np.stack(Y, axis=1)
X = Tensor(X)
Y = Tensor(Y)

model = RNNet(input_size=vocab_size, hidden_size=512, num_layers=2, dropout=0.5, output_size=vocab_size)
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


smooth_loss = -np.log(1.0/vocab_size)
print(f"Expected initial loss: {smooth_loss:.4f}")
n = 0
while True:
  for i in range(0, X.shape[1]-bs, bs):
    if i == 0:
      hprev = Tensor.zeros(2, bs, 512)
    x_batch = X[:, i:i+bs]
    y_batch = Y[:, i:i+bs]
    # sample from the model now and then
    if n % 100 == 0:
      sample_idxs = sample(hprev[:, :1], x_batch[0, :1], 200)
      txt = "".join(idx_to_char[idx] for idx in sample_idxs)
      print(f"---\n {txt} \n---")

    loss, hprev = training_step(x_batch, y_batch, hprev)

    if n % 100 == 0:
      print(f"iter {n}, loss: {loss.numpy():.4f}")
    
    n += bs

sys.exit()
n, p = 0, 0
smooth_loss = -np.log(1.0/vocab_size)*seq_len
print(f"Expected initial loss: {smooth_loss:.4f}")
for i in range(steps):
  n = i * bs
  hprev = Tensor.zeros(bs, hidden_size, requires_grad=False)
  samples = Tensor.randint(bs, high=X.shape[1])
  x_batch = X[:, samples] # (seq_len, batch_size, vocab_size)
  y_batch = Y[:, samples] # (seq_len, batch_size)
  # sample from the model now and then
  if n % 100 == 0:
    sample_idxs = sample(hprev[0], x_batch[0, 0], 200)
    txt = "".join(idx_to_char[idx] for idx in sample_idxs)
    print(f"----\n {txt} \n----")
  
  loss, hprev = training_step(x_batch, y_batch, hprev)

  if n % 100 == 0:
    print(f"iter {n}, loss: {loss.numpy():.4f}")
  

while True:
  if p+seq_len+1 >= len(data) or n == 0:
    hprev = Tensor.zeros(hidden_size, requires_grad=False) # reset RNN memory
    p = 0 # go from start of data
  inputs = Tensor.zeros(seq_len, vocab_size).contiguous()
  for i, ch in enumerate(data[p:p+seq_len]): 
    inputs[i, char_to_idx[ch]] = 1
  targets = Tensor([char_to_idx[ch] for ch in data][p+1:p+seq_len+1])

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