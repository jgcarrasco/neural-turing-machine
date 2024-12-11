"""
Character-level Vanilla RNN model implementation in Tinygrad. Inspired by Andrej Karpathy
https://gist.github.com/karpathy/d4dee566867f8291f086
"""

from typing import Optional
import numpy as np
from tinygrad import Tensor, TinyJit
import tinygrad.nn as nn

# Loading data
data = open("lorem.txt", "r").read()
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print(f"data has {data_size} characters, {vocab_size} unique")
char_to_idx = {ch:i for i,ch in enumerate(chars)}
idx_to_char = {i:ch for i,ch in enumerate(chars)}

# hyperparameters
hidden_size = 256
seq_len = 25
lr = 3e-4

# model
class RNN:
  def __init__(self, vocab_size, hidden_size):
    self.Wxh = Tensor.randn(hidden_size, vocab_size)*0.01 # input to hidden
    self.Whh = Tensor.randn(hidden_size, hidden_size)*0.01 # hidden to hidden
    self.Why = Tensor.randn(vocab_size, hidden_size)*0.01 # hidden to output
    self.bh = Tensor.zeros(hidden_size) # hidden bias
    self.by = Tensor.zeros(vocab_size) # output bias
    self.hidden_size = hidden_size

  def __call__(self, x: Tensor, h: Optional[Tensor]=None):
    # x is a vector of size vocab_size
    if h is None: h = Tensor.zeros(self.hidden_size)

    h_new = (self.Wxh.dot(x) + self.Whh.dot(h) + self.bh).tanh()
    y = self.Why.dot(h_new) + self.by
    return y.realize(), h_new.realize()
  

model = RNN(vocab_size=vocab_size, hidden_size=hidden_size)
opt = nn.optim.Adam(nn.state.get_parameters(model), lr=lr)

@Tensor.test()
def sample(h, seed_idx, n):
  """
  sample a sequence of integers from the model
  h is memory state, seed_idx is seed letter for the first timestep 
  """ 
  x = Tensor.zeros(vocab_size).contiguous()
  x[seed_idx] = 1
  idxs = []
  for _ in range(n):
    y, h = model(x, h)
    p = y.softmax()
    idx = int(np.random.choice(range(vocab_size), p=p.numpy()))
    x = Tensor.zeros(vocab_size).contiguous()
    x[idx] = 1
    idxs.append(idx)
  return idxs

@Tensor.train()
def training_step(inputs, targets, hprev):
  h = hprev
  y_pred = Tensor.zeros(len(targets), vocab_size).contiguous() 
  loss = None 
  h = hprev
  loss = 0.0
  for c, t in zip(inputs, targets):
    x = Tensor.zeros(vocab_size).contiguous()
    x[c] = 1
    y_pred, h = model(x, h)
    loss_i = y_pred.sparse_categorical_crossentropy(t)
    loss = loss + loss_i
    print(x.shape, y_pred.shape, h.shape)
    # assert np.allclose(loss_i.numpy(), -1.0 * y_pred.softmax()[t].log().numpy())
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
    hprev = Tensor.zeros(hidden_size) # reset RNN memory
    p = 0 # go from start of data
  inputs = Tensor([char_to_idx[ch] for ch in data[p:p+seq_len]])
  targets = Tensor([char_to_idx[ch] for ch in data[p+1:p+seq_len+1]])

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