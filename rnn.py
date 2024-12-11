"""
Character-level Vanilla RNN model implementation in Tinygrad. Inspired by Andrej Karpathy
https://gist.github.com/karpathy/d4dee566867f8291f086
"""

# TODO: Look at tinygrad LSTM implementation https://github.com/tinygrad/tinygrad/blob/master/extra/models/rnnt.py#L118
# TODO: Look at the tests to see input format https://github.com/tinygrad/tinygrad/blob/d462f8ace059a75a6205a234fccd0dc07e271cde/test/models/test_rnnt.py#L9
# TODO: Try to use a single LSTM layer to generate text

import sys
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
bs = 2
steps = 70

X = []
Y = []
for p in range(0, data_size-seq_len-1, seq_len):
  inputs = Tensor.zeros(seq_len, vocab_size).contiguous()
  for i, ch in enumerate(data[p:p+seq_len]): 
    inputs[i, char_to_idx[ch]] = 1
  targets = Tensor([char_to_idx[ch] for ch in data][p+1:p+seq_len+1])
  X.append(inputs)
  Y.append(targets)
X = Tensor.stack(*X, dim=1) # (seq_len, N, vocab_size)
Y = Tensor.stack(*Y, dim=1)

class RNNCell:
  def __init__(self, input_size, hidden_size, dropout=None):
    if dropout is None: dropout = 0.
    self.dropout = dropout

    self.weight_ih = Tensor.uniform(hidden_size, input_size)
    self.weight_hh = Tensor.uniform(hidden_size, hidden_size)
    self.bias_ih = Tensor.uniform(hidden_size)
    self.bias_hh = Tensor.uniform(hidden_size)
  
  def __call__(self, x, h: Optional[Tensor]=None):
      return (x.linear(self.weight_ih, self.bias_ih) + \
              h.linear(self.weight_hh, self.bias_hh)).tanh().dropout(self.dropout)


sys.exit()

# model
class RNN:
  def __init__(self, vocab_size, hidden_size):
    self.Wxh = Tensor.randn(vocab_size, hidden_size)*0.01 # input to hidden
    self.Whh = Tensor.randn(hidden_size, hidden_size)*0.01 # hidden to hidden
    self.Why = Tensor.randn(hidden_size, vocab_size)*0.01 # hidden to output
    self.bh = Tensor.zeros(hidden_size) # hidden bias
    self.by = Tensor.zeros(vocab_size) # output bias
    self.hidden_size = hidden_size

  def __call__(self, x: Tensor, h: Optional[Tensor]=None):
    # x is a vector of size vocab_size
    if h is None: h = Tensor.zeros(self.hidden_size)

    h_new = (x.dot(self.Wxh) + h.dot(self.Whh) + self.bh).tanh()
    y = h_new.dot(self.Why) + self.by
    return y.realize(), h_new.realize()
  

model = RNN(vocab_size=vocab_size, hidden_size=hidden_size)
opt = nn.optim.Adam(nn.state.get_parameters(model), lr=lr)

@Tensor.test()
def sample(h, seed_x, n):
  """
  sample a sequence of integers from the model
  h is memory state, seed_idx is seed letter for the first timestep 
  """ 
  idxs = []
  x = seed_x
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
  # inputs is (seq_len, batch_size, vocab_size) targets is (seq_len, batch_size)
  h = hprev
  loss = 0.0
  for x_i, y in zip(inputs, targets):
    y_pred, h = model(x_i, h)
    loss_i = y_pred.sparse_categorical_crossentropy(y)
    loss = loss + loss_i
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