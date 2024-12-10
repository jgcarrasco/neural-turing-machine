"""
Character-level Vanilla RNN model implementation in Tinygrad. Inspired by Andrej Karpathy
https://gist.github.com/karpathy/d4dee566867f8291f086
"""

from typing import Optional
import numpy as np
from tinygrad import Tensor

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

    h = (x.linear(self.Wxh) + h.linear(self.Whh, self.bh)).tanh() # update hidden state
    y = h.linear(self.Why, self.by) # unnormalized logprobs for next char
    return y, h
  

model = RNN(vocab_size=vocab_size, hidden_size=hidden_size)

@Tensor.test()
def sample(h, seed_idx, n):
  """
  sample a sequence of integers from the model
  h is memory state, seed_idx is seed letter for the first timestep 
  """ 
  x = Tensor.zeros(vocab_size).contiguous()
  x[seed_idx] = 1
  idxs = []
  h = None
  for t in range(n):
    y, h = model(x, h)
    p = y.softmax()
    idx = int(np.random.choice(range(vocab_size), p=p.numpy()))
    x = Tensor.zeros(vocab_size).contiguous()
    x[idx] = 1
    idxs.append(idx)
  return idxs

char = 'A'

idx = char_to_idx[char]
h = Tensor.zeros(hidden_size)
sample_idxs = sample(h, idx, 100)
print("".join(idx_to_char[idx] for idx in sample_idxs))