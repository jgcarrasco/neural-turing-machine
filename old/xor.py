import sys
import numpy as np

import matplotlib.pyplot as plt

import torch
import torch.nn as torch_nn
from torch.optim import Adam as torch_Adam

from tinygrad import Tensor, TinyJit
import tinygrad.nn as tiny_nn
from tinygrad.nn.optim import Adam as tiny_Adam
from ntm.models import RNNCell


def generate_delayed_xor_dataset(seq_len=10, batch_size=2):
  x = np.zeros((seq_len, batch_size, 1), dtype=np.float32)
  y = np.zeros_like(x)
  for i in range(batch_size):
    x[:, i, 0] = np.random.randint(0, 2, size=seq_len).astype(float)
  y[0] = x[0]  # First output is just the first input (XOR with 0)
  y[1:] = np.logical_xor(x[1:], x[:-1])
  return x, y

x_numpy, y_numpy = generate_delayed_xor_dataset(seq_len=10, batch_size=100) 

torch_cell = torch_nn.RNNCell(input_size=1, hidden_size=1) 
opt = torch_Adam(torch_cell.parameters(), lr=1e-2)
loss_fn = torch_nn.MSELoss()

x = torch.tensor(x_numpy, dtype=torch.float32)
y = torch.tensor(y_numpy, dtype=torch.float32)

losses_torch = []

for i in range(600):
  h_prev = None
  y_pred = []
  for x_t, y_t in zip(x, y):
    h_prev = torch_cell(x_t, h_prev)
    y_pred.append(h_prev)
  y_pred = torch.stack(y_pred)
  loss = loss_fn(y_pred, y)
  opt.zero_grad()
  loss.backward()
  opt.step()
  print(f"Loss at iteration {i}: {loss.item():.6f}")
  losses_torch.append(loss.item())

tiny_cell = RNNCell(input_size=1, hidden_size=1)
tiny_opt = tiny_Adam(tiny_nn.state.get_parameters(tiny_cell))

x = Tensor(x_numpy)
y = Tensor(y_numpy)

losses_tiny = []

def forward(x, h_prev):
  y_pred = None
  for x_t in x:
    h_prev = tiny_cell(x_t, h_prev)
    if y_pred is None:
      y_pred = h_prev[None, ...]
    else:
      y_pred = y_pred.cat(h_prev[None, ...], dim=0).realize() 
  return y_pred

Tensor.training = True
for i in range(600):
  h_prev = Tensor.zeros(1, 1) 
  y_pred = forward(x, h_prev) 
  loss = (y_pred - y).square().mean()
  tiny_opt.zero_grad()
  loss.backward()
  tiny_opt.step()
  print(f"Loss at iteration {i}: {loss.item():.6f}")
  losses_tiny.append(loss.item())



plt.plot(losses_torch, color="orange", label="torch")
plt.plot(losses_tiny, color="blue", label="tiny")
plt.legend()
plt.savefig("fig.png")

