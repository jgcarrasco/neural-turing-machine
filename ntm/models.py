import math
from typing import Optional, Tuple

from tinygrad import Tensor, TinyJit
import tinygrad.nn as nn


class RNNCell:
  def __init__(self, input_size, hidden_size, dropout=None):
    if dropout is None: dropout = 0.
    self.dropout = dropout
    self.hidden_size = hidden_size

    stdv = 1.0 / math.sqrt(hidden_size)

    self.weight_ih = Tensor.uniform(hidden_size, input_size, low=-stdv, high=stdv)
    self.weight_hh = Tensor.uniform(hidden_size, hidden_size, low=-stdv, high=stdv)
    self.bias_ih = Tensor.uniform(hidden_size)
    self.bias_hh = Tensor.uniform(hidden_size)
  
  def __call__(self, x, h):
    return (x.linear(self.weight_ih.T, self.bias_ih) + \
            h.linear(self.weight_hh.T, self.bias_hh)).tanh().dropout(self.dropout).realize()


class RNN:
  def __init__(self, input_size, hidden_size, num_layers, dropout):
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.num_layers = num_layers

    self.cells = [RNNCell(input_size, hidden_size, dropout) if i == 0 else \
                  RNNCell(hidden_size, hidden_size, dropout if i != num_layers - 1 else 0) for i in range(num_layers)]

  def __call__(self, x, h=None):
    @TinyJit
    def _do_step(x_, h_):
      return self.do_step(x_, h_)
    
    if h is None:
      h = Tensor.zeros(self.num_layers, x.shape[1], self.hidden_size, requires_grad=False)
    
    output = None
    for t in range(x.shape[0]):
      h = _do_step(x[t] + 1 - 1, h) # If you don't add the +1 terms, JIT throws an error related to offset
      if output is None:
        output = h[-1:]
      else:
        output = output.cat(h[-1:], dim=0).realize()
    
    return output, h


  def do_step(self, x, h):
    # x is (N, I)
    new_h = [x]
    for i, cell in enumerate(self.cells):
      new_h.append(cell(new_h[i], h[i]))
    return Tensor.stack(*new_h[1:]).realize()

class RNNet:
  def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
    self.RNN = RNN(input_size, hidden_size, num_layers, dropout)
    self.proj = nn.Linear(hidden_size, output_size)
  def __call__(self, x, h=None):
    output, h  = self.RNN(x, h)
    output = self.proj(output)
    return output, h

class LSTM:
  """
  Vanilla LSTM used in the experiments of the Neural Turing Machines paper. This is simply a
  stack of LSTM cells followed by a Linear layer to project into the output space.

  Args:
    input_size: The number of expected features in the input `x`. `x` has shape `(batch_size, input_size)`
    hidden_size: The number of features in the hidden state `h`
    n_layers: Number of stacked LSTM cells 
  """
  def __init__(self, input_size=8, hidden_size=256, n_layers=3):
    self.n_layers = n_layers
    self.hidden_size = hidden_size
    output_sizes = n_layers * [hidden_size]
    input_sizes = [input_size] + output_sizes[1:]  
    self.cells = [nn.LSTMCell(input_size=i, hidden_size=o) for i, o in zip(input_sizes, output_sizes)]
    self.linear = nn.Linear(in_features=hidden_size, out_features=input_size)


  def __call__(self, x: Tensor, hc: Optional[Tuple[Tensor, Tensor]]=None) -> Tuple[Tensor, Tensor]:
    # h and c have shape (n_layers, batch_size, hidden_size)
    # x has shape (batch_size, input_size)
    if hc is None: hc = (Tensor.zeros(self.n_layers, x.size(0), self.hidden_size,
                                      dtype=x.dtype, device=x.device),) * 2
    
    h_new = []
    c_new = []

    x_i = x
    for i, cell in enumerate(self.cells):
      hc_i = (hc[0][i], hc[1][i]) # previous state of ith cell
      h, c = cell(x_i, hc=hc_i)
      x_i = h
      h_new.append(h)
      c_new.append(c)
    
    h_new = Tensor.stack(*h_new, dim=0)
    c_new = Tensor.stack(*c_new, dim=0)
    return self.linear(x_i), (h_new, c_new)    