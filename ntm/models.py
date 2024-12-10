from typing import Optional, Tuple

from tinygrad import Tensor
import tinygrad.nn as nn

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