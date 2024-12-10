from tinygrad import Tensor
import tinygrad.nn as nn

from ntm.models import LSTM

def test_shapes():
  hidden_size = 256
  input_size = 8
  seq_len = 4
  x = Tensor.randint(seq_len, input_size, low=0, high=2) # 8-bit sequence of 10 elements
  model_1 = nn.LSTMCell(input_size=input_size, hidden_size=hidden_size)
  model_2 = LSTM(input_size=input_size, hidden_size=hidden_size, n_layers=3)

  hc_1 = None
  hc_2 = None
  for x_i in x:
    x_i = x_i[None, ...] # add batch dim
    hc_1 = model_1(x_i, hc_1)
    _, hc_2 = model_2(x_i, hc_2) 
    
    assert hc_1[0].shape == (1, hidden_size)
    assert hc_2[0].shape == (3, 1, hidden_size)
    
    hc_1[0].shape, hc_2[0].shape