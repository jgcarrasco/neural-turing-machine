import unittest

from tinygrad import Tensor
import tinygrad.nn as nn

from ntm.models import RNNCell, RNN

import torch
import numpy as np

class TestRNN(unittest.TestCase):
  def test_rnncell(self):
    BS, SQ, IS, HS = 2, 20, 40, 128

    with torch.no_grad():
      torch_layer = torch.nn.RNNCell(IS, HS)
    layer = RNNCell(IS, HS)

    with torch.no_grad():
      layer.weight_ih.assign(Tensor(torch_layer.weight_ih.numpy()))      
      layer.weight_hh.assign(Tensor(torch_layer.weight_hh.numpy()))      
      layer.bias_ih.assign(Tensor(torch_layer.bias_ih.numpy()))      
      layer.bias_hh.assign(Tensor(torch_layer.bias_hh.numpy()))      

    # test  unbatched input
    for _ in range(3):
      x = Tensor.randn(IS)
      h_new = layer(x, None)
      torch_x = torch.tensor(x.numpy())
      torch_h_new = torch_layer(torch_x)
      np.testing.assert_allclose(h_new.numpy(), torch_h_new.detach().numpy(), atol=5e-3, rtol=5e-3)

    # test batched input
    for _ in range(3):
      x = Tensor.randn(BS, IS)
      h_new = layer(x, None)
      torch_x = torch.tensor(x.numpy())
      torch_h_new = torch_layer(torch_x)
      np.testing.assert_allclose(h_new.numpy(), torch_h_new.detach().numpy(), atol=5e-3, rtol=5e-3)

  def test_rnn(self):
    BS, SQ, IS, HS, L = 2, 20, 40, 128, 2

    # create in torch
    with torch.no_grad():
      torch_layer = torch.nn.RNN(IS, HS, L)
    
    # create in tinygrad
    layer = RNN(IS, HS, L, 0.0)

    # copy weights
    with torch.no_grad():
      layer.cells[0].weight_ih.assign(Tensor(torch_layer.weight_ih_l0.numpy()))
      layer.cells[0].weight_hh.assign(Tensor(torch_layer.weight_hh_l0.numpy()))
      layer.cells[0].bias_ih.assign(Tensor(torch_layer.bias_ih_l0.numpy()))
      layer.cells[0].bias_hh.assign(Tensor(torch_layer.bias_hh_l0.numpy()))
      layer.cells[1].weight_ih.assign(Tensor(torch_layer.weight_ih_l1.numpy()))
      layer.cells[1].weight_hh.assign(Tensor(torch_layer.weight_hh_l1.numpy()))
      layer.cells[1].bias_ih.assign(Tensor(torch_layer.bias_ih_l1.numpy()))
      layer.cells[1].bias_hh.assign(Tensor(torch_layer.bias_hh_l1.numpy()))

      # test initial hidden
      for _ in range(3):
        x = Tensor.randn(SQ, BS, IS)
        z, h = layer(x, None)
        torch_x = torch.tensor(x.numpy())
        torch_z, torch_h = torch_layer(torch_x)
        np.testing.assert_allclose(z.numpy(), torch_z.detach().numpy(), atol=5e-3, rtol=5e-3)

if __name__ == "__main__":
  unittest.main()