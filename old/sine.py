import numpy as np
import matplotlib.pyplot as plt
import torch

from ntm.models import RNNet
import tinygrad
from tinygrad import Tensor

def generate_sinus_wave(train_len, valid_len):
  time_steps = np.linspace(0, 8*np.pi, train_len+valid_len)
  data = np.sin(time_steps)

  xs = data[:train_len-1]
  ys = data[1:train_len] # as discussed in class, targets are shifted by 1 step

  train_x = xs.reshape(-1, 1, 1)
  train_y = ys
  return data, time_steps, train_x, train_y

seq_length = 200 #total sequence length
portion_train = 0.3 #portion of the sequence length used for training

train_len = int(seq_length*portion_train)
valid_len = seq_length-train_len
data, time_steps, train_x, train_y = \
    generate_sinus_wave(train_len = train_len, valid_len = valid_len)

x_train_torch = torch.tensor(train_x, dtype=torch.float32)
y_train_torch = torch.tensor(train_y, dtype=torch.float32)

class RNNModel(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.rnn = torch.nn.RNN(input_size=1, hidden_size=10)
    self.linear = torch.nn.Linear(10, 1)
  def forward(self, x, h=None):
    y, h = self.rnn(x, h)
    return self.linear(y), h

torch_model = RNNModel()
loss_fn_torch = torch.nn.MSELoss()
opt_torch = torch.optim.Adam(torch_model.parameters(), lr=1e-2)

epochs = 500
for epoch in range(epochs):
  output, _ = torch_model(x_train_torch)
  loss = loss_fn_torch(output.view(-1), y_train_torch)
  opt_torch.zero_grad()
  loss.backward()
  opt_torch.step()
  if epoch % 10 == 0:
    print(f"Epoch {epoch+1}: loss {loss.item()}")

tiny_model = RNNet(input_size=1, hidden_size=10, num_layers=1, output_size=1, dropout=0.)
opt_tiny = tinygrad.nn.optim.Adam(tinygrad.nn.state.get_parameters(tiny_model), lr=1e-2)

x_train_tiny = tinygrad.Tensor(train_x)
y_train_tiny = tinygrad.Tensor(train_y)

@tinygrad.TinyJit
@tinygrad.Tensor.train()
def train_tiny():
  for epoch in range(epochs):
    output, _ = tiny_model(x_train_tiny)
    loss = (output.view(-1) - y_train_tiny).square().mean()
    opt_tiny.zero_grad()
    loss.backward()
    opt_tiny.step()
    if epoch % 10 == 0:
      print(f"Epoch {epoch+1}: loss {loss.numpy()}")

train_tiny()

# tiny_model.RNN.cells[0].weight_ih.assign(Tensor(torch_model.rnn.weight_ih_l0.detach().numpy()))
# tiny_model.RNN.cells[0].weight_hh.assign(Tensor(torch_model.rnn.weight_hh_l0.detach().numpy()))
# tiny_model.RNN.cells[0].bias_ih.assign(Tensor(torch_model.rnn.bias_ih_l0.detach().numpy()))
# tiny_model.RNN.cells[0].bias_hh.assign(Tensor(torch_model.rnn.bias_hh_l0.detach().numpy()))
# tiny_model.proj.weight.assign(Tensor(torch_model.linear.weight.detach().numpy()))
# tiny_model.proj.bias.assign(Tensor(torch_model.linear.bias.detach().numpy()))

@tinygrad.Tensor.test()
def make_predictions_train():
    predictions = []
    predictions_tiny = []
    hidden_prev = None
    hidden_prev_tiny = None
    # we will go over all points in out training sequence
    for i in range(x_train_torch.shape[0]):
        input = x_train_torch[i]
        input = input.view(1, 1, 1)
        # we will give the current (single) point and the (current) 
        # hidden state as input to our model
        input_tiny = x_train_tiny[i]
        input_tiny = input_tiny.view(1, 1, 1) 
        # we carry over the previous hidden state
        pred, hidden_prev = torch_model(input, hidden_prev) 
        predictions.append(pred.data.numpy()[0][0])
        pred_tiny, hidden_prev_tiny = tiny_model(input_tiny, hidden_prev_tiny)
        predictions_tiny.append(pred_tiny.numpy()[0][0])

    return predictions, hidden_prev, predictions_tiny, hidden_prev_tiny

@tinygrad.Tensor.test()
def generate_unseen_sequence(length, starting_point, starting_point_tiny, hidden_state, hidden_state_tiny):
    predicts=[]
    predicts_tiny = []
    input = torch.Tensor(starting_point).view(1,1,1)
    input_tiny = tinygrad.Tensor(starting_point_tiny).view(1, 1, 1)
    for i in range(length):
        pred, hidden_state = torch_model(input, hidden_state)
        predicts.append(pred.data.numpy()[0][0])
        input = pred
        pred_tiny, hidden_state_tiny = tiny_model(input_tiny, hidden_state_tiny)
        predicts_tiny.append(pred_tiny.numpy()[0][0])
        input_tiny = pred_tiny
    return predicts, predicts_tiny
  
predictions_train, hidden_prev, predictions_train_tiny, hidden_prev_tiny = make_predictions_train()

generated_points, generated_points_tiny = generate_unseen_sequence(valid_len, starting_point=predictions_train[-1], starting_point_tiny=predictions_train_tiny[-1], hidden_state=hidden_prev, hidden_state_tiny=hidden_prev_tiny)
predictions = predictions_train+generated_points #concatenate two lists
predictions_tiny = predictions_train_tiny+generated_points_tiny

fig, (ax1, ax2) = plt.subplots(figsize=(20,10), nrows=2)
ax1.scatter(time_steps, data, s=90, label='actual')
ax1.scatter(time_steps[1:train_len], predictions[:train_len-1], label='predicted')
ax1.scatter(time_steps[train_len:], predictions[train_len-1:], label='generated')
ax1.legend()
ax2.scatter(time_steps, data, s=90, label='actual')
ax2.scatter(time_steps[1:train_len], predictions_tiny[:train_len-1], label='predicted')
ax2.scatter(time_steps[train_len:], predictions_tiny[train_len-1:], label='generated')
ax2.legend()
plt.savefig("fig.png")