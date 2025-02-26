import torch.nn as nn
import torch

# Relu
class NN1(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, act_fun=nn.ReLU(), dropout=0):
        super(NN1, self).__init__()

        self.input_size = input_size
        self.hidden_size_1 = hidden_size[0]
        self.hidden_size_2 = hidden_size[1]
        self.hidden_size_3 = hidden_size[2]
        self.hidden_size_4 = hidden_size[3]
        self.output_size = output_size

        self.act = act_fun
        self.i2h1 = nn.Linear(self.input_size, self.hidden_size_1)
        self.act = act_fun
        self.h2h2 = nn.Linear(self.hidden_size_1, self.hidden_size_2)
        self.act = act_fun
        self.h2h3 = nn.Linear(self.hidden_size_2, self.hidden_size_3)
        self.act = act_fun
        self.h2h4 = nn.Linear(self.hidden_size_3, self.hidden_size_4)
        self.dropout = nn.Dropout(dropout)
        self.h2o = nn.Linear(self.hidden_size_4, self.output_size)

    def forward(self, x):
        x = self.act(self.i2h1(x))
        x = self.dropout(x)
        x = self.act(self.h2h2(x))
        x = self.dropout(x)
        x = self.act(self.h2h3(x))
        x = self.dropout(x)
        x = self.act(self.h2h4(x))
        x = self.dropout(x)
        outputs = self.h2o(x)

        return (outputs)

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0)

class NN(nn.Module):
    def __init__(self, input_size, hidden_layer, output_size, act_fun=nn.ReLU(), dropout=0):
        super(NN, self).__init__()
        self.use_softmax = False
        self.softmax = torch.nn.Softmax(dim=1)

        self.input_size = input_size
        self.output_size = output_size

        self.layer_input = nn.Linear(input_size, hidden_layer[0])
        self.layer_output = nn.Linear(hidden_layer[-1], output_size)
        self.layer_hidden = []
        for h in range(len(hidden_layer)-1):
            self.layer_hidden.append(nn.Linear(hidden_layer[h], hidden_layer[h+1]))

        self.act = act_fun
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer_input(x)

        for l in self.layer_hidden:
            x = l(x)
            x = self.act(x)
            x = self.dropout(x)

        outputs = self.layer_output(x)
        if self.use_softmax:
            outputs = self.softmax(outputs)

        return (outputs)

    def init_weights(self):
        for l in self.layer_hidden:
            torch.nn.init.xavier_uniform_(l.weight)
            l.bias.data.fill_(0)

        torch.nn.init.xavier_uniform_(self.layer_input.weight)
        l.bias.data.fill_(0)

        torch.nn.init.xavier_uniform_(self.layer_output.weight)
        l.bias.data.fill_(0)

    def get_var_trainable(self):
        var_trainable = [self.layer_input.weight, self.layer_input.bias, self.layer_output.weight, self.layer_output.bias]
        for h in self.layer_hidden:
            var_trainable.append(h.weight)
            var_trainable.append(h.bias)
        return var_trainable