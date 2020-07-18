import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import RandLinear


def irnn_initializer(m: nn.Module):
    nn.init.eye_(m.weight.data)
    nn.init.zeros_(m.bias.data)

def gaussian_initializer(m: nn.Module):
    nn.init.normal_(m.weight.data, 0, 0.001)
    nn.init.zeros_(m.bias.data)


class MNISTIRNN(nn.Module):
    """
    IRNN where the only output that is returned is the logits for a distribution over classes.
    """
    kCompatibleDataset = 'mnist'

    def __init__(self, hidden_size, num_classes=10, rp_args=None):
        super(MNISTIRNN, self).__init__()
        rp_args = {} if rp_args is None else rp_args
        kept_keys = ['keep_frac', 'full_random', 'sparse']
        kept_dict = {key: rp_args[key] for key in kept_keys}

        self._hidden_size = hidden_size

        if kept_dict['keep_frac'] == 1.:
            self.i2h = nn.Linear(1, hidden_size)
            self.h2h = nn.Linear(hidden_size, hidden_size)
            self.h2o = nn.Linear(hidden_size, num_classes)
            print("Using nn.Linear layers")
        else:
            self.i2h = RandLinear(1, hidden_size, **kept_dict)
            self.h2h = RandLinear(hidden_size, hidden_size, **kept_dict)
            self.h2o = RandLinear(hidden_size, num_classes, **kept_dict)
            print("Using RandLinear layers")

        self.h2h.apply(irnn_initializer)
        self.i2h.apply(gaussian_initializer)
        self.h2o.apply(gaussian_initializer)

    def forward(self, inputs):
        batch_size = inputs.shape[0]
        hidden = self._init_hidden(batch_size)
        inputs = inputs.view(batch_size, -1, 1).permute(1, 0, 2)   # (781, batch_size, 1)
        for t in range(inputs.shape[0]):
            hidden_part = self.h2h(hidden)
            input_part = self.i2h(inputs[t])
            hidden = F.relu(hidden_part + input_part)
        output_logits = F.log_softmax(self.h2o(hidden), dim=1)
        return output_logits

    def _init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        return weight.new(batch_size, self._hidden_size).zero_()


class MNISTIRNNPyTorch(nn.RNN, nn.Module):
    """
    Implementation of MNISTIRNN that uses the torch.nn.RNN implementation. Can't be used for Rand experiments.
    """

    kCompatibleDataset = 'mnist'

    def __init__(self, hidden_size, *args, **kwargs):
        super().__init__(input_size=1, hidden_size=hidden_size, num_layers=1, nonlinearity='relu', *args, **kwargs)

        self._hidden_size = hidden_size

        self.h2o = nn.Linear(hidden_size, 10)
        self.h2o.apply(gaussian_initializer)

        # Initialise parameters. We assume single layer for now
        nn.init.normal_(self.weight_ih_l0, 0., 0.001)
        nn.init.zeros_(self.bias_ih_l0)

        nn.init.eye_(self.weight_hh_l0)
        nn.init.zeros_(self.bias_hh_l0)

    def forward(self, inputs, hx=None):
        batch_size = inputs.shape[0]
        hidden = self._init_hidden(batch_size)
        inputs = inputs.view(batch_size, -1, 1).permute(1, 0, 2)  # (781, batch_size, 1)
        outputs, hidden = super().forward(inputs, hidden)
        output_logits = F.log_softmax(self.h2o(outputs[-1]), dim=1)
        return output_logits

    def _init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        return weight.new(1, batch_size, self._hidden_size).zero_()


class IRNN(nn.Module):
    """
    Implementation of IRNN from "A Simple Way to Initialize Recurrent Networks of Rectified Linear Units" by Le. et al,
    2015.

    It works just like a basic RNN but with the weights initialised to identity and biases to zero in addition to using
    a ReLU activation function instead of the typical Tanh.
    """

    def __init__(self, input_size, hidden_size):
        super(IRNN, self).__init__()

        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.h2h.apply(irnn_initializer)

        #self.apply(irnn_initializer)

    def forward(self, inputs, hidden):
        hidden = hidden[0]
        outputs = []
        for i in range(inputs.shape[0]):
            #combined = torch.cat((inputs[i], hidden), dim=1)
            hidden_part = self.h2h(hidden)
            input_part = self.i2h(inputs[i])
            hidden = F.relu(hidden_part + input_part)
            # output = self.i2o(combined)
            output = hidden
            outputs.append(output)
        outputs = torch.stack(outputs)
        return outputs, hidden.view(1, hidden.size(0), hidden.size(1))


class RandIRNN(nn.Module):
    """
    Implementation of random projection version of IRNN
    """

    def __init__(self, input_size, hidden_size, keep_frac=0.9, full_random=False, sparse=False):
        super(RandIRNN, self).__init__()

        self.i2h = RandLinear(
            input_size, hidden_size, keep_frac=keep_frac, full_random=full_random, sparse=sparse
        )
        self.h2h = RandLinear(
            hidden_size, hidden_size, keep_frac=keep_frac, full_random=full_random, sparse=sparse
        )
        self.h2h.apply(irnn_initializer)

        #self.apply(irnn_initializer)

    def forward(self, inputs, hidden):

        hidden = hidden[0]
        outputs = []
        for i in range(inputs.shape[0]):
            #combined = torch.cat((inputs[i], hidden), dim=1)
            hidden_part = self.h2h(hidden)
            input_part = self.i2h(inputs[i])
            hidden = F.relu(hidden_part + input_part)
            # output = self.i2o(combined)
            output = hidden
            outputs.append(output)
        outputs = torch.stack(outputs)
        return outputs, hidden.view(1, hidden.size(0), hidden.size(1))

