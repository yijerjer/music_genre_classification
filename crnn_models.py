import torch
import torch.nn as nn
import torch.nn.functional as functional
from cnn_models import ConvBlock, MelCNNSquare, CQTCNNSquare


class CQTCNNSquareForCRNN(nn.Module):
    def __init__(self):
        super(CQTCNNSquareForCRNN, self).__init__()
        self.conv1 = ConvBlock("basic", 1, 8, (3, 3))
        self.conv2 = ConvBlock("basic", 8, 32, (3, 3))
        self.conv3 = ConvBlock("basic", 32, 128, (3, 3))
        self.conv4 = ConvBlock("basic", 128, 256, (3, 3))

        self.maxpool1 = nn.MaxPool2d((2, 2))
        self.maxpool2 = nn.MaxPool2d((3, 2))
        self.maxpool3 = nn.MaxPool2d((4, 4))

    def forward(self, input):
        input = input[:, None, :, :]
        output = self.maxpool1(self.conv1(input))
        output = self.maxpool1(self.conv2(output))
        output = self.maxpool2(self.conv3(output))
        output = self.maxpool3(self.conv4(output))

        output = output.squeeze(2)
        output = output.transpose(1, 2)
        return output


class CQTCNNFreqForCRNN(nn.Module):
    def __init__(self):
        super(CQTCNNFreqForCRNN, self).__init__()
        self.conv1 = ConvBlock("basic", 1, 16, (1, 5))
        self.conv2 = ConvBlock("basic", 16, 16, (1, 5))
        self.conv3 = ConvBlock("basic", 16, 32, (1, 5))
        self.conv4 = ConvBlock("basic", 32, 32, (1, 5))

        self.maxpool1 = nn.MaxPool2d((1, 2))
        self.maxpool2 = nn.MaxPool2d((1, 4))
    
    def forward(self, input):
        input = input[:, None, :, :]
        output = self.maxpool1(self.conv1(input))
        output = self.maxpool1(self.conv2(output))
        output = self.maxpool1(self.conv3(output))
        output = self.maxpool2(self.conv4(output))

        output = output.view(-1, output.shape[1] * output.shape[2], output.shape[3])
        output = output.transpose(1, 2)
        return output


class CQTCRNN(nn.Module):
    def __init__(self, cnn, cnn_output_size, add_name="", rnn_type="lstm", bidirectional=False, num_layers=1, hidden_size=128):
        super(CQTCRNN, self).__init__()
        self.name = f"cqt_crnn_{rnn_type}{'_bi' if bidirectional else ''}{add_name}"
        self.bidirectional, self.rnn_type, self.num_layers  = bidirectional, rnn_type, num_layers
        self.hidden_size = hidden_size
        self.rnn_output_size = (self.hidden_size * 2) if bidirectional else self.hidden_size
        self.cnn_output_size = cnn_output_size

        self.cnn = cnn
        if rnn_type == "lstm":
            self.rnn = nn.LSTM(self.cnn_output_size, self.hidden_size, batch_first=True, bidirectional=bidirectional, num_layers=num_layers)
        elif rnn_type == "gru":
            self.rnn = nn.GRU(self.cnn_output_size, self.hidden_size, batch_first=True, bidirectional=bidirectional, num_layers=num_layers)

        self.fc1 = nn.Linear(self.rnn_output_size, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 8)

        self.dropout = nn.Dropout()

    def forward(self, input, hidden):
        output = self.cnn(input)
        output, hidden = self.rnn(output, hidden)

        if self.bidirectional:
            forward_output = output[:, -1, :self.hidden_size]
            backward_output = output[:, 0, self.hidden_size:]
            output = torch.cat((forward_output, backward_output), dim=1)
        else:
            output = output[:, -1, :]

        output = output.view(-1, self.rnn_output_size)
        output = functional.relu(self.fc1(self.dropout(output)))
        output = functional.relu(self.fc2(self.dropout(output)))
        output = self.fc3(self.dropout(output))
        return output

    def init_hidden(self, batch_size, cuda=False):
        device = torch.device("cuda:0" if cuda else "cpu")
        num_hidden = (2 if self.bidirectional else 1) * self.num_layers
        cell_state = torch.zeros(num_hidden, batch_size, self.hidden_size).to(device)
        hidden_state = torch.zeros(num_hidden, batch_size, self.hidden_size).to(device)

        if self.rnn_type == "lstm":
            return (hidden_state, cell_state)
        elif self.rnn_type == "gru":
            return hidden_state


class CQTCRNNParallel(nn.Module):
    def __init__(self, cnn, cnn_output_size, add_name="", rnn_type="lstm", bidirectional=True, num_layers=1, hidden_size=128):
        super(CQTCRNNParallel, self).__init__()
        self.name = f"cqt_crnn_parallel_{rnn_type}{add_name}"
        self.bidirectional, self.rnn_type, self.num_layers = bidirectional, rnn_type, num_layers
        self.hidden_size = hidden_size
        self.rnn_output_size = (self.hidden_size * 2) if bidirectional else self.hidden_size
        self.cnn_output_size = cnn_output_size

        self.cnn = cnn

        self.avgpool1 = nn.AvgPool1d(20)
        if rnn_type == "lstm":
            self.rnn = nn.LSTM(84, self.hidden_size, batch_first=True, bidirectional=bidirectional, num_layers=num_layers)
        elif rnn_type == "gru":
            self.rnn = nn.GRU(84, self.hidden_size, batch_first=True, bidirectional=bidirectional, num_layers=num_layers)

        self.fc1 = nn.Linear(self.rnn_output_size + self.cnn_output_size, 256)
        self.fc2 = nn.Linear(256, 32)
        self.fc3 = nn.Linear(32, 8)

        self.dropout = nn.Dropout()

    def forward(self, input, hidden):
        cnn_output = self.cnn(input)

        rnn_input = self.avgpool1(input)
        rnn_input = rnn_input.transpose(1, 2)
        rnn_output, hidden = self.rnn(rnn_input, hidden)
        if self.bidirectional:
            forward_output = rnn_output[:, -1, :self.hidden_size]
            backward_output = rnn_output[:, 0, self.hidden_size:]
            rnn_output = torch.cat((forward_output, backward_output), dim=1)
        else:
            rnn_output = rnn_output[:, -1, :]
        rnn_output = rnn_output.view(-1, self.rnn_output_size)

        output = torch.cat((cnn_output, rnn_output), dim=1)
        output = functional.relu(self.fc1(self.dropout(output)))
        output = functional.relu(self.fc2(self.dropout(output)))
        output = self.fc3(self.dropout(output))
        return output

    def init_hidden(self, batch_size, cuda=False):
        device = torch.device("cuda:0" if cuda else "cpu")
        num_hidden = (2 if self.bidirectional else 1) * self.num_layers
        cell_state = torch.zeros(num_hidden, batch_size, self.hidden_size).to(device)
        hidden_state = torch.zeros(num_hidden, batch_size, self.hidden_size).to(device)

        if self.rnn_type == "lstm":
            return (hidden_state, cell_state)
        elif self.rnn_type == "gru":
            return hidden_state
