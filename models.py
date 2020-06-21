import torch
import torch.nn as nn
import torch.nn.functional as functional


# takes chroma input of 12 x 640
class ChromaCNNFreq(nn.Module):
    def __init__(self, add_name=""):
        super(ChromaCNNFreq, self).__init__()
        self.name = f"chroma_cnn_freq{add_name}"

        self.conv1 = nn.Conv1d(12, 12, 5)
        self.conv2 = nn.Conv1d(12, 12, 4)
        self.conv3 = nn.Conv1d(12, 12, 4)
        self.conv4 = nn.Conv1d(12, 12, 4)
        self.maxpool1 = nn.MaxPool1d(4)
        self.maxpool2 = nn.MaxPool1d(6)

        self.fc1 = nn.Linear(12, 26)
        self.fc2 = nn.Linear(26, 8)

    def forward(self, input):
        output = self.maxpool1(functional.relu(self.conv1(input)))
        output = self.maxpool1(functional.relu(self.conv2(output)))
        output = self.maxpool1(functional.relu(self.conv2(output)))
        output = self.maxpool2(functional.relu(self.conv2(output)))

        output = output.view(-1, 12)
        output = functional.relu(self.fc1(output))
        output = self.fc2(output)

        return output


# takes chroma input of 12 x 640
class ChromaCNNFreqFilters(nn.Module):
    def __init__(self, add_name=""):
        super(ChromaCNNFreqFilters, self).__init__()
        self.name = f"chroma_cnn_freq_filters{add_name}"

        self.conv1 = nn.Conv1d(12, 12 * 4, 5)
        self.conv2 = nn.Conv1d(12 * 4, 12 * 4, 4)
        self.conv3 = nn.Conv1d(12 * 4, 12 * 8, 4)
        self.conv4 = nn.Conv1d(12 * 8, 12 * 8, 4)
        self.maxpool1 = nn.MaxPool1d(4)
        self.maxpool2 = nn.MaxPool1d(6)

        self.fc1 = nn.Linear(12 * 8, 35)
        self.fc2 = nn.Linear(35, 8)

    def forward(self, input):
        output = self.maxpool1(functional.relu(self.conv1(input)))
        output = self.maxpool1(functional.relu(self.conv2(output)))
        output = self.maxpool1(functional.relu(self.conv3(output)))
        output = self.maxpool2(functional.relu(self.conv4(output)))

        output = output.view(-1, 96)
        output = functional.relu(self.fc1(output))
        output = self.fc2(output)
        return output


# takes chroma input of 12 x 640
class ChromaCNNTemporal(nn.Module):
    def __init__(self, add_name=""):
        super(ChromaCNNTemporal, self).__init__()
        self.name = f"chroma_cnn_temporal{add_name}"

        self.conv1 = nn.Conv2d(1, 4, (12, 5))
        self.conv2 = nn.Conv1d(4, 16, 4)
        self.conv3 = nn.Conv1d(16, 64, 4)
        self.conv4 = nn.Conv1d(64, 256, 4)

        self.maxpool1 = nn.MaxPool2d((1, 4))
        self.maxpool2 = nn.MaxPool1d(4)
        self.maxpool3 = nn.MaxPool1d(6)

        self.fc1 = nn.Linear(256, 110)
        self.fc2 = nn.Linear(110, 36)
        self.fc3 = nn.Linear(36, 8)

    def forward(self, input):
        output = input[:, None, :, :]
        output = self.maxpool1(functional.relu(self.conv1(output)))
        output = torch.squeeze(output, dim=2)
        output = self.maxpool2(functional.relu(self.conv2(output)))
        output = self.maxpool2(functional.relu(self.conv3(output)))
        output = self.maxpool3(functional.relu(self.conv4(output)))

        output = output.view(-1, 256)
        output = functional.relu(self.fc1(output))
        output = functional.relu(self.fc2(output))
        output = self.fc3(output)
        return output


# takes chroma input of 12 x 640
class ChromaCNNSquare(nn.Module):
    def __init__(self, add_name=""):
        super(ChromaCNNSquare, self).__init__()
        self.name = f"chroma_cnn_square{add_name}"

        self.conv1 = nn.Conv2d(1, 4, 5)
        self.conv2 = nn.Conv2d(4, 16, 4)
        self.conv3 = nn.Conv2d(16, 64, 4)
        self.conv4 = nn.Conv2d(64, 256, (2, 4))

        self.maxpool1 = nn.MaxPool2d((1, 4))
        self.maxpool2 = nn.MaxPool2d((1, 4))
        self.maxpool3 = nn.MaxPool2d((1, 6))

        self.fc1 = nn.Linear(256, 110)
        self.fc2 = nn.Linear(110, 36)
        self.fc3 = nn.Linear(36, 8)
    
    def forward(self, input):
        input = input[:, None, :, :]

        output = self.maxpool1(functional.relu(self.conv1(input)))
        output = self.maxpool1(functional.relu(self.conv2(output)))
        output = self.maxpool2(functional.relu(self.conv3(output)))
        output = self.maxpool3(functional.relu(self.conv4(output)))
        
        output = torch.squeeze(output, dim=2)
        output = output.view(-1, 256)
        output = functional.relu(self.fc1(output))
        output = functional.relu(self.fc2(output))
        output = self.fc3(output)

        return output


# takes chroma input of 12 x 640
class ChromaCNNRectangle(nn.Module):
    def __init__(self, add_name=""):
        super(ChromaCNNRectangle, self).__init__()
        self.name = f"chroma_cnn_rectangle{add_name}"

        self.conv1 = nn.Conv2d(1, 4, 3)
        self.conv2 = nn.Conv2d(4, 16, 3)
        self.conv3 = nn.Conv2d(16, 64, 3)
        self.conv4 = nn.Conv2d(64, 256, (1, 3))

        self.maxpool1 = nn.MaxPool2d((2, 4))
        self.maxpool2 = nn.MaxPool2d((1, 4))
        self.maxpool3 = nn.MaxPool2d((1, 6))

        self.fc1 = nn.Linear(256, 110)
        self.fc2 = nn.Linear(110, 36)
        self.fc3 = nn.Linear(36, 8)
    
    def forward(self, input):
        input = input[:, None, :, :]

        output = self.maxpool1(functional.relu(self.conv1(input)))
        output = self.maxpool2(functional.relu(self.conv2(output)))
        output = self.maxpool2(functional.relu(self.conv3(output)))
        output = self.maxpool3(functional.relu(self.conv4(output)))
        
        output = output.view(-1, 256)
        output = functional.relu(self.fc1(output))
        output = functional.relu(self.fc2(output))
        output = self.fc3(output)

        return output


# takes chroma input of 12 x 640
class ChromaCNNRectangle2d(nn.Module):
    def __init__(self, add_name=""):
        super(ChromaCNNRectangle2d, self).__init__()
        self.name = f"chroma_cnn_rectangle2d{add_name}"

        self.conv1 = nn.Conv2d(1, 4, 3)
        self.conv2 = nn.Conv2d(4, 16, 3)
        self.conv3 = nn.Conv2d(16, 64, 3)

        self.maxpool1 = nn.MaxPool2d((2, 4))
        self.maxpool2 = nn.MaxPool2d((1, 4))

        self.fc1 = nn.Linear(9 * 64, 110)
        self.fc2 = nn.Linear(110, 36)
        self.fc3 = nn.Linear(36, 8)
    
    def forward(self, input):
        input = input[:, None, :, :]

        output = self.maxpool1(functional.relu(self.conv1(input)))
        output = self.maxpool2(functional.relu(self.conv2(output)))
        output = self.maxpool2(functional.relu(self.conv3(output)))
        
        output = output.view(-1, 9 * 64)
        output = functional.relu(self.fc1(output))
        output = functional.relu(self.fc2(output))
        output = self.fc3(output)

        return output


# takes chroma input of 128 x 640
class MelCNNFreq(nn.Module):
    def __init__(self, add_name=""):
        super(MelCNNFreq, self).__init__()
        self.name = f"mel_cnn_filter{add_name}s"

        self.conv1 = nn.Conv1d(128, 128 * 4, 5)
        self.conv2 = nn.Conv1d(128 * 4, 128 * 4, 4)
        self.conv3 = nn.Conv1d(128 * 4, 128 * 8, 4)
        self.conv4 = nn.Conv1d(128 * 8, 128 * 8, 4)

        self.maxpool1 = nn.MaxPool1d(4)
        self.maxpool2 = nn.MaxPool1d(6)

        self.fc1 = nn.Linear(128 * 8, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 8)

    def forward(self, input):
        output = self.maxpool1(functional.relu(self.conv1(input)))
        output = self.maxpool1(functional.relu(self.conv2(output)))
        output = self.maxpool1(functional.relu(self.conv3(output)))
        output = self.maxpool2(functional.relu(self.conv4(output)))

        output = output.view(-1, 128 * 8)
        output = functional.relu(self.fc1(output))
        output = functional.relu(self.fc2(output))
        output = self.fc3(output)
        return output


# takes chroma input of 128 x 640
class MelCNNTemporal(nn.Module):
    def __init__(self, add_name=""):
        super(MelCNNTemporal, self).__init__()
        self.name = f"mel_cnn_temporal{add_name}"

        self.conv1 = nn.Conv2d(1, 4, (128, 5))
        self.conv2 = nn.Conv1d(4, 16, 4)
        self.conv3 = nn.Conv1d(16, 64, 4)
        self.conv4 = nn.Conv1d(64, 256, 4)

        self.maxpool1 = nn.MaxPool2d((1, 4))
        self.maxpool2 = nn.MaxPool1d(4)
        self.maxpool3 = nn.MaxPool1d(6)

        self.fc1 = nn.Linear(256, 110)
        self.fc2 = nn.Linear(110, 36)
        self.fc3 = nn.Linear(36, 8)

    def forward(self, input):
        output = self.maxpool1(functional.relu(self.conv1(input)))
        output = torch.squeeze(output, dim=2)
        output = self.maxpool2(functional.relu(self.conv2(output)))
        output = self.maxpool2(functional.relu(self.conv3(output)))
        output = self.maxpool3(functional.relu(self.conv4(output)))

        output = output.view(-1, 256)
        output = functional.relu(self.fc1(output))
        output = functional.relu(self.fc2(output))
        output = self.fc3(output)
        return output


# takes chroma input of 128 x 640
class MelCNNSquare(nn.Module):
    def __init__(self, add_name=""):
        self.name = f"mel_cnn_square{add_name}"

        self.conv1 = nn.Conv2d(1, 8, 3)
        self.conv2 = nn.Conv2d(8, 32, 3)
        self.conv3 = nn.Conv2d(32, 128, 3)
        self.conv4 = nn.Conv2d(128, 256, 3)

        self.maxpool1 = nn.MaxPool2d((2, 4))
        self.maxpool2 = nn.MaxPool2d((3, 4))
        self.maxpool3 = nn.MaxPool2d((4, 4))

        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 8)

    def forward(self, input):
        output = self.maxpool1(functional.relu(self.conv1(input)))
        output = self.maxpool2(functional.relu(self.conv2(output)))
        output = self.maxpool3(functional.relu(self.conv3(output)))
        output = self.maxpool4(functional.relu(self.conv4(output)))

        output = output.view(-1, 256)
        output = functional.relu(self.fc1(output))
        output = functional.relu(self.fc2(output))
        output = self.fc3(output)
        return output


