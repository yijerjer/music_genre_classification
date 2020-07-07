import torch
import torch.nn as nn
import torch.nn.functional as functional


class ConvBlock(nn.Module):
    def __init__(self, type, in_channels, out_channels, kernel_size, padding=None):
        super(ConvBlock, self).__init__()
        self.type = type
        if not padding:
            padding = (int((kernel_size[0] - 1) / 2), int((kernel_size[1] - 1) / 2))
        if type == "basic":
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
            self.bn1 = nn.BatchNorm2d(out_channels)
        elif type == "resnet":
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
            self.bn2 = nn.BatchNorm2d(out_channels)
            self.conv_sc = nn.Conv2d(in_channels, out_channels, (1, 1))
        elif type == "densenet":
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.conv2 = nn.Conv2d(in_channels + out_channels, in_channels + out_channels, kernel_size, padding=padding)
            self.bn2 = nn.BatchNorm2d(in_channels + out_channels)
            self.conv3 = nn.Conv2d((in_channels + out_channels) * 2, out_channels, 1)
            self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, input):
        if self.type == "basic":
            output = functional.relu(self.bn1(self.conv1(input)))
        elif self.type == "resnet":
            output = functional.relu(self.bn1(self.conv1(input)))
            output = self.bn2(self.conv2(output))
            output += self.conv_sc(input)
            output = functional.relu(output)
        elif self.type == "densenet":
            output1 = functional.relu(self.bn1(self.conv1(input)))
            input2 = torch.cat((input, output1), dim=1)
            output2 = functional.relu(self.bn2(self.conv2(input2)))
            output = torch.cat((input, output1, output2), dim=1)
            output = functional.relu(self.bn3(self.conv3(output)))

        return output


# takes chroma input of 12 x 640
class ChromaCNNFreq(nn.Module):
    def __init__(self, add_name=""):
        super(ChromaCNNFreq, self).__init__()
        self.name = f"chroma_cnn_freq{add_name}"

        self.conv1 = ConvBlock("basic", 1, 16, (1, 5))
        self.conv2 = ConvBlock("basic", 16, 16, (1, 5))
        self.conv3 = ConvBlock("basic", 16, 32, (1, 5))
        self.conv4 = ConvBlock("basic", 32, 32, (1, 5))

        self.maxpool1 = nn.MaxPool2d((1, 4))
        self.maxpool2 = nn.MaxPool2d((1, 6))

        self.fc1 = nn.Linear(32 * 12, 256)
        self.fc2 = nn.Linear(256, 32)
        self.fc3 = nn.Linear(32, 8)

        self.dropout = nn.Dropout()

    def forward(self, input):
        input = input[:, None, :, :]
        output = self.maxpool1(self.conv1(input))
        output = self.maxpool1(self.conv2(output))
        output = self.maxpool1(self.conv3(output))
        output = self.maxpool2(self.conv4(output))

        output = output.view(-1, 32 * 12)
        output = functional.relu(self.fc1(self.dropout(output)))
        output = functional.relu(self.fc2(self.dropout(output)))
        output = self.fc3(self.dropout(output))
        return output


# takes chroma input of 12 x 640
class ChromaCNNTemporal(nn.Module):
    def __init__(self, add_name=""):
        super(ChromaCNNTemporal, self).__init__()
        self.name = f"chroma_cnn_temporal{add_name}"

        self.conv1 = ConvBlock("basic", 1, 64, (12, 5), padding=(0, 2))
        self.conv2 = ConvBlock("basic", 64, 128, (1, 5))
        self.conv3 = ConvBlock("basic", 128, 256, (1, 5))
        self.conv4 = ConvBlock("basic", 256, 512, (1, 5))

        self.maxpool1 = nn.MaxPool2d((1, 4))
        self.maxpool2 = nn.MaxPool2d((1, 6))

        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 32)
        self.fc3 = nn.Linear(32, 8)

        self.dropout = nn.Dropout()

    def forward(self, input):
        input = input[:, None, :, :]
        output = self.maxpool1(self.conv1(input))
        output = self.maxpool1(self.conv2(output))
        output = self.maxpool1(self.conv3(output))
        output = self.maxpool2(self.conv4(output))

        output = output.view(-1, 512)
        output = functional.relu(self.fc1(self.dropout(output)))
        output = functional.relu(self.fc2(self.dropout(output)))
        output = self.fc3(self.dropout(output))
        return output


# takes chroma input of 12 x 640
class ChromaCNNSquare(nn.Module):
    def __init__(self, add_name=""):
        super(ChromaCNNSquare, self).__init__()
        self.name = f"chroma_cnn_square{add_name}"

        self.conv1 = ConvBlock("basic", 1, 8, (3, 3))
        self.conv2 = ConvBlock("basic", 8, 32, (3, 3))
        self.conv3 = ConvBlock("basic", 32, 128, (3, 3))
        self.conv4 = ConvBlock("basic", 128, 256, (1, 5))

        self.maxpool1 = nn.MaxPool2d((2, 4))
        self.maxpool2 = nn.MaxPool2d((1, 6))

        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 8)

        self.dropout = nn.Dropout()
    
    def forward(self, input):
        input = input[:, None, :, :]
        output = self.maxpool1(self.conv1(input))
        output = self.maxpool1(self.conv2(output))
        output = self.maxpool1(self.conv3(output))
        output = self.maxpool2(self.conv4(output))
        
        output = output.view(-1, 256)
        output = functional.relu(self.fc1(self.dropout(output)))
        output = functional.relu(self.fc2(self.dropout(output)))
        output = self.fc3(self.dropout(output))

        return output


# takes mel input of 128 x 640
class MelCNNFreq(nn.Module):
    def __init__(self, add_name=""):
        super(MelCNNFreq, self).__init__()
        self.name = f"mel_cnn_freq{add_name}"

        self.conv1 = ConvBlock("basic", 1, 16, (1, 5))
        self.conv2 = ConvBlock("basic", 16, 16, (1, 5))
        self.conv3 = ConvBlock("basic", 16, 32, (1, 5))
        self.conv4 = ConvBlock("basic", 32, 32, (1, 5))

        self.maxpool1 = nn.MaxPool2d((1, 4))
        self.maxpool2 = nn.MaxPool2d((1, 6))

        self.fc1 = nn.Linear(128 * 32, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 32)
        self.fc4 = nn.Linear(32, 8)

        self.dropout = nn.Dropout()

    def forward(self, input):
        input = input[:, None, :, :]
        output = self.maxpool1(self.conv1(input))
        output = self.maxpool1(self.conv2(output))
        output = self.maxpool1(self.conv3(output))
        output = self.maxpool2(self.conv4(output))

        output = output.view(-1, 128 * 32)
        output = functional.relu(self.fc1(self.dropout(output)))
        output = functional.relu(self.fc2(self.dropout(output)))
        output = functional.relu(self.fc3(self.dropout(output)))
        output = self.fc4(self.dropout(output))

        return output


# takes mel input of 128 x 640
class MelCNNTemporal(nn.Module):
    def __init__(self, add_name=""):
        super(MelCNNTemporal, self).__init__()
        self.name = f"mel_cnn_temporal{add_name}"

        self.conv1 = ConvBlock("basic", 1, 64, (128, 5), padding=(0, 2))
        self.conv2 = ConvBlock("basic", 64, 128, (1, 5))
        self.conv3 = ConvBlock("basic", 128, 256, (1, 5))
        self.conv4 = ConvBlock("basic", 256, 512, (1, 5))

        self.maxpool1 = nn.MaxPool2d((1, 4))
        self.maxpool2 = nn.MaxPool2d((1, 6))

        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 32)
        self.fc3 = nn.Linear(32, 8)

        self.dropout = nn.Dropout()

    def forward(self, input):
        input = input[:, None, :, :]
        output = self.maxpool1(self.conv1(input))
        output = self.maxpool1(self.conv2(output))
        output = self.maxpool1(self.conv3(output))
        output = self.maxpool2(self.conv4(output))

        output = output.view(-1, 512)
        output = functional.relu(self.fc1(self.dropout(output)))
        output = functional.relu(self.fc2(self.dropout(output)))
        output = self.fc3(self.dropout(output))
        return output


# takes mel input of 128 x 640
class MelCNNSquare(nn.Module):
    def __init__(self, add_name="", for_crnn=False):
        super(MelCNNSquare, self).__init__()
        self.name = f"mel_cnn_square{add_name}"
        self.for_crnn = for_crnn

        self.conv1 = ConvBlock("basic", 1, 8, (3, 3))
        self.conv2 = ConvBlock("basic", 8, 32, (3, 3))
        self.conv3 = ConvBlock("basic", 32, 128, (3, 3))
        self.conv4 = ConvBlock("basic", 128, 256, (3, 3))

        self.maxpool1 = nn.MaxPool2d((2, 4))
        self.maxpool2 = nn.MaxPool2d((3, 4))
        self.maxpool3 = nn.MaxPool2d((6, 6))

        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 8)

        self.dropout = nn.Dropout()

    def forward(self, input):
        input = input[:, None, :, :]
        output = self.maxpool1(self.conv1(input))
        output = self.maxpool1(self.conv2(output))
        output = self.maxpool2(self.conv3(output))
        output = self.maxpool3(self.conv4(output))

        output = output.view(-1, 256)
        if self.for_crnn:
            return output 
        else:
            output = functional.relu(self.fc1(self.dropout(output)))
            output = functional.relu(self.fc2(self.dropout(output)))
            output = self.fc3(self.dropout(output))
            return output


# takes cqt input of 84 x 640
class CQTCNNFreq(nn.Module):
    def __init__(self, add_name="", conv_types=["basic"]*4, for_crnn=False):
        super(CQTCNNFreq, self).__init__()
        self.name = f"cqt_cnn_freq{add_name}"
        self.for_crnn = for_crnn

        self.conv1 = ConvBlock(conv_types[0], 1, 16, (1, 5))
        self.conv2 = ConvBlock(conv_types[1], 16, 16, (1, 5))
        self.conv3 = ConvBlock(conv_types[2], 16, 32, (1, 5))
        self.conv4 = ConvBlock(conv_types[3], 32, 32, (1, 5))

        self.maxpool1 = nn.MaxPool2d((1, 4))
        self.maxpool2 = nn.MaxPool2d((1, 6))

        self.fc1 = nn.Linear(84 * 32, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 32)
        self.fc4 = nn.Linear(32, 8)

        self.dropout = nn.Dropout()
    
    def forward(self, input):
        input = input[:, None, :, :]
        output = self.maxpool1(self.conv1(input))
        output = self.maxpool1(self.conv2(output))
        output = self.maxpool1(self.conv3(output))
        output = self.maxpool2(self.conv4(output))

        output = output.view(-1, 84 * 32)
        output = functional.relu(self.fc1(self.dropout(output)))
        if self.for_crnn:
            return output
        else:
            output = functional.relu(self.fc2(self.dropout(output)))
            output = functional.relu(self.fc3(self.dropout(output)))
            output = self.fc4(self.dropout(output))
            return output



# takes cqt input of 84 x 640
class CQTCNNTemporal(nn.Module):
    def __init__(self, add_name=""):
        super(CQTCNNTemporal, self).__init__()
        self.name = f"cqt_cnn_temporal{add_name}"

        self.conv1 = ConvBlock("basic", 1, 64, (84, 5), padding=(0, 2))
        self.conv2 = ConvBlock("basic", 64, 128, (1, 5))
        self.conv3 = ConvBlock("basic", 128, 256, (1, 5))
        self.conv4 = ConvBlock("basic", 256, 512, (1, 5))

        self.maxpool1 = nn.MaxPool2d((1, 4))
        self.maxpool2 = nn.MaxPool2d((1, 6))

        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 32)
        self.fc3 = nn.Linear(32, 8)

        self.dropout = nn.Dropout()

    def forward(self, input):
        input = input[:, None, :, :]
        output = self.maxpool1(self.conv1(input))
        output = self.maxpool1(self.conv2(output))
        output = self.maxpool1(self.conv3(output))
        output = self.maxpool2(self.conv4(output))

        output = output.view(-1, 512)
        output = functional.relu(self.fc1(self.dropout(output)))
        output = functional.relu(self.fc2(self.dropout(output)))
        output = self.fc3(self.dropout(output))
        return output


# takes cqt input of 84 x 640
class CQTCNNSquare(nn.Module):
    def __init__(self, add_name="", for_crnn=False):
        super(CQTCNNSquare, self).__init__()
        self.name = f"cqt_cnn_square{add_name}"
        self.for_crnn = for_crnn

        self.conv1 = ConvBlock("basic", 1, 8, (3, 3))
        self.conv2 = ConvBlock("basic", 8, 32, (3, 3))
        self.conv3 = ConvBlock("basic", 32, 128, (3, 3))
        self.conv4 = ConvBlock("basic", 128, 256, (3, 3))

        self.maxpool1 = nn.MaxPool2d((2, 4))
        self.maxpool2 = nn.MaxPool2d((3, 4))
        self.maxpool3 = nn.MaxPool2d((5, 6))

        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 8)
    
        self.dropout = nn.Dropout()

    def forward(self, input):
        input = input[:, None, :, :]
        output = self.maxpool1(self.conv1(input))
        output = self.maxpool1(self.conv2(output))
        output = self.maxpool2(self.conv3(output))
        output = self.maxpool3(self.conv4(output))

        output = output.view(-1, 256)
        if self.for_crnn:
            return output
        else:
            output = functional.relu(self.fc1(self.dropout(output)))
            output = functional.relu(self.fc2(self.dropout(output)))
            output = self.fc3(self.dropout(output))
            return output