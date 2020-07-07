import torch
import torch.nn as nn
import torch.nn.functional as functional


class InceptionBlock(nn.Module):
    def __init__(self, in_channels, stream_channels):
        super(InceptionBlock, self).__init__()

        self.maxpool1_1 = nn.MaxPool2d(3, stride=1, padding=1)
        self.conv2_1 = nn.Conv2d(in_channels, stream_channels, 1)
        self.conv3_1 = nn.Conv2d(in_channels, stream_channels, 1)
        self.conv4_1 = nn.Conv2d(in_channels, stream_channels, 1)
        self.conv1_2 = nn.Conv2d(in_channels, stream_channels, 1)
        self.conv2_2 = nn.Conv2d(stream_channels, stream_channels, 3, padding=1)
        self.conv3_2 = nn.Conv2d(stream_channels, stream_channels, 5, padding=2)

        self.bn2_1 = nn.BatchNorm2d(in_channels)
        self.bn3_1 = nn.BatchNorm2d(in_channels)
        self.bn4_1 = nn.BatchNorm2d(in_channels)
        self.bn1_2 = nn.BatchNorm2d(in_channels)
        self.bn2_2 = nn.BatchNorm2d(stream_channels)
        self.bn3_2 = nn.BatchNorm2d(stream_channels)

    def forward(self, input):
        output_1 = self.maxpool1_1(input)
        output_1 = self.conv1_2(functional.relu(self.bn1_2(output_1)))
        output_2 = self.conv2_1(functional.relu(self.bn2_1(input)))
        output_2 = self.conv2_2(functional.relu(self.bn2_2(output_2)))
        output_3 = self.conv3_1(functional.relu(self.bn3_1(input)))
        output_3 = self.conv3_2(functional.relu(self.bn3_2(output_3)))
        output_4 = self.conv4_1(functional.relu(self.bn4_1(input)))

        output = torch.cat((input, output_1, output_2, output_3, output_4), dim=1)
        del output_1, output_2, output_3, output_4

        return output


class CQTBBNN(nn.Module):
    def __init__(self, add_name=""):
        super(CQTBBNN, self).__init__()
        self.name = f"cqt_bbnn{add_name}"

        # Initial Block
        self.conv1 = nn.Conv2d(1, 16, 3)
        self.maxpool1 = nn.MaxPool2d((1, 4))
        self.bnI = nn.BatchNorm2d(16)

        self.inception1 = InceptionBlock(16, 16)
        self.inception2 = InceptionBlock(80, 16)
        self.inception3 = InceptionBlock(144, 16)

        # Transition Block
        self.bnT = nn.BatchNorm2d(208)
        self.convT = nn.Conv2d(208, 32, 1)
        self.avgpoolT = nn.AvgPool2d((2, 2))

        # Final Block
        self.bnF = nn.BatchNorm2d(32)
        self.avgpoolF = nn.AvgPool2d((41, 79))
        self.dropout = nn.Dropout()
        self.fc = nn.Linear(32, 8)

    def forward(self, input):
        input = input[:, None, :, :]

        # Initial Block
        output = functional.relu(self.bnI(self.conv1(input)))
        output = self.maxpool1(output)

        # Inception Blocks
        output = self.inception1(output)
        output = self.inception2(output)
        output = self.inception3(output)

        # Transition Block
        output = self.convT(functional.relu(self.bnT(output)))
        output = self.avgpoolT(output)

        # Final Block
        output = functional.relu(self.bnF(output))
        output = self.avgpoolF(output)
        output = output.view(-1, 32)
        output = self.fc(self.dropout(output))

        return output
