import torch
import torch.nn as nn
import torch.nn.functional as F


class Inspector(nn.Module):

    def __init__(self, input_channel: int):
        self.conv1 = nn.Conv2d(input_channel, 64, 5, 2)
        self.conv2 = nn.Conv2d(64, 128, 3, 1)
        self.conv3 = nn.Conv2d(64, 128, 5, 1)
        self.conv4 = nn.Conv2d(128, 256, 5, 1)
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(256 * 5 * 5, 128)
        self.dense2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = F.relu(x)

        x = self.conv4(x)
        x = F.relu(x)

        x = self.flatten(x)
        x = self.dense1(x)
        x = F.relu(x)
        x = self.dense2(x)
        x = torch.sigmoid(x)

        return x
