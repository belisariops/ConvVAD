import torch.nn as neural_network
import torch.nn.functional as funtional


class NeuralNetwork(neural_network.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.conv1 = neural_network.Conv2d(in_channels=1, out_channels=20, kernel_size=6)
        self.conv2 = neural_network.Conv2d(in_channels=20, out_channels=15, kernel_size=4)
        self.conv3 = neural_network.Conv2d(in_channels=15, out_channels=10, kernel_size=4)
        self.conv4 = neural_network.Conv2d(in_channels=10, out_channels=5, kernel_size=2)
        self.conv5 = neural_network.Conv2d(in_channels=5, out_channels=3, kernel_size=2)
        self.conv5_bn = neural_network.BatchNorm2d(3)
        self.pool = neural_network.MaxPool2d(2,2)
        self.fc1 = neural_network.Linear(3*4*6, 30)
        self.fc2 = neural_network.Linear(30, 13)
        self.dense1_bn = neural_network.BatchNorm1d(13)
        self.fc3 = neural_network.Linear(13, 2)

    def forward(self, x):
        # Arquitectura de red
        x = self.pool(funtional.relu(self.conv1(x)))
        x = self.pool(funtional.relu(self.conv2(x)))
        x = self.pool(funtional.relu(self.conv3(x)))
        x = self.pool(funtional.relu(self.conv4(x)))
        x = self.pool(self.conv5_bn(funtional.relu(self.conv5(x))))
        x = x.view(-1, 3*4*6)
        x = funtional.relu(self.fc1(x))
        x = funtional.relu(self.dense1_bn(self.fc2(x)))
        x = self.fc3(x)
        return x
