import torch.nn as nn

class RedisDNN(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RedisDNN, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.fc1 = nn.Linear(self.in_channel,16)
        self.fc2 = nn.Linear(16,4)
        self.fc3 = nn.Linear(4,self.out_channel)
        self.relu = nn.ReLU()

    def forward(self,X):
        X = X.float()
        hidden1 = self.relu(self.fc1(X))
        hidden2 = self.relu(self.fc2(hidden1))
        hidden3 = self.fc3(hidden2)
        return hidden3

class RedisDoubleDNN(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RedisDNN, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.fc1 = nn.Linear(self.in_channel,16)
        self.fc2 = nn.Linear(16,4)
        self.out1 = nn.Linear(4,1)
        self.out2 = nn.Linear(4,1)
        self.relu = nn.ReLU()

    def forward(self,X):
        X = X.float()
        hidden1 = self.relu(self.fc1(X))
        hidden2 = self.relu(self.fc2(hidden1))
        output1 = self.out1(hidden2)
        output2 = self.out2(hidden2)
        return output1, output2