import torch.nn as nn

class RedisSingleDNN(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RedisSingleDNN, self).__init__()
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

class RedisTwiceDNN(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RedisTwiceDNN, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.fc1 = nn.Linear(self.in_channel,16)
        self.fc2 = nn.Linear(16,4)
        self.outs = []
        for _ in range(self.out_channel):
            self.outs.append(nn.Linear(4,1))
        self.relu = nn.ReLU()

    def forward(self,X):
        X = X.float()
        hidden1 = self.relu(self.fc1(X))
        hidden2 = self.relu(self.fc2(hidden1))
        outputs = []
        for output in self.outs:
            outputs.append(output(hidden2))
        return outputs