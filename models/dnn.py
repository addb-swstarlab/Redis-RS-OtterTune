import torch.nn as nn

class RedisSingleDNN(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RedisSingleDNN, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.fc1 = nn.Linear(self.in_channel,32)
        self.fc1_1 = nn.Linear(32,64)
        self.fc2 = nn.Linear(64,16)
        self.fc3 = nn.Linear(16,self.out_channel)
        self.relu = nn.Tanh()

    def forward(self,X):
        X = X.float()
        hidden1 = self.relu(self.fc1(X))
        hidden1 = self.relu(self.fc1_1(hidden1))
        hidden2 = self.relu(self.fc2(hidden1))
        hidden3 = self.fc3(hidden2)
        return hidden3

class RedisTwiceDNN(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RedisTwiceDNN, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.fc1 = nn.Linear(self.in_channel,2*self.in_channel)
        self.fc2 = nn.Linear(2*self.in_channel,4*self.in_channel)
        self.fc3 = nn.Linear(4*self.in_channel,2*self.in_channel)
        self.fc4 = nn.Linear(2*self.in_channel,self.in_channel)
        self.outs = []
        for _ in range(self.out_channel):
            self.outs.append(nn.Linear(self.in_channel,1))
        self.relu = nn.ReLU()
        self.af = [nn.Tanh(), nn.Sigmoid()]

    def forward(self,X):
        X = X.float()
        hidden1 = self.relu(self.fc1(X))
        hidden2 = self.relu(self.fc2(hidden1))
        hidden3 = self.relu(self.fc3(hidden2))
        hidden4 = self.relu(self.fc4(hidden3))
        outputs = []
        for i, output in enumerate(self.outs):
            outputs.append(self.af[i](output(hidden4)))
        return outputs