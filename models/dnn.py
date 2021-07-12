import torch.nn as nn

class RedisSingleDNN(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RedisSingleDNN, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.fc1 = nn.Linear(self.in_channel,128)
        self.fc2 = nn.Linear(128,256)
        self.fc3 = nn.Linear(256,128)
        self.fc4 = nn.Linear(128,self.out_channel)
        self.tanh = nn.Tanh()

    def forward(self,X):
        X = X.float()
        hidden1 = self.tanh(self.fc1(X))
        hidden2 = self.tanh(self.fc2(hidden1))
        hidden3 = self.tanh(self.fc3(hidden2))
        hidden4 = self.fc4(hidden3)
        return hidden4

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
        self.tanh = nn.Tanh()

    def forward(self,X):
        X = X.float()
        hidden1 = self.tanh(self.fc1(X))
        hidden2 = self.tanh(self.fc2(hidden1))
        hidden3 = self.tanh(self.fc3(hidden2))
        hidden4 = self.tanh(self.fc4(hidden3))
        outputs = []
        for _, output in enumerate(self.outs):
            outputs.append(output(hidden4))
        return outputs