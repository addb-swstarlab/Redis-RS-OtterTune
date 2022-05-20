import torch
import torch.nn as nn

class SingleNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SingleNet, self).__init__()
        self.input_dim = input_dim # 22
        self.hidden_dim = hidden_dim # 32
        self.output_dim = output_dim # 1
        self.knob_fc = nn.Sequential(nn.Linear(self.input_dim, self.hidden_dim), nn.Sigmoid())
        self.im_fc = nn.Sequential(nn.Linear(self.hidden_dim, self.output_dim))

    def forward(self, x):
        self.x_kb = self.knob_fc(x)
        self.x_im = self.im_fc(self.x_kb)
        return self.x_im

class ReshapeNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, group_dim, wk_num):
        super(ReshapeNet, self).__init__()
        self.input_dim = input_dim - wk_num
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.group_dim = group_dim
        self.wk_num = wk_num

        self.embedding = nn.Linear(self.wk_num, self.hidden_dim)
        self.knob_fc = nn.Sequential(nn.Linear(self.input_dim, self.hidden_dim*self.group_dim), nn.Sigmoid()) # (22, 1) -> (group*hidden, 1)
        self.attention = nn.MultiheadAttention(self.hidden_dim, 1)
        self.active = nn.Sigmoid()
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, x):
        wk = x[:, -self.wk_num:] # only workload information
        x = x[:, :-self.wk_num] # only knobs
        
        self.embed_wk = self.embedding(wk) # (batch, 4) -> (batch, dim)
        self.embed_wk = self.embed_wk.unsqueeze(1) # (batch, 1, dim)
        self.x = self.knob_fc(x) # (batch, 22) -> (batch, group*hidden)
        self.res_x = torch.reshape(self.x, (-1, self.group_dim, self.hidden_dim)) # (batch, group, hidden)
        
        # attn_ouptut = (1, batch, hidden), attn_weights = (batch, 1, group)
        self.attn_output, self.attn_weights = self.attention(self.embed_wk.permute((1,0,2)), self.res_x.permute((1,0,2)), self.res_x.permute((1,0,2)))
        self.attn_output = self.active(self.attn_output.squeeze())
        outs = self.attn_output
        self.outputs = self.fc(outs)
        return self.outputs
