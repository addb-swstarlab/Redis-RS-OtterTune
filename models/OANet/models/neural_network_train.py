import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from datetime import datetime
import numpy as np
from sklearn.metrics import r2_score
from models.OANet.models.network import SingleNet, ReshapeNet
from models.OANet.models.utils import get_filename
# from pytorchtools import EarlyStopping

class NeuralModel():
    def __init__(self, logger, mode, batch_size, lr, epochs, input_dim, hidden_dim, 
                 output_dim, lamb, group_dim=None, wk_num=None, dot=False):
        self.mode = mode
        self.batch_size = batch_size
        self.lr = lr                    # learning rate
        self.epochs = epochs
        self.dot = dot                  # dot loss          
        self.lamb = lamb

        # add logger
        self.logger = logger
                
        if self.mode == 'single':
            self.model = SingleNet(input_dim=input_dim, hidden_dim=hidden_dim, 
                                   output_dim=output_dim).cuda()
        elif self.mode == 'reshape':
            self.model = ReshapeNet(input_dim=input_dim, hidden_dim=hidden_dim, 
                                    output_dim=output_dim, group_dim=group_dim, wk_num=wk_num).cuda()
    
    def forward(self, X, y):
        self.fit(X, y)

    def fit(self, X, y):
        self.X_tr, self.X_te = X
        self.y_tr, self.y_te = y
        dataset_tr = TensorDataset(self.X_tr, self.y_tr)
        dataset_te = TensorDataset(self.X_te, self.y_te)
        dataloader_tr = DataLoader(dataset_tr, batch_size=self.batch_size, shuffle=True)
        dataloader_te = DataLoader(dataset_te, batch_size=self.batch_size, shuffle=True)        
        best_loss = np.inf
        self.logger.info(f"[Train MODE] Training Model") 
        name = get_filename('model_save', 'model', '.pt')


        for epoch in range(self.epochs):
            loss_tr, dot_loss_tr, _ = self.train(self.model, dataloader_tr)
            loss_te, dot_loss_te, te_outputs, r2_res = self.valid(self.model, dataloader_te)

            time = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
            if self.dot:
                # print(f"{time}[{epoch:02d}/{self.epochs}] loss_tr: {loss_tr:.8f}\tloss_te:{loss_te:.8f} \tdot loss_tr: {dot_loss_tr:.8f}\tdot loss_te:{dot_loss_te:.8f}")
                self.logger.info(f"[{epoch:02d}/{self.epochs}] loss_tr: {loss_tr:.8f}\tloss_te:{loss_te:.8f} \tdot loss_tr: {dot_loss_tr:.8f}\tdot loss_te:{dot_loss_te:.8f}\t r2:{r2_res:.4f}")
            else:
                # print(f"{time}[{epoch:02d}/{self.epochs}] loss_tr: {loss_tr:.8f}\tloss_te:{loss_te:.8f}")
                self.logger.info(f"[{epoch:02d}/{self.epochs}] loss_tr: {loss_tr:.8f}\tloss_te:{loss_te:.8f}\t r2:{r2_res:.4f}")
            
            if best_loss > loss_te:
                best_loss = loss_te
                self.best_model = self.model
                self.best_outputs = te_outputs
                self.best_epoch = epoch
                torch.save(self.best_model, os.path.join('model_save', name))
        self.logger.info(f"best loss is {best_loss:.4f} in [{self.best_epoch}/{self.epochs}]epoch, save model to {os.path.join('model_save', name)}")                
        print(f"best loss is {best_loss:.4f} in [{self.best_epoch}/{self.epochs}]epoch")

        return self.best_outputs

    def predict(self, X):
        return self.model(X)

    def train(self, model, train_loader):        
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        model.train()

        total_loss = 0.
        total_dot_loss = 0.
        outputs = torch.Tensor().cuda()

        for data, target in train_loader:
            optimizer.zero_grad()
            ## predict
            if self.mode == 'reshape':
                output = model(data) # output.shape = (batch_size, 1)

                if self.dot:
                    ## dot
                    dot = torch.bmm(model.res_x, model.res_x.transpose(1,2))    # bmm : batch matrix multiplication --> [B, n, m] x [B, m, p] = [B, n, p]

                    dot_loss = F.mse_loss(dot, torch.eye(dot.size(1)).repeat(data.shape[0], 1, 1).cuda())
                    ## loss
                    loss = (1-self.lamb)*F.mse_loss(output, target) + self.lamb*dot_loss
                    total_dot_loss += dot_loss.item()
                else:
                    loss = F.mse_loss(output, target)
            elif self.mode == 'single':
                output = model(data)
                loss = F.mse_loss(output, target)
            


            ## backpropagation
            loss.backward()
            optimizer.step()
            ## Logging
            total_loss += loss.item()
            outputs = torch.cat((outputs, output))
        total_loss /= len(train_loader)
        total_dot_loss /= len(train_loader)

        return total_loss, total_dot_loss, outputs 

    def valid(self, model, valid_loader):
        model.eval()

        ## Valid start    
        total_loss = 0.
        total_dot_loss = 0.
        outputs = torch.Tensor().cuda()
        r2_res = 0
        with torch.no_grad():
            for data, target in valid_loader:
                if self.mode == 'reshape':
                    output = model(data)
                    
                    if self.dot:
                        ## dot
                        dot = torch.bmm(model.res_x, model.res_x.transpose(1,2))
                        dot_loss = F.mse_loss(dot, torch.eye(dot.size(1)).repeat(data.shape[0], 1, 1).cuda())
                        loss = (1-self.lamb)*F.mse_loss(output, target) + self.lamb*dot_loss # mean squared error
                        total_dot_loss += dot_loss.item()
                    else:
                        loss = F.mse_loss(output, target)
                elif self.mode == 'single':
                    output = model(data)
                    loss = F.mse_loss(output, target)

                true = target.cpu().detach().numpy().squeeze()
                pred = output.cpu().detach().numpy().squeeze()            
                r2_res += r2_score(true, pred)
                total_loss += loss.item()
                outputs = torch.cat((outputs, output))
        total_loss /= len(valid_loader)
        total_dot_loss /= len(valid_loader)
        r2_res /= len(valid_loader)

        return total_loss, total_dot_loss, outputs, r2_res
