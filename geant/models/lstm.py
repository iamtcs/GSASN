
import  numpy as np
import  torch
import  torch.nn as nn
import  torch.optim as optim
from    matplotlib import pyplot as plt
from torch.utils.data import Dataset,DataLoader
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torchinfo import summary
import torch.nn.functional as F
from models.models import  MyGAT,MyGAT1,MyGAT2,MyGAT3,MyGAT3_1,ResGAT

class LSTM(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layer=3, seq_len=12, pre_len=3):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(
            in_dim, hidden_dim, n_layer, batch_first=True, dropout=0.5)
        print(self.lstm)
        self.fc = nn.Linear(hidden_dim, 1)
        self.time_linear = nn.Linear(seq_len, pre_len)

    def forward(self, x):
        # BS,seq_len = x.size()
        x = x.unsqueeze(-1)  # bs,t,1
        x, _ = self.lstm(x)  # BS,T,h
        x = self.fc(x)  # BS,T,1
        x = self.time_linear(x.squeeze(-1))

        return x


class LSTM_TM(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layer=1, seq_len=12, pre_len=1):
        super(LSTM_TM, self).__init__()
        self.lstm = nn.LSTM(
            in_dim, hidden_dim, n_layer, batch_first=True)
        self.fc = nn.Linear(hidden_dim, in_dim)
        self.time_linear = nn.Linear(seq_len, pre_len)


    def forward(self, data,adj,is_training=False):
        x=data["flow_x"]
        x=x.view(x.shape[0],-1,x.shape[3]).transpose(1,2)
        # BS,seq_len,f = x.size()
        x, _ = self.lstm(x)  # BS,T,h
        # x = self.fc(x)  # BS,T,f
        return x[:,-1,:].reshape(-1,23,23,1)



class LSTM(nn.Module):
    def __init__(self, in_dim, hidden_dim,device, n_layer=1, seq_len=12, pre_len=1):
        super(LSTM, self).__init__()
        self.device=device
        self.lstm = nn.LSTM(
            in_dim, hidden_dim, n_layer, batch_first=True)
        self.fc = nn.Linear(hidden_dim, in_dim)
        self.time_linear = nn.Linear(seq_len, pre_len)


    def forward(self, data,adj,is_training=True):
        x = data["flow_x"].to(self.device)  # [bs,N,N,H]
        bs, N, N, H = x.shape
        x = x.view(-1, H, 1)
        ouput, (ouput_hn,ouput_cn) = self.lstm(x)  # BS,H,1
        # for i in range(23):
        #     for j in range(23):
        #         ouput, ouput_n = self.rnn(x[:,i,j].unsqueeze(-1))  # BS,H,1
        #         result = torch.cat([result,ouput_n],dim=-1)
        return ouput_hn.reshape(-1, 23, 23, 1)

class LSTM_GAT(nn.Module):
    def __init__(self, in_dim, hidden_dim,device, n_layer=1, seq_len=12, pre_len=1):
        super(LSTM_GAT, self).__init__()
        self.device=device
        self.lstm = nn.LSTM(
            in_dim, hidden_dim, n_layer, batch_first=True)
        self.gat=ResGAT(nfeat=23,
                    nhid=23,
                    nclass=23,
                    dropout=0.1,
                    nheads=1,
                    alpha=0.2,
                    device=device)
        # self.fc = nn.Linear(hidden_dim, in_dim)
        # self.time_linear = nn.Linear(seq_len, pre_len)


    def forward(self, data,adj,is_training=True):
        x = data["flow_x"]  # [bs,N,N,H]
        bs, N, N, H = x.shape
        x = x.view(-1, H, 1).to(self.device)
        ouput, (ouput_hn,ouput_cn) = self.lstm(x)  # BS,H,1
        # for i in range(23):
        #     for j in range(23):
        #         ouput, ouput_n = self.rnn(x[:,i,j].unsqueeze(-1))  # BS,H,1
        #         result = torch.cat([result,ouput_n],dim=-1)
        lstm_output=ouput_hn.reshape(-1, 23, 23, 1)
        result=self.gat({"flow_x":lstm_output},adj)
        return result

class LSTM_GAT1(nn.Module):
    def __init__(self, in_dim, hidden_dim,device, n_layer=1, seq_len=12, pre_len=1):
        super(LSTM_GAT1, self).__init__()
        self.lstm = LSTM(in_dim=1, hidden_dim=1,device=device)
        self.gat=ResGAT(nfeat=23,
                    nhid=23,
                    nclass=23,
                    dropout=0,
                    nheads=1,
                    alpha=0.2,
                    device=device)
        self.a = nn.Parameter(torch.rand(2,1).float())
        # self.a = nn.Parameter(torch.rand(23, 23, 2).float())
        # self.fc = nn.Linear(hidden_dim, in_dim)
        # self.time_linear = nn.Linear(seq_len, pre_len)


    def forward(self, data,adj,is_training=True):
        x1=self.lstm(data,adj=None)
        x2=self.gat(data,adj)
        x = torch.cat([x1, x2], dim=-1)
        # yz = F.softmax(self.a, dim=-1)
        yz = self.a / self.a.sum(dim=0, keepdim=True)
        result = (x @ yz).sum(dim=-1)
        return result.unsqueeze(-1)



class GRU_GAT(nn.Module):
    def __init__(self, in_dim, hidden_dim,device, n_layer=1, seq_len=12, pre_len=1):
        super(GRU_GAT, self).__init__()
        self.gru = GRU(in_dim=1, hidden_dim=1)
        self.gat=ResGAT(nfeat=23,
                    nhid=23,
                    nclass=23,
                    dropout=0,
                    nheads=1,
                    alpha=0.2,
                    device=device)
        self.a = nn.Parameter(torch.rand(2,1).float())
        # self.a = nn.Parameter(torch.rand(23, 23, 2).float())
        # self.fc = nn.Linear(hidden_dim, in_dim)
        # self.time_linear = nn.Linear(seq_len, pre_len)


    def forward(self, data,adj,is_training=True):
        x1=self.gru(data,adj=None)
        x2=self.gat(data,adj)
        x = torch.cat([x1, x2], dim=-1)
        # yz = F.softmax(self.a, dim=-1)
        yz = self.a / self.a.sum(dim=0, keepdim=True)
        result = (x @ yz).sum(dim=-1)
        return result.unsqueeze(-1)

class LSTM_GAT2(nn.Module):
    def __init__(self, in_dim, hidden_dim,device, n_layer=1, seq_len=12, pre_len=1):
        super(LSTM_GAT1, self).__init__()
        self.lstm = LSTM(in_dim=1, hidden_dim=1,device=device)
        self.gat=MyGAT3_1(nfeat=23,
                    nhid=23,
                    nclass=23,
                    dropout=0.1,
                    nheads=1,
                    alpha=0.2,
                    device=device)
        self.a = nn.Parameter(torch.rand(2,1).float())
        # self.a = nn.Parameter(torch.rand(23, 23, 2).float())
        # self.fc = nn.Linear(hidden_dim, in_dim)
        # self.time_linear = nn.Linear(seq_len, pre_len)


    def forward(self, data,adj,is_training=True):
        x1=self.lstm(data,adj=None)
        x2=self.gat(data,adj)
        x = torch.cat([x1, x2], dim=-1)
        # yz = F.softmax(self.a, dim=-1)
        yz = self.a / self.a.sum(dim=0, keepdim=True)
        result = (x @ yz).sum(dim=-1)
        return result.unsqueeze(-1)




class RNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layer=1, seq_len=12, pre_len=1):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(
            in_dim, hidden_dim, n_layer, batch_first=True)
        self.fc = nn.Linear(hidden_dim, in_dim)
        self.time_linear = nn.Linear(seq_len, pre_len)


    def forward(self, data,adj,is_training=True):
        # result = torch.rand(0)
        x=data["flow_x"] #[bs,N,N,H]
        bs,N,N,H=x.shape
        x=x.view(-1,H,1)
        ouput, ouput_n = self.rnn(x)  # BS,H,1
        # for i in range(23):
        #     for j in range(23):
        #         ouput, ouput_n = self.rnn(x[:,i,j].unsqueeze(-1))  # BS,H,1
        #         result = torch.cat([result,ouput_n],dim=-1)
        return ouput_n.reshape(-1,23,23,1)


class GRU(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layer=1, seq_len=12, pre_len=1):
        super(GRU, self).__init__()
        self.gru = nn.GRU(
            in_dim, hidden_dim, n_layer, batch_first=True)
        self.fc = nn.Linear(hidden_dim, in_dim)
        self.time_linear = nn.Linear(seq_len, pre_len)


    def forward(self, data,adj,is_training=True):
        # result = torch.rand(0)
        x=data["flow_x"] #[bs,N,N,H]
        bs,N,N,H=x.shape
        x=x.view(-1,H,1)
        ouput, ouput_n = self.gru(x)  # BS,H,1
        # for i in range(23):
        #     for j in range(23):
        #         ouput, ouput_n = self.rnn(x[:,i,j].unsqueeze(-1))  # BS,H,1
        #         result = torch.cat([result,ouput_n],dim=-1)
        return ouput_n.reshape(-1,23,23,1)