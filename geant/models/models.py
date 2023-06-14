import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers import GraphAttentionLayer,GraphAttentionLayer1, SpGraphAttentionLayer
import numpy as np

def mySoftMax(x):
    x1,x2=x.shape
    result=np.array([])
    for i in range(x1):
        result=np.append(result,x[i] / np.sum(x[i]))
    return result.reshape(-1,x2)

class MyGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads,device):
        """Dense version of GAT."""
        super(MyGAT, self).__init__()
        self.dropout = dropout
        self.device=device
        w=torch.rand(23, 23,23).float()
        b = torch.rand(23, 23, 1).float()
        # for i in range(23):
        #     for j in range(23):
        #         w[i,j,j]=1
        self.W = nn.Parameter(w)
        self.B = nn.Parameter(b)
        # nn.init.xavier_uniform_(self.W.data, gain=1.414)
        # self.attentions = [GraphAttentionLayer1(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        # for i, attention in enumerate(self.attentions):
        #     self.add_module('attention_{}'.format(i), attention)

        # self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)  # 第二层(最后一层)的attention layer

    def forward(self, data, adj,is_training=True):
        data = data["flow_x"][:,:,:,-1].squeeze(-1).to(self.device) #bs,N,N,1
        B, N, N = data.size()
        # data=F.dropout(data,self.dropout,training=is_training)
        for i in range(23):
            adj[i][i]=1

        min_value=self.W.min(dim=-1,keepdim=True).values
        min_value = torch.where(min_value > 0., torch.zeros_like(min_value).double(), min_value.double()).float()
        # min_value=torch.where(min_value>0.,0.,min_value.double()).float()
        attention=self.W-min_value
        # self.W.requires_grad_(False)
        for i in range(23):
            for j in range(23):
                if adj[i][j] == 0:
                    attention[:, i, j] = 0
        # self.W.requires_grad_(True)
        attention=attention / attention.sum(dim=-1, keepdim=True)
        # for i in range(23):
        #     adj[i][i]=1
        # for i in range(23):
        #     for j in range(23):
        #         if adj[i][j]==0:
        #             attention[:,i,j]=0

        # attention=torch.where(torch.tensor(adj)>0,attention,0)

        # attention = attention / attention.sum(dim=-1, keepdim=True)
        x = torch.zeros(B, N, N).to(self.device)
        for j in range(B):
            x[j] = torch.cat([attention[i] @ (data[j, :, i].unsqueeze(-1)) for i in range(23)], dim=1).t()
        # cj=(x-data)/data
        # x = F.dropout(x, self.dropout, training=is_training)
        # x=F.elu(x)

        # x=data["flow_x"].squeeze(-1).squeeze(0)
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = torch.cat([att(x, adj,is_training) for att in self.attentions], dim=-1)  # 将每层attention拼接
        # x = F.dropout(x, self.dropout, training=self.training)
        # # x = F.elu(self.out_att(x, adj))   # 第二层的attention layer
        return x.unsqueeze(-1)


class ResGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads,device):
        """Dense version of GAT."""
        super(ResGAT, self).__init__()
        self.dropout = dropout
        self.device=device
        self.gat = MyGAT3_1(nfeat=23,
                         nhid=23,
                         nclass=23,
                         dropout=0.1,
                         nheads=1,
                         alpha=0.2,
                         device=device)
        self.a = nn.Parameter(torch.rand(1,23,23, 2).float())

    def forward(self, data, adj,is_training=True):
        out=self.gat(data,adj)
        x=data["flow_x"][:,:,:,-1].unsqueeze(-1).to(self.device)
        out = torch.cat([x, out], dim=-1)
        yz = self.a / self.a.sum(dim=-1, keepdim=True)
        # yz=F.softmax(self.a, dim=-1)
        out=(out*yz).sum(dim=-1,keepdim=True)
        return out

class ResGAT1(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads,device):
        """Dense version of GAT."""
        super(ResGAT1, self).__init__()
        self.dropout = dropout
        self.device=device
        self.gat = MyGAT3_1(nfeat=23,
                         nhid=23,
                         nclass=23,
                         dropout=0.1,
                         nheads=1,
                         alpha=0.2,
                         device=device)
        self.a = nn.Parameter(torch.rand(1,23,23, 2).float())

    def forward(self, data, adj,is_training=True):
        out=self.gat(data,adj)
        x=data["flow_x"][:,:,:,-1].unsqueeze(-1).to(self.device)
        out = x+out
        # yz = self.a / self.a.sum(dim=-1, keepdim=True)
        # # yz=F.softmax(self.a, dim=-1)
        # out=(out*yz).sum(dim=-1,keepdim=True)
        return out

class ResGATByNum(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads,res_gat_num,device):
        """Dense version of GAT."""
        super(ResGATByNum, self).__init__()
        self.dropout = dropout

        self.resgats = [ResGAT(nfeat=23,nhid=23,nclass=23,dropout=0.1,nheads=1,alpha=0.2,device=device) for _ in range(res_gat_num)]
        for i, resgat in enumerate(self.resgats):
            self.add_module('resgat_{}'.format(i), resgat)

        # self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)  # 第二层(最后一层)的attention layer

    def forward(self, data, adj,is_training=True):
        out=data["flow_x"]
        for resgat in self.resgats:
            out = resgat({"flow_x": out}, adj)
        # x = torch.cat([att(x, adj,is_training) for att in self.attentions], dim=-1)  # 将每层attention拼接
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = F.elu(self.out_att(x, adj))   # 第二层的attention layer
        return out

class ResGAT1ByNum(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads,res_gat_num,device):
        """Dense version of GAT."""
        super(ResGAT1ByNum, self).__init__()
        self.dropout = dropout

        self.resgats = [ResGAT1(nfeat=23,nhid=23,nclass=23,dropout=0.1,nheads=1,alpha=0.2,device=device) for _ in range(res_gat_num)]
        for i, resgat in enumerate(self.resgats):
            self.add_module('resgat_{}'.format(i), resgat)

        # self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)  # 第二层(最后一层)的attention layer

    def forward(self, data, adj,is_training=True):
        out=data["flow_x"]
        for resgat in self.resgats:
            out = resgat({"flow_x": out}, adj)
        # x = torch.cat([att(x, adj,is_training) for att in self.attentions], dim=-1)  # 将每层attention拼接
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = F.elu(self.out_att(x, adj))   # 第二层的attention layer
        return out

class DoubleResGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads,device):
        """Dense version of GAT."""
        super(DoubleResGAT, self).__init__()
        self.dropout = dropout
        self.resgat = ResGAT(nfeat=23,
                         nhid=23,
                         nclass=23,
                         dropout=0.1,
                         nheads=1,
                         alpha=0.2,
                         device=device)
        self.resgat1 = ResGAT(nfeat=23,
                             nhid=23,
                             nclass=23,
                             dropout=0.1,
                             nheads=1,
                             alpha=0.2,
                             device=device)
        self.a = nn.Parameter(torch.rand(1,23,23, 2).float())

    def forward(self, data, adj,is_training=True):
        out=self.resgat(data,adj)
        out=self.resgat1({"flow_x":out},adj)
        return out


class MyGAT1(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads,device):
        """Dense version of GAT."""
        super(MyGAT1, self).__init__()
        self.dropout = dropout
        self.device = device
        w = torch.rand(23, 23, 23).float()
        b = torch.rand(23, 23, 1).float()
        # for i in range(23):
        #     for j in range(23):
        #         w[i, j, j] = 1
        self.W = nn.Parameter(w)
        self.B = nn.Parameter(b)
        # nn.init.xavier_uniform_(self.W.data, gain=1.414)
        # self.attentions = [GraphAttentionLayer1(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        # for i, attention in enumerate(self.attentions):
        #     self.add_module('attention_{}'.format(i), attention)

        # self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)  # 第二层(最后一层)的attention layer

    def forward(self, data, adj,is_training=True):
        data = data["flow_x"][:, :, :, -1].squeeze(-1)
        data=data.permute(0,2,1).to(self.device)
        B,N,N=data.size()

        # data = data["flow_x"][:,:,:,-1].squeeze(-1).squeeze(0).t().to(self.device)
        # data = F.dropout(data, self.dropout, training=is_training)

        for i in range(23):
            adj[i][i]=1

        min_value=self.W.min(dim=-1,keepdim=True).values
        min_value=torch.where(min_value>0.,0.,min_value.double()).float()
        attention=self.W-min_value
        for i in range(23):
            for j in range(23):
                if adj[i][j] == 0:
                    attention[:, i, j] = 0

        # for i in range(23):
        #     adj[i][i]=1
        # self.W.requires_grad_(False)
        # for i in range(23):
        #     for j in range(23):
        #         if adj[i][j]==0:
        #             self.W[:,i,j]=0
        # self.W.requires_grad_(True)
        attention=attention / attention.sum(dim=-1, keepdim=True)

        # attention=torch.where(torch.tensor(adj)>0,attention,0)
        # attention = attention / attention.sum(dim=-1, keepdim=True)
        x=torch.zeros(B,N,N).to(self.device)
        for j in range(B):
            x[j]=torch.cat([attention[i]@(data[j,:,i].unsqueeze(-1)) for i in range(23)],dim=1).t()
        # x = F.dropout(x, self.dropout, training=is_training)
        # x = F.elu(x)
        # x=data["flow_x"].squeeze(-1).squeeze(0)
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = torch.cat([att(x, adj,is_training) for att in self.attentions], dim=-1)  # 将每层attention拼接
        # x = F.dropout(x, self.dropout, training=self.training)
        # # x = F.elu(self.out_att(x, adj))   # 第二层的attention layer
        # x=torch.where(x<0,0,x)
        return x.unsqueeze(-1)



class MyGAT2(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads,device):
        """Dense version of GAT."""
        super(MyGAT2, self).__init__()
        self.dropout = dropout
        self.device = device
        self.W1 = nn.Parameter(torch.rand(23, 23,23).float())
        nn.init.xavier_uniform_(self.W1.data, gain=1.414)
        self.W2 = nn.Parameter(torch.rand(23, 23, 23).float())
        nn.init.xavier_uniform_(self.W2.data, gain=1.414)
        # self.attentions = [GraphAttentionLayer1(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        # for i, attention in enumerate(self.attentions):
        #     self.add_module('attention_{}'.format(i), attention)

        # self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)  # 第二层(最后一层)的attention layer

    def forward(self, data, adj,is_training=True):
        for i in range(23):
            adj[i][i]=1
        # self.W1.requires_grad_(False)
        # self.W2.requires_grad_(False)
        # for i in range(23):
        #     for j in range(23):
        #         if adj[i][j]==0:
        #             self.W1[:,i,j]=0
        #             self.W2[:, i, j] = 0
        # self.W1.requires_grad_(True)
        # self.W2.requires_grad_(True)
        data = data["flow_x"].squeeze(-1).squeeze(0).to(self.device)
        min_value1 = self.W1.min(dim=-1, keepdim=True).values
        min_value1=torch.where(min_value1>0.,0.,min_value1.double()).float()
        attention1 = self.W1 - min_value1
        for i in range(23):
            for j in range(23):
                if adj[i][j] == 0:
                    attention1[:, i, j] = 0
        attention1=attention1 / attention1.sum(dim=-1, keepdim=True)
        # for i in range(23):
        #     for j in range(23):
        #         if adj[i][j]==0:
        #             attention[:,i,j]=0
        # attention=torch.where(torch.tensor(adj)>0,attention,0)
        # attention = attention / attention.sum(dim=-1, keepdim=True)
        x1=torch.cat([attention1[i]@(data[:,i].unsqueeze(-1)) for i in range(23)],dim=1)
        data=data.t()
        min_value2 = self.W2.min(dim=-1, keepdim=True).values
        min_value2=torch.where(min_value2>0.,0.,min_value2.double()).float()
        attention2 = self.W2 - min_value2
        for i in range(23):
            for j in range(23):
                if adj[i][j] == 0:
                    attention2[:, i, j] = 0
        attention2 = attention2 / attention2.sum(dim=-1, keepdim=True)
        x2 = torch.cat([attention2[i] @ (data[:, i].unsqueeze(-1)) for i in range(23)], dim=1).t()
        x=(x1+x2)/2
        # x=data["flow_x"].squeeze(-1).squeeze(0)
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = torch.cat([att(x, adj,is_training) for att in self.attentions], dim=-1)  # 将每层attention拼接
        # x = F.dropout(x, self.dropout, training=self.training)
        # # x = F.elu(self.out_att(x, adj))   # 第二层的attention layer
        return x.unsqueeze(0).unsqueeze(-1)


class MyGAT3(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads,device):
        """Dense version of GAT."""
        super(MyGAT3, self).__init__()
        self.dropout = dropout
        self.gat = MyGAT(nfeat=23,
                       nhid=23,
                       nclass=23,
                       dropout=0.1,
                       nheads=1,
                       alpha=0.2,
                       device=device)
        self.gat1 = MyGAT1(nfeat=23,
                          nhid=23,
                          nclass=23,
                          dropout=0.1,
                          nheads=1,
                          alpha=0.2,
                          device=device)
        self.a = nn.Parameter(torch.rand(2,1).float())

        # nn.init.xavier_uniform_(self.W.data, gain=1.414)
        # self.attentions = [GraphAttentionLayer1(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        # for i, attention in enumerate(self.attentions):
        #     self.add_module('attention_{}'.format(i), attention)

        # self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)  # 第二层(最后一层)的attention layer

    def forward(self, data, adj,is_training=True):
        x1=self.gat(data,adj)
        x2=self.gat1(data,adj)
        x = torch.cat([x1,x2],dim=-1)
        # yz=F.softmax(self.a, dim=0)
        yz = self.a / self.a.sum(dim=0, keepdim=True)
        x=x@yz
        return x

class MyGAT3_1(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads,device):
        """Dense version of GAT."""
        super(MyGAT3_1, self).__init__()
        self.dropout = dropout
        self.gat = MyGAT(nfeat=23,
                       nhid=23,
                       nclass=23,
                       dropout=0.1,
                       nheads=1,
                       alpha=0.2,
                       device=device)
        self.gat1 = MyGAT1(nfeat=23,
                          nhid=23,
                          nclass=23,
                          dropout=0.1,
                          nheads=1,
                          alpha=0.2,
                          device=device)
        self.a = nn.Parameter(torch.rand(23,23,2).float())

    def forward(self, data, adj,is_training=True):
        x1=self.gat(data,adj) #[B,N,N,1]
        x2=self.gat1(data,adj) #[B,N,N,1]
        x = torch.cat([x1,x2],dim=-1)
        yz = self.a / self.a.sum(dim=-1, keepdim=True)
        # yz=F.softmax(self.a, dim=-1)
        x=(x*yz).sum(dim=-1)
        return x.unsqueeze(-1)


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer1(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        # self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)  # 第二层(最后一层)的attention layer

    def forward(self, data, adj,is_training=True):
        x = data["flow_x"][:, :, :, -1].unsqueeze(-1).squeeze(-1).squeeze(0)
        # x=data["flow_x"].squeeze(-1).squeeze(0)
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj,is_training) for att in self.attentions], dim=-1)  # 将每层attention拼接
        x = F.dropout(x, self.dropout, training=self.training)
        # x = F.elu(self.out_att(x, adj))   # 第二层的attention layer
        return x.unsqueeze(-1)


class SpGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Sparse version of GAT."""
        super(SpGAT, self).__init__()
        self.dropout = dropout

        self.attentions = [SpGraphAttentionLayer(nfeat, 
                                                 nhid, 
                                                 dropout=dropout, 
                                                 alpha=alpha, 
                                                 concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = SpGraphAttentionLayer(nhid * nheads, 
                                             nclass, 
                                             dropout=dropout, 
                                             alpha=alpha, 
                                             concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)

