from models.lstm import GRU
import torch.nn as nn
import torch
from chebnet import ChebNet


class GCN(nn.Module):
    def __init__(self,in_c,hid_c,out_c,device):
        super(GCN,self).__init__()
        self.linear_1=nn.Linear(in_c,hid_c)
        self.linear_2=nn.Linear(hid_c,out_c)
        self.linear_3 = nn.Linear(2, 1)
        self.my_w=torch.randn(23,23,2,requires_grad=True)
        self.device=device
        # print(self.my_w)
        # self.my_w=self.my_w.to(device)
        # print(self.my_w)
        self.act=nn.ReLU()
        self.sig=nn.Sigmoid()

    def forward(self,data,adj,is_training=False):
        device=self.device
        graph_data=data["graph"].to(device)[0]

        graph_data=GCN.process_graph(graph_data)
        b=graph_data.to(torch.device("cpu")).numpy()
        flow_x=data["flow_x"].to(device)
        B, N, H, D = flow_x.size()
        flow_x = flow_x.view(B, N, -1) # [B,N,H*D]
        # print("flow_x[:,1,17]:",flow_x[:,1,17])
        output_1=self.linear_1(flow_x) #[B,N,hid_c]
        # print("output_1[:,1,17]:", output_1[:, 1, 17])
        # graph_data=graph_data.unsqueeze(0).expand(64,23,23)
        # output_1_0numpy=output_1[0].to(torch.device("cpu")).detach().numpy()
        # graph_data_numpy=graph_data.to(torch.device("cpu")).detach().numpy()
        output_1=self.act(torch.matmul(graph_data,output_1)) #[N,N]*[B,N,Hid_c]=[B,N,Hid_c]
        # print("output_1[:,1,17]:", output_1[:, 1, 17])

        output_2=self.linear_2(output_1)
        # print("output_2[:,1,17]:", output_2[:, 1, 17])
        output_2 = self.sig(torch.matmul(graph_data, output_2))
        spatial_output=output_2.unsqueeze(-1) # [B,N,N]->[B,N,N,1]
        time_output=flow_x.unsqueeze(-1) #[B,N,hid_c]->[B,N,hid_c,1]
        output_3=torch.cat((spatial_output, time_output), -1)

        # output_3=self.linear_3(output_3)
        output_3=output_3*self.my_w.to(device) #[B,N,hid_c,2]
        output_3=output_3[:,:,:,0]+output_3[:,:,:,1]

        # print("output_2[:,1,17]:", output_2[:, 1, 17])
        return output_3.unsqueeze(-1)


    @staticmethod
    def process_graph(graph_data):
        N=graph_data.size(0)
        matrix_i=torch.eye(N,dtype=graph_data.dtype,device=graph_data.device)
        graph_data+=matrix_i
        degree_matrix=torch.sum(graph_data,dim=-1,keepdim=False)
        degree_matrix=degree_matrix.pow(-1)
        degree_matrix[degree_matrix==float("inf")]=0
        degree_matrix=torch.diag(degree_matrix)
        return torch.mm(degree_matrix,graph_data)


class SGCRU(nn.Module):
    def __init__(self, in_dim, hidden_dim,device, n_layer=1, seq_len=12, pre_len=1):
        super(SGCRU, self).__init__()
        self.gru = nn.GRU(
            in_dim, hidden_dim, n_layer, batch_first=True)
        self.gcn=ChebNet(in_c=23,hid_c=23,out_c=23,K=2,device=device)
        self.fc = nn.Linear(hidden_dim, in_dim)
        self.time_linear = nn.Linear(seq_len, pre_len)


    def forward(self, data,adj,is_training=True):
        # result = torch.rand(0)
        x=data["flow_x"] #[bs,N,N,H]
        bs,N,N,H=x.shape
        x=x.view(-1,H,1)
        ouput, ouput_n = self.gru(x)  # BS,H,1
        result=self.gcn({"flow_x":ouput_n.reshape(-1,23,23,1),"graph":data["graph"]},adj,is_training=True)
        # for i in range(23):
        #     for j in range(23):
        #         ouput, ouput_n = self.rnn(x[:,i,j].unsqueeze(-1))  # BS,H,1
        #         result = torch.cat([result,ouput_n],dim=-1)
        return result