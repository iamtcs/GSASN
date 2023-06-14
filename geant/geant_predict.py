import os
import torch
import time
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
# from traffic_dataset import LoadData
from geant_dataset import LoadGeantData
import numpy as np
from torch.utils.data import Dataset
import argparse
import  matplotlib.pyplot as plt
from chebnet import ChebNet
from sklearn.metrics import mean_absolute_error, mean_squared_error,mean_absolute_percentage_error
from models.models import MyGAT,GAT, SpGAT,MyGAT1,MyGAT2,MyGAT3,MyGAT3_1,ResGAT,DoubleResGAT,ResGATByNum,ResGAT1,ResGAT1ByNum
from models.lstm import LSTM_TM,LSTM,RNN,GRU,LSTM_GAT,LSTM_GAT1,LSTM_GAT2,GRU_GAT
from models.sgcrn import SGCRU
from models.stgcn import STGCNChebGraphConv,STGCNGraphConv

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
        self.sig=nn.ReLU()

    def forward(self,data,adj,is_training=False):
        device=self.device
        graph_data=data["graph"].to(device)[0]

        graph_data=GCN.process_graph(graph_data)
        b=graph_data.to(torch.device("cpu")).numpy()
        flow_x = data["flow_x"][:, :, :, -1].unsqueeze(-1).to(device)
        # flow_x=data["flow_x"].to(device)
        B, N, H, D = flow_x.size()
        flow_x = flow_x.view(B, N, -1) # [B,N,H*D]
        # print("flow_x[:,1,17]:",flow_x[:,1,17])
        output_1=self.linear_1(flow_x) #[B,N,hid_c]
        # print("output_1[:,1,17]:", output_1[:, 1, 17])
        # graph_data=graph_data.unsqueeze(0).expand(64,23,23)
        # output_1_0numpy=output_1[0].to(torch.device("cpu")).detach().numpy()
        # graph_data_numpy=graph_data.to(torch.device("cpu")).detach().numpy()
        output_1=torch.matmul(graph_data,output_1) #[N,N]*[B,N,Hid_c]=[B,N,Hid_c]
        # print("output_1[:,1,17]:", output_1[:, 1, 17])

        return output_1.unsqueeze(-1)

        # output_2=self.linear_2(output_1)
        # output_2 = torch.matmul(graph_data, output_2)
        # spatial_output=output_2.unsqueeze(-1) # [B,N,N]->[B,N,N,1]
        # time_output=flow_x.unsqueeze(-1) #[B,N,hid_c]->[B,N,hid_c,1]
        # output_3=torch.cat((spatial_output, time_output), -1)
        #
        # output_3=output_3*self.my_w.to(device) #[B,N,hid_c,2]
        # output_3=output_3[:,:,:,0]+output_3[:,:,:,1]
        #
        # return output_3.unsqueeze(-1)


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


class STM(nn.Module):
    def __init__(self,tm,sp,device):
        super(STM,self).__init__()
        self.tm=tm
        self.sp = sp
        self.device=device

    def forward(self,data,adj,is_training=False):
        flow_x=data["flow_x"].to(self.device)
        B,N,N,H=flow_x.size()
        x=self.tm(data,adj)
        # flow_x=flow_x.view(B,N,-1)
        output=self.sp({"flow_x":x,"graph":data["graph"]},adj) #[B,out_channels,N,N]
        return output.view(B,N,N,1)

class STM1(nn.Module):
    def __init__(self,tm,sp,device):
        super(STM1,self).__init__()
        self.tm=tm
        self.sp = sp
        self.device=device
        self.a = nn.Parameter(torch.rand(2, 1).float())

    def forward(self,data,adj,is_training=False):
        x1 = self.tm(data, adj)
        # data["flow_x"] = data["flow_x"][:, :, :, -1].unsqueeze(-1)
        x2 = self.sp(data, adj)
        x = torch.cat([x1, x2], dim=-1)
        # yz = F.softmax(self.a, dim=-1)
        yz = self.a / self.a.sum(dim=0, keepdim=True)
        result = (x @ yz).sum(dim=-1)
        return result.unsqueeze(-1)

        # flow_x=data["flow_x"].to(self.device)
        # B,N,N,H=flow_x.size()
        # x=self.tm(data,adj)
        # # flow_x=flow_x.view(B,N,-1)
        # output=self.sp({"flow_x":x,"graph":torch.tensor(adj)},adj) #[B,out_channels,N,N]
        # return output.view(B,N,N,1)


class CNN_T(nn.Module):
    def __init__(self,device):
        super(CNN_T,self).__init__()
        self.device=device
        self.cnn1=nn.Conv1d(in_channels=1,out_channels = 1, kernel_size = 4,stride=2)
        self.cnn2 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=4, stride=1)
        self.cnn3 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=4, stride=1)
        # self.layer=nn.Linear(in_c,out_c)

    def forward(self,data,adj,is_training=False):
        flow_x=data["flow_x"].to(self.device)
        B,N,N,H=flow_x.size()
        x=flow_x.view(-1,1,H)
        # flow_x=flow_x.view(B,N,-1)
        x=self.cnn1(x) #[B,out_channels,N,N]
        x = self.cnn2(x)
        output = self.cnn3(x)
        return output.view(B,N,N,1)

class CNN_S(nn.Module):
    def __init__(self,in_c,out_c,device):
        super(CNN_S,self).__init__()
        self.device=device
        self.cnn=nn.Conv2d(
                in_channels=1,    # 输入图片的高度
                out_channels=1,  # 输出图片的高度
                kernel_size=3,    # 5x5的卷积核，相当于过滤器
                stride=1,         # 卷积核在图上滑动，每隔一个扫一次
                padding=1,        # 给图外边补上0
            )
        self.layer=nn.Linear(in_c,out_c)

    def forward(self,data,adj,is_training=False):
        flow_x=data["flow_x"].to(self.device)
        B,N,N,H=flow_x.size()
        x=flow_x[:,:,:,-1].view(B,1,N,N)
        # flow_x=flow_x.view(B,N,-1)
        output=self.cnn(x) #[B,out_channels,N,N]
        return output.view(B,N,N,1)


class FFNN_T(nn.Module):
    def __init__(self,in_c,out_c,device):
        super(FFNN_T,self).__init__()
        self.device=device
        self.layer=nn.Linear(in_c,out_c)

    def forward(self,data,adj,is_training=False):
        flow_x=data["flow_x"].to(self.device)
        B,N,N,H=flow_x.size()
        # flow_x=flow_x.view(B,N,-1)
        output=self.layer(flow_x)
        return output



class FFNN(nn.Module):
    def __init__(self,in_c,out_c,device):
        super(FFNN,self).__init__()
        self.device=device
        self.layer=nn.Linear(in_c,out_c)

    def forward(self,data,adj,is_training=False):
        flow_x=data["flow_x"].to(self.device)
        B,N,H,D=flow_x.size()
        flow_x=flow_x.view(B,N,-1)
        output=self.layer(flow_x)
        return output.unsqueeze(-1)

class LSTM_FFNN(nn.Module):
    def __init__(self,in_c,out_c,device):
        super(LSTM_FFNN,self).__init__()
        self.lstm=LSTM_TM(in_dim=529, hidden_dim=529)
        self.ffnn=FFNN(in_c=23, out_c=23,device=device)
        self.device=device

    def forward(self,data,adj,is_training=False):
        x=self.lstm(data,adj,is_training=False)
        x=self.ffnn({"flow_x":x},adj,is_training=False)
        return x

def get_parameters():
    parser = argparse.ArgumentParser(description='STGCN')
    parser.add_argument('--enable_cuda', type=bool, default=True, help='enable CUDA, default as True')
    parser.add_argument('--seed', type=int, default=42, help='set the random seed for stabilizing experiment results')
    parser.add_argument('--dataset', type=str, default='metr-la', choices=['metr-la', 'pems-bay', 'pemsd7-m'])
    parser.add_argument('--n_his', type=int, default=18)
    parser.add_argument('--n_pred', type=int, default=3, help='the number of time interval for predcition, default as 3')
    parser.add_argument('--time_intvl', type=int, default=5)
    parser.add_argument('--Kt', type=int, default=3)
    parser.add_argument('--stblock_num', type=int, default=2)
    parser.add_argument('--act_func', type=str, default='glu', choices=['glu', 'gtu'])
    parser.add_argument('--Ks', type=int, default=3, choices=[3, 2])
    parser.add_argument('--graph_conv_type', type=str, default='cheb_graph_conv', choices=['cheb_graph_conv', 'graph_conv'])
    parser.add_argument('--gso_type', type=str, default='sym_norm_lap', choices=['sym_norm_lap', 'rw_norm_lap', 'sym_renorm_adj', 'rw_renorm_adj'])
    parser.add_argument('--enable_bias', type=bool, default=True, help='default as True')
    parser.add_argument('--droprate', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weight_decay_rate', type=float, default=0.0005, help='weight decay (L2 penalty)')
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--epochs', type=int, default=100, help='epochs, default as 10000')
    parser.add_argument('--opt', type=str, default='adam', help='optimizer, default as adam')
    parser.add_argument('--step_size', type=int, default=10)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--patience', type=int, default=100, help='early stopping patience')
    parser.add_argument('--res_gat_num', type=int, default=4, help='early stopping patience')
    parser.add_argument('--model', type=str, default="cheb_graph_conv", help='FFNN LSTM_FFNN GCN ChebNet GAT MyGAT ResGAT ResGAT1 MyGAT1 MyGAT2 MyGAT3 MyGAT3_1 Lstm_tm Lstm RNN GRU LSTM_GAT GRU_GAT SGCRU LSTM_GAT1 DoubleResGAT ResGATByNum ResGAT1ByNum')
    args = parser.parse_args()

    Ko = args.n_his - (args.Kt - 1) * 2 * args.stblock_num
    blocks = []
    blocks.append([1])
    for l in range(args.stblock_num):
        blocks.append([64, 16, 64])
    if Ko == 0:
        blocks.append([128])
    elif Ko > 0:
        blocks.append([128, 128])
    blocks.append([1])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adj=torch.zeros(529,529)
    for i in range(23):
        for j in range(23):
            index1=i*23+j
            for k in range(23):
                index2=k*23+j
                index3=i*23+k
                adj[index1,index2]=1
                adj[index1, index3] = 1

    args.gso=adj.to(device)
    return args,blocks

def print_errors(y_true, y_pred):
    y_true=y_true.view(y_true.size(0),-1)
    y_pred = y_pred.view(y_pred.size(0), -1)
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mape=mean_absolute_percentage_error(y_true,y_pred)*100
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    print('TEST RESULT:',
          'mse:{:.6}'.format(mse),
          'mae:{:.6}'.format(mae),
          'mape:{:.6}'.format(mape),
          'rmse:{:.6}'.format(rmse),
          )

def r2_loss(output, target): #output==predict，target==true
    target_mean = torch.mean(target)
    ss_tot = torch.sum((target - target_mean) ** 2)
    ss_res = torch.sum((target - output) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2

def print_errors1(y_true, y_pred):
    y_true = y_true.view(y_true.size(0), -1).numpy()
    y_pred = y_pred.view(y_pred.size(0), -1).numpy()
    mse = np.mean(np.square(y_true - y_pred))
    rmse = np.sqrt(np.mean(np.square(y_true - y_pred)))
    mae = np.mean(np.abs(y_true - y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    print('TEST RESULT:',
          'mse:{:.6}'.format(mse),
          'mae:{:.6}'.format(mae),
          'mape:{:.6}'.format(mape),
          'rmse:{:.6}'.format(rmse),
          )

def print_images(true_value,pred_value):
    # 保存打印行
    for i in range(23):
        for j in range(23):
            plt.plot(true_value[-200:-1, i, j], color='b', label="true_y")
            plt.plot(pred_value[-200:-1, i, j], color='r', label="predict_y")
            plt.legend()
            plt.title("node:" + str(i*23+j))
            f = plt.gcf()  # 获取当前图像
            f.savefig("./images/diff_gcn/node"  + str(i*23+j) + ".png")
            # plt.show()
            f.clear()  # 释放内存

def predict_data(model,print_log,device,criterion,args,pred_n=1,batch_size=1):
    if pred_n == 1:
        test_start_time = time.time()
    test_data = LoadGeantData(data_path=["./simple/geant_a_metrix.csv", "./simple/geant_norm.npz"], num_nodes=23,
                              divide_days=[45, 14],
                              time_interval=5, histroy_length=args.n_his, train_mode="test",predict_length=pred_n)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0)
    my_net=model
    my_net.eval()
    with torch.no_grad():
        total_loss = 0.0
        before_test_data = torch.rand(0)
        predict_value_before = torch.rand(0)
        true_value_before = torch.rand(0)
        pred_value_recover = torch.rand(0)
        true_value_recover = torch.rand(0)
        for data in test_loader:
            data["flow_x"] = data["flow_x"].to(device)

            for i in range(pred_n):
                predict_value = my_net(data, test_data.graph, is_training=False).to(torch.device("cpu")) #[B,N,N,1]
                loss = criterion(predict_value, data["flow_y"][:,:,:,i].unsqueeze(-1))
                data["flow_x"]=torch.cat([data["flow_x"][:,:,:,1:],predict_value.to(device)],dim=-1)
                predict_value_before = torch.cat([predict_value_before, predict_value.detach().clone().reshape(-1)], dim=0)
                true_value_before = torch.cat([true_value_before, data["flow_y"][:,:,:,i].detach().clone().reshape(-1)], dim=0)
                # recovered_true = LoadGeantData.recover_data(max_data=test_data.flow_norm[0],
                #                                             min_data=test_data.flow_norm[1],
                #                                             data=data["flow_y"][:,:,:,i].unsqueeze(-1))
                # true_value_recover = torch.cat([true_value_recover, recovered_true], dim=0)
                # recovered_predict = LoadGeantData.recover_data(max_data=test_data.flow_norm[0],
                #                                                min_data=test_data.flow_norm[1],
                #                                                data=predict_value)
                # pred_value_recover = torch.cat([pred_value_recover, recovered_predict], dim=0)
                total_loss += loss.item()

        # print("befor_mse:",mean_squared_error(true_value_before, before_test_data))
        # print("befor_mae:", mean_absolute_error(true_value_before, before_test_data))

        if pred_n == 1:
            test_end_time = time.time()
            print("每个TM测试时间:", (test_end_time - test_start_time) / len(test_loader) * 1000, "毫秒（ms）")
            print("每个TM测试时间:", (test_end_time - test_start_time) / len(test_loader) * 1000, "毫秒（ms）", file=print_log)

        val_mse = mean_squared_error(true_value_before, predict_value_before)
        val_mae = mean_absolute_error(true_value_before, predict_value_before)
        val_rmse = mean_squared_error(true_value_before, predict_value_before) ** 0.5
        r2 = r2_loss(predict_value_before, true_value_before)
        print("----------------", pred_n, "----------------")
        print("----------------", pred_n, "----------------", file=print_log)
        print("mse:", val_mse)
        print("mae:", val_mae)
        print("rmse:", val_rmse)
        print("r2:", r2)
        print("mse:", val_mse, file=print_log)
        print("mae:", val_mae, file=print_log)
        print("rmse:", val_rmse, file=print_log)
        print("r2:", r2, file=print_log)
        print("----------------", pred_n, "----------------")
        print("----------------", pred_n, "----------------", file=print_log)



def main(model_name,epochs,n_his=16,model=None,model_n=None,is_slx=False,is_long_term=False,batch_size=-1):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args,blocks=get_parameters()
    args.model=model_name
    args.epochs = epochs
    args.n_his = n_his
    if model_name=="MyGAT3_1" or model_name=="GAT":
        args.n_his=1
    if batch_size!=-1:
        args.batch_size=batch_size
    # if model_name == "STM" and (model_n.find("MyGAT3_1") != -1 or model_n.find("GAT") != -1 or model_n.find("GCN") != -1):
    #     args.n_his = 1

    os.environ["CUDA_VISIBLE_DEVICES"]="0"

    #Loading Dataset
    train_data=LoadGeantData(data_path=["./simple/geant_a_metrix.csv","./simple/geant_norm.npz"],num_nodes=23,divide_days=[45,14],
                        time_interval=5,histroy_length=args.n_his,train_mode="train")
    train_loader=DataLoader(train_data,batch_size=args.batch_size,shuffle=True,num_workers=0)



    #Loading Model


    # blocks: settings of channel size in st_conv_blocks and output layer,
    # using the bottleneck design in st_conv_blocks


    if args.model == 'cheb_graph_conv':
        my_net = STGCNChebGraphConv(args, blocks, 529,device)
    elif args.model=='STM':
        my_net=model
    elif args.model=='CNN_T':
        my_net=CNN_T(device)
    elif args.model=='CNN_S':
        my_net=CNN_S(23,23,device)
    elif args.model=='FFNN_T':
        my_net=FFNN_T(args.n_his,1,device)
    elif args.model == 'graph_conv':
        my_net = STGCNGraphConv(args, blocks, 529,device)
    elif args.model=='FFNN':
        my_net = FFNN(in_c=23, out_c=23,device=device)
    elif args.model=='LSTM_FFNN':
        my_net = LSTM_FFNN(in_c=23, out_c=23,device=device)
    elif args.model=='Lstm_tm':
        my_net=LSTM_TM(in_dim=529, hidden_dim=529)
    elif args.model=='Lstm':
        my_net=LSTM(in_dim=1, hidden_dim=1,device=device)
    elif args.model=='GRU_GAT':
        my_net=GRU_GAT(in_dim=1, hidden_dim=1,device=device)
    elif args.model=='LSTM_GAT':
        my_net=LSTM_GAT(in_dim=1, hidden_dim=1,device=device)
    elif args.model == 'LSTM_GAT1':
        my_net = LSTM_GAT1(in_dim=1, hidden_dim=1,device=device)
    elif args.model == 'LSTM_GAT2':
        my_net = LSTM_GAT1(in_dim=1, hidden_dim=1,device=device)
    elif args.model=='RNN':
        my_net=RNN(in_dim=1, hidden_dim=1)
    elif args.model=='GRU':
        my_net=GRU(in_dim=1, hidden_dim=1)
    elif args.model == 'SGCRU':
        my_net = SGCRU(in_dim=1, hidden_dim=1,device=device)
    elif args.model=='GCN':
        my_net=GCN(in_c=23*1,hid_c=23*1,out_c=23*1,device=device)
        # my_net = ChebNet(in_c=23, hid_c=23, out_c=23, K=3, device=device)
    elif args.model=='ChebNet':
        my_net=ChebNet(in_c=23,hid_c=23,out_c=23,K=2,device=device)
    elif args.model=='GAT':
        my_net = GAT(nfeat=23,
                    nhid=23,
                    nclass=23,
                    dropout=0.1,
                    nheads=1,
                    alpha=0.2)
    elif args.model=='MyGAT':
        my_net = MyGAT(nfeat=23,
                    nhid=23,
                    nclass=23,
                    dropout=0,
                    nheads=1,
                    alpha=0.2,
                    device=device)
    elif args.model=='ResGAT':
        my_net = ResGAT(nfeat=23,
                    nhid=23,
                    nclass=23,
                    dropout=0.1,
                    nheads=1,
                    alpha=0.2,
                    device=device)
    elif args.model=='ResGAT1':
        my_net = ResGAT1(nfeat=23,
                    nhid=23,
                    nclass=23,
                    dropout=0.1,
                    nheads=1,
                    alpha=0.2,
                    device=device)
    elif args.model=='DoubleResGAT':
        my_net = DoubleResGAT(nfeat=23,
                    nhid=23,
                    nclass=23,
                    dropout=0.1,
                    nheads=1,
                    alpha=0.2,
                    device=device)
    elif args.model=='ResGATByNum':
        my_net = ResGATByNum(nfeat=23,
                    nhid=23,
                    nclass=23,
                    dropout=0.1,
                    nheads=1,
                    alpha=0.2,
                    res_gat_num=args.res_gat_num,
                    device=device)
    elif args.model=='ResGAT1ByNum':
        my_net = ResGAT1ByNum(nfeat=23,
                    nhid=23,
                    nclass=23,
                    dropout=0.1,
                    nheads=1,
                    alpha=0.2,
                    res_gat_num=args.res_gat_num,
                    device=device)
    elif args.model=='MyGAT1':
        my_net = MyGAT1(nfeat=23,
                    nhid=23,
                    nclass=23,
                    dropout=0.1,
                    nheads=1,
                    alpha=0.2,
                    device=device)
    elif args.model=='MyGAT2':
        my_net = MyGAT2(nfeat=23,
                    nhid=23,
                    nclass=23,
                    dropout=0.1,
                    nheads=1,
                    alpha=0.2,
                    device=device)
    elif args.model=='MyGAT3':
        my_net = MyGAT3(nfeat=23,
                    nhid=23,
                    nclass=23,
                    dropout=0.1,
                    nheads=1,
                    alpha=0.2,
                    device=device)
    elif args.model=='MyGAT3_1':
        my_net = MyGAT3_1(nfeat=23,
                    nhid=23,
                    nclass=23,
                    dropout=0.1,
                    nheads=1,
                    alpha=0.2,
                    device=device)
    my_net=my_net.to(device)
    criterion=nn.MSELoss()
    optimizer = optim.Adam(params=my_net.parameters(),lr=0.001)
    # optimizer=optim.Adam([{"params": my_net.parameters()},
    #                    {"params": my_net.my_w}])

    #Train model
    ts = time.strftime("%m-%d@%H-%M-%S", time.localtime())
    file_name=args.model+"_"+ts+"_epochs_"+str(args.epochs)+"_squeLength_"+str(args.n_his)+"_batchSize_"+str(args.batch_size)
    if model_name =='STM':
        file_name+='_'+model_n
    print_log = open("./print_dir/"+file_name, 'w')
    my_net.train()
    lr_list = []
    for epoch in range(args.epochs):
        if is_slx==True and (epoch==0 or epoch==1 or epoch%10==0) :
            predict_data(my_net, print_log, device, criterion, args, pred_n=1,batch_size=args.batch_size)
        my_net.train()
        epoch_loss=0.0
        start_time=time.time()
        for data in train_loader:
            my_net.zero_grad()
            data["flow_x"]=data["flow_x"].to(device)
            predict_value = my_net(data, train_data.graph).to(device=torch.device("cpu"))
            # predict_value=my_net(data).to(torch.device("cpu"))
            loss=criterion(predict_value,data["flow_y"])
            epoch_loss+=loss
            loss.backward()
            optimizer.step()

        # if epoch==20:
        #     for p in optimizer.param_groups:
        #         p['lr'] = 0.0003  # 注意这里
        # if epoch==30:
        #     for p in optimizer.param_groups:
        #         p['lr'] = 0.0001  # 注意这里

        if epoch%4==0:
            for p in optimizer.param_groups:
                p['lr'] *= 0.9  # 注意这里
            lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])





        end_time = time.time()
        print("Epoch:{:04d},Loss:{:.4f},Time:{:02.2f} mins".format(epoch,10000*epoch_loss/len(train_data),(end_time-start_time)/60))
        print("Epoch:{:04d},Loss:{:.4f},Time:{:02.2f} mins".format(epoch, 10000 * epoch_loss / len(train_data),(end_time - start_time) / 60),file=print_log)

    # plt.plot(range(len(lr_list)), lr_list, color='r')
    # plt.show()

    # predict_data(my_net, print_log, device, criterion, args, pred_n=2)
    if is_long_term==False :
        for i in range(1):
            predict_data(my_net, print_log, device, criterion, args, pred_n=i+1,batch_size=args.batch_size)
            # predict_data(my_net, print_log, device, criterion, args, pred_n=i+1)
    else:
        for i in range(1):
            predict_data(my_net, print_log, device, criterion, args, pred_n=i+1,batch_size=args.batch_size)

    # # 定义总参数量、可训练参数量及非可训练参数量变量
    # Total_params = 0
    # Trainable_params = 0
    # NonTrainable_params = 0
    #
    # # 遍历model.parameters()返回的全局参数列表
    # for param in my_net.parameters():
    #     mulValue = np.prod(param.size())  # 使用numpy prod接口计算参数数组所有元素之积
    #     Total_params += mulValue  # 总参数量
    #     if param.requires_grad:
    #         Trainable_params += mulValue  # 可训练参数量
    #     else:
    #         NonTrainable_params += mulValue  # 非可训练参数量
    #
    # print(f'Total params: {Total_params}')
    # print(f'Trainable params: {Trainable_params}')
    # print(f'Non-trainable params: {NonTrainable_params}')
    # print(f'Total params: {Total_params}', file=print_log)
    # print(f'Trainable params: {Trainable_params}', file=print_log)
    # print(f'Non-trainable params: {NonTrainable_params}', file=print_log)

    print_log.close()

def print_single_prediction(epochs=50):
    # FFNN LSTM_FFNN GCN ChebNet GAT  MyGAT ResGAT ResGAT1 MyGAT1 MyGAT2 MyGAT3 MyGAT3_1 Lstm_tm Lstm RNN GRU LSTM_GAT GRU_GAT LSTM_GAT1 DoubleResGAT ResGATByNum ResGAT1ByNum

    # print("==================start LSTM 模型预测===========================")
    # main("Lstm_tm", epochs, n_his=16,is_long_term=True)
    # print("====================end LSTM 模型预测============================")
    # print("==================start RNN-FBF 模型预测===========================")
    # main("RNN", epochs, n_his=16,is_long_term=True)
    # print("====================end RNN-FBF 模型预测============================")
    # print("==================start LSTM-FBF 模型预测===========================")
    # main("Lstm", epochs=epochs, n_his=16,is_long_term=True,is_slx=True)
    # print("====================end LSTM-FBF 模型预测============================")
    #
    # print("==================start GRU-FBF 模型预测===========================")
    # main("GRU", epochs=epochs, n_his=16,is_long_term=True,is_slx=True)
    # print("====================end GRU-FBF 模型预测============================")
    # print("==================start GCN(1st) 模型预测===========================")
    # main("GCN", epochs, n_his=1,is_long_term=True)
    # print("====================end GCN(1st) 模型预测============================")
    print("==================start GCN(Cheb) 模型预测===========================")
    main("ChebNet", epochs, n_his=1,is_long_term=True)
    print("====================end GCN(Cheb) 模型预测============================")
    print("==================start GAT 模型预测===========================")
    main("GAT", epochs, n_his=1,is_long_term=True)
    print("====================end GAT 模型预测============================")
    print("==================start GSASN 模型预测===========================")
    main("ResGAT", epochs, n_his=1,is_long_term=True,is_slx=True)
    print("====================end GSASN 模型预测============================")
    print("==================start LSTM-FFNN 模型预测===========================")
    main("LSTM_FFNN", epochs, n_his=16,is_long_term=True)
    print("====================end LSTM-FFNN 模型预测============================")
    print("==================start STGCN(Cheb) 模型预测===========================")
    main("cheb_graph_conv", epochs, n_his=16,is_long_term=True)
    print("====================end STGCN(Cheb) 模型预测============================")
    print("==================start STGCN(1st) 模型预测===========================")
    main("graph_conv", epochs, n_his=16,is_long_term=True)
    print("====================end STGCN(1st) 模型预测============================")
    print("==================start FFNN-GSASN 模型预测===========================")
    print_models_zh1(["FFNN_T"],["ResGAT"],epochs,is_long_term=True)
    print("====================end FFNN-GSASN 模型预测============================")
    print("==================start CNN-GSASN 模型预测===========================")
    print_models_zh1(["CNN_T"], ["ResGAT"], epochs,is_long_term=True)
    print("====================end CNN-GSASN 模型预测============================")


    print("==================start GRU-GSASN 模型预测===========================")
    print_models_zh1(["GRU"], ["ResGAT"], epochs,is_long_term=True,is_slx=True)
    print("====================end GRU-GSASN 模型预测============================")
    # print("==================start LSTM-GSASN 模型预测===========================")
    # print_models_zh1(["Lstm"], ["ResGAT"], epochs, is_long_term=True,is_slx=True)
    # print("====================end LSTM-GSASN 模型预测============================")

def print_models_zh1(time_models_name,space_models_name,epochs=50,is_sequence_length=False,is_slx=False,is_long_term=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args, blocks = get_parameters()
    # main("MyGAT3_1",epochs=2)
    # models=["GRU_GAT","cheb_graph_conv","graph_conv","LSTM_FFNN","Lstm","Lstm_tm","GAT","MyGAT3_1",]
    # time_models_name = [ "Lstm","FFNN_T","Lstm_tm","GRU","CNN_T"] #

    # time_models_name = ["FFNN_T"]
    # space_models_name = ["ResGAT"]

    # time_models_name = ["GRU", "Lstm", "FFNN_T", "CNN_T", "Lstm_tm"]
    # space_models_name = ["ChebNet", "ResGAT", "GCN", "CNN_S", "GAT"]

    # space_models_name = ["ResGAT", "cheb_graph_conv", "graph_conv", "CNN_S"]

    if is_sequence_length==True:
        sq_len=[12,14,16,18,20,22,24]
    else:
        sq_len = [16]

    # for tm_name in time_models_name:
    #     print("=============start ",tm_name, "==================")
    #     for l in sq_len:
    #         print("=============sq_len ", l, "===============")
    #         main(tm_name,50,l)
    #         print("======================================")
    #     print("=============end ",tm_name,"==================")

    for l in sq_len:
        for tm_name in time_models_name:
            for sp_name in space_models_name:
                if tm_name == 'CNN_T':
                    tm = CNN_T(device)
                elif tm_name == 'FFNN_T':
                    tm = FFNN_T(l, 1, device)
                elif tm_name == 'Lstm_tm':
                    tm = LSTM_TM(in_dim=529, hidden_dim=529)
                elif tm_name == 'Lstm':
                    tm = LSTM(in_dim=1, hidden_dim=1, device=device)
                elif tm_name == 'GRU':
                    tm = GRU(in_dim=1, hidden_dim=1)
                if sp_name == 'CNN_S':
                    sp = CNN_S(23, 23, device)
                elif sp_name == 'GAT':
                    sp = GAT(nfeat=23,
                             nhid=23,
                             nclass=23,
                             dropout=0.1,
                             nheads=1,
                             alpha=0.2)
                elif sp_name == 'MyGAT3_1':
                    sp = MyGAT3_1(nfeat=23,
                                  nhid=23,
                                  nclass=23,
                                  dropout=0.1,
                                  nheads=1,
                                  alpha=0.2,
                                  device=device)
                elif sp_name == 'GCN':
                    sp = GCN(in_c=23 * 1, hid_c=23 * 1, out_c=23 * 1, device=device)
                elif sp_name == 'ChebNet':
                    sp = ChebNet(in_c=23, hid_c=23, out_c=23, K=2, device=device)
                elif sp_name == 'CNN_S':
                    sp = CNN_S(23, 23, device)
                elif sp_name == 'ResGAT':
                    sp = ResGAT(nfeat=23,
                                nhid=23,
                                nclass=23,
                                dropout=0.1,
                                nheads=1,
                                alpha=0.2,
                                device=device)
                stm = STM1(tm, sp, device)
                print("=============start", tm_name, "-", sp_name, "===============")
                print("=============sq_len ", l, "===============")
                main("STM",epochs,l,model=stm,model_n=tm_name+"-"+sp_name,is_slx=is_slx,is_long_term=is_long_term)
                print("======================================")
                print("=============end", tm_name, "-", sp_name, "===============")

def print_model_effect(epochs=50):
    # epochs = 1
    print("==================start GRU-GSASN 模型预测===========================")
    print_models_zh1(["GRU"], ["ResGAT"], epochs)
    print("====================end GRU-GSASN 模型预测============================")
    print("==================start GRU-FBF 模型预测===========================")
    main("GRU", epochs, n_his=1)
    print("====================end GRU-FBF 模型预测============================")
    print("==================start GSASN-SOD 模型预测===========================")
    main("ResGAT", epochs, n_his=1)
    print("====================end GSASN-SOD 模型预测============================")
    print("==================start GSASN-SO 模型预测===========================")
    main("MyGAT", epochs, n_his=1)
    print("====================end GSASN-SO 模型预测============================")
    print("==================start GSASN-SD 模型预测===========================")
    main("MyGAT1", epochs, n_his=1)
    print("====================end GSASN-SD 模型预测============================")
    print("==================start GAT 模型预测===========================")
    main("GAT", epochs, n_his=1)
    print("====================end GAT 模型预测============================")

def print_sequence_length(epochs=50):

    sq_len = [12, 14, 16, 18, 20, 22, 24]
    # sq_len = [12, 24]
    # print("==================start FFNN-GSASN 模型预测===========================")
    # print_models_zh1(["FFNN_T"], ["ResGAT"], epochs,is_sequence_length=True)
    # print("====================end FFNN-GSASN 模型预测============================")
    print("==================start GRU-GSASN 模型预测===========================")
    print_models_zh1(["GRU"], ["ResGAT"], epochs, is_sequence_length=True)
    print("====================end GRU-GSASN 模型预测============================")
    print("==================start STGCN(Cheb) 模型预测===========================")
    for l in sq_len:
        print("=============sq_len ", l, "===============")
        main("cheb_graph_conv", epochs, n_his=l)
        print("======================================")
    print("====================end STGCN(Cheb) 模型预测============================")
    print("==================start STGCN(1st) 模型预测===========================")
    for l in sq_len:
        print("=============sq_len ", l, "===============")
        main("graph_conv", epochs, n_his=l)
        print("======================================")
    print("====================end STGCN(1st) 模型预测============================")
    print("==================start LSTM-FFNN 模型预测===========================")
    for l in sq_len:
        print("=============sq_len ", l, "===============")
        main("LSTM_FFNN", epochs, n_his=l)
        print("======================================")
    print("====================end LSTM-FFNN 模型预测============================")
    # print("==================start LSTM 模型预测===========================")
    # for l in sq_len:
    #     print("=============sq_len ", l, "===============")
    #     main("Lstm_tm", epochs, n_his=l)
    #     print("======================================")
    # print("====================end LSTM 模型预测============================")

def print_slx(epochs = 50):

    print("==================start LSTM 模型预测===========================")
    main("Lstm_tm", epochs, n_his=16, is_slx=True)
    print("====================end LSTM 模型预测============================")
    print("==================start RNN-FBF 模型预测===========================")
    main("RNN", epochs, n_his=16,is_slx=True)
    print("====================end RNN-FBF 模型预测============================")
    print("==================start LSTM-FBF 模型预测===========================")
    main("Lstm", epochs, n_his=16,is_slx=True)
    print("====================end LSTM-FBF 模型预测============================")
    print("==================start GRU-FBF 模型预测===========================")
    main("GRU", epochs, n_his=16,is_slx=True)
    print("====================end GRU-FBF 模型预测============================")
    print("==================start GCN(1st) 模型预测===========================")
    main("GCN", epochs, n_his=1, is_slx=True)
    print("====================end GCN(1st) 模型预测============================")
    print("==================start GCN(Cheb) 模型预测===========================")
    main("ChebNet", epochs, n_his=1, is_slx=True)
    print("====================end GCN(Cheb) 模型预测============================")
    print("==================start GAT 模型预测===========================")
    main("GAT", epochs, n_his=1, is_slx=True)
    print("====================end GAT 模型预测============================")
    print("==================start GSASN 模型预测===========================")
    main("ResGAT", epochs, n_his=1, is_slx=True)
    print("====================end GSASN 模型预测============================")
    print("==================start LSTM-FFNN 模型预测===========================")
    main("LSTM_FFNN", epochs, n_his=16, is_slx=True)
    print("====================end LSTM-FFNN 模型预测============================")
    print("==================start STGCN(Cheb) 模型预测===========================")
    main("cheb_graph_conv", epochs, n_his=16, is_slx=True)
    print("====================end STGCN(Cheb) 模型预测============================")
    print("==================start STGCN(1st) 模型预测===========================")
    main("graph_conv", epochs, n_his=16, is_slx=True)
    print("====================end STGCN(1st) 模型预测============================")
    print("==================start LSTM-GSASN 模型预测===========================")
    print_models_zh1(["Lstm"], ["ResGAT"], epochs, is_slx=True)
    print("====================end LSTM-GSASN 模型预测============================")
    print("==================start FFNN-GSASN 模型预测===========================")
    print_models_zh1(["FFNN_T"], ["ResGAT"], epochs, is_slx=True)
    print("====================end FFNN-GSASN 模型预测============================")
    print("==================start CNN-GSASN 模型预测===========================")
    print_models_zh1(["CNN_T"], ["ResGAT"], epochs, is_slx=True)
    print("====================end CNN-GSASN 模型预测============================")
    print("==================start GRU-GSASN 模型预测===========================")
    print_models_zh1(["GRU"], ["ResGAT"], epochs, is_slx=True)
    print("====================end GRU-GSASN 模型预测============================")



def print_st_models(epochs=50):
    print_models_zh1(["GRU", "Lstm", "FFNN_T", "CNN_T", "Lstm_tm"],["ChebNet", "ResGAT", "GCN", "CNN_S", "GAT"],epochs=epochs)
    # print_models_zh1(["GRU", "Lstm", "FFNN_T", "CNN_T", "Lstm_tm"], ["GAT","ChebNet", "ResGAT", "GCN", "CNN_S"],
    #                  epochs=1)

def print_models_len():
    # lstm - gat
    # gru - gat
    # stgcn
    # stgcn(cheb)
    # LSTM - FFNN
    # LSTM - FBF
    # LSTM
    print("=============start ", "graph_conv", "==================")
    # print("=============sq_len ", 22, "===============")
    # main("cheb_graph_conv", 15, 22)
    # print("======================================")
    print("=============sq_len ", 24, "===============")
    main("graph_conv", 15, 24)
    print("======================================")
    print("=============end", "graph_conv", "==================")


    models=["LSTM_FFNN","Lstm_tm"]
    sq_len=[12,14,16,18,20,22,24]
    for tm_name in models:
        print("=============start ", tm_name, "==================")
        for l in sq_len:
            print("=============sq_len ", l, "===============")
            main(tm_name,15,l)
            print("======================================")
        print("=============end",tm_name,"==================")

def print_models_efficent():
    models = ["GRU_GAT","cheb_graph_conv","graph_conv","LSTM_FFNN","Lstm","Lstm_tm"]
    # STGCN（cheb）
    # STGCN
    # LSTM - FFNN
    # LSTM - FBF
    # LSTM

    sq_len = [12, 14, 16, 18, 20, 22, 24]
    for tm_name in models:
        print("=============start ", tm_name, "==================")
        for l in sq_len:
            print("=============sq_len ", l, "===============")
            main(tm_name,1,l)
            print("======================================")
        print("=============end",tm_name,"==================")


if __name__=='__main__':
    print_single_prediction(epochs=50) #打印短长期预测 14
    print_model_effect(epochs=50) #打印模型各个模块有效性 6
    print_sequence_length(epochs=50) #打印各个模型随序列长度的变化 6*7
    print_slx(epochs=50) #打印收敛性 10
    print_st_models(epochs=50) #打印各个时间和空间组合性能比较 30