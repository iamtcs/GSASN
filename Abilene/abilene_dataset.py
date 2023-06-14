import csv
import math
import torch
import numpy as np
import  matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

def get_adjacent_matrix(distance_file: str,num_nodes: int,id_file: str=None,graph_type="connect") -> np.array:
    A=np.zeros([int(num_nodes),int(num_nodes)])
    with open(distance_file,"r") as f_d:
        f_d.readline()
        reader=csv.reader(f_d)
        for item in reader:
            if len(item)!=3:
                continue
            i,j,distance=int(item[0]),int(item[1]),float(item[2])
            if graph_type=="connect":
                A[i,j],A[j,i]=1.,1.
            elif graph_type=="distance":
                A[i,j]=1./distance
                A[j,i]=1./distance
            else:
                raise ValueError("graph type is not correct (connect or distance)")
        return A

def get_flow_data(flow_file: str)-> np.array:
    flow_data = np.load(flow_file)
    print([key for key in flow_data.keys()])
    flow_data = flow_data['data'].transpose([1, 0, 2])[:, :, 0][:, :, np.newaxis]
    return flow_data

class LoadAbileneData(Dataset):
    def __init__(self,data_path,num_nodes,divide_days,time_interval,histroy_length,train_mode,predict_length=1):
        # divide_days: list,[days of train data,days of test data]
        # time_interval:int ,(mins)
        # train_mode: list,["train","test"]
        self.data_path=data_path
        self.num_nodes=num_nodes
        self.train_mode=train_mode
        self.train_days=divide_days[0]
        self.test_days = divide_days[1]
        self.history_length=histroy_length
        self.predict_length=predict_length
        self.time_interval=time_interval

        self.one_day_length=int(24*60/self.time_interval)
        self.graph=get_adjacent_matrix(distance_file=data_path[0],num_nodes=num_nodes)
        #对时间维度进行归一化
        data_norm_npz = np.load(data_path[1])
        self.flow_norm,self.flow_data = data_norm_npz["base"], data_norm_npz["data"]
        self.nums=self.flow_data.shape[-1]
        self.nums=self.nums-histroy_length-predict_length+1
        self.train_nums=math.floor(self.nums*0.7)
        self.test_nums =self.nums-self.train_nums
        # self.flow_norm,self.flow_data=self.pre_process_data(data=get_flow_data(data_path[1]),norm_dim=1)

    def __len__(self):
        if self.train_mode=='train':
            return self.train_nums
        elif self.train_mode=='test':
            return self.test_nums
        else:
            raise ValueError("train mode:[{}] is not defined".format(self.train_mode))

    def __getitem__(self,index):
        if self.train_mode=='train':
            index=index
        elif self.train_mode=='test':
            index+=self.train_nums
        else:
            raise ValueError("train mode:[{}] is not defined".format(self.train_mode))

        data_x,data_y=LoadAbileneData.slice_data(self.flow_data,self.history_length,index,self.predict_length,self.train_mode)
        data_x=LoadAbileneData.to_tensor(data_x)
        data_y=LoadAbileneData.to_tensor(data_y)

        return {"graph":LoadAbileneData.to_tensor(self.graph),"flow_x":data_x,"flow_y":data_y}

    @staticmethod
    def slice_data(data,history_length,index,predict_length,train_mode):
        # if train_mode=="train":
        start_index=index
        end_index=index+history_length
        # elif train_mode=="test":
        #     start_index=index-history_length
        #     end_index=index
        # else:
        #     raise ValueError("train mode:[{}] is not defined".format(train_mode))

        data_x=data[:,:,start_index:end_index]
        data_y=data[:,:,end_index:end_index+predict_length]

        return data_x,data_y


    @staticmethod
    def pre_process_data(data,norm_dim):
        norm_base=LoadAbileneData.normalize_base(data,norm_dim)
        norm_data=LoadAbileneData.normalize_data(norm_base[0],norm_base[1],data)
        return norm_base,norm_data

    @staticmethod
    def normalize_base(data,norm_dim):
        max_data=np.max(data,norm_dim,keepdims=True)
        min_data=np.min(data,norm_dim,keepdims=True)
        return max_data,min_data

    @staticmethod
    def normalize_data(max_data,min_data,data):
        mid=min_data
        base=max_data-min_data
        normalized_data=(data-mid)/base
        return normalized_data

    @staticmethod
    def recover_data(max_data,min_data,data):
        mid=min_data
        base=max_data-min_data
        recovered_data=data*base+mid
        return recovered_data

    @staticmethod
    def to_tensor(data):
        return torch.tensor(data,dtype=torch.float)

if __name__=='__main__':
    train_data=LoadAbileneData(data_path=["./simple/abilene_a_metrix.csv","./simple/abilene_norm.npz"],num_nodes=12,divide_days=[45,14],
                        time_interval=5,histroy_length=6,train_mode="train")
    print(len(train_data))
    flow_x=train_data[0]["flow_x"]
    print(type(flow_x))
    flow_x_numpy=flow_x.numpy()
    print(train_data[0]["flow_x"].size())
    print(train_data[0]["flow_y"].size())
    test_data = LoadAbileneData(data_path=["./simple/geant_a_metrix.csv", "./simple/geant_norm.npz"], num_nodes=23,
                          divide_days=[45, 14],
                          time_interval=5, histroy_length=6, train_mode="test")
    # test_data = LoadGeantData(data_path=["./simple/geant_a_metrix.csv", "./simple/geant_diff_norm.npz"], num_nodes=23,
    #                           divide_days=[45, 14],
    #                           time_interval=5, histroy_length=1, train_mode="test")
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False, num_workers=0)
    print(len(test_data))
    print(test_data[0]["flow_x"].size())
    print(test_data[0]["flow_y"].size())

    true_value_recover = torch.rand(0)
    for data in test_loader:
        recovered_true = LoadAbileneData.recover_data(max_data=train_data.flow_norm[0], min_data=train_data.flow_norm[1],
                                               data=data["flow_y"])
        true_value_recover = torch.cat([true_value_recover, recovered_true], dim=0)
    print(true_value_recover.shape)
    plt.plot(true_value_recover[-200:-1, 1, 17], color='b', label="true_y")
    # plt.plot(pred_value_recover[-200:-1,1,17], color='r',label="predict_y")
    plt.legend()
    plt.show()
    a=1