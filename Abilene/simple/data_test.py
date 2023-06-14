#头文件
import os
import numpy as np
import pandas as pd
import math
import time
from pandas import DataFrame
from xml.etree import ElementTree as ET
import matplotlib.pyplot as plt
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.optimizers import Adam
# from sklearn.preprocessing import MinMaxScaler
# from sklearn import metrics
# from tensorflow.keras.layers import Dense, Dropout, LSTM, Conv1D, MaxPooling1D
# from tensorflow.keras.layers import *
# from datetime import date
# from tensorflow.keras.models import *
# import tensorflow as tf
import matplotlib.pyplot as plt
import csv
import torch
import numpy as np
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

def save_features():
    data_norm_npz = np.load('./geant_diff_norm.npz')
    norm_base, norm_data = data_norm_npz["base"], data_norm_npz["data"]
    # 保存打印行
    for i in range(23):
        for j in range(23):
            plt.plot(np.arange(len(norm_data[:300, i*23+j])), norm_data[:300, i*23+j], label=str(i*23+j))
        plt.legend()
        plt.title("行:" + str(i))
        f = plt.gcf()  # 获取当前图像
        f.savefig("./images/hang" + "/行#" + str(i) + ".png")
        # plt.show()
        f.clear()  # 释放内存
    # 保存打印列
    for i in range(23):
        for j in range(23):
            plt.plot(np.arange(len(norm_data[:300, j*23+i])), norm_data[:300, j*23+i], label=str(j*23+i))
        plt.legend()
        plt.title("列:" + str(i))
        f = plt.gcf()  # 获取当前图像
        f.savefig("./images/lie" + "/列#" + str(i) + ".png")
        # plt.show()
        f.clear()  # 释放内存
    # plt.plot(np.arange(len(norm_data[:300,276])), norm_data[:300,276], label="276")
    # plt.plot(np.arange(len(norm_data[:300, 289])), norm_data[:300, 289], label="289")
    # plt.plot(np.arange(len(norm_data[:300, 286])), norm_data[:300, 286], label="286")
    # plt.plot(np.arange(len(norm_data[:300, 297])), norm_data[:300, 297], label="297")
    # plt.legend()
    # plt.title("node:" + str(0))
    # f = plt.gcf()  # 获取当前图像
    # f.savefig("./images" + "/node#" + str(0) + ".png")
    # plt.show()
    #
    # f.clear()  # 释放内存
    # plt.show()


class LoadData(Dataset):
    def __init__(self,data_path,num_nodes,divide_days,time_interval,histroy_length,train_mode):
        # divide_days: list,[days of train data,days of test data]
        # time_interval:int ,(mins)
        # train_mode: list,["train","test"]
        self.data_path=data_path
        self.num_nodes=num_nodes
        self.train_mode=train_mode
        self.train_days=divide_days[0]
        self.test_days = divide_days[1]
        self.history_length=histroy_length
        self.time_interval=time_interval

        self.one_day_length=int(24*60/self.time_interval)
        self.graph=get_adjacent_matrix(distance_file=data_path[0],num_nodes=num_nodes)
        self.flow_norm,self.flow_data=self.pre_process_data(data=get_flow_data(data_path[1]),norm_dim=1)

    def __len__(self):
        if self.train_mode=='train':
            return self.train_days*self.one_day_length-self.history_length
        elif self.train_mode=='test':
            return self.test_days*self.one_day_length
        else:
            raise ValueError("train mode:[{}] is not defined".format(self.train_mode))

    def __getitem__(self,index):
        if self.train_mode=='train':
            index=index
        elif self.train_mode=='test':
            index+=self.train_days*self.one_day_length
        else:
            raise ValueError("train mode:[{}] is not defined".format(self.train_mode))

        data_x,data_y=LoadData.slice_data(self.flow_data,self.history_length,index,self.train_mode)
        data_x=LoadData.to_tensor(data_x)
        data_y=LoadData.to_tensor(data_y).unsqueeze(1)

        return {"graph":LoadData.to_tensor(self.graph),"flow_x":data_x,"flow_y":data_y}

    @staticmethod
    def slice_data(data,history_length,index,train_mode):
        if train_mode=="train":
            start_index=index
            end_index=index+history_length
        elif train_mode=="test":
            start_index=index-history_length
            end_index=index
        else:
            raise ValueError("train mode:[{}] is not defined".format(train_mode))

        data_x=data[:,start_index:end_index]
        data_y=data[:,end_index]

        return data_x,data_y


    @staticmethod
    def pre_process_data(data,norm_dim):
        norm_base=LoadData.normalize_base(data,norm_dim)
        norm_data=LoadData.normalize_data(norm_base[0],norm_base[1],data)
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
    # A=get_adjacent_matrix("./geant_a_metrix.csv",23)
    # print(A)
    # save_features()
    data_npz = np.load('./abilene_norm.npz')
    data = data_npz["data"] #[N,N,T]

    data_40=data[1,0,:]

    # print(data.shape)
    # data=data.reshape(23,23,-1)
    # plt.plot(data[1,17,-200:-1], color='r', label="40")
    # plt.show()
    # # np.savez('./geant_diff.npz', data=data)
    # norm_base, norm_data = LoadData.pre_process_data(data=data, norm_dim=2)
    # norm_data = np.nan_to_num(norm_data)
    # np.savez('./geant_norm.npz', data=norm_data, base=norm_base)

    data_norm_npz = np.load('./abilene_norm.npz')
    norm_base, norm_data=data_norm_npz["base"],data_norm_npz["data"]
    # norm_data[norm_data == float("inf")] = 0
    # norm_data[norm_data == float("nan")] = 0
    # norm_data[norm_data == float("NAN")] = 0
    # norm_data=np.nan_to_num(norm_data)
    # norm_base,norm_data=LoadData.pre_process_data(data=data,norm_dim=2)
    recovered_data=LoadData.recover_data(norm_base[0],norm_base[1],norm_data)
    plt.plot(recovered_data[1,0,-200:-1], color='r', label="40")
    plt.show()
    # print(recovered_data.shape)
    arr=recovered_data-data
    bool=recovered_data==data
    # print((recovered_data==data).all())

    # np.savez('./geant_diff_norm.npz', data=norm_data,base=norm_base)
    # plt.plot(np.arange(len(norm_data[:300,276])), norm_data[:300,276], label="276")
    # plt.plot(np.arange(len(norm_data[:300, 289])), norm_data[:300, 289], label="289")
    # plt.plot(np.arange(len(norm_data[:300, 286])), norm_data[:300, 286], label="286")
    # plt.plot(np.arange(len(norm_data[:300, 297])), norm_data[:300, 297], label="297")
    # plt.show()
    a=0
    # data=np.diff(data,n=1,axis=0)
    # np.savez('./data/geant_diff.npz', data=data)


    # a = np.array([[1,3,5],[2,4,6],[1,5,6]])
    # print(a)
    # b=np.diff(a)
    # print(b)
    # print(a)
    # print(np.diff(a))
    # print(np.diff(a, n=1, axis=1))