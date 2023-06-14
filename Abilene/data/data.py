import csv
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch
import matplotlib.pyplot as plt

def readData(filename):
# filename = './output/tm.2004-03-01.00-00-00.csv'
    data = []
    with open(filename) as csvfile:
        csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
        # header = next(csv_reader)        # 读取第一行每一列的标题
        matrix = np.zeros((12, 12))
        i=0
        for row in csv_reader: # 将csv 文件中的数据保存到data中
            if i==0:
                i+=1
                continue
            for j in range(12):
                matrix[i-1][j]=float(row[j+1])*1000000
            # print(row)
            # print(len(row))
            # print(type(row[5]))
            # data.append(row[5])  # 选择某一列加入到data数组中
            i+=1
        return matrix


def data_load():
    path = 'output'
    FileList = os.listdir(path)
    FileList.sort()
    i=0
    dataset = []
    data = []
    for file in FileList:
        oldDirPath = path + '/' + file
        matrix=readData(oldDirPath)
        # print("oldDirPath:",oldDirPath)
        # per = ET.parse(oldDirPath)
        # p = per.findall('./IntraTM/src')
        # matrix = np.zeros((12, 12))
        # for child in p:
        #     src = child.attrib['id']
        #     c = child.findall('dst')
        #     for i in c:
        #         dst = i.attrib['id']
        #         matrix[int(src) - 1][int(dst) - 1] = i.text
        dataset.append(matrix)
    datasets = np.array(dataset)
    return datasets

def save_a_matrix():

    result=[]
    for i in range(12):
        for j in range(12):
            result.append(i)
            result.append(j)
            result.append(1)
    result=np.array(result)
    result=result.reshape(-1,3)
    df2 = pd.DataFrame(data=result,
                       columns=['from', 'to', 'cost'])
    df2.to_csv('../simple/abilene_a_metrix.csv', index=False)


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


# save_a_matrix()
datasets=data_load()
datasets=datasets.transpose([1,2,0])
plt.plot(datasets[1,0,-200:-1], color='r', label="40")
plt.show()

norm_base, norm_data = LoadData.pre_process_data(data=datasets, norm_dim=2)
norm_data = np.nan_to_num(norm_data)
np.savez('../simple/abilene_norm.npz', data=norm_data, base=norm_base)

