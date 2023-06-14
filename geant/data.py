#头文件
import os
import numpy as np
import pandas as pd
import math
# from pandas import DataFrame
from xml.etree import ElementTree as ET
# import matplotlib.pyplot as plt
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler,StandardScaler
# from sklearn import metrics
# from tensorflow.keras.layers import Dense, Dropout, LSTM, Conv1D, MaxPooling1D
# from tensorflow.keras.layers import *
# from datetime import date
# from tensorflow.keras.models import *
# import tensorflow as tf
import matplotlib.pyplot as plt



def loadData():
    dataset = []
    data = []
    def data_load():
        path = 'traffic-matrices'
        FileList = os.listdir(path)
        FileList.sort()
        for files in FileList:
            oldDirPath = path + '/' + files
            # print("oldDirPath:",oldDirPath)
            per = ET.parse(oldDirPath)
            p = per.findall('./IntraTM/src')
            matrix = np.zeros((23, 23))
            for child in p:
                src = child.attrib['id']
                c = child.findall('dst')
                for i in c:
                    dst = i.attrib['id']
                    matrix[int(src) - 1][int(dst) - 1] = i.text
            dataset.append(matrix)
        datasets = np.array(dataset)
        return datasets

    datasets = data_load()
    print("datasets:",datasets.shape)
    for i in range(len(datasets)):
        arr = []
        for row in datasets[i]:
            arr.extend(row)
        data.append(arr)
    data = np.array(data)
    return data

data=loadData()
print(data.shape)

# scaler1 = MinMaxScaler(feature_range=(0, 1))
# scaler2=StandardScaler()

def getDataByNode(node): #获取节点训练数据和测试数据
    node_data = data[:, node]  #(10772,)
    return node_data

def getData(node_data,start,end,scaler,time_stamp=64,is_train=None):
    # node_data=getDataByNode(node)
    print(node_data.shape)
    node_data=node_data.reshape(-1,1)
    print(node_data.shape)
    node_data=node_data[start:end]
    print(node_data.shape)
    print(node_data[0:time_stamp].reshape(1,-1).shape)
    x=[]
    y=[]
    if is_train is None:
        print("train_node_data min:",node_data.min())
        print("train_node_data max:",node_data.max())
        print("train_0:",node_data[0])
        if scaler is not None:
            node_data=scaler.fit_transform(node_data)
    else:
        print("test_node_data min:",node_data.min())
        print("test_node_data max:",node_data.max())
        if scaler is not None:
            node_data=scaler.transform(node_data)
    for i in range(time_stamp,len(node_data)):
        y.append(node_data[i][0])
        x.append(node_data[i-time_stamp:i].reshape(1,-1))
    x,y=np.array(x),np.array(y)
    print("x.shape:",x.shape)
    print("y.shape:",y.shape)
    return x,y

# def test_print_mape(node):
#     node_data = getDataByNode(node)
#     true = node_data[10600:-3]
#     pred = node_data[10599:-4]
#     mape = []
#     for i in range(len(true)):
#         if true[i] != 0:
#             mape.append(np.abs(true[i] - pred[i]) / true[i] * 100)
#     mape_cz=[]
#     for i in range(1,len(mape)):
#         mape_cz.append(mape[i]-mape[i-1])
#     mape_cz.sort(reverse=True)
#     print((node," -> mape_cz_max:",mape_cz[:30]))
#     print((node," -> mape_cz_min:",mape_cz[-30:]))
#     print(node," -> mape:", np.mean(mape))

if __name__ == '__main__':
    data=data.transpose([1, 0]).reshape(23,23,-1)
    print(data.shape)
    plt.plot(data[1,17,-200:-1], color='r', label="40")
    plt.show()
    np.savez('./simple/geant.npz', data=data)

    # data=np.diff(data,n=1,axis=0)
    # np.savez('./simple/geant_diff.npz', data=data)

    # pa = np.array([[1,3,5],[2,4,6],[1,5,6]])
    # df1 = pd.DataFrame(data=pa,
    #                    columns=['filepath', 'label','test'])
    # df1.to_csv('./simple/filename1.csv')
    # df1.to_csv('./simple/filename2.csv', index=False)
    #
    # data=data[0]
    # result=[]
    # for i in range(len(data)):
    #     if data[i]!=0:
    #         start=i//23
    #         end=i%23
    #         result.append(start)
    #         result.append(end)
    #         result.append(1)
    # result=np.array(result)
    # result=result.reshape(-1,3)
    # df2 = pd.DataFrame(data=result,
    #                    columns=['from', 'to', 'cost'])
    # df2.to_csv('./simple/geant_a_metrix.csv', index=False)

    # time_stamp = 1
    # node=91
    # node_data=getDataByNode(node)
    # print(node_data.shape)
    # plt.plot(np.arange(len(node_data[-1000:])), node_data[-1000:]/1024, label="true_y", color='b')
    # plt.show()

