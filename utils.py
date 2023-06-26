#!/usr/bin/env python
# coding:utf-8
"""
Author  : Xiwen Lu
Time    : 11/1/22 3:25 PM
Desc    : 
"""
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

def combine_npy(classify:str):
    """

    :return: combine the simulation data and return a boxplot csv
    """

    data1 = np.load('Data/data_temp/'+classify+'/d_10.0_r_1.5_num_4000_velocity_[20, 0, 0]_0.npy')
    data1 = np.concatenate((data1,20*np.ones((1,50))),axis=0)
    data2 = np.load('Data/data_temp/'+classify+'/d_10.0_r_1.5_num_4000_velocity_[30, 0, 0]_0.npy')
    data2 = np.concatenate((data2,30*np.ones((1,50))),axis=0)
    data3 = np.load('Data/data_temp/'+classify+'/d_10.0_r_1.5_num_4000_velocity_[40, 0, 0]_0.npy')
    data3 = np.concatenate((data3,40*np.ones((1,50))),axis=0)
    data4 = np.load('Data/data_temp/'+classify+'/d_10.0_r_1.5_num_4000_velocity_[50, 0, 0]_0.npy')
    data4 = np.concatenate((data4,50*np.ones((1,50))),axis=0)
    data = np.concatenate((data1,data2,data3,data4),axis=1)
    df1 = pd.DataFrame(data[1],columns=["$$n_i$$"])
    df2 = pd.DataFrame(data[0],columns=["signals"])
    df3 = pd.DataFrame(data[2],columns=[classify])
    df = pd.concat([df1,df2,df3],axis=1)
    df.to_csv('pics/pic_datas/data_boxplot_'+classify+'.csv',index=False)
    # np.save('Data/data_new_5000/data_boxplot_d_10.0_r_1.5_num_100_vs_N.npy',data)
    print(data.shape)

def write_results():
    eps = np.finfo(float).eps
    results_transformer=np.array([0.059596,0.028283,0.009091,0.016162,0.004040,0.002424,0.002020])
    results_MAP = np.array([0.063,0.03,0.008,0.012,0.002,0.0046,0.001])
    results_DNN = np.array([0.103,0.058,0.029,0.031,0.012,0.0082,0.005])
    velocity = np.array([20,25,30,35,40,45,50])
    df0 = pd.DataFrame(velocity,columns=["velocity"])
    df1 = pd.DataFrame(results_transformer,columns=["MCFormer"])
    df2 = pd.DataFrame(results_MAP,columns=["MAP"])
    df3 = pd.DataFrame(results_DNN,columns=["DNN"])
    df = pd.concat([df0,df1,df2,df3],axis=1)
    df.to_csv('pics/pic_datas/BER_results.csv',index=False)
    print(df)

def combine_logs():
    for v in [20,25,30,35,40,45,50]:
        losses = np.load('logs/train_loss/Model_Transformer_brownian_n_4000_r_1.5_v_{}.pth.npy'.format(v))
        losses_c = [np.sum(losses[i*101:(i+1)*101]) for i in range(50)]
        np.save('logs/Model_Transformer_brownian_n_4000_r_1.5_v_{}.pth.npy'.format(v),np.array(losses_c))
        # print(losses_c)

def write_train_logs(log_name='val_acc'):
    losses_20=np.load('logs/{}/Model_Transformer_brownian_n_4000_r_1.5_v_20.pth.npy'.format(log_name))
    losses_25=np.load('logs/{}/Model_Transformer_brownian_n_4000_r_1.5_v_25.pth.npy'.format(log_name))
    losses_30=np.load('logs/{}/Model_Transformer_brownian_n_4000_r_1.5_v_30.pth.npy'.format(log_name))
    losses_35=np.load('logs/{}/Model_Transformer_brownian_n_4000_r_1.5_v_35.pth.npy'.format(log_name))
    losses_40=np.load('logs/{}/Model_Transformer_brownian_n_4000_r_1.5_v_40.pth.npy'.format(log_name))
    losses_45=np.load('logs/{}/Model_Transformer_brownian_n_4000_r_1.5_v_45.pth.npy'.format(log_name))
    losses_50=np.load('logs/{}/Model_Transformer_brownian_n_4000_r_1.5_v_50.pth.npy'.format(log_name))
    df_0 = pd.DataFrame(np.arange(1,51,1),columns=["epoch"])
    df_20=pd.DataFrame(losses_20,columns=["velocity: 20"])
    df_25=pd.DataFrame(losses_25,columns=["velocity: 25"])
    df_30=pd.DataFrame(losses_30,columns=["velocity: 30"])
    df_35=pd.DataFrame(losses_35,columns=["velocity: 35"])
    df_40=pd.DataFrame(losses_40,columns=["velocity: 40"])
    df_45=pd.DataFrame(losses_45,columns=["velocity: 45"])
    df_50=pd.DataFrame(losses_50,columns=["velocity: 50"])
    df = pd.concat([df_0,df_20,df_25,df_30,df_35,df_40,df_45,df_50],axis=1)
    df.to_csv('pics/pic_datas/val_acc_results.csv',index=False)
    print(df)

# def combine_simulation_data():
#     n_hit_all = np.load('Data/data_temp/d_10.0_r_1.5_num_4000_velocity_[20, 0, 0]_0.npy')
#     n_hit_sample = np.load('Data/data_temp/d_10.0_r_1.5_num_4000_velocity_[20, 0, 0]_0.npy')
#     n_hit_all = np.load('Data/data_temp/d_10.0_r_1.5_num_4000_velocity_[20, 0, 0]_0.npy')
#
#     data = np.concatenate((data1,data2,data3,data4),axis=1)
#     df1 = pd.DataFrame(data[1],columns=["$$n_i$$"])
#     df2 = pd.DataFrame(data[0],columns=["signals"])
#     df3 = pd.DataFrame(data[2],columns=[classify])
#     df = pd.concat([df1,df2,df3],axis=1)
#     df.to_csv('pics/pic_datas/data_boxplot_'+classify+'.csv',index=False)

def prepare_3d_data_simulation_data():
    n0 = np.load('Data/data_temp/0.npy')
    n3 = np.load('Data/data_temp/3.npy')
    n4 = np.load('Data/data_temp/4.npy')
    n5 = np.load('Data/data_temp/5.npy')
    n7 = np.load('Data/data_temp/7.npy')
    df1 = pd.DataFrame(n0,columns=[0])
    df2 = pd.DataFrame(n3,columns=[3])
    df3 = pd.DataFrame(n4,columns=[4])
    df4 = pd.DataFrame(n5,columns=[5])
    df5 = pd.DataFrame(n7,columns=[7])
    df = pd.concat([df1,df2,df3,df4,df5],axis=1)
    df.to_csv('pics/pic_datas/data_3d_simulation.csv',index=False)
    return df

def prepare_nhit_sample():
    n_hit_all = np.load('Data/data_temp/all_nhit.npy')
    release_signals = np.array([1,0,0,1,1,1,0,1])
    sample_index = 200
    df1 = pd.DataFrame(n_hit_all,columns=['all_nhit'])
    df2 = pd.DataFrame(release_signals,columns=['signals'])
    df3 = pd.DataFrame(np.arange(sample_index-1,1600,sample_index),columns=['sample_x'])
    df4 = pd.DataFrame(n_hit_all[sample_index - 1::sample_index],columns=['sample_y'])
    df = pd.concat([df1,df2,df3,df4],axis=1)
    df.to_csv('pics/pic_datas/data_simulation_8_signals.csv',index=False)

if __name__ == '__main__':
    # combine_npy('v')
    # write_results()
    write_train_logs()
    # combine_logs()
    # prepare_3d_data_simulation_data()
    # prepare_nhit_sample()
