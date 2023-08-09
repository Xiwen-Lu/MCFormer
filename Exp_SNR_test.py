#!/usr/bin/env python
# coding:utf-8
"""
Author  : Xiwen Lu
Time    : 6/15/23 12:59 PM
Desc    : test the BER under different SNR by using MCFormer and MAP
"""
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats

import train_DNN
from Model_MCFormer import *
from calculate_MAP_sequence_accelerate import *

import yaml

configs = {
    'particle_num': 4000,
    'receiver_radius': 1.5,
    'distance': 10,
    'diffusion': 79.4,
    'delta_t': 0.001,
    'total_time': 1000,
    'release_time': 0.2,
    'sample_time': 0.2,
    'velocity': 40
}
configs = Struct(**configs)


def generate_SNR_data(src_data,dst_file,SNRs):
    alldata = np.load(src_data)
    label = alldata[0]
    data = alldata[1].copy()
    signal_avg_power = np.mean(np.square(data))
    for SNR in SNRs:
        signal = data.copy()
        noise_var = signal_avg_power / (10 ** (SNR / 10))
        noise = np.random.normal(0, math.sqrt(noise_var), len(data))
        signal += noise
        signal[signal<0] = 0
        alldata[1] = signal
        alldata = alldata.astype(int)
        np.save(dst_file+'snr_{}/snr_{}'.format(SNR,SNR),alldata)


def MCFormer_SNR_test(SNRs,file_folder):
    results = np.asarray(SNRs,float)
    settings = yaml.safe_load(open("./settings/settings_40.yaml"))
    for SNR in SNRs:
        print("======SNR: {}======".format(SNR))
        settings['data_filepath']['test'] = file_folder+"/snr_{}".format(SNR)
        BER = test_currunt_model(settings=settings)
        results[SNRs==SNR]=BER
    return results


def MAP_SNR_test(SNRs,file_folder):
    results = np.asarray(SNRs,float)
    for SNR in SNRs:
        N_hit_sample = np.load(os.path.join(file_folder,"snr_{}/snr_{}.npy".format(SNR,SNR)))
        signals = np.array(N_hit_sample[0])
        signals_hat = calc_MAP_sequence(N_hit_sample[1],configs,L=10)
        wrong_signals_array = np.bitwise_xor(signals.astype(bool),signals_hat.astype(bool))
        BER = np.sum(wrong_signals_array) / len(signals)
        print("The MAP BER is {} at SNR {}".format(BER,SNR))
        results[SNRs==SNR]=BER
    return results

def DNN_SNR_test(SNRs,file_folder):
    results = np.asarray(SNRs,float)
    settings = yaml.safe_load(open("./settings/settings_40.yaml"))
    for SNR in SNRs:
        print("======SNR: {}======".format(SNR))
        settings['data_filepath']['test'] = file_folder+"/snr_{}".format(SNR)
        BER = train_DNN.test_currunt_model(settings=settings)
        results[SNRs==SNR]=BER
    return results

if __name__ == '__main__':
    SNRs = np.array([10,15,20,25,30,35,40])
    dst_file_folder = './Data/data_num_4000_r_1.5_SNR_v40/'
    # generate_SNR_data(src_data='./Data/data_num_4000_r_1.5/v_40/test/d_10.0_r_1.5_num_4000_velocity_[40, 0, 0]_0.npy',
    #                   dst_file=dst_file_folder,
    #                   SNRs=SNRs)

    results_MCFormer = MCFormer_SNR_test(SNRs=SNRs,file_folder = dst_file_folder)
    results_MAP = MAP_SNR_test(SNRs=SNRs,file_folder = dst_file_folder)
    results_DNN = DNN_SNR_test(SNRs=SNRs,file_folder = dst_file_folder)

    df0 = pd.DataFrame(SNRs,columns=["SNR"])
    df1 = pd.DataFrame(results_MCFormer,columns=["MCFormer"])
    df2 = pd.DataFrame(results_MAP,columns=["MAP"])
    df3 = pd.DataFrame(results_DNN,columns=["DNN"])
    df = pd.concat([df0,df1,df2,df3],axis=1)
    df.to_csv('pics/pic_datas/SNR_results.csv',index=False)
    print(df)