#!/usr/bin/env python
# coding:utf-8
"""
Author  : Xiwen Lu
Time    : 6/21/23 12:08 PM
Desc    : Reproducing experiments without imposed noise in paper
"""
from Model_MCFormer import *
import train_DNN
from calculate_MAP_sequence_accelerate import *
import os

if __name__ == '__main__':
    # MCFormer Experiment
    # for v in [20,25,30,35,40,45,50]:
    #     settings_name = "./settings/settings_{}.yaml".format(v)
    #     settings = yaml.safe_load(open(settings_name))
    #     print("MCFormer at v={}.".format(v))
    #     train_mcformer(settings=settings,nlayers=3,nfeedforward=128)
    #     test_currunt_model(settings=settings)

    # DNN Experiment
    nlayers = 1
    for v in [25,30,35,40,45,50]:
        settings_name = "./settings/settings_{}.yaml".format(v)
        settings = yaml.safe_load(open(settings_name))
        settings['weights_name']['dnn'] = "Model_DNN_{}layers_brownian_n_4000_r_1.5_v_{}.pth".format(nlayers, v)
        print("DNN at v={}\tnlayers={}.".format(v, nlayers))
        train_DNN.train(settings=settings, nlayer=nlayers)
        train_DNN.test_currunt_model(settings=settings)

    # MAP Experiment
    # configs = {
    #     'particle_num':4000,
    #     'receiver_radius':1.5,
    #     'distance':10,
    #     'diffusion':79.4,
    #     'delta_t':0.001,
    #     'total_time':1000,
    #     'release_time':0.2,
    #     'sample_time':0.2,
    #     'velocity':30
    # }
    # configs = Struct(**configs)
    # for v in [20,25,30,35,40,45,50]:
    #     configs.velocity = v
    #     settings_name = "./settings/settings_{}.yaml".format(v)
    #     settings = yaml.safe_load(open(settings_name))
    #     file_folder = settings['data_filepath']['test']
    #     N_hit_sample = np.load(os.path.join(file_folder,os.listdir(file_folder)[0]))
    #     signals = np.array(N_hit_sample[0])
    #     signals_hat = calc_MAP_sequence(N_hit_sample[1],configs,L=10)
    #     wrong_signals_array = np.bitwise_xor(signals.astype(bool),signals_hat.astype(bool))
    #     BER = np.sum(wrong_signals_array) / len(signals)
    #     print("The BER is {} at v={}.".format(BER,v))