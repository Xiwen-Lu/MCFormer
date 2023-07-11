#!/usr/bin/env python
# coding:utf-8
"""
Author  : Xiwen Lu
Time    : 6/21/23 12:08 PM
Desc    : 
"""
from Model_MCFormer import *
import train_DNN

if __name__ == '__main__':
    # MCFormer Experiment
    for v in [20,25,30,35,40,45,50]:
        settings_name = "./settings/settings_{}.yaml".format(v)
        settings = yaml.safe_load(open(settings_name))
        print("v: {}\t.".format(v))
        train_mcformer(settings=settings,nlayers=3,nfeedforward=128)
        test_currunt_model(settings=settings)

    # DNN Experiment
    nlayers = 1
    for v in [20,25,30,35,40,45,50]:
        settings_name = "./settings_{}.yaml".format(v)
        settings = yaml.safe_load(open(settings_name))
        settings['weights_name'] = "Model_DNN_{}layers_brownian_n_4000_r_1.5_v_{}.pth".format(nlayers, v)
        print("[v: {} \t nlayers: {}]--------".format(v, nlayers))
        train_DNN.train(settings=settings, nlayer=nlayers)
        _ = train_DNN.test_currunt_model(settings=settings)
