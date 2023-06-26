#!/usr/bin/env python
# coding:utf-8
"""
Author  : Xiwen Lu
Time    : 6/21/23 12:08 PM
Desc    : 
"""
from Model_Transformer import *

if __name__ == '__main__':
    for v in [20,25,30,35,40,50]:
        settings_name = "./settings/settings_{}.yaml".format(v)
        settings = yaml.safe_load(open(settings_name))
        print("v: {}\t.".format(v))
        train_transformer(settings=settings,nlayers=3,nfeedforward=128)
        test_currunt_model(settings=settings)
