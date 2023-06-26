#!/usr/bin/env python
# coding:utf-8
"""
Author  : Xiwen Lu
Time    : 10/17/22 2:01 PM
Desc    : This file will calculate the BER by sequence MAP (use matrix to accelerate)
"""
import os
from tqdm import tqdm,trange
import math
import numpy as np
# import cupy
from scipy.ndimage import shift
from scipy.stats import poisson

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

def probability(configs, m: int):
    '''

    :param R: the radius of the receiver
    :param d: the distance between the emitter and the receiver
    :param D: the diffusion param
    :param m: the time slot
    :return:
    '''
    R = configs.receiver_radius
    d = configs.distance
    D = configs.diffusion
    ts = configs.delta_t
    Vm = configs.velocity
    t = ts * m
    return configs.particle_num*4*math.pi*pow(R,3) /3/pow(4*math.pi*D*t,1.5) * math.exp(-pow((d-Vm*t),2) / (4*D * t))

def getAffectStep(configs):
    sample_index = np.round(configs.sample_time/configs.delta_t)
    for step in range(1,20):
        if probability(configs,step*sample_index)-0<1e-6*configs.particle_num:
            break
    return step

def calc_theoretical_nhit_num(configs,release_signals:np.ndarray):
    sample_indexs = int(np.round(configs.release_time / configs.delta_t))
    p = [probability(configs, i) for i in range(1, sample_indexs * len(release_signals))]
    p = [0, *p]
    # p_true = np.zeros(len(p))
    # for i in range(len(release_signals)):
    #     if release_signals[i] == 0:
    #         continue
    #     p_true = p_true + shift(p, i * sample_indexs, cval=0)
    tmp = np.array([shift(p, i * sample_indexs, cval=0) for i in range(len(release_signals))])
    p_true = np.sum(tmp*release_signals[:,np.newaxis],axis=0)

    return p_true[sample_indexs-1::sample_indexs]

def get_sample_probability(configs,k:int):
    sample_indexs = int(np.round(configs.release_time/configs.delta_t))
    p = [probability(configs, i) for i in range(1, sample_indexs*k)]
    p=[0,*p]
    return p[sample_indexs-1::sample_indexs]

def calc_MAP_sequence(N_hit_values,configs,L:int):
    # the sequence length should larger than L
    signals_hat = np.zeros(len(N_hit_values)).astype(int)
    affect_step = getAffectStep(configs)
    group_steps = int(np.floor(len(N_hit_values)/L))
    fake_signals = np.arange(2**L,dtype=int).tolist()
    fake_signals_strs = [np.binary_repr(s,width=L) for s in fake_signals]
    fix_signal_matrix = np.array([np.fromiter(s,dtype=int) for s in fake_signals_strs])
    p = get_sample_probability(configs,affect_step)
    for g in trange(group_steps):
        if g*L-affect_step >= 0:
            temp_signals = signals_hat[g*L-affect_step:g*L].astype(int)
        else:
            #           when the g is 0, then the signals need padding
            temp_signals = np.zeros(affect_step,dtype=int)
            # temp_signals = shift(signals_hat[g*L:g*L+affect_step],affect_step,cval=0).astype(int)
        fix_signal_matrix_append = np.concatenate((np.tile(temp_signals,(2**L,1)),fix_signal_matrix),axis=1)
        # fix_signal_matrixs = np.tile(fix_signal_matrix,(affect_step,1,1))
        signal_matrixs = np.array([shift(fix_signal_matrix_append,[0,i],cval=0) for i in range(affect_step)])
        temp = signal_matrixs*np.array(p).reshape(affect_step,1,1)
        e_num_lamda = np.sum(temp[:,:,affect_step:],axis=0)
        joint_p_array = poisson.pmf(np.tile(N_hit_values[g*L:g*L+L],(2**L,1)),e_num_lamda)
        maxP_l = np.argmax(np.prod(joint_p_array,axis=1))
        signals_hat[g * L:g * L + L] = np.fromiter(np.binary_repr(maxP_l,width=L),dtype=int)

    # if len(N_hit_values) mod L != 0, do some specific caculate
    remain_signal = len(N_hit_values)-group_steps*L
    if remain_signal>0:
        L=remain_signal
        fake_signals = np.arange(2 ** L, dtype=int).tolist()
        fake_signals_strs = [np.binary_repr(s, width=L) for s in fake_signals]
        fix_signal_matrix = np.array([np.fromiter(s, dtype=int) for s in fake_signals_strs])
        temp_signals = signals_hat[g * L - affect_step:g * L].astype(int)
        fix_signal_matrix_append = np.concatenate((np.tile(temp_signals,(2**L,1)),fix_signal_matrix),axis=1)
        # fix_signal_matrixs = np.tile(fix_signal_matrix,(affect_step,1,1))
        signal_matrixs = np.array([shift(fix_signal_matrix_append,[0,i],cval=0) for i in range(affect_step)])
        temp = signal_matrixs*np.array(p).reshape(affect_step,1,1)
        e_num_lamda = np.sum(temp[:,:,affect_step:],axis=0)
        joint_p_array = poisson.pmf(np.tile(N_hit_values[g*L:g*L+L],(2**L,1)),e_num_lamda)
        maxP_l = np.argmax(np.prod(joint_p_array,axis=1))
        signals_hat[len(N_hit_values)-L:] = np.fromiter(np.binary_repr(maxP_l,width=L),dtype=int)

    return signals_hat

if __name__ == "__main__":
    configs = {
        'particle_num':4000,
        'receiver_radius':1.5,
        'distance':10,
        'diffusion':79.4,
        'delta_t':0.001,
        'total_time':1000,
        'release_time':0.2,
        'sample_time':0.2,
        'velocity':30
    }
    configs = Struct(**configs)
    file_folder = './Data/data_num_4000_r_1.5/v_30/test'
    # N_hit = np.load('{}d_10.0_r_1_num_10000_nhit.npy'.format(file_folder))
    N_hit_sample = np.load(os.path.join(file_folder,"d_10.0_r_1.5_num_4000_velocity_[30, 0, 0]_0.npy"))
    signals = np.array(N_hit_sample[0])
    # p_true = true_signal_probability(signals,configs=configs)
    signals_hat = calc_MAP_sequence(N_hit_sample[1],configs,L=10)
    wrong_signals_array = np.bitwise_xor(signals.astype(bool),signals_hat.astype(bool))
    BER = np.sum(wrong_signals_array) / len(signals)
    print("The BER is {}".format(BER))