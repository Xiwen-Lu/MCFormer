#!/usr/bin/env python
# coding:utf-8
"""
Author  : Xiwen Lu
Time    : 10/17/22 2:01 PM
Desc    : This file will calculate the BER by sequence MAP
"""
from tqdm import tqdm,trange
import math
import numpy as np
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

def true_signal_probability(release_signals,configs):
    sample_indexs = int(np.round(configs.release_time/configs.delta_t))
    p = [probability(configs, i) for i in range(1, sample_indexs*len(release_signals))]
    p=[0,*p]
    p_true = np.zeros(len(p))
    for i in range(len(release_signals)):
        if release_signals[i] == 0:
            continue
        p_true = p_true + shift(p, i * sample_indexs, cval=0)

    return p_true[sample_indexs-1::sample_indexs]

def calc_MAP_sequence(N_hit_values,configs,L:int):
    # the sequence length should larger than L
    signals_hat = np.zeros(len(N_hit_values)).astype(int)
    affect_step = getAffectStep(configs)
    group_steps = int(np.floor(len(N_hit_values)/L))
    for g in trange(group_steps):
        step_array = np.zeros((2**L,L))
        for l in range(2**L):
            signal_curr = np.fromiter(np.binary_repr(l,width=L),dtype=int)
            signals_hat[g*L:g*L+L]=signal_curr
            if g*L-affect_step >= 0:
                temp_signals = signals_hat[g*L-affect_step:g*L+L].astype(int)
            else:
                #           when the g is 0, then the signals need padding
                temp_signals = shift(signals_hat[g*L:g*L+L+affect_step],affect_step,cval=0).astype(int)
            signal_p_calc = true_signal_probability(temp_signals,configs)
            step_array[l] = poisson.pmf(N_hit_values[g*L:g*L+L],signal_p_calc[affect_step:])
        maxP_l = np.argmax(np.prod(step_array,axis=1))
        signals_hat[g * L:g * L + L] = np.fromiter(np.binary_repr(maxP_l,width=L),dtype=int)

    # if len(N_hit_values) mod L != 0, do some specific caculate
    remain_signal = len(N_hit_values)-group_steps*L
    if remain_signal>0:
        L=remain_signal
        step_array = np.zeros((2**L,L))
        for l in range(2**L):
            signal_curr = np.fromiter(np.binary_repr(l,width=L),dtype=int)
            signals_hat[len(N_hit_values)-L:]=signal_curr
            signal_p_calc = true_signal_probability(signals_hat[len(N_hit_values)-L-affect_step:],configs)
            step_array[l] = poisson.pmf(N_hit_values[len(N_hit_values)-L:],signal_p_calc[affect_step:])
        maxP_l = np.argmax(np.prod(step_array,axis=0))
        signals_hat[len(N_hit_values)-L:] = np.fromiter(np.binary_repr(maxP_l,width=L),dtype=int)

    return signals_hat

if __name__ == "__main__":
    configs = {
        'particle_num':10000,
        'receiver_radius':1,
        'distance':10,
        'diffusion':79.4,
        'delta_t':0.001,
        'total_time':10,
        'release_time':0.2,
        'sample_time':0.2,
        'velocity':40
    }
    configs = Struct(**configs)
    file_folder = './Data/test_data/'
    # N_hit = np.load('{}d_10.0_r_1_num_10000_nhit.npy'.format(file_folder))
    N_hit_sample = np.load('{}d_10.0_r_1_num_10000_sample.npy'.format(file_folder))
    signals = np.array(N_hit_sample[0])
    # p_true = true_signal_probability(signals,configs=configs)
    signals_hat = calc_MAP_sequence(N_hit_sample[1],configs,L=10)
    wrong_signals_array = np.bitwise_xor(signals.astype(bool),signals_hat.astype(bool))
    BER = np.sum(wrong_signals_array) / len(signals)
    print("The ber is {}".format(BER))