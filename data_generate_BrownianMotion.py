#!/usr/bin/env python
# coding:utf-8
"""
Author  : Xiwen Lu
Time    : 2022/9/28 15:50
Desc    : generate the signal through the brownian motion, use array to accelerate
"""
import os
# import torch
import numpy as np
import math
import matplotlib.pyplot as plt
import torch
from tqdm import trange
from scipy.ndimage import shift
import time


def probability(configs, m: int):
    '''

    :param R: the radius of the receiver
    :param d: the distance between the emitter and the receiver
    :param D: the diffusion param
    :param m: the time slot
    :return:
    '''
    R = configs["receiver_radius"]
    d = configs["distance"]
    D = configs["diffusion"]
    ts = configs["delta_t"]
    Vm = configs["velocity"][0]
    particle_num = configs["particle_num"]
    t = ts * m
    return particle_num*4*math.pi*pow(R,3) /3/pow(4*math.pi*D*t,1.5) * math.exp(-pow((d-Vm*t),2) / (4*D * t))


def getAffectStep(configs):
    for step in range(1, 30):
        if probability(configs, step * configs["sample_index"]) - 0 < 1e-6 * configs["particle_num"]:
            break

    return step


def BrownianMotionAndDrift_pytorch_accelerate(configs,release_signals,file_folder="./Data/data_new_5000/",is_gendata_only=False):
    particle_num = configs["particle_num"]
    total_time = configs["total_time"]
    delta_t = configs["delta_t"]
    sample_time = configs["sample_time"]
    release_time = configs["release_time"]
    receiver_radius = configs["receiver_radius"]
    distance = configs["distance"]
    receiver_location = configs["receiver_location"]
    velocity = configs["velocity"]
    diffusion = configs["diffusion"]
    sample_index = int(np.round(sample_time / delta_t))
    release_times = configs["release_times"]
    release_indexs = int(np.round(release_time / delta_t))
    affect_step = getAffectStep(configs)

    # start torch gpu accelerate data generating.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # create a total array than contains all the information
    num_hit = torch.zeros(int(total_time/delta_t),device=device)
    num_hit_sample = torch.zeros(release_times,device=device)
    # do iteration for the release times, caculate the brownian motion of the particles
    t = trange(release_times)
    for i in t:
        t.set_postfix({'Sample_index': j})
        if release_signals[i] == 0:
            continue
        total_data = torch.zeros((affect_step * release_indexs, particle_num, 4),device=device)

        # for j in range(particle_num):
        # curr_release_time = int(np.round(i * release_time/delta_t))
        curr_release_time = 0
        remain_steps = total_data.shape[0] - curr_release_time - 1
        total_data[curr_release_time + 1:, :, 0] = torch.cumsum(torch.normal(
            velocity[0] * delta_t, math.sqrt(2 * diffusion * delta_t),
            size=(remain_steps, particle_num)), dim=0)
        # consider that the velocity y = z, so here otal_data.shape[0]-curr_release_time-1combain the two part
        total_data[curr_release_time + 1:, :, 1:3] = torch.cumsum(torch.normal(
            velocity[1] * delta_t, math.sqrt(2 * diffusion * delta_t),
            size=(remain_steps, particle_num, 2)), dim=0)

        # here, if just generate data, then sample first, and don't save num_hit
        if is_gendata_only:
            total_data = total_data[sample_index-1::sample_index,:,:]
            tmp = torch.sqrt(torch.sum((total_data[:, :, 0:3] - torch.tensor(receiver_location,device=device)) ** 2, dim=2))
            try:
                num_hit_sample[i:i+affect_step]+=torch.sum(tmp<receiver_radius, dim=1)
            except:
                num_hit_sample[i:]+=torch.sum(tmp<receiver_radius, dim=1)[:len(num_hit_sample)-i]
        else:
            tmp = torch.sqrt(torch.sum((total_data[:, :, 0:3] - torch.tensor(receiver_location,device=device)) ** 2, dim=2))
            try:
                num_hit[i * release_indexs:(i + affect_step) * release_indexs] += torch.sum(tmp<receiver_radius, dim=1)
            except:
                num_hit[i * release_indexs:] += torch.sum(tmp<receiver_radius, dim=1)[:len(num_hit) - i * release_indexs]
    np.save('./Data/data_new_5000/d_{}_r_{}_num_{}_{}_nhit'.format(distance,receiver_radius,particle_num,j),num_hit)
    # here, the start symbol from sample_index-1, to ensure the sample dots equal the signals
    if is_gendata_only==False:
        num_hit_sample = num_hit[sample_index - 1::sample_index].cpu().detach().numpy()


    np.save('{}d_{}_r_{}_num_{}_{}'.format(file_folder, distance, receiver_radius, particle_num, j),
            [release_signals[:num_hit_sample.shape[0]], num_hit_sample.cpu().detach().numpy()])

    return num_hit.cpu().detach().numpy(),num_hit_sample

def BrownianMotionAndDrift(configs,release_signals,file_folder="./Data/data_new_5000/"):
    particle_num = configs["particle_num"]
    total_time = configs["total_time"]
    delta_t = configs["delta_t"]
    sample_time = configs["sample_time"]
    release_time = configs["release_time"]
    receiver_radius = configs["receiver_radius"]
    distance = configs["distance"]
    receiver_location = configs["receiver_location"]
    velocity = configs["velocity"]
    diffusion = configs["diffusion"]
    sample_index = int(np.round(sample_time / delta_t))
    release_times = configs["release_times"]
    release_indexs = int(np.round(release_time / delta_t))

    affect_step = getAffectStep(configs)
    print(affect_step)
    affect_step = 3

    # create a total array than contains all the information
    num_hit = np.zeros(int(total_time / delta_t))
    num_hit_sample = np.zeros(release_times)
    # do iteration for the release times, caculate the brownian motion of the particles
    t = trange(release_times)
    for i in t:
        t.set_postfix({'Sample_index': j})
        if release_signals[i] == 0:
            continue
        total_data = np.zeros((affect_step * release_indexs, particle_num, 4))

        # for j in range(particle_num):
        # curr_release_time = int(np.round(i * release_time/delta_t))
        curr_release_time = 0
        remain_steps = total_data.shape[0] - curr_release_time - 1
        tx = np.random.normal(
            velocity[0] * delta_t, math.sqrt(2 * diffusion * delta_t),
            size=(remain_steps, particle_num))
        total_data[curr_release_time + 1:, :, 0] = np.cumsum(tx,axis=0)
        tyz = np.random.normal(
            velocity[1] * delta_t, math.sqrt(2 * diffusion * delta_t),
            size=(remain_steps, particle_num, 2))
        total_data[curr_release_time + 1:, :, 1:3] = np.cumsum(tyz,axis=0)
        # total_data[curr_release_time + 1:, :, 0] = np.cumsum(np.random.normal(
        #     velocity[0] * delta_t, math.sqrt(2 * diffusion * delta_t),
        #     size=(remain_steps, particle_num)), axis=0)
        # # consider that the velocity y = z, so here otal_data.shape[0]-curr_release_time-1combain the two part
        # total_data[curr_release_time + 1:, :, 1:3] = np.cumsum(np.random.normal(
        #     velocity[1] * delta_t, math.sqrt(2 * diffusion * delta_t),
        #     size=(remain_steps, particle_num, 2)), axis=0)

        tmp = np.sqrt(np.sum((total_data[:, :, 0:3] - receiver_location) ** 2, axis=2))
        total_data[tmp < receiver_radius, 3] = 1

        try:
            num_hit[i * release_indexs:(i + affect_step) * release_indexs] += np.sum(total_data[:, :, 3], axis=1)
            # np.save('Data/data_temp/{}'.format(i),np.sum(total_data[:, :, 3], axis=1))
        except:
            num_hit[i * release_indexs:] += np.sum(total_data[:, :, 3], axis=1)[:len(num_hit) - i * release_indexs]
            # np.save('Data/data_temp/{}'.format(i),np.sum(total_data[:, :, 3], axis=1)[:len(num_hit) - i * release_indexs])

    # np.save('Data/data_temp/avg/d_{}_r_{}_num_{}_{}_nhit'.format(distance,receiver_radius,particle_num,j),num_hit)
    # np.save('Data/data_temp/all_nhit',num_hit)
    # here, the start symbol from sample_index-1, to ensure the sample dots equal the signals
    num_hit_sample = num_hit[sample_index - 1::sample_index]

    # np.save('{}d_{}_r_{}_num_{}_velocity_{}_{}'.format(file_folder, distance, receiver_radius, particle_num, velocity,j),
    #       [release_signals[:num_hit_sample.shape[0]], num_hit_sample])

    return num_hit,num_hit_sample

def calc_theoretical_nhit_num(configs,release_signals:np.ndarray):
    sample_indexs = int(np.round(configs["release_time"] / configs["delta_t"]))
    p = [probability(configs, i) for i in range(1, sample_indexs * len(release_signals))]
    p = [0, *p]
    # p_true = np.zeros(len(p))
    # for i in range(len(release_signals)):
    #     if release_signals[i] == 0:
    #         continue
    #     p_true = p_true + shift(p, i * sample_indexs, cval=0)
    tmp = np.array([shift(p, i * sample_indexs, cval=0) for i in range(len(release_signals))])
    p_true = np.sum(tmp*release_signals[:,np.newaxis],axis=0)

    return p_true

def draw_simulate_results(configs,single_simulation,avg_simulation,release_signals,num_hit_sample):
    # sample_index = int(np.round(configs["sample_time"] / configs["delta_t"]))
    theoretical_nums = calc_theoretical_nhit_num(configs, release_signals)
    np.save('Data/data_temp/theoretical_nums.npy',theoretical_nums)
    # plt.figure(figsize=(10,8))
    plt.plot(single_simulation,color='orange')
    plt.plot(avg_simulation,color='green')
    plt.plot(theoretical_nums,color='blue')
    plt.scatter(configs["sample_index"]*np.arange(1,num_hit_sample.shape[0]+1)*release_signals,num_hit_sample*release_signals,color='red',s=50,zorder=3)
    plt.scatter(configs["sample_index"]*np.arange(1,num_hit_sample.shape[0]+1)*(1-release_signals),num_hit_sample*(1-release_signals),color='blue',s=50,zorder=3)
    plt.title('d={},r={},num={},v={}\nsignals={}'.format(configs["distance"],configs["receiver_radius"],configs["particle_num"],configs["velocity"],release_signals))
    # plt.savefig('pics/new.svg',dpi=600)
    plt.show()


if __name__ == '__main__':
    # here, wo should ensure that all time params can be divided by delta_time
    """
    unit:
    total_time: s
    delta_time: s
    diffusion: μm^2/s
    sphere_radius: μm
    velocity:μm/s
    """
    configs = {
        'particle_num': 4000,
        'receiver_radius': 1.5,
        'distance': 10.,
        'receiver_location': [10, 0, 0],
        'diffusion': 79.4,
        'delta_t': 0.001,
        'total_time': 1000,
        'release_time': 0.2,
        'velocity': [30, 0, 0]
    }
    # computational attributes
    configs["sample_time"] = configs["release_time"]
    configs["sample_index"] = int(np.round(configs["sample_time"] / configs["delta_t"]))
    configs["release_times"] = release_times = int(np.round(configs["total_time"] / configs["release_time"]))

    # release_signals = np.ones(release_times)
    num_hit_avg = np.zeros(int(configs["total_time"] / configs["delta_t"]))
    start_runing_time = time.time()
    random_samples = 1
    # to generate different data, this code should be put in the loop
    # release_signals = np.random.binomial(1, 0.5, release_times)
    # release_signals = np.array([1,0,0,1,1,1,0,1])
    release_signals = np.ones(release_times)


    for j in range(random_samples):
        # release_signals = np.random.binomial(1, 0.5, release_times)
        single_start_time = time.time()
        num_hit, num_hit_sample = BrownianMotionAndDrift(configs,release_signals,file_folder="./Data/speed_test/")
        single_end_time = time.time()
        print("Single Simulation Finished.\tRunning time is {}.".format(single_end_time - single_start_time))
        # num_hit_avg += num_hit
    # num_hit_avg = num_hit_avg / random_samples
    end_running_time = time.time()
    # np.save('Data/data_temp/avg/num_hit_avg',num_hit_avg)
    draw_simulate_results(configs,num_hit,num_hit_avg,release_signals,num_hit_sample)
    print("Simulation Finished.\nTotal running time is {}.".format(end_running_time - start_runing_time))
