#!/usr/bin/env python
# coding:utf-8
"""
Author  : Xiwen Lu
Time    : 2022/5/23 23:13
Desc    : 
"""
from functools import partial

from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import pandas as pd
import yaml
import os
import scipy.io
import seaborn

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager, ticker
from matplotlib.animation import FuncAnimation

# plt.style.use('MyPapers.yaml')
plt.style.use('science.yaml')

paper_pic_filefolder = "/home/luxiwen/Documents/MyPapers/001-MCFormer/MCFormer_CL/pics/"
# paper_pic_filefolder = "./pics/"

def update_3d_pic(frame, pic_moleculars, pic_title, positions):
    slice = frame
    x = positions[:1000, slice, 0]
    y = positions[:1000, slice, 1]
    z = positions[:1000, slice, 2]
    # moleculars.set_3d_properties(x,y,z)
    pic_moleculars._offsets3d = (x, y, z)
    pic_title.set_text('Brownian Motion, time slot={}'.format(frame))

    # plt.gca().view_init(elev=10*i, azim=10*i)
    return pic_moleculars, pic_title,


def draw_3d_brownian_simulation(simulation_data_path, slice=50, elevation_angle=0, azimuthal_angle=-90,
                                is_generate_gif=False):
    # Creating figure
    fig = plt.figure(figsize=(10, 7))
    ax = plt.axes(projection="3d")

    # Creating plot
    data_positions = np.load(simulation_data_path)
    x = data_positions[:1000, slice, 0]
    y = data_positions[:1000, slice, 1]
    z = data_positions[:1000, slice, 2]
    moleculars = ax.scatter3D(x, y, z, s=10, marker='o', c=1 / (y * y + z * z + 1e-3), cmap='summer')
    ax.scatter3D(0., 0., 0., color="black", s=80)
    r = 1.5
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    xx = r * np.outer(np.cos(u), np.sin(v)) + 10
    yy = r * np.outer(np.sin(u), np.sin(v))
    zz = r * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(xx, yy, zz, rstride=1, cstride=1, color='purple', alpha=1)
    ax.plot_wireframe(xx, yy, zz, rstride=10, cstride=10, color='black', linewidth=0.5)

    # plt.gca().set_box_aspect((3, 5, 2))
    plt.axis('off')
    plt.gca().view_init(elev=elevation_angle, azim=azimuthal_angle)
    plt.savefig('pics/MC_system3.svg', bbox_inches='tight', dpi=600, format='svg')
    title = plt.title("radius={},slice={}".format(r, slice))
    plt.show()

    if is_generate_gif:
        anim = FuncAnimation(fig,
                             partial(update_3d_pic, pic_moleculars=moleculars, pic_title=title,
                                     positions=data_positions),
                             frames=np.arange(0, 100), interval=200)
        anim.save('pics/3d_brownian_simulation.gif', dpi=120, writer='ffmpeg')


def draw_pic_data_simulations(N_hit_all_path, N_hit_avg_path, N_hit_and_signal_path, N_hit_theoretical_path):
    N_hit = np.load(N_hit_all_path)
    N_hit_avg = np.load(N_hit_avg_path)
    N_hit_sample = np.load(N_hit_and_signal_path)
    N_hit_theoretical = np.load(N_hit_theoretical_path)
    print("N_hit_sample[0]: All signals     . e.g. [0 0 1  1  1 ...]")
    print("N_hit_sample[1]: All N_hit values. e.g. [0 0 19 21 18...]")
    print("N_hit: All N_hit values for each delta_t.")
    plt.figure(figsize=(6, 3.5))
    # orange,green,black
    # plt.plot(N_hit,color='orange',label='Simulation result '+r'$n_{i}$')
    # plt.plot(N_hit_avg,color='green' , label='Average simulation result '+r'$\bar n_{i}$')
    # plt.plot(N_hit_theoretical,color='black', linewidth=1, label='Derived '+r'$n_{i}^{*}$')
    # try another palette
    plt.plot(N_hit, color='#efd6d6', label='Simulation result ' + r'$n_{i}$')
    plt.plot(N_hit_avg, color='#18bc55', label='Average simulation result ' + r'$\bar n_{i}$')
    plt.plot(N_hit_theoretical, color='#490649', linewidth=1, label='Derived ' + r'$n_{i}^{*}=\mathrm{M}\times p(t)$')
    sample_index = int(N_hit.shape[0] / N_hit_sample.shape[1])
    signal_0_x = []
    signal_0_y = []
    signal_1_x = []
    signal_1_y = []
    for i in range(N_hit_sample.shape[1]):
        if N_hit_sample[0][i] == 0:
            signal_0_x.append(i + 1)
            signal_0_y.append(N_hit_sample[1][i])
        else:
            signal_1_x.append(i + 1)
            signal_1_y.append(N_hit_sample[1][i])
    plt.scatter(sample_index * np.array(signal_1_x), signal_1_y, color='red', s=18, zorder=3,
                label='Sample result @ ' + r'$s_{i}=1$')
    plt.scatter(sample_index * np.array(signal_0_x), signal_0_y, color='blue', s=18, zorder=2,
                label='Sample result @ ' + r'$s_{i}=0$')
    # plt.title('d_10.0_r_1_num_10000')
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles[:], labels[:])
    # plt.legend()
    plt.ylim(top=60)
    plt.ylabel('Received Molecules ' + r'$n_i$')
    plt.xlabel('Time ' + r'$(s)$')
    plt.xticks([200 * i for i in range(6)], labels=[0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.savefig(paper_pic_filefolder + 'data_simulation.eps', dpi=600, bbox_inches='tight', format='eps')
    plt.savefig('pics/data_simulation.svg', dpi=600, bbox_inches='tight', format='svg')
    plt.show()


def draw_pic_data_simulations_without_axis(N_hit_all_path):
    df = pd.read_csv(N_hit_all_path)
    N_hit = df['all_nhit']
    signals = np.array(df['signals'])
    sample_x = np.array(df['sample_x'])
    sample_y = np.array(df['sample_y'])
    plt.figure(figsize=(3, 1.5))
    # plt.plot(N_hit,color='#8EA0C9')
    signal_0_x = []
    signal_0_y = []
    signal_1_x = []
    signal_1_y = []
    for i in range(len(signals)):
        if signals[i] == 0:
            signal_0_x.append(sample_x[i])
            signal_0_y.append(sample_y[i])
        else:
            signal_1_x.append(sample_x[i])
            signal_1_y.append(sample_y[i])
    c1 = ['#490649', '#490649']
    c2 = ['red', 'blue']
    c = c2
    plt.scatter(signal_1_x, signal_1_y, color=c[0], s=23, zorder=3,
                label='Sample result @ ' + r'$s_{i}=1$')
    plt.scatter(signal_0_x, signal_0_y, color=c[1], s=23, zorder=3,
                label='Sample result @ ' + r'$s_{i}=0$')
    # plt.title('d_10.0_r_1_num_10000')
    # handles, labels = plt.gca().get_legend_handles_labels()
    # plt.legend(handles[:], labels[:])
    plt.axis('off')
    plt.ylim(top=54)
    # plt.ylabel('Received Molecules '+r'$n_i$')
    # plt.xlabel('Time '+r'$(s)$')
    # plt.xticks([200*i for i in range(6)],labels=[0,0.2,0.4,0.6,0.8,1])
    # plt.savefig(paper_pic_filefolder+'data_simulation.eps',dpi=600,bbox_inches='tight',format='eps')
    plt.savefig('pics/data_simulation_without_axis2.svg', dpi=600, bbox_inches='tight', format='svg')
    plt.show()


def draw_pic_3d_data_simulation(nhit_list, elevation_angle=0, azimuthal_angle=-90):
    df = pd.read_csv(nhit_list)

    z = np.linspace(0, 1.6, 8)
    T = np.linspace(0, 1.6, 1600)
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    s = [1, 0, 0, 1, 1, 1, 0, 1]
    s_name = ['0', '1', '2', '3', '4', '5', '6', '7']
    x_start = [0, 200, 400, 600, 800, 1000, 1200, 1400]
    x_middle = [200, 400, 600, 800, 1000, 1200, 1400, 1600]
    x_end = [1000, 1200, 1400, 1600, 1600, 1600, 1600, 1600]
    for i in range(8):
        if s[i] == 0:
            continue
        else:
            ax.plot(T[x_start[i]:x_middle[i]], 0.2 * i * np.ones(200), np.array(df[s_name[i]])[0:200], color='blue')
            l = x_end[i] - x_middle[i]
            ax.plot(T[x_middle[i]:x_end[i]], 0.2 * i * np.ones(l), np.array(df[s_name[i]])[200:200 + l], color='grey')

    # poly = PolyCollection(verts, facecolors=(1, 1, 1, 1), edgecolors=(0, 0, 1, 1))
    # ax.add_collection3d(poly, zs=z[:, 0], zdir='y')
    ax.set_xlim3d(np.min(T), np.max(T))
    ax.set_ylim3d(np.min(z), np.max(z))
    ax.set_zlim3d(0, 50)
    ax.set_xticks([])
    # ax.set_xlabel("Time "+r"$(s)$")
    # ax.set_yticks([0, 0.2, 0.4, 0.8, 1.2], ['','','', '', '', ''])
    # ax.yaxis.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4])
    # plt.yticks([0, 0.2, 0.4, 0.6,2])
    # plt.yticks([0, 0.2, 0.4, 0.6,2])
    plt.yticks([])
    # ax.set_ylabel("Release Signal "+r"$s_i$")
    ax.set_zticks([])
    # ax.set_zlabel("Molecules "+r"$n_i$")
    # plt.grid(True)
    plt.gca().view_init(elev=elevation_angle, azim=azimuthal_angle)
    plt.savefig('pics/3d_data_simulation.svg', bbox_inches='tight', dpi=600, format='svg')
    plt.show()


# def draw_pic_attention(datafilepath=None):
#     # uniform_data = np.random.rand(10,12)
#     embeddings = np.load(datafilepath)
#     # print(embeddings)
#     ax = seaborn.heatmap(embeddings[0])
#     plt.show()
#     bx = seaborn.heatmap(embeddings[3])
#     plt.show()
#     cx = seaborn.heatmap(embeddings[388])
#     plt.show()
#     seaborn.heatmap(embeddings[388] - embeddings[0])
#     plt.show()


def draw_pic_data_boxplot(N_hit_and_signal_data_path, classify: str = "M"):
    plt.figure(figsize=(4, 2.8))
    df = pd.read_csv(N_hit_and_signal_data_path)
    # Grouped boxplot, and that boxplot can be replaced by violinplot
    ax = seaborn.boxplot(x=classify, y='$$n_i$$', hue="signals", data=df, palette="Pastel1", whis=4)
    seaborn.swarmplot(data=df, x=classify, y='$$n_i$$', hue="signals", size=2, dodge=True)
    handles, labels = ax.get_legend_handles_labels()
    # ax.legend(handles[:2], labels[:2])
    ax.legend(handles[:2], ['Transmit signal ' + r'$s_{i}=0$', 'Transmit signal ' + r'$s_{i}=1$'])
    plt.ylabel('Received Molecules ' + r'$n_i$')
    plt.xlabel('Drift Velocity ' + r'$v$')
    plt.ylim(top=40)
    plt.savefig(paper_pic_filefolder + 'data_simulation_swarm_' + classify + '.eps', dpi=600, bbox_inches='tight',
                format='eps')
    plt.savefig('pics/data_simulation_swarm_' + classify + '.svg', dpi=600, bbox_inches='tight', format='svg')
    plt.show()


# def draw_pic_theory_distribution():

def draw_pic_BER_results(results_file_path,colormap="Accent"):
    plt.figure(figsize=(4, 2.8))
    df = pd.read_csv(results_file_path)
    df.plot(x='velocity', y=['MCFormer', 'MAP', 'DNN'], logy=True, colormap=colormap, kind='bar')
    plt.ylabel("BER")
    plt.xlabel("Drift Velocity " + r'$v$')
    label = np.array(df['velocity'])
    ticks = np.arange(start=0, stop=len(label), step=1, dtype=int)
    plt.xticks(ticks=ticks, labels=label, rotation=0)
    plt.savefig(paper_pic_filefolder + 'results_BER.eps', dpi=600, bbox_inches='tight', format='eps')
    plt.savefig('pics/results_BER.svg', dpi=600, bbox_inches='tight', format='svg')
    plt.show()


def draw_pic_nlayers_results(results_file_path,colormap="Accent"):
    plt.figure(figsize=(4, 4))
    df = pd.read_csv(results_file_path)
    ax = df.plot(x='nlayers', y=['DNN @ v=20', 'DNN @ v=25', 'MCFormer @ v=20', 'MCFormer @ v=25'],
                 logy=True, colormap=colormap, kind='line', style=['+--', '.:', 'o-', 's-'])
    plt.xticks([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
    plt.xlabel("Number of Layers")
    plt.ylabel("BER")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles,
              ['DNN @ ' + r'$v=20$', 'DNN @ ' + r'$v=25$', 'MCFormer @ ' + r'$v=20$', 'MCFormer @ ' + r'$v=25$'])
    plt.savefig(paper_pic_filefolder + 'results_nlayers.eps', dpi=600, bbox_inches='tight', format='eps')
    plt.savefig('pics/results_nlayers.svg', dpi=600, bbox_inches='tight', format='svg')
    plt.show()


def draw_pic_losses_results(results_losses_path,colormap="Accent",
                            ylabel="Validation Accuracy",
                            save_filename="results_accs"):
    plt.figure(figsize=(4, 4))
    df = pd.read_csv(results_losses_path)
    ax = df.plot(x='epoch', y=['velocity: 20', 'velocity: 25', 'velocity: 30', 'velocity: 35',
                               'velocity: 40', 'velocity: 45', 'velocity: 50', ], colormap=colormap, kind='line')
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, [r'$v=20$', r'$v=25$', r'$v=30$', r'$v=35$', r'$v=40$', r'$v=45$', r'$v=50$'])
    plt.savefig(paper_pic_filefolder + '{}.eps'.format(save_filename), dpi=600, bbox_inches='tight', format='eps')
    plt.savefig('pics/{}.svg'.format(save_filename), dpi=600, bbox_inches='tight', format='svg')
    plt.show()

def draw_pic_SNR_results(results_file_path,colormap="Accent"):
    plt.figure(figsize=(4, 2.8))
    df = pd.read_csv(results_file_path)
    df.plot(x='SNR', y=['MCFormer', 'MAP', 'DNN'], logy=True, colormap=colormap
            , kind='line',style=['+--', '.:', 'o-'])
    plt.ylabel("BER")
    plt.xlabel("SNR (dB)")
    label = np.array(df['SNR']).astype(int)
    ticks = np.arange(start=10, stop=len(label)*5+10, step=5, dtype=int)
    plt.xticks(ticks=ticks, labels=label, rotation=0)
    plt.savefig(paper_pic_filefolder + 'results_SNR.pdf', dpi=600, bbox_inches='tight', format='pdf')
    plt.savefig('pics/results_SNR.svg', dpi=600, bbox_inches='tight', format='svg')
    plt.show()

if __name__ == '__main__':
    # draw_3d_brownian_simulation(simulation_data_path='pics/pic_datas/d=10.0,r=2.5,num=500,iterations=1.npy',
    #                             slice=50,
    #                             elevation_angle=0,
    #                             azimuthal_angle=-90,
    #                             is_generate_gif=False)
    # draw_pic_data_simulations(N_hit_all_path='pics/pic_datas/data_simulation_nhit_all.npy',
    #                           N_hit_avg_path='pics/pic_datas/num_hit_avg.npy',
    #                           N_hit_and_signal_path='pics/pic_datas/data_simulation_nhit_sample.npy',
    #                           N_hit_theoretical_path='pics/pic_datas/theoretical_nums.npy')
    # draw_pic_data_simulations_without_axis(N_hit_all_path='pics/pic_datas/data_simulation_8_signals.csv')
    # draw_pic_3d_data_simulation(nhit_list='pics/pic_datas/data_3d_simulation.csv',
    #                             elevation_angle=20,
    #                             azimuthal_angle=-60)
    # draw_pic_attention(datafilepath='logs/embeddings.npy')
    draw_pic_data_boxplot(N_hit_and_signal_data_path='pics/pic_datas/data_boxplot_v.csv',
                    classify='v')

    cmap='Accent'
    draw_pic_BER_results(results_file_path='pics/pic_datas/BER_results.csv',
                         colormap=cmap)
    # draw_pic_nlayers_results(results_file_path='pics/pic_datas/nlayers_results.csv',
    #                          colormap=cmap)
    # draw_pic_losses_results(results_losses_path='pics/pic_datas/losses_results.csv',
    #                         colormap=cmap,
    #                         ylabel="Training Losses",
    #                         save_filename="results_losses")
    # draw_pic_losses_results(results_losses_path='pics/pic_datas/val_acc_results.csv',
    #                         colormap=cmap,
    #                         ylabel="Validation Accuracy",
    #                         save_filename="results_accs")

    draw_pic_SNR_results(results_file_path='pics/pic_datas/SNR_results.csv',
                         colormap=cmap)
    print("Picture Drawing Finish.")
