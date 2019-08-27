# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 11:16:21 2018

@author: ut
"""
import matplotlib as mpl
mpl.use('Agg')
mpl.rcParams['figure.dpi'] = 300
import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('ggplot')
import matplotlib.patches as mpatches
import os
colors = ['#ff0022', 'mediumblue', 'black', 'green']
import matplotlib.pyplot as plt


def calculate_conf_inf(numbers, z=1.96):
    num_std = np.std(numbers)
    num_mean = np.mean(numbers)
    critical = z*num_std/math.sqrt(len(numbers))
    conf_high = num_mean + critical
    conf_low = num_mean - critical
    return conf_low, conf_high


def get_data(file_path="../result/Adam/Adam_CE_13/", method="train/csv/", file_name="accuracy.csv"):
    mean_list = []
    mlow_list = []
    mhigh_list = []
    df = pd.read_csv(file_path + method + file_name, header=None)
    # DF as numpy
    df_as_np = df.values
    for i in range(int(df_as_np.shape[0])):
        np_row = df_as_np[i, :]
        mean_list.append(np.mean(np_row))  # 1.1 for DOMAINS
        q25, q75 = calculate_conf_inf(np_row, 3)
        mhigh_list.append(q75)
        mlow_list.append(q25)
    return mean_list, mlow_list, mhigh_list


def get_multiple_data(file_names=["accuracy.csv",
                                  "accuracy_black.csv",
                                  "accuracy_white.csv",
                                  "cofindence_black.csv",
                                  "confidence_white.csv",
                                  "IOU.csv"],
                      file_path="../result/Adam/Adam_CE_13", method="train/csv/"):
    data_dict = {}
    for single_file_name in file_names:
        mean_list, mlow_list, mhigh_list = get_data(file_path, method, single_file_name)
        data_dict[single_file_name[:-4]] = (mean_list, mlow_list, mhigh_list)

    return data_dict


def line_plot(train_data, test_data, save_folder_name, data_name, y_lim=[0, 1]):
    plt.ioff()
    fig, ax1 = plt.subplots()
    fig.set_size_inches(9, 5, forward=True)
    """# Grid
    plt.gca().yaxis.grid(True, linestyle='-', linewidth=1, alpha=0.5)
    plt.gca().xaxis.grid(True, linestyle='-', linewidth=1, alpha=0.25)
    # X Axis
    ax1.set_xlim([0, 247])
    ax1.set_xticks([0, 50, 100, 150, 200, 247], minor=False)
    ax1.set_xticklabels([0, 50, 100, 150, 200, 250])
    ax1.set_xlabel('Iteration', color='black', fontsize=20)
    ax1.tick_params(axis='x', labelsize=16)
    # Y Axis

    ax1.set_yticks([0, 2, 4, 6, 8], minor=False)
    # ax1.set_yticklabels(['0', 0.25, '0.50', 0.75, '1.00'], minor=False)
    ax1.set_ylabel(r'Magnitude of the Gradient', color='black', fontsize=20)
    ax1.tick_params(axis='y', labelsize=16)
    """
    ax1.set_xlim([0, 1000])
    ax1.set_ylim(y_lim)
    ax1.xaxis.set_tick_params(labelsize=12)
    ax1.yaxis.set_tick_params(labelsize=12)
    row_max = len(train_data[0])
    # Plot Train
    ax1.plot([x*5 for x in range(row_max)], train_data[0],
             label='Mean', linewidth=2, color="blue", linestyle='-', alpha=1)
    """
    ax1.fill_between([x for x in range(row_max)],
                     train_data[1],
                     train_data[2],
                     color="blue",
                     label='Softmax 95% Confidence Interval',
                     alpha=0.2)
    """

    # Plot Test
    ax1.plot([x*5 for x in range(row_max)], test_data[0],
             label='Mean', linewidth=2, color="red", linestyle='-', alpha=1)
    """
    ax1.fill_between([x for x in range(row_max)],
                     test_data[1],
                     test_data[2],
                     color="red",
                     label='Softmax 95% Confidence Interval',
                     alpha=0.2)
    """
    ax1.set_xlabel("epoch", fontsize=18,fontname="Arial")
    if len(data_name.split("_"))>1:
        data_split = data_name.split("_")
        y_label = data_split[0] + " " + data_split[1]
    else:
        y_label = data_name
    ax1.set_ylabel(y_label, fontsize=18,fontname="Arial")
    from matplotlib.lines import Line2D

    custom_lines = [Line2D([0], [0], color='blue', lw=2, ls='-'),
                    Line2D([0], [0], color='red', lw=2, ls='-')]
    legend2 = plt.legend(custom_lines, ["train", "test"],
                         framealpha=0.5, loc="lower right", fontsize=16)
    legend2.get_frame().set_edgecolor('black')

    # plt.show()
    loss_name = save_folder_name.split("/")[-2]
    plt.savefig(save_folder_name+data_name+"_"+loss_name+'.png',
                bbox_inches='tight', format='png', dpi=800)


if __name__ == '__main__':
    data_types = ["accuracy",
                  "accuracy_black",
                  "accuracy_white",
                  "cofindence_black",
                  "confidence_white",
                  "IOU"]
    y_lims = {"accuracy": [0.86, 1],
              "accuracy_black": [0.85, 1.2],
              "accuracy_white": [0, 3.5],
              "cofindence_black": [0.5, 1],
              "confidence_white": [0.17, 1],
              "IOU": [0, 1]}
    init = ''
    folder_names = ["../result/Adam/Focal_5_Adam_"+init+"init/",
                    "../result/Adam/Focal_3_Adam_"+init+"init/",
                    "../result/Adam/Focal_2_Adam_"+init+"init/",
                    "../result/Adam/Focal_1_Adam_"+init+"init/",
                    "../result/Adam/CE_11_Adam_"+init+"init/",
                    "../result/Adam/CE_12_Adam_"+init+"init/",
                    "../result/Adam/CE_13_Adam_"+init+"init/",
                    "../result/Adam/CE_14_Adam_"+init+"init/"]
    save_names = ["../graphs/Focal_5_Adam_"+init+"init/",
                  "../graphs/Focal_3_Adam_"+init+"init/",
                  "../graphs/Focal_2_Adam_"+init+"init/",
                  "../graphs/Focal_1_Adam_"+init+"init/",
                  "../graphs/CE_11_Adam_"+init+"init/",
                  "../graphs/CE_12_Adam_"+init+"init/",
                  "../graphs/CE_13_Adam_"+init+"init/",
                  "../graphs/CE_14_Adam_"+init+"init/"]

    for folder_name, save_name in zip(folder_names, save_names):
        train_data_dict = get_multiple_data(
            file_path=folder_name, method="train/csv/")
        test_data_dict = get_multiple_data(
            file_path=folder_name, method="test/csv/")
        if not os.path.exists(save_name):
            os.makedirs(save_name)
        for data_type in data_types:
            print(data_type)
            line_plot(train_data_dict[data_type], test_data_dict[data_type],
                      y_lim=y_lims[data_type], save_folder_name=save_name, data_name=data_type)
