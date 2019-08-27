# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 11:16:21 2018

@author: Pyeong
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


def sigmoid(x):
    sigmoid_output = 1/(1 + np.exp(-x))
    return sigmoid_output


def grad_sigmoid(x):
    # Gradient of sigmoid
    return np.exp(-x)/((1+np.exp(-x))**2)


def cross_entropy_loss(p):
    ce = -np.log(p)
    return ce


def focal_loss(p, gamma):
    fl = -np.multiply(np.log(p), (1-p)**gamma)
    return fl


def grad_cross_entropy_loss(p, grad_p):
    grad_ce = -1/p*grad_p
    return grad_ce


def grad_focal_loss(p, grad_p, gamma):
    grad_fl = ((1 - p)**(gamma - 1))*grad_p*(gamma*np.log(p) - (1 - p)/p)
    return grad_fl


def line_plot(input_x, loss_list, color_dict, save_folder_name, save_name, x_label, y_label, legend_list, x_lim=None, y_lim=None):

    if not os.path.exists(save_folder_name):
        os.makedirs(save_folder_name)

    plt.ioff()
    fig, ax1 = plt.subplots()
    fig.set_size_inches(9, 5, forward=True)
    for loss_name, loss in loss_list:
        ax1.plot(input_x, loss,
                 label=loss_name, linewidth=2, color=color_dict[loss_name], linestyle='-', alpha=1)
    ax1.legend(legend_list)
    if y_lim:
        ax1.set_ylim(y_lim)
    if x_lim:
        ax1.set_xlim(x_lim)
    ax1.set_ylabel(y_label)
    ax1.set_xlabel(x_label)
    plt.savefig(save_folder_name+save_name,
                bbox_inches='tight', format='png', dpi=800)

    return None


if __name__ == '__main__':
    color_dict = {"FL_g1": "blue",
                  "FL_g2": "red",
                  "FL_g3": "gray",
                  "FL_g5": "green",
                  "CE": "black"}

    x = np.arange(-10.1, 10.1, 0.01)
    p = sigmoid(x)
    grad_p = grad_sigmoid(x)
    folder_dir = "../loss_graphs/"
    graph_name = "loss_vs_p_ce.png"
    legend_list = ["CEL", "FL, $\gamma$=1", "FL, $\gamma$=2", "FL, $\gamma$=3", "FL, $\gamma$=5"]
    #x_label = "output logit of ground truth class "+r"($g(\theta,X)$)"
    x_label = "Softmax prediction (prediction confidence of ground truth class $p_t$)"
    y_label = "Loss"
    ylim = [0, 5]
    #ylim = None
    xlim = [0, 1]
    
    ce = cross_entropy_loss(p)
    fl_g1 = focal_loss(p, 1)
    fl_g2 = focal_loss(p, 2)
    fl_g3 = focal_loss(p, 3)
    fl_g5 = focal_loss(p, 5)
    loss_list = [("CE", ce),
                 ("FL_g1", fl_g1),
                 ("FL_g2", fl_g2),
                 ("FL_g3", fl_g3),
                 ("FL_g5", fl_g5)]
    line_plot(p, loss_list, color_dict, folder_dir,
              graph_name, x_label, y_label, legend_list, y_lim=ylim, x_lim=xlim)
    
    '''
    grad_ce = grad_cross_entropy_loss(p, grad_p)
    grad_fl_g1 = grad_focal_loss(p, grad_p, 1)
    grad_fl_g2 = grad_focal_loss(p, grad_p, 2)
    grad_fl_g3 = grad_focal_loss(p, grad_p, 3)
    grad_fl_g5 = grad_focal_loss(p, grad_p, 5)
    grad_loss_list = [("CE", grad_ce),
                      ("FL_g1", grad_fl_g1),
                      ("FL_g2", grad_fl_g2),
                      ("FL_g3", grad_fl_g3),
                      ("FL_g5", grad_fl_g5)]

    line_plot(p, grad_loss_list, color_dict, folder_dir,
              graph_name, x_label, y_label, legend_list, y_lim=ylim, x_lim=xlim)
    '''
