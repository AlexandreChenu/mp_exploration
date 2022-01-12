import sys
import os
import uuid
from datetime import datetime
import gym
from envs import *
import argparse

from os import path

import pickle


import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib import collections as mc

from ex.ex_dubinsmazeenv import Ex


import seaborn

def plot_trajectory(plot_ax, planner, env, dir_path):

    print("planner.transitions[-1] = ", planner.transitions[-1])
    path = [planner.transitions[-1][1]]
    actions = []

    prevs = {tuple(sp): tuple(s) for [s,sp] in planner.transitions}

    print("prevs = ", prevs)

    print("Reconstructing path...")
    while path[0][0] != 0.5 and path[0][1] != 0.5:
        #print("path[0] = ", path[0])
        # print("path[0] = ", path[0])
        actions = [planner.transitions_w_actions[tuple([prevs[tuple(path[0])][0], prevs[tuple(path[0])][1], prevs[tuple(path[0])][2], prevs[tuple(path[0])][3], path[0][0], path[0][1], path[0][2], path[0][3]])]] + actions
        path = [prevs[tuple(path[0])]] + path

    L_X = [state[0] for state in path]
    L_Y = [state[1] for state in path]
    # L_Z = [state[2] for state in path]

    env.draw(plot_ax, paths=False)

    plot_ax.plot(L_X,L_Y,c="red")

    plot_ax.autoscale()
    plot_ax.margins(0.1)

    plot_ax.set_title("Ex | Maze Environment")

    plot_ax.set_xlabel("X")
    plot_ax.set_ylabel("Y")


    demo = {}
    demo["obs"] = path
    demo["actions"] = actions

    with open(dir_path + '/DubinsMazeEnv5.demo', 'wb') as handle:
        pickle.dump(demo, handle, protocol=pickle.HIGHEST_PROTOCOL)



def plot_tree(plot_ax, planner, env):

    env.draw(plot_ax, paths=False)

    transitions = [[s[:2], sp[:2]] for [s,sp] in planner.transitions]

    plot_ax.add_collection(mc.LineCollection(transitions, linewidths=2., color = "red"))

    plot_ax.autoscale()
    plot_ax.margins(0.1)

    plot_ax.set_title("Ex | Maze Environment")

    plot_ax.set_xlabel("X")
    plot_ax.set_ylabel("Y")

    # plt.show()

def reward_fun(old_obs, action, repeat, obs):
    return 0


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Argument for Ex.')
    parser.add_argument('-b', help='budget')
    parsed_args = parser.parse_args()

    args = {}
    args["budget"] = float(parsed_args.b)

    # create new directory
    # now = datetime.now()
    # dt_string = now.strftime("%d_%m_%Y_%H:%M:%S")

    now = datetime.now()
    dt_string = '_%s_%s' % (datetime.now().strftime('%Y%m%d_%H:%M'), str(os.getpid()))

    cur_path = os.getcwd()
    dir_path = cur_path + "/xp/EX_MP_" + dt_string

    try:
        os.mkdir(dir_path)
    except OSError:
        print ("Creation of the directory %s failed" % dir_path)
    else:
        print ("Successfully created the directory %s " % dir_path)


    env = gym.make('DubinsMazeEnv-v2')

    ex = Ex(env, dir_path, resolution=.15, stochastic=True, budget = args["budget"])
    ex.insert([0.5,0.5,0.,0.])
    success, cnt = ex.search((4.5, 4.5))

    seaborn.set()

    fig, ax = plt.subplots()
    plot_trajectory(ax, ex, env, dir_path)
    plt.savefig(dir_path +"/exploration_traj.png")
    plt.show()
    plt.close(fig)

    fig, ax = plt.subplots()
    plot_tree(ax, ex, env)
    plt.savefig(dir_path +"/exploration_tree.png")
    plt.show()
    plt.close(fig)
