import sys
import os
import uuid
from datetime import datetime
import gym
import gym_marblemaze

from os import path


print('gym_marblemaze.__path__:', gym_marblemaze.__path__)
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib import collections as mc


from ex_mazeenv import Ex


import seaborn



def plot_tree(plot_ax, planner, env):

    env.draw(plot_ax, paths=False)

    plot_ax.add_collection(mc.LineCollection(planner.transitions, linewidths=2., color = "red"))

    plot_ax.autoscale()
    plot_ax.margins(0.1)

    plot_ax.set_title("Ex | Maze Environment")

    plot_ax.set_xlabel("X")
    plot_ax.set_ylabel("Y")

    # plt.show()

def reward_fun(old_obs, action, repeat, obs):
    return 0


if __name__ == '__main__':

    # create new directory
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H:%M:%S")

    cur_path = os.getcwd()
    dir_path = cur_path + "/EX_MP_" + dt_string + "_" + str(np.random.randint(1000))

    # use different name is directory already exists
    new_path = dir_path
    i = 0
    while path.exists(new_path):
        new_path = dir_path + "_" + str(i)
        i += 1

    dir_path = new_path

    try:
        os.mkdir(dir_path)
    except OSError:
        print ("Creation of the directory %s failed" % dir_path)
    else:
        print ("Successfully created the directory %s " % dir_path)


    env = gym.make('MazeEnv-v0')

    ex = Ex(env, new_path, resolution=.15, stochastic=True, budget = 100000)
    ex.insert([0.5,0.5])
    success, cnt = ex.search((14., 14.))

    seaborn.set()
    fig, ax = plt.subplots()

    plot_tree(ax, ex, env)

    plt.savefig(new_path +"/exploration_tree.png")
    plt.close(fig)
