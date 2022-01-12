import sys
import os
import uuid
from datetime import datetime
import gym
import gym_marblemaze
import argparse

from os import path


print('gym_marblemaze.__path__:', gym_marblemaze.__path__)
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib import collections as mc
from mpl_toolkits import mplot3d


from ex_humanoid_su import Ex
#from ex_antmaze_low_energy import Ex


import seaborn

parser = argparse.ArgumentParser(description='Test Ex in antmaze')
parser.add_argument('--budget', dest='budget',default=1000, type=int)
parser.add_argument('--resolution', dest='resolution',default=0.1, type=float)
parser.add_argument('--frameskip', dest='frameskip',default=1, type=int)
args = parser.parse_args()



def plot_tree(plot_ax, planner, env, title):

    L_X = []
    L_Y = []
    L_Z = []

    transitions = [[s[:3], sp[:3]] for [s,sp] in planner.transitions]

    for [_,sp] in planner.transitions:
        L_X.append(sp[0])
        L_Y.append(sp[1])
        L_Z.append(sp[2])

    ax.scatter3D(L_X, L_Y, L_Z)

def play_behavior(env, planner):

    prevs = {tuple(sp): tuple(s) for [s,sp] in planner.transitions}
    actions_dict = {(tuple(s), tuple(sp)): a for [s, sp, a] in planner.full_transitions}

    for [_, sp] in planner.transitions:
        if sp[2] > 0.3:
            break

    actions = []
    path = [tuple(sp)]
    while tuple(path[0]) in prevs.keys():
        path = [prevs[tuple(path[0])]] + path
        actions.append(actions_dict[(path[0], path[1])])

    env.reset()
    # for a in actions:
    #     env.render()
    #     obs, _, _, _ = env.step(a)

    n_act = env.action_space.shape[0]
    for i in range(100):
        env.render()
        act = np.random.uniform(-1, 1, n_act)
        obs, _, _, _ = env.step(act)


    env.close()

    return
    # plt.show()

def reward_fun(old_obs, action, repeat, obs):
    return 0


if __name__ == '__main__':

    # create new directory
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H:%M:%S")

    cur_path = os.getcwd()
    dir_path = cur_path + "/EX_MP_humanoid_" + dt_string + "_" + str(np.random.randint(1000))

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


    env = gym.make("HumanoidEnvNOWS-v0")

    s_init = env._get_obs()

    budget = args.budget
    resolution = args.resolution
    frameskip = args.frameskip

    ex = Ex(env, new_path, frameskip= frameskip, resolution=resolution, stochastic=True, budget = budget)
    ex.insert(s_init)
    ex.save_sim(s_init, env.get_state_sim())

    success, cnt = ex.search((0., 0., 2.))

    #seaborn.set()
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    title = "Ex | Humanoid | budget = " + str(budget) + " | resolution = " + str(resolution) + " | frameskip = " + str(frameskip)

    plot_tree(ax, ex, env, title)

    play_behavior(env, ex)

    plt.show()
    plt.savefig(new_path +"/exploration_tree.png")
    plt.close(fig)
