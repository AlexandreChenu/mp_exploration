import gym
from gym.envs.registration import register

# dubins variant of mazeenv
from .dubins_mazeenv.mazeenv_speed import DubinsMazeEnv#, FreeDubinsMazeEnv
# from .dubins_mazeenv.mazeenv_wrappers import DubinsMazeEnvGCPHERSB3


# # fetch environment from Go-Explore 2 (Ecoffet et al.)
# from .fetchenv.fetch_env import MyComplexFetchEnv
# from .fetchenv.fetchenv_wrappers import ComplexFetchEnvGCPHERSB3
#
# # humanoid environment from mujoco
# from .humanoid.humanoidenv import MyHumanoidEnv
# from .humanoid.humanoidenv_wrappers import HumanoidEnvGCPHERSB3

## mazeenv from Guillaume Matheron with a Dubins car
print("REGISTERING DubinsMazeEnv")
register(
    id='DubinsMazeEnv-v2',
    entry_point='envs.dubins_mazeenv.mazeenv_speed:DubinsMazeEnv')

#
# ## fetch environment from Go-Explore 2 (Ecoffet et al.)
# print("REGISTERING FetchEnv-v0 & FetchEnvGCPHERSB3-v0")
# register(
#     id='FetchEnv-v0',
#     entry_point='envs.fetchenv.fetch_env:MyComplexFetchEnv',
# )
#
# register(
#     id='FetchEnvGCPHERSB3-v0',
#     entry_point='envs.fetchenv.fetchenv_wrappers:ComplexFetchEnvGCPHERSB3',
# )
#
# ## Humanoid environment from Mujoco
# print("REGISTERING HumanoidEnv-v0 & HumanoidEnvGCPHERSB3-v0")
# register(
#     id='HumanoidEnv-v0',
#     entry_point='envs.humanoid.humanoidenv:MyHumanoidEnv',
# )
#
# register(
#     id='HumanoidEnvGCPHERSB3-v0',
#     entry_point='envs.humanoid.humanoidenv_wrappers:HumanoidEnvGCPHERSB3',
# )
