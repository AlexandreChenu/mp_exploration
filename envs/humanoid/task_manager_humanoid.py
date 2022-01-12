import sys
import os
sys.path.append(os.getcwd())

import numpy as np
import math
import random
import copy
from collections import OrderedDict
import torch



class TasksManager():

	def __init__(self, L_full_demonstration, L_full_inner_demonstration, L_states, L_actions, L_full_observations, L_goals, L_inner_states, L_budgets, m_goals, std_goals, env_option):

		self.L_full_demonstration = L_full_demonstration
		self.L_full_inner_demonstration = L_full_inner_demonstration
		self.L_states = L_states
		self.L_actions = L_actions
		self.L_full_observations = L_full_observations
		self.L_inner_states = L_inner_states
		self.L_budgets = L_budgets
		self.L_goals = L_goals

		## set of starting state for subgoal adaptation
		self.starting_inner_state_set = [[inner_state] for inner_state in self.L_inner_states]
		self.starting_state_set = [[state] for state in self.L_states]

		## monitor task success rates, task feasibility, overshoot success rates and feasibility
		self.L_tasks_results = [[] for _ in self.L_states]
		self.L_overshoot_results = [[[]] for _ in self.L_states] ## a list of list of results per task
		self.L_tasks_feasible = [False for _ in self.L_states]
		self.L_skipping_feasible = [False for _ in self.L_states]

		self.task_window = 10
		self.max_size_starting_state_set = 15

		## task sampling strategy (weighted or uniform)
		self.weighted_sampling = True

		self.delta_step = 1
		self.dist_threshold = 0.08

		self.min_dist_threshold = 0.05
		self.max_dist_threshold = 0.15

		self.nb_tasks = len(self.L_states)-1

		self.env_option = env_option

		self.incl_extra_full_state = 1

		self.subgoal_adaptation = False
		self.task_skipping = True

		self.ratio_skipping = 0.7
		self.skipping = False

		self.do_overshooting = True

		self.L_goals = [self.project_to_goal_space(state) for state in self.L_states]
		self.shift_lookup_table = self._init_lookup_table()


	def _init_lookup_table(self,
		):
		"""
		associate goal to next goal
		"""

		lookup_table = {}

		for i in range(0, len(self.L_states)):
			if i < len(self.L_states)-1:
				lookup_table[tuple(self.project_to_goal_space(self.L_states[i]).astype(np.float32))] = tuple(self.project_to_goal_space(self.L_states[i+1]).astype(np.float32))
			else:
				lookup_table[tuple(self.project_to_goal_space(self.L_states[i]).astype(np.float32))] = tuple(self.project_to_goal_space(self.L_states[i]).astype(np.float32))

		return lookup_table

	def skip_task(self, task_indx):
		"""
		Remove task if skipping successful
		"""

		print("remove task ", task_indx)
		self.remove_subgoal(task_indx)
		return 0


	def divide_task(self, task_indx):
		"""
		Divide a task into two subtask if the task is too difficult
		"""
		print("devide task ", task_indx)
		new_subgoal, new_inner_subgoal, new_budget = self.get_new_subgoal(task_indx)
		self.insert_new_subgoal(new_subgoal, new_inner_subgoal, new_budget, task_indx)

		return 0

	def insert_new_subgoal(self, new_subgoal, new_inner_subgoal, new_budget, task_indx):
		"""
		Insert the new subgoal
		"""
		size_L_states = len(self.L_states)
		size_L_inner_states = len(self.L_inner_states)

		assert size_L_states == size_L_inner_states

		self.L_states = self.L_states[:task_indx] + [new_subgoal] + self.L_states[task_indx:]
		self.L_inner_states = self.L_inner_states[:task_indx] + [new_inner_subgoal] + self.L_inner_states[task_indx:]

		assert len(self.L_states) == size_L_states + 1

		self.starting_inner_state_set = self.starting_inner_state_set[:task_indx] + [[new_inner_subgoal]] + self.starting_inner_state_set[task_indx:]
		self.starting_state_set = self.starting_state_set[:task_indx] + [[new_subgoal]] + self.starting_state_set[task_indx:]

		self.L_tasks_results = self.L_tasks_results[:task_indx] + [[]] + self.L_tasks_results[task_indx:]
		self.L_overshoot_results = self.L_overshoot_results[:task_indx] + [[[]]] + self.L_overshoot_results[task_indx:]

		self.L_budgets = self.L_budgets[:task_indx] + [new_budget, new_budget] + self.L_budgets[task_indx+1:]

		self.L_tasks_feasible = self.L_tasks_feasible[:task_indx] + [False] + self.L_tasks_feasible[task_indx:]
		self.L_skipping_feasible = self.L_skipping_feasible[:task_indx] + [False] + self.L_skipping_feasible[task_indx:]

		## adapt lookup table
		self.shift_lookup_table[tuple(self.project_to_goal_space(self.L_states[task_indx-1]).astype(np.float32))] = tuple(self.project_to_goal_space(new_subgoal).astype(np.float32))
		self.shift_lookup_table[tuple(self.project_to_goal_space(new_subgoal).astype(np.float32))] = tuple(self.project_to_goal_space(self.L_states[task_indx]).astype(np.float32))

		# for i in range(len(self.L_tasks_feasible)):
		# 	self.L_tasks_feasible[i] = False
		# for i in range(len(self.L_skipping_feasible)):
		# 	self.L_skipping_feasible[i] = False

		self.nb_tasks = len(self.L_states)-1

		return 0

	def remove_subgoal(self, task_indx):
		"""
		Insert the new subgoal
		"""
		size_L_states = len(self.L_states)
		size_L_inner_states = len(self.L_inner_states)

		assert size_L_states == size_L_inner_states

		self.shift_lookup_table[tuple(self.project_to_goal_space(self.L_states[task_indx-1]).astype(np.float32))] = self.shift_lookup_table[tuple(self.project_to_goal_space(self.L_states[task_indx+1]).astype(np.float32))]

		self.L_states = self.L_states[:task_indx] + self.L_states[task_indx+1:]
		self.L_goals = self.L_goals[:task_indx] + self.L_goals[task_indx+1:]
		self.L_inner_states = self.L_inner_states[:task_indx] + self.L_inner_states[task_indx+1:]

		assert len(self.L_states) == size_L_states - 1

		self.starting_inner_state_set = self.starting_inner_state_set[:task_indx] + self.starting_inner_state_set[task_indx+1:]
		self.starting_state_set = self.starting_state_set[:task_indx] + self.starting_state_set[task_indx+1:]

		self.L_tasks_results = self.L_tasks_results[:task_indx] + self.L_tasks_results[task_indx+1:]
		self.L_overshoot_results = self.L_overshoot_results[:task_indx] + self.L_overshoot_results[task_indx+1:]

		extra_budget = self.L_budgets[task_indx]
		self.L_budgets = self.L_budgets[:task_indx] + self.L_budgets[task_indx+1:]
		self.L_budgets[task_indx-1] += extra_budget

		self.L_tasks_feasible = self.L_tasks_feasible[:task_indx] + self.L_tasks_feasible[task_indx+1:]
		self.L_skipping_feasible = self.L_skipping_feasible[:task_indx] + self.L_skipping_feasible[task_indx+1:]

		for i in range(len(self.L_tasks_feasible)):
			self.L_tasks_feasible[i] = False
		for i in range(len(self.L_skipping_feasible)):
			self.L_skipping_feasible[i] = False

		self.nb_tasks = len(self.L_states)-1

		return 0

	def get_new_subgoal(self, task_indx):
		"""
		Get the new subgoal
		"""

		L_full_demonstration = [list(el_demo) for el_demo in self.L_full_demonstration]
		indx_start  = L_full_demonstration.index(list(self.L_states[task_indx-1]))
		indx_goal = L_full_demonstration.index(list(self.L_states[task_indx]))

		indx_subgoal = int((indx_start + indx_goal)/2)

		budget = int((indx_goal - indx_start)/2)*2

		assert indx_subgoal >= indx_start
		assert indx_subgoal <= indx_goal

		subgoal = self.L_full_demonstration[indx_subgoal]
		inner_subgoal = self.L_full_inner_demonstration[indx_subgoal]
		return subgoal, inner_subgoal, budget


	def add_new_starting_state(self, task_indx, inner_state, state):
		"""
		add new starting state to a given task i.e. add a new subgoal.
		ONLY IF SUBGOAL ADAPTATION ENABLED (i.e. self.subgoal_adaptation=True)
		"""
		add = True
		for starting_state in self.starting_state_set[task_indx]:

			## reject new subgoal if too close to already saved subgoals
			if self.compute_distance_in_goal_space(self.project_to_goal_space(state), self.project_to_goal_space(starting_state)) < self.min_dist_threshold:
				add = False
				break

			## reject new subgoal if too far from actual subgoal (should avoid replacing grasped subgoal with non-grasped subgoals and enable adaptative resolution)
			if self.compute_distance_in_goal_space(self.project_to_goal_space(state), self.project_to_goal_space(starting_state)) > self.max_dist_threshold:
				add = False
				break

		if add and len(self.starting_inner_state_set[task_indx]) < self.max_size_starting_state_set:
			self.starting_inner_state_set[task_indx].append(inner_state)
			self.starting_state_set[task_indx].append(state)

			self.L_overshoot_results[task_indx].append([])

		return

	def update_overshoot_result(self, subgoal_task_indx, subgoal_state, success_bool):
		"""
		update the overshoot score of a given subgoal.
		"""
		for starting_state, subgoal_state_indx in zip(self.starting_state_set[subgoal_task_indx], list(np.arange(0,len(self.starting_state_set[subgoal_task_indx])))):
			bool = np.array_equal(self.project_to_goal_space(starting_state), subgoal_state)
			if bool :
				break

		## the subgoal_state should be contained in the corresponding starting_state_set
		assert bool

		self.L_overshoot_results[subgoal_task_indx][subgoal_state_indx].append(int(success_bool))

		if len(self.L_overshoot_results[subgoal_task_indx][subgoal_state_indx]) > 20:
			self.L_overshoot_results[subgoal_task_indx][subgoal_state_indx].pop(0)

		print("overshoot result = ", self.L_overshoot_results[subgoal_task_indx][subgoal_state_indx])

		return

	def get_task(self, task_indx):
		"""
		get starting state, length and goal associated to a given task
		"""
		assert task_indx > 0
		assert task_indx < len(self.L_states), "indx too large: task indx = " + str(task_indx)

		self.indx_start = task_indx - self.delta_step
		self.indx_goal = task_indx

		length_task = sum(self.L_budgets[self.indx_start:self.indx_goal])

		starting_state, starting_inner_state = self.get_starting_state(task_indx - self.delta_step)
		goal_state = self.get_goal_state(task_indx)

		return starting_inner_state, length_task, goal_state

	def advance_task(self):
		"""
		shift task by one and get task goal and task length if possible.
		if no more task -> return False signal
		"""
		self.indx_goal += 1

		if self.indx_goal < len(self.L_states) and self.do_overshooting:
			assert self.indx_goal < len(self.L_states)

			length_task = sum(self.L_budgets[self.indx_goal - self.delta_step:self.indx_goal])

			goal_state = self.get_goal_state(self.indx_goal, overshoot = True)

			return goal_state, length_task, True

		else:
			return None, None, False


	def add_success(self, task_indx):
		"""
		Monitor successes for a given task
		"""
		self.L_tasks_feasible[task_indx] = True

		if self.skipping:
			self.L_skipping_feasible[task_indx-1] = True

		self.L_tasks_results[task_indx].append(1)

		if len(self.L_tasks_results[task_indx]) > self.task_window:
			self.L_tasks_results[task_indx].pop(0)

		return

	def add_failure(self, task_indx):
		"""
		Monitor failues for a given task
		"""
		self.L_tasks_results[task_indx].append(0)

		if len(self.L_tasks_results[task_indx]) > self.task_window:
			self.L_tasks_results[task_indx].pop(0)

		return

	def get_task_success_rate(self, task_indx):

		nb_tasks_success = self.L_tasks_results[task_indx].count(1)

		s_r = float(nb_tasks_success/len(self.L_tasks_results[task_indx]))

		## on cape l'inversion
		if s_r <= 0.1:
			s_r = 10
		else:
			s_r = 1./s_r

		return s_r

	def get_tasks_success_rates(self):

		L_rates = []

		for i in range(self.delta_step, len(self.L_states)):
			L_rates.append(self.get_task_success_rate(i))

		return L_rates


	def sample_task_indx(self):
		"""
		Sample a task indx.
		2 cases:
			- weighted sampling of task according to task success rates
			- uniform sampling
		"""
		weights_available = True
		for i in range(self.delta_step,len(self.L_tasks_results)):
			if len(self.L_tasks_results[i]) == 0:
				weights_available = False

		if self.weighted_sampling and weights_available: ## weighted sampling

			L_rates = self.get_tasks_success_rates()

			assert len(L_rates) == len(self.L_states) - self.delta_step

			## weighted sampling
			total_rate = sum(L_rates)
			pick = random.uniform(0, total_rate)

			current = 0
			for i in range(0,len(L_rates)):
				s_r = L_rates[i]
				current += s_r
				if current > pick:
					break

			i = i + self.delta_step

		else: ## uniform sampling
			i = random.randint(self.delta_step, len(self.L_states)-1)
			# i = 2

		return i

	def select_task(self):
		"""
		Select a task and return corresponding starting state, budget and goal
		"""
		self.skipping = False
		task_indx = self.sample_task_indx()
		## task indx coorespond to a goal indx
		self.indx_start = task_indx - self.delta_step

		if self.task_skipping and task_indx - 2*self.delta_step >= 0:
			sample = random.random()
			if sample >= self.ratio_skipping:
				# print("\n\n SKIPPING \n\n")
				self.indx_start =  task_indx - 2*self.delta_step
				self.skipping = True

		self.indx_goal = task_indx
		length_task = sum(self.L_budgets[self.indx_start:self.indx_goal])
		starting_state, starting_inner_state = self.get_starting_state(self.indx_start)

		goal_state = self.get_goal_state(self.indx_goal)
		return starting_inner_state, length_task, goal_state

	def get_starting_state(self, indx_start):

		return self.starting_state_set[indx_start][0], self.starting_inner_state_set[indx_start][0]

	def get_goal_state(self, indx_goal, overshoot=False):

		if self.subgoal_adaptation and not overshoot:
			## uniform sampling of goal state in the starting_state_set
			indx_goal_state = random.randint(0,len(self.starting_state_set[indx_goal])-1)

			return self.starting_state_set[indx_goal][indx_goal_state]

		else:
			return self.starting_state_set[indx_goal][0]

	def project_to_goal_space(self, state):
		"""
		Project a state in the goal space depending on the environment.
		"""

		com_pos = self.get_com_pos(state)
		com_vel = self.get_com_vel(state)

		#norm_gripper_velp = np.linalg.norm(gripper_velp)

		if "vel" in self.env_option:
			return np.concatenate((np.array(com_pos), np.array(com_vel)))

		else:
			return np.array(com_pos)

	def _normalize_goal(self, goal):
		"""
		Normalize goal according to previously computed mean and standard deviation.
		"""

		if self.m_goals is not None:
			norm_goal = (np.array(goal) - self.m_goals) / self.std_goals
			return list(norm_goal)
		else:
			return goal

	def compute_distance_in_goal_space(self, in_goal1, in_goal2):
		"""
		goal1 = achieved_goal
		goal2 = desired_goal
		"""

		goal1 = copy.deepcopy(in_goal1)
		goal2 = copy.deepcopy(in_goal2)

		if "com" in self.env_option:

			if len(goal1.shape) ==  1:
				euclidian_goal1 = goal1[:3]
				euclidian_goal2 = goal2[:3]

				return np.linalg.norm(euclidian_goal1 - euclidian_goal2, axis=-1)

			else:
				euclidian_goal1 = goal1[:,:3]
				euclidian_goal2 = goal2[:,:3]

				return np.linalg.norm(euclidian_goal1[:,:] - euclidian_goal2[:,:], axis=-1)

		else:
			if len(goal1.shape) ==  1:
				return np.linalg.norm(goal1 - goal2, axis=-1)
			else:
				return np.linalg.norm(goal1[:,:] - goal2[:,:], axis=-1)

	## TODO: get coordinates in state
	def get_com_pos(self, state):
		"""
		get center of mass position from full state for torso?
		"""
		# print("len(list(state)) = ", len(list(state)))
		assert len(list(state))== 378

		com_pos = state[:3]
		# gripper_pos = state[102:105]

		assert len(list(com_pos)) == 3

		return com_pos

	def get_com_vel(self, state):
		"""
		get center of mass velocity from full state for torso
		"""
		assert len(list(state)) == 378

		object_pos = state[105:108]
		# object_pos = state[123:126]
		assert len(list(object_pos))==3

		# print("indx object pos = ", indx)

		return object_pos
