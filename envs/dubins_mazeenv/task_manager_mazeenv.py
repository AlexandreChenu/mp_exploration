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

	def __init__(self, L_full_demonstration, L_states, L_inner_states, L_actions, L_full_observations, L_goals, L_budgets):

		self.L_full_demonstration = L_full_demonstration
		self.L_states = L_states
		self.L_inner_states = L_inner_states
		self.L_actions = L_actions
		self.L_full_observations = L_full_observations
		self.L_goals = L_goals
		self.L_budgets = L_budgets

		## set of starting states (for subgoal adaptation)
		self.starting_inner_state_set = [[inner_state] for inner_state in self.L_inner_states]
		self.starting_state_set = [[state] for state in self.L_states]

		self.L_tasks_results = [[] for _ in self.L_states]
		self.L_overshoot_results = [[[]] for _ in self.L_states] ## a list of list of results per task

		self.task_window = 10
		self.max_size_starting_state_set = 100

		self.weighted_sampling = False

		self.delta_step = 1
		self.dist_threshold = 0.1

		self.nb_tasks = len(self.L_states)-1

		self.subgoal_adaptation = False
		self.add_noise_starting_state = False

		self.shift_lookup_table = self._init_lookup_table()

	def _init_lookup_table(self,
		):
		"""
		associate goal to next goal for fast computation of reward bonus
		"""

		lookup_table = {}

		for i in range(0, len(self.L_states)):
			if i < len(self.L_states)-1:
				lookup_table[tuple(self.project_to_goal_space(self.L_states[i]).astype(np.float32))] = tuple(self.project_to_goal_space(self.L_states[i+1]).astype(np.float32))
			else:
				lookup_table[tuple(self.project_to_goal_space(self.L_states[i]).astype(np.float32))] = tuple(self.project_to_goal_space(self.L_states[i]).astype(np.float32))

		return lookup_table


	def add_new_starting_state(self, task_indx, inner_state, state):
		"""
		add new starting state to a given task i.e. add a new subgoal

		ONLY IF SUBGOAL ADAPTATION ENABLED
		"""
		add = True

		for starting_state in self.starting_state_set[task_indx]:
			#if np.linalg.norm(self.project_to_goal_space(state) - self.project_to_goal_space(starting_state)) < self.dist_threshold:
			if self.compute_distance_in_goal_space(self.project_to_goal_space(state), self.project_to_goal_space(starting_state)) < self.dist_threshold:
				add = False
				break

		if add and len(self.starting_inner_state_set[task_indx]) < self.max_size_starting_state_set:
			self.starting_inner_state_set[task_indx].append(inner_state)
			self.starting_state_set[task_indx].append(state)
			self.L_overshoot_results[task_indx].append([])

		return

	# def add_starting_state(self, task_indx_inner_state, state):
	#

	def update_overshoot_result(self, subgoal_task_indx, subgoal_state, success_bool):
		"""
		update the overshoot score of a given subgoal
		"""

		## replace method index() for list of arrays
		for starting_state, subgoal_state_indx in zip(self.starting_state_set[subgoal_task_indx], list(np.arange(0,len(self.starting_state_set[subgoal_task_indx])))):
			bool = np.array_equal(starting_state, subgoal_state)
			if bool :
				# print("subgoal_state_indx = ", subgoal_state_indx)
				break

		## the subgoal_state should be contained in the corresponding starting_state_set
		assert bool

		#subgoal_state_indx = self.starting_state_set[subgoal_task_indx].index(subgoal_state)
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
		assert task_indx < len(self.L_states)

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

		if self.indx_goal < len(self.L_states) :
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
		self.L_tasks_results[task_indx].append(1)

		if len(self.L_tasks_results[task_indx]) > self.task_window:
			self.L_tasks_results[task_indx].pop(0)

		return

	def add_failure(self, task_indx):
		"""
		Monitor failures for a given task
		"""
		self.L_tasks_results[task_indx].append(0)

		if len(self.L_tasks_results[task_indx]) > self.task_window:
			self.L_tasks_results[task_indx].pop(0)

		return

	def get_task_success_rate(self, task_indx):

		nb_tasks_success = self.L_tasks_results[task_indx].count(1)

		s_r = float(nb_tasks_success/len(self.L_tasks_results[task_indx]))

		## on cap l'inversion
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

		return i

	def select_task(self):
		"""
		Select a task and return corresponding starting state, budget and goal
		"""
		task_indx = self.sample_task_indx()
		## task indx coorespond to a goal indx
		self.indx_start = task_indx - self.delta_step
		self.indx_goal = task_indx
		length_task = sum(self.L_budgets[self.indx_start:self.indx_goal])
		starting_state, starting_inner_state = self.get_starting_state(self.indx_start)

		## diverse choice of goal
		# if self.indx_goal < len(self.L_states) - 1:
		#     goal_state = random.choice(self.starting_state_set[self.indx_goal])
		# else:
		#     goal_state = self.starting_state_set[self.indx_goal][0]

		goal_state = self.get_goal_state(self.indx_goal)
		return starting_inner_state, length_task, goal_state

	# def get_starting_state(self, indx_start):
	#
	# 	starting_state = self.starting_state_set[indx_start][0]
	# 	starting_inner_state = self.starting_inner_state_set[indx_start][0]
	#
	# 	return starting_state, starting_inner_state

	def get_starting_state(self, indx_start):

		starting_state = copy.deepcopy(self.starting_state_set[indx_start][0])
		starting_inner_state = copy.deepcopy(self.starting_inner_state_set[indx_start][0])

		# if indx_start ==1:
		# 	delta_x = (np.random.random()-0.5)*2.*0.3
		# 	delta_y = (np.random.random()-0.5)*2.*0.3
		# 	delta_theta = (np.random.random()-0.5)*2.*np.pi
		#
		# 	x = starting_state[0] + delta_x
		# 	if x < starting_state[0] + 0.3 and x > starting_state[0] - 0.3:
		# 		starting_state[0] = x
		#
		# 	y = starting_state[1] + delta_y
		# 	if y < starting_state[1] + 0.3 and y > starting_state[1] - 0.3:
		# 		starting_state[1] = y
		#
		# 	theta = starting_state[2] + delta_theta
		# 	if theta < starting_state[2] + np.pi and theta > starting_state[2] - np.pi:
		# 		starting_state[2] = theta
		#
		# 	starting_inner_state = copy.deepcopy(starting_state)

		return starting_state, starting_inner_state

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

		return np.array(state[:2])

	def compute_distance_in_goal_space(self, goal1, goal2):

		goal1 = np.array(goal1)
		goal2 = np.array(goal2)

		if len(goal1.shape) ==  1:
			return np.linalg.norm(goal1 - goal2, axis=-1)
		else:
			return np.linalg.norm(goal1[:,:] - goal2[:,:], axis=-1)
