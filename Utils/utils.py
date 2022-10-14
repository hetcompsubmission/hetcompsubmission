import numpy as np
import os
import random
import pickle
import time
import torch
import copy
from Utils.evaluation import *

## helper functions
def load_pickle(path, filename):
	with open(path + filename, 'rb') as f:
		obj = pickle.load(f)

	return obj

def to_np(x):
	return x.data.cpu().numpy()


def dict_set(base_dict, u_id, i_id, val):
	if u_id in base_dict:
		base_dict[u_id][i_id] = val
	else:
		base_dict[u_id] = {i_id: val}


def is_visited(base_dict, u_id, i_id):
	if u_id in base_dict and i_id in base_dict[u_id]:
		return True
	else:
		return False


def list_to_dict(base_list):
	result = {}
	for u_id, i_id, value in base_list:
		dict_set(result, u_id, i_id, value)
	
	return result


def dict_to_list(base_dict):
	result = []

	for u_id in base_dict:
		for i_id in base_dict[u_id]:
			result.append((u_id, i_id, 1))
	
	return result
	
## for data load
def read_file(f):
	total_ints = []
	u_count, i_count = 0, 0
	u_id_dict = {}
	i_id_dict = {}

	for line in f.readlines():
		u_raw_id, i_raw_id, _, _ = line.strip().split(',')

		if u_raw_id not in u_id_dict:
			u_id_dict[u_raw_id] = len(u_id_dict)

		u_id = u_id_dict[u_raw_id]

		if i_raw_id not in i_id_dict:
			i_id_dict[i_raw_id] = len(i_id_dict)

		i_id = i_id_dict[i_raw_id]

		u_id = int(u_id)
		i_id = int(i_id)

		u_count = max(u_count, u_id)
		i_count = max(i_count, i_id)

		total_ints.append((u_id, i_id, 1))

	return u_count, i_count, total_ints


def load_data(test_ratio=0.2, random_seed=0):
	
	np.random.seed(random_seed)

	with open('ratings_Digital_Music.csv') as f:
		u_count, i_count, total_int_tmp = read_file(f)

	u_count_dict, i_count_dict = get_count_dict(total_int_tmp)
	u_count, i_count, total_ints = get_total_ints(total_int_tmp, u_count_dict, i_count_dict, count_filtering = [10, 10])
	
	print(u_count, i_count, len(total_ints))
	total_mat = list_to_dict(total_ints)

	train_mat, valid_mat, test_mat = {}, {}, {}

	for u in total_mat:
		is = list(total_mat[u].keys())
		np.random.shuffle(is)

		num_test_is = int(len(is) * test_ratio)
		test_is = is[:num_test_is]
		valid_is = is[num_test_is: num_test_is*2]
		train_is = is[num_test_is*2:]

		for i in test_is:
			dict_set(test_mat, u, i, 1)

		for i in valid_is:
			dict_set(valid_mat, u, i, 1)

		for i in train_is:
			dict_set(train_mat, u, i, 1)
			
	train_mat_R = {}

	for u in train_mat:
		for i in train_mat[u]:
			dict_set(train_mat_R, i, u, 1)
			
	for u in list(valid_mat.keys()):
		for i in list(valid_mat[u].keys()):
			if i not in train_mat_R:
				del valid_mat[u][i]
		if len(valid_mat[u]) == 0:
			del valid_mat[u]
			del test_mat[u]
			
	for u in list(test_mat.keys()):
		for i in list(test_mat[u].keys()):
			if i not in train_mat_R:
				del test_mat[u][i]
		if len(test_mat[u]) == 0:
			del test_mat[u]
			del valid_mat[u]
	
	train_ints = []
	for u in train_mat:
		for i in train_mat[u]:
			train_ints.append([u, i, 1])
			
	return u_count, i_count, train_mat, train_ints, valid_mat, test_mat


def get_count_dict(total_ints, spliter="\t"):

	u_count_dict, i_count_dict = {}, {}

	for line in total_ints:
		u, i, rating = line
		u, i, rating = int(u), int(i), float(rating)

		if u in u_count_dict:
			u_count_dict[u] += 1
		else: 
			u_count_dict[u] = 1

		if i in i_count_dict:
			i_count_dict[i] += 1
		else: 
			i_count_dict[i] = 1

	return u_count_dict, i_count_dict


def get_total_ints(total_int_tmp, u_count_dict, i_count_dict, is_implicit=True, count_filtering = [10, 10], spliter="\t"):

	total_ints = []
	u_dict, i_dict = {}, {}
	u_count, i_count = 0, 0

	for line in total_int_tmp:
		u, i, rating = line
		u, i, rating = int(u), int(i), float(rating)

		# count filtering
		if u_count_dict[u] < count_filtering[0]:
			continue
		if i_count_dict[i] < count_filtering[1]:
			continue

		# u indexing
		if u in u_dict:
			u_id = u_dict[u]
		else:
			u_id = u_count
			u_dict[u] = u_id
			u_count += 1

		# i indexing
		if i in i_dict:
			i_id = i_dict[i]
		else:
			i_id = i_count
			i_dict[i] = i_id
			i_count += 1

		if is_implicit:
			rating = 1.

		total_ints.append((u_id, i_id, rating))

	return u_count + 1, i_count + 1, total_ints


def load_teacher_trajectory(path, model_list):

	state_dict = {}
	for model_idx, model_type in enumerate(model_list):
		state_dict[model_type] = []
		for e in range(4):
			# we load the precomputed importance for ensemble (Eq.8)
			with open('./Teachers/' + model_type + "_E_" +  str(e), 'rb') as f:
				state_dict[model_type].append(np.load(f))

	# initial top/pos permutations
	p_results = load_pickle(path, 'observed')
	t_results = load_pickle(path, 't_results')

	exception_ints = []
	for u in range(p_results.shape[0]):
		for i in p_results[u][:K//2]:
			exception_ints.append((u, i, 1))
		for i in t_results[u][:K//2]:
			exception_ints.append((u, i, 1))

	train_dataset = train_dataset(u_count, i_count, train_mat, 1, train_ints, exception_ints)
	test_dataset = test_dataset(u_count, i_count, valid_mat, test_mat)
	train_loader = data.DataLoader(train_dataset, batch_size=1024, shuffle=True)

	R = np.zeros((u_count, i_count))
	for u in train_mat:
		is = list(train_mat[u].keys())
		R[u][is] = 1.

	last_max_idx = np.zeros((6, u_count))
	next_idx = np.clip(last_max_idx + 1, a_min=0, a_max=3)

	sorted_mat = g_torch(state_dict, torch.zeros((u_count, 6)))
	t_results = sorted_mat[:, :K]   
	p_results = p_results[:, :K]
	p_results = torch.LongTensor(p_results).to(gpu)  

	return state_dict, t_results, p_results


#############
def g(importance_mats):
	
	result = 0
	for importance_mat in importance_mats:
		result += importance_mat

	return result
