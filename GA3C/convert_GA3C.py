import numpy as np
import collections
import pickle as pkl
import h5py
import argparse
import os, sys
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join('../..', 'gym_ca')))
from gym_collision_avoidance.envs.policies.GA3C_CADRL.network import Actions, Actions_Plus

parser = argparse.ArgumentParser(prog='Dataset conversion', description='Split data by episode')
parser.add_argument('--filename', '-f', type=str, required=True)
args = parser.parse_args()

filename = args.filename
if not os.path.isfile(filename):
    assert(0), f'{filename} does not exist'
name = filename.split('.')[0]

dataset = {}

with h5py.File(filename, 'r') as f:
    for key in list(f.keys()):
        if key == 'metadata':
            continue
        dataset[key] = f[key][()]

N = dataset['rewards'].shape[0]
data_ = collections.defaultdict(list)

use_timeouts = False
if 'timeouts' in dataset:
    use_timeouts = True

episode_step = 0
paths = []
d_act = 0
for i in tqdm(range(N)):
    done_bool = bool(dataset['terminals'][i])
    if use_timeouts:
        final_timestep = dataset['timeouts'][i]
    else:
        final_timestep = False
    for k in ['observations', 'next_observations', 'c_actions', 'd_actions', 'rewards', 'terminals']:
        print(f'{k} | {dataset[k].shape}')
        data_[k].append(dataset[k][i])
    if done_bool or final_timestep:
        episode_step = 0
        episode_data = {}
        for k in data_:
            episode_data[k] = np.array(data_[k])
        paths.append(episode_data)
        data_ = collections.defaultdict(list)
    episode_step += 1
    assert(0)
# print(f'd_actions: {d_act}')

returns = np.array([np.sum(p['rewards']) for p in paths])
num_samples = np.sum([p['rewards'].shape[0] for p in paths])
print(f'Number of samples collected: {num_samples}')
print(f'Trajectory returns: mean = {np.mean(returns):.2f}, std = {np.std(returns):.2f}, max = {np.max(returns):.2f}, min = {np.min(returns):.2f}')

with open(f'{name}.pkl', 'wb') as f:
    pkl.dump(paths, f)
