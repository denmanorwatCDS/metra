import collections
import numpy as np
import torch
import copy

class StateSequence:
    def __init__(self, num_parallel_envs):
        self._trajectory_data = {
            'observations': [], 'next_observations': [], 'actions': [],
            'options': [], 'obj_idxs': [],
            'terminated': [], 'truncated': []
        }
        self.local_trajectory_bounds = [[] for i in range(num_parallel_envs)]
        self._num_parallel_envs = num_parallel_envs
        self._length = 0

    def update(self, states, next_states, actions, options, obj_idxs, terminated, truncated):
        self._trajectory_data['observations'].append(states)
        self._trajectory_data['next_observations'].append(next_states)
        self._trajectory_data['actions'].append(actions)
        self._trajectory_data['options'].append(options)
        self._trajectory_data['obj_idxs'].append(obj_idxs)
        self._trajectory_data['terminated'].append(terminated)
        self._trajectory_data['truncated'].append(truncated)
        self._length += 1

    def finalize_trajectory_data(self):
        self._trajectory_data['observations'] = np.stack(self._trajectory_data['observations'], axis = 1)
        self._trajectory_data['next_observations'] = np.stack(self._trajectory_data['next_observations'], axis = 1)
        self._trajectory_data['actions'] = np.stack(self._trajectory_data['actions'], axis = 1)
        self._trajectory_data['options'] = np.stack(self._trajectory_data['options'], axis = 1)
        self._trajectory_data['obj_idxs'] = np.stack(self._trajectory_data['obj_idxs'], axis = 1)
        self._trajectory_data['terminated'] = np.stack(self._trajectory_data['terminated'], axis = 1)
        self._trajectory_data['truncated'] = np.stack(self._trajectory_data['truncated'], axis = 1)
        
        start = [0 for i in range(self._num_parallel_envs)]
        done = np.logical_or(self._trajectory_data['terminated'], self._trajectory_data['truncated'])
        index_of_environment = np.arange(self._num_parallel_envs)
        for i in range(self._length):
            finished_envs = index_of_environment[done[:, i]]
            for finished_env in finished_envs:
                self.local_trajectory_bounds[finished_env].append((start[finished_env], i))
                start[finished_env] = None
                if i != self._length - 1:
                    start[finished_env] = i + 1
        
        for i, s in enumerate(start):
            if s is not None:
                self.local_trajectory_bounds[i].append((s, None))

    def __len__(self):
        return self._length
    
    def __getitem__(self, key):
        return self._trajectory_data[key]
    
    def keys(self):
        return self._trajectory_data.keys()
        
class CyclicBuffer:
    def __init__(self, total_transitions, env_qty, seed, device,
                 observation_shape, observation_dtype, 
                 action_shape, action_dtype, 
                 option_shape, option_dtype):
        
        self.transitions_per_env, self.env_qty = total_transitions // env_qty, env_qty
        self.device = device
        self.rng = np.random.default_rng(seed)
        self._data = {
            'observations': np.array(np.zeros((env_qty, self.transitions_per_env, *observation_shape), dtype = observation_dtype)),
            'next_observations': np.array(np.zeros((env_qty, self.transitions_per_env, *observation_shape), dtype = observation_dtype)),
            'actions': np.array(np.zeros((env_qty, self.transitions_per_env, *action_shape), dtype = action_dtype)),
            'options': np.array(np.zeros((env_qty, self.transitions_per_env, option_shape), dtype = option_dtype)),
            'obj_idxs': np.array(np.zeros((env_qty, self.transitions_per_env), dtype = int)),
            'terminated': np.array(np.zeros((env_qty, self.transitions_per_env), dtype = bool)),
            'truncated': np.array(np.zeros((env_qty, self.transitions_per_env), dtype = bool))
        }
        self._mask = np.ones((env_qty, self.transitions_per_env), dtype = bool)
        
        self._next_free_index = np.zeros(env_qty, dtype = int)
        self._oldest_occupied_index = np.ma.array(data = np.zeros(env_qty, dtype = int), mask = np.ones(env_qty, dtype = bool))
        
        self._start_to_end = [{} for i in range(env_qty)]
        self._start_of_non_ended_trajectories = [None for i in range(env_qty)]

    def update_buffer(self, state_sequence):
        for env_idx in range(self.env_qty):
            while (not (np.ma.is_masked(self._oldest_occupied_index[env_idx]))) and\
                  len(state_sequence) > ((self._oldest_occupied_index[env_idx] - self._next_free_index[env_idx]) % self.transitions_per_env):
                start = self._oldest_occupied_index[env_idx]
                end = self._start_to_end[env_idx].pop(start)
                self._oldest_occupied_index[env_idx] = (end + 1) % self.transitions_per_env
                
                if end < start:
                    end += self.transitions_per_env
                removed_indexes = np.arange(start, end + 1) % self.transitions_per_env
                self._mask[env_idx][removed_indexes] = True
            
            if np.ma.is_masked(self._oldest_occupied_index[env_idx]):
                self._oldest_occupied_index[env_idx] = 0
            
            overwritten_indexes = np.arange(self._next_free_index[env_idx], 
                                            self._next_free_index[env_idx] + len(state_sequence)) % self.transitions_per_env
            for key in state_sequence.keys():
                self._data[key][env_idx][overwritten_indexes] = state_sequence[key][env_idx]
            self._mask[env_idx][overwritten_indexes] = False
            for local_start, local_end in state_sequence.local_trajectory_bounds[env_idx]:
                global_start = (self._next_free_index[env_idx] + local_start) % self.transitions_per_env
                if self._start_of_non_ended_trajectories[env_idx] is not None:
                    global_start = self._start_of_non_ended_trajectories[env_idx]
                    self._start_of_non_ended_trajectories[env_idx] = None

                if local_end is None:
                    self._start_of_non_ended_trajectories[env_idx] = global_start
                    local_end = len(state_sequence) - 1
                global_end = (self._next_free_index[env_idx] + local_end) % self.transitions_per_env
                self._start_to_end[env_idx][global_start] = global_end
            self._next_free_index[env_idx] = global_end + 1

    def prepare_for_sampling(self):
        valid_entries = np.logical_not(self._mask).astype(bool)
        all_indexes_in_memory = np.arange(self.env_qty * self.transitions_per_env).\
            reshape(self.env_qty, self.transitions_per_env)[valid_entries]
        self.sampling_indexes = all_indexes_in_memory
        pass
    
    def sample(self, size):
        samples = self.rng.choice(self.sampling_indexes, size = size, replace = False)
        env_idx, buffer_idx = samples // self.transitions_per_env, samples % self.transitions_per_env
        batch = {}
        for key in self._data.keys():
            batch[key] = torch.from_numpy(self._data[key][env_idx, buffer_idx]).to(self.device)
        return batch
    
    @property
    def n_transitions_stored(self):
        return len(self.sampling_indexes)