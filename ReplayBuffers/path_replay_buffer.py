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


class PathBuffer:
    """A replay buffer that stores and can sample whole paths.

    This buffer only stores valid steps, and doesn't require paths to
    have a maximum length.

    Args:
        capacity_in_transitions (int): Total memory allocated for the buffer.

    """

    def __init__(self, capacity_in_transitions, batch_size, pixel_keys, seed):
        self._capacity = capacity_in_transitions
        self.batch_size = batch_size
        self._first_idx_of_next_path = 0
        # Each path in the buffer has a tuple of two ranges in
        # self._path_segments. If the path is stored in a single contiguous
        # region of the buffer, the second range will be range(0, 0).
        # The "left" side of the deque contains the oldest path.
        self._path_segments = collections.deque()
        self._buffer = {}
        self.rng = np.random.default_rng(seed)
        self._pixel_keys = pixel_keys
        self.relevant_indicies = []

    def _next_path_segments(self, n_indices):
        """Compute where the next path should be stored.

        Args:
            n_indices (int): Path length.

        Returns:
            tuple: Lists of indices where path should be stored.

        Raises:
            ValueError: If path length is greater than the size of buffer.

        """
        if n_indices > self._capacity:
            raise ValueError('Path is too long to store in buffer.')
        start = self._first_idx_of_next_path
        end = start + n_indices
        if end > self._capacity:
            second_end = end - self._capacity
            return (range(start, self._capacity), range(0, second_end))
        else:
            return (range(start, end), range(0, 0))
        
    @staticmethod
    def _segments_overlap(seg_a, seg_b):
        """Compute if two segments overlap.

        Args:
            seg_a (range): List of indices of the first segment.
            seg_b (range): List of indices of the second segment.

        Returns:
            bool: True iff the input ranges overlap at at least one index.

        """
        # Empty segments never overlap.
        if not seg_a or not seg_b:
            return False
        
        first, second = seg_a, seg_b
        if seg_b.start < seg_a.start:
            first, second = seg_b, seg_a

        return first.stop > second.start
        
    @staticmethod
    def _get_path_length(path):
        """Get path length.

        Args:
            path (dict): Path.

        Returns:
            length: Path length.

        Raises:
            ValueError: If path is empty or has inconsistent lengths.

        """
        length_key = None
        length = None
        for key, value in path.items():
            if length is None:
                length = len(value)
                length_key = key
            elif len(value) != length:
                raise ValueError('path has inconsistent lengths between '
                                 '{!r} and {!r}.'.format(length_key, key))
        if not length:
            raise ValueError('Nothing in path')
        return length

    def add_path(self, path):
        """Add a path to the buffer.
        Args:
            path (dict): A dict of array of shape (path_len, flat_dim).
        Raises:
            ValueError: If a key is missing from path or path has wrong shape.
        """
        path_len = self._get_path_length(path)
        first_seg, second_seg = self._next_path_segments(path_len)
        # Remove paths which will overlap with this one.
        while (self._path_segments and self._segments_overlap(
                first_seg, self._path_segments[0][0])):
            self._path_segments.popleft()
        while (self._path_segments and self._segments_overlap(
                second_seg, self._path_segments[0][0])):
            self._path_segments.popleft()
        self._path_segments.append((first_seg, second_seg))
        for key, array in path.items():
            buf_arr = self._get_or_allocate_key(key, array)
            buf_arr[first_seg.start: first_seg.stop] = array[:len(first_seg)]
            buf_arr[second_seg.start: second_seg.stop] = array[len(first_seg):]
        if second_seg.stop != 0:
            self._first_idx_of_next_path = second_seg.stop
        else:
            self._first_idx_of_next_path = first_seg.stop

    def prepare_sampling(self):
        relevant_indicies = []
        for path in self._path_segments:
            relevant_indicies += list(path[0]) + list(path[1])
        self.relevant_indicies = np.array(relevant_indicies)

    def fetch_transitions(self, idxs):
        batch = {key: buf_arr[idxs] for key, buf_arr in self._buffer.items()}
        for key in self._pixel_keys:
            batch[key] = ((batch[key].astype(np.float32) - (255 / 2)) / (255 / 2))
        return batch

    def sample_transitions(self, batch_size = None):
        """Sample a batch of transitions from the buffer.

        Args:
            batch_size (int): Number of transitions to sample.

        Returns:
            dict: A dict of arrays of shape (batch_size, flat_dim).

        """
        if batch_size is None:
            batch_size = self.batch_size
        idxs = np.random.choice(self.relevant_indicies, batch_size)
        return self.fetch_transitions(idxs)

    def _get_or_allocate_key(self, key, array):
        """Get or allocate key in the buffer.

        Args:
            key (str): Key in buffer.
            array (numpy.ndarray): Array corresponding to key.

        Returns:
            numpy.ndarray: A NumPy array corresponding to key in the buffer.

        """
        buf_arr = self._buffer.get(key, None)
        if buf_arr is None:
            buf_arr = np.zeros((self._capacity,) + array.shape[1:], array.dtype)
            self._buffer[key] = buf_arr
        return buf_arr

    def preprocess_data(self, paths):
        data = copy.deepcopy(paths)
        if 'obs' in self._pixel_keys:
            assert np.bitwise_and(np.all(data['observations'] > -1.01), np.all(data['observations'] < 1.01)),\
                'Expected normalized images'
            data['observations'] = np.rint((data['observations'] * 255 / 2) + 255 / 2).astype(np.uint8)

        if 'next_obs' in self._pixel_keys:
            assert np.bitwise_and(np.all(data['next_observations'] > -1.01), np.all(data['next_observations'] < 1.01)),\
                'Expected normalized images'
            data['next_observations'] = np.rint((data['next_observations'] * 255 / 2) + 255 / 2).astype(np.uint8)
        return data

    def update_replay_buffer(self, data):
        data = dict(self.preprocess_data(data))
        for i in range(len(data['actions'])):
            path = {}
            for key in data.keys():
                path[key] = data[key][i]
            self.add_path(path)

    @property
    def n_transitions_stored(self):
        """Return the size of the replay buffer.

        Returns:
            int: Size of the current replay buffer.

        """
        return self.relevant_indicies.shape[0]