"""Replay Memory"""

import numpy as np
import random
from sumTree import SumTree

class ReplayMemory:
    """Store and replay (sample) memories."""
    def __init__(self,
                max_size,
                window_size,
                input_shape):
        """Setup memory.
        You should specify the maximum size o the memory. Once the
        memory fills up oldest values are removed.
        """
        self._max_size = max_size
        self._window_size = window_size
        self._WIDTH = input_shape[0]
        self._HEIGHT = input_shape[1]
        self._memory = []


    def append(self, old_state, action, reward, new_state, is_terminal):
        """Add a list of samples to the replay memory."""
        num_sample = len(old_state)

        if len(self._memory) >= self._max_size:
            del(self._memory[0:num_sample])

        for o_s, a, r, n_s, i_t in zip(old_state, action, reward, new_state, is_terminal):
            self._memory.append((o_s, a, r, n_s, i_t))


    def sample(self, batch_size, indexes=None):
        """Return samples from the memory.
        Returns
        --------
        (old_state_list, action_list, reward_list, new_state_list, is_terminal_list, frequency_list)
        """
        samples = random.sample(self._memory, min(batch_size, len(self._memory)))
        zipped = list(zip(*samples))
        zipped[0] = np.reshape(zipped[0], (-1, self._WIDTH, self._HEIGHT, self._window_size))
        zipped[3] = np.reshape(zipped[3], (-1, self._WIDTH, self._HEIGHT, self._window_size))
        return zipped


class PriorityExperienceReplay:
    '''
    Almost copy from
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    '''
    def __init__(self,
                max_size,
                window_size,
                input_shape):
        self.tree = SumTree(max_size)
        self._max_size = max_size
        self._window_size = window_size
        self._WIDTH = input_shape[0]
        self._HEIGHT = input_shape[1]
        self.e = 0.01
        self.a = 0.6

    def _getPriority(self, error):
        return (error + self.e) ** self.a

    def append(self, old_state, action, reward, new_state, is_terminal):
        for o_s, a, r, n_s, i_t in zip(old_state, action, reward, new_state, is_terminal):
            # 0.5 is the maximum error
            p = self._getPriority(0.5)
            self.tree.add(p, data=(o_s, a, r, n_s, i_t)) 

    def sample(self, batch_size, indexes=None):
        data_batch = []
        idx_batch = []
        p_batch = []
        segment = self.tree.total_and_count()[0] / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            data_batch.append(data)
            idx_batch.append(idx)
            p_batch.append(p)

        zipped = list(zip(*data_batch))
        zipped[0] = np.reshape(zipped[0], (-1, self._WIDTH, self._HEIGHT, self._window_size))
        zipped[3] = np.reshape(zipped[3], (-1, self._WIDTH, self._HEIGHT, self._window_size))

        sum_p, count = self.tree.total_and_count()
        return zipped, idx_batch, p_batch, sum_p, count

    def update(self, idx_list, error_list):
        for idx, error in zip(idx_list, error_list):
            p = self._getPriority(error)
            self.tree.update(idx, p)