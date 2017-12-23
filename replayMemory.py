"""Replay Memory"""

import numpy as np
import random

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


    def clear(self):
        """Reset the memory. Deletes all references to the samples."""
        self._memory = []