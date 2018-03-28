"""Wrapped atari environment"""

import numpy as np
import gym
import sys
from preprocessor import Preprocessor
import time

SKIP_START_FRAME_NUM = 42


class Environment:
    def __init__(self,
                env_name,
                window_size,
                input_shape,
                num_frame_per_action):
        self.env_name = env_name
        self.window_size = window_size
        self.env = gym.make(self.env_name)
        self.env.reset()
        self.num_actions = self.env.action_space.n
        self.num_frame_per_action = num_frame_per_action
        self.preprocessor = Preprocessor(window_size, input_shape)
        self.preprocessor.reset()

    def take_action(self, action):
        self.reward = 0
        self.action = action
        self.old_state = self.preprocessor.get_state()
        for _ in range(self.num_frame_per_action):
            state, intermediate_reward, is_terminal, _ = self.env.step(action)
            self.preprocessor.process_state_for_memory(state)
            self.reward += intermediate_reward
            if is_terminal:
                self.is_terminal=True
                self.reset()
                break
            else:
                self.is_terminal=False
        self.new_state = self.preprocessor.get_state()

    def reset(self):
        self.env.reset()
        self.preprocessor.reset()
        for _ in range(SKIP_START_FRAME_NUM-self.window_size):
            self.env.step(0)
        for _ in range(self.window_size):
            state, _, _, _ = self.env.step(0)
            self.preprocessor.process_state_for_memory(state)

    def get_state(self):
        return self.preprocessor.get_state()

    def get_train_tuple(self):
        return self.old_state, self.action, self.reward, self.new_state, self.is_terminal


    def close(self):
        self.env.close()
