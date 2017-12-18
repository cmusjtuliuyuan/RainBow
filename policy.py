"""RL Policy classes."""

import numpy as np


class Policy:
    """Base class representing an MDP policy.
    Policies are used by the agent to choose actions.
    Policies are designed to be stacked to get interesting behaviors
    of choices. For instances in a discrete action space the lowest
    level policy may take in Q-Values and select the action index
    corresponding to the largest value. If this policy is wrapped in
    an epsilon greedy policy then with some probability epsilon, a
    random action will be chosen.
    """

    def select_action(self, **kwargs):
        """Used by agents to select actions.
        Returns
        -------
        Any:
          An object representing the chosen action. Type depends on
          the hierarchy of policy instances.
        """
        raise NotImplementedError('This method should be overriden.')


class UniformRandomPolicy(Policy):
    """Chooses a discrete action with uniform random probability.
    This is provided as a reference on how to use the policy class.
    Parameters
    ----------
    num_actions: int
      Number of actions to choose from. Must be > 0.
    Raises
    ------
    ValueError:
      If num_actions <= 0
    """

    def __init__(self, num_actions, batch_size):
        assert num_actions >= 1
        self.num_actions = num_actions
        self.batch_size = batch_size

    def select_action(self, **kwargs):
        """Return a random action index.
        This policy cannot contain others (as they would just be ignored).
        Returns
        -------
        int:
          Action index in range [0, num_actions)
        """
        return np.random.randint(0, self.num_actions, size=(self.batch_size,))

    def get_config(self):
        return {'num_actions': self.num_actions,
                'batch_size': self.batch_size}


class GreedyPolicy(Policy):
    """Always returns best action according to Q-values.
    This is a pure exploitation policy.
    """

    def select_action(self, q_values, **kwargs):
        """q_values shape: [batch_size, num_actions]"""
        return np.argmax(q_values, axis=1)


class GreedyEpsilonPolicy(Policy):
    """Selects greedy action or with some probability a random action.
    Standard greedy-epsilon implementation. With probability epsilon
    choose a random action. Otherwise choose the greedy action.
    Parameters
    ----------
    epsilon: float
     Initial probability of choosing a random action. Can be changed
     over time.
    """
    def __init__(self, epsilon):
        self._epsilon = epsilon
        self._greedy_policy = GreedyPolicy()


    def select_action(self, q_values, epsilon_override=None, **kwargs):
        """Run Greedy-Epsilon for the given Q-values.
        Parameters
        ----------
        q_values: array-like
          Array-like structure of floats representing the Q-values for
          each action.
        epsilon_override: float
          Optional epsilon to use, instead of one passed in constructor.
        Returns
        -------
        int:
          The action index chosen.
        """
        if epsilon_override is None:
            epsilon_override = self._epsilon

        if np.random.rand() < epsilon_override:
          return np.random.randint(0, q_values.shape[1], size=(q_values.shape[0],))

        return self._greedy_policy.select_action(q_values, **kwargs)


class LinearDecayGreedyEpsilonPolicy(Policy):
    """Policy with a parameter that decays linearly.
    Like GreedyEpsilonPolicy but the epsilon decays from a start value
    to an end value over k steps.
    Parameters
    ----------
    start_value: int, float
      The initial value of the parameter
    end_value: int, float
      The value of the policy at the end of the decay.
    num_steps: int
      The number of steps over which to decay the value.
      A step is a change. So 3 steps from 1 to 10 = [1, 4, 7, 10].
    """

    def __init__(self, start_value, end_value, num_steps):  # noqa: D102
        self._epsilon_policy = GreedyEpsilonPolicy(start_value)
        self._start_value = start_value
        self._increment = (end_value - start_value) / (1.0 * num_steps)

        self._current_step = 0  # Mutable.
        self._num_steps = num_steps


    def select_action(self, q_values, decay_epsilon=True, **kwargs):
        """Decay parameter and select action.
        Parameters
        ----------
        q_values: np.array
          The Q-values for each action.
        decay_epsilon: bool, optional
          If true then parameter will be decayed. Defaults to true.
          If false, then parameter will use the last value, which could be a
          partially decayed value.
        Returns
        -------
        Any:
          Selected action.
        """
        epsilon = self._start_value + self._current_step * self._increment
        if decay_epsilon and self._current_step < self._num_steps:
            self._current_step += 1

        return self._epsilon_policy.select_action(
            q_values, epsilon_override=epsilon, **kwargs)


    def reset(self):
        """Start the decay over at the start value."""
        self._current_step = 0