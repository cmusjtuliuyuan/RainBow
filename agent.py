"""Main DQN agent."""

import numpy as np
import tensorflow as tf
from PIL import Image
import random

class DQNAgent:
    """Class implementing DQN.
    Parameters
    ----------
    gamma: float
      Discount factor.
    target_update_freq: float
      Frequency to update the target network. You can either provide a
      number representing a soft target update (see utils.py) or a
      hard target update (see utils.py and Atari paper.)
    batch_size: int
      How many samples in each minibatch.
    is_double_dqn: boolean
      Whether to treat the online/target models as a double dqn.
    """
    def __init__(self,
                 online_model,
                 target_model,
                 memory,
                 policies,
                 gamma,
                 update_freq,
                 target_update_freq,
                 update_target_params_ops,
                 batch_size,
                 is_double_dqn):
        self._online_model = online_model
        self._target_model = target_model
        self._memory = memory
        self._policies = policies
        self._gamma = gamma
        self._update_freq = update_freq
        self._target_update_freq = target_update_freq
        self._update_target_params_ops=update_target_params_ops
        self._batch_size = batch_size
        self._is_double_dqn = is_double_dqn
        self._update_times = 0


    def calc_q_values(self, sess, state, model):
        """Given a state (or batch of states) calculate the Q-values.
        Return
        ------
        Q-values for the state(s)
        """
        state = state.astype(np.float32) / 255.0
        feed_dict = {model['input_frames']: state}
        q_values = sess.run(model['q_values'], feed_dict=feed_dict)
        return q_values


    def select_action(self, sess, state, policy, model):
        """Select the action based on the current state.
        Returns
        --------
        selected action(s)
        """
        q_values = self.calc_q_values(sess, state, model)
        return policy.select_action(q_values=q_values)

    def get_mean_max_Q(self, sess, samples):
        mean_max = []
        INCREMENT = 1000
        for i in range(0, len(samples), INCREMENT):
            feed_dict = {self._online_model['input_frames']:
                samples[i: i + INCREMENT].astype(np.float32)/255.0}
            mean_max.append(sess.run(self._online_model['mean_max_Q'],
                feed_dict = feed_dict))
        return np.mean(mean_max)

    def evaluate(self, sess, env, num_episode):
        """Evaluate num_episode games by online model.
        Parameters
        ----------
        sess: tf.Session
        env: batchEnv.BatchEnvironment
          This is your paralleled Atari environment.
        num_episode: int
          This is the number of episode of games to evaluate
        Returns
        -------
        reward list for each episode
        """
        num_environment = env.num_process
        env.reset()
        reward_of_each_environment  = np.zeros(num_environment)
        rewards_list = []

        num_finished_episode = 0

        while num_finished_episode < num_episode:
            old_state, action, reward, new_state, is_terminal = env.get_state()
            action = self.select_action(sess, new_state,
                        self._policies['evaluate_policy'], self._online_model)
            env.take_action(action)
            for i, r, is_t in zip(range(num_environment), reward, is_terminal):
                if not is_t:
                    reward_of_each_environment[i] += r
                else:
                    rewards_list.append(reward_of_each_environment[i])
                    reward_of_each_environment[i] = 0
                    num_finished_episode += 1
        return np.mean(rewards_list), np.std(rewards_list)


    def fit(self, sess, env, num_iterations, do_train=True):
        """Fit your model to the provided batched environment.
        Its a good idea to print out things like loss, average reward,
        Q-values, etc to see if your agent is actually improving.
        You should probably also periodically save your network
        weights and any other useful info.
        This is where you should sample actions from your network,
        collect experience samples and add them to your replay memory,
        and update your network parameters.
        Parameters
        ----------
        sess: tf.Session
        env: batchEnv.BatchEnvironment
          This is your paralleled Atari environment.
        num_iterations: int
          How many samples to get from the env.
        do_train: boolean
          Whether to train the model or skip training (e.g. for burn in).
        """

        num_environment = env.num_process
        env.reset()

        for t in range(0, num_iterations, num_environment):
            old_state, action, reward, new_state, is_terminal = env.get_state()
            # Clip the reward to -1, 0, 1
            reward = np.sign(reward)
            self._memory.append(old_state, reward, action, new_state, is_terminal)

            next_action = self.select_action(sess, new_state, self._policies['train_policy'], self._online_model)
            env.take_action(next_action)

            #If train, first decide how many batch update to do, then train.
            if do_train:
                num_update = [1 if i%self._update_freq == 0 else 0 for i in range(t, t+num_environment)]
                for _ in num_update:
                    old_state_list, reward_list, action_list, new_state_list, is_terminal_list \
                                    = self._memory.sample(self._batch_size)

                    # calculate y_j
                    Q_values = self.calc_q_values(sess, new_state_list, self._target_model)
                    if self._is_double_dqn:
                        target_action_list = self.calc_q_values(
                            sess, new_state_list, self._online_model).argmax(axis=1)
                        max_q = [Q_values[i, j] for i, j in enumerate(target_action_list)]
                    else:
                        max_q = Q_values.max(axis=1)
                    y = np.array(reward_list)
                    # TODO following three line can be simplifed
                    for i in range(len(is_terminal_list)):
                      if not is_terminal_list[i]:
                          y[i] += self._gamma * max_q[i]

                    # Train on memory sample.
                    self._update_times += 1
                    old_state_list = old_state_list.astype(np.float32) / 255.0
                    feed_dict = {self._online_model['input_frames']: old_state_list,
                                 self._online_model['Q_vector_indexes']: list(enumerate(action_list)),
                                 self._online_model['y_ph']: y}
                    sess.run([self._online_model['train_step']], feed_dict=feed_dict)
                    
                    # Assign online_model to target_model 
                    if self._update_times%self._target_update_freq == 0:
                        sess.run(self._update_target_params_ops)




