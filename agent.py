"""Main DQN agent."""

import numpy as np
import tensorflow as tf
from PIL import Image
import random
from huberLoss import mean_huber_loss, weighted_huber_loss

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
                 is_double_dqn,
                 is_per,
                 learning_rate,
                 rmsp_decay,
                 rmsp_momentum,
                 rmsp_epsilon):
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
        self._is_per = is_per
        self._learning_rate = learning_rate
        self._rmsp_decay = rmsp_decay
        self._rmsp_momentum = rmsp_momentum
        self._rmsp_epsilon = rmsp_epsilon
        self._update_times = 0
        self._beta = 0.5
        self._beta_increment = (1.0-0.5)/(5000000*0.8)

        self._action_ph = tf.placeholder(tf.int32, [None, 2], name ='action_ph')
        self._reward_ph = tf.placeholder(tf.float32, name='reward_ph')
        self._is_terminal_ph = tf.placeholder(tf.float32, name='is_terminal_ph')
        self._action_chosen_by_online_ph = tf.placeholder(tf.int32, [None, 2], name ='action_chosen_by_online_ph')
        self._huber_loss_weight_ph = tf.placeholder(tf.float32, name='huber_loss_weight_ph')
        self._error_op, self._train_op = self._get_error_and_train_op(self._reward_ph,
                self._is_terminal_ph, self._action_ph, self._action_chosen_by_online_ph, self._huber_loss_weight_ph)


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

    def _get_error_and_train_op(self,
                               reward_ph,
                               is_terminal_ph,
                               action_ph,
                               action_chosen_by_online_ph,
                               huber_loss_weight_ph):
        """Select the action based on the current state.
        Inputs
        --------
        reward_ph: tensorflow place holder for reward, [batch_size,] float32
        is_terminal_ph: tensorflow place holder for terminal signal, [batch_size,] float32
        action_ph: tensorflow place holder for action, [batch_size, 2] int
        action_chosen_by_online_ph: tensorflow place holder for action chosen by online_model
                                according to the new state list, [batch_size, 2] int
        Returns
        --------
        train operation
        """
        # calculate y_j

        Q_values_target = self._target_model['q_values']
        Q_values_online = self._online_model['q_values']

        if self._is_double_dqn:
            online_action_list = tf.argmax(Q_values_online, axis=1)
            max_q = tf.gather_nd(Q_values_target, action_chosen_by_online_ph)
        else:
            max_q = tf.reduce_max(Q_values_target, axis = 1)

        target = reward_ph + (1.0 - is_terminal_ph) * self._gamma * max_q
        gathered_outputs = tf.gather_nd(Q_values_online, action_ph, name='gathered_outputs')

        if self._is_per == 1:
            loss = weighted_huber_loss(target, gathered_outputs, huber_loss_weight_ph)
        else:
            loss = mean_huber_loss(target, gathered_outputs)

        train_op = tf.train.RMSPropOptimizer(self._learning_rate,
            decay=self._rmsp_decay, momentum=self._rmsp_momentum, epsilon=self._rmsp_epsilon).minimize(loss)
        error_op = tf.abs(gathered_outputs - target, name='abs_error')
        return error_op, train_op

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

            # TODO error and next_action can be combined together
            if self._is_per==1:
                error = self._get_error(sess, old_state, action, reward, new_state, is_terminal)
                self._memory.append(old_state, action, reward, new_state, is_terminal, error)
            else:
                self._memory.append(old_state, action, reward, new_state, is_terminal)

            next_action = self.select_action(sess, new_state, self._policies['train_policy'], self._online_model)
            env.take_action(next_action)

            #If train, first decide how many batch update to do, then train.
            if do_train:
                num_update = sum([1 if i%self._update_freq == 0 else 0 for i in range(t, t+num_environment)])
                for _ in range(num_update):
                    if self._is_per==1:
                        (old_state_list, action_list, reward_list, new_state_list, is_terminal_list),\
                                idx_list, p_list, sum_p, count = self._memory.sample(self._batch_size)
                    else:
                        old_state_list, action_list, reward_list, new_state_list, is_terminal_list \
                                    = self._memory.sample(self._batch_size)

                    feed_dict = {self._target_model['input_frames']: new_state_list.astype(np.float32)/255.0,
                                 self._online_model['input_frames']: old_state_list.astype(np.float32)/255.0,
                                 self._action_ph: list(enumerate(action_list)),
                                 self._reward_ph: np.array(reward_list).astype(np.float32),
                                 self._is_terminal_ph: np.array(is_terminal_list).astype(np.float32),
                                 }

                    if self._is_double_dqn:
                        action_chosen_by_online = sess.run(self._online_model['action'], feed_dict={
                                    self._online_model['input_frames']: new_state_list.astype(np.float32)/255.0})
                        feed_dict[self._action_chosen_by_online_ph] = list(enumerate(action_chosen_by_online))

                    if self._is_per == 1:
                        # Annealing weight beta
                        feed_dict[self._huber_loss_weight_ph] = (np.array(p_list)*count/sum_p)**(-self._beta)
                        error, _ = sess.run([self._error_op, self._train_op], feed_dict=feed_dict)
                        self._memory.update(idx_list, error)
                    else:
                        sess.run(self._train_op, feed_dict=feed_dict)

                    self._update_times += 1
                    if self._beta < 1:
                        self._beta += self._beta_increment

                    if self._update_times%self._target_update_freq == 0:
                        sess.run(self._update_target_params_ops)

    def _get_error(self, sess, old_state, action, reward, new_state, is_terminal):
        '''
        Get error for Prioritized Experience Replay
        '''
        feed_dict = {self._target_model['input_frames']: new_state.astype(np.float32)/255.0,
                     self._online_model['input_frames']: old_state.astype(np.float32)/255.0,
                     self._action_ph: list(enumerate(action)),
                     self._reward_ph: np.array(reward).astype(np.float32),
                     self._is_terminal_ph: np.array(is_terminal).astype(np.float32),
                     }

        if self._is_double_dqn:
            action_chosen_by_online = sess.run(self._online_model['action'], feed_dict={
                        self._online_model['input_frames']: new_state.astype(np.float32)/255.0})
            feed_dict[self._action_chosen_by_online_ph] = list(enumerate(action_chosen_by_online))

        error = sess.run(self._error_op, feed_dict=feed_dict)
        return error



