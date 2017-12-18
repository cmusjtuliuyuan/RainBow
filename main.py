import argparse
import gym
import numpy as np
import random
import tensorflow as tf

from batchEnv import BatchEnvironment
from replayMemory import ReplayMemory
from policy import GreedyPolicy, LinearDecayGreedyEpsilonPolicy, UniformRandomPolicy
from model import create_deep_q_network, create_duel_q_network, create_model
from agent import DQNAgent

NUM_FRAME_PER_ACTION = 4
REPLAYMEMORY_SIZE = 100000
MAX_EPISODE_LENGTH = 100000
RMSP_EPSILON = 0.01
RMSP_DECAY = 0.95
RMSP_MOMENTUM =0.95
MAX_EPISODE_LENGTH = 100000
NUM_FIXED_SAMPLES = 10000
NUM_BURN_IN = 50000
LINEAR_DECAY_LENGTH = 4000000

def main():
    parser = argparse.ArgumentParser(description='Run DQN on Atari Space Invaders')
    parser.add_argument('--env', default='SpaceInvaders-v0', help='Atari env name')
    parser.add_argument('--seed', default=10703, type=int, help='Random seed')
    parser.add_argument('--input_shape', default=(84,84), help='Input shape')
    parser.add_argument('--gamma', default=0.99, help='Discount factor')
    parser.add_argument('--epsilon', default=0.1, help='Exploration probability in epsilon-greedy')
    parser.add_argument('--learning_rate', default=0.00025, help='Training learning rate.')
    parser.add_argument('--window_size', default=4, type = int, help=
                                'Number of frames to feed to the Q-network')
    parser.add_argument('--batch_size', default=4, type = int, help=
                                'Batch size of the training part')
    parser.add_argument('--num_iteration', default=20000000, type = int, help=
                                'number of iterations to train')
    parser.add_argument('--eval_every', default=0.001, type = float, help=
                                'What fraction of num_iteration to run between evaluations.')
    parser.add_argument('--is_duel', default=0, type = int, help=
                                'Whether use duel DQN, 0 means no, 1 means yes.')
    parser.add_argument('--is_double', default=0, type = int, help=
                                'Whether use double DQN, 0 means no, 1 means yes.')


    args = parser.parse_args()
    args.input_shape = tuple(args.input_shape)
    print('Environment: %s.'%(args.env,))
    env = gym.make(args.env)
    num_actions = env.action_space.n
    print('number_actions: %d.'%(num_actions,))
    env.close()


    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)


    batch_environment = BatchEnvironment(args.env, args.batch_size,
                args.window_size, args.input_shape, NUM_FRAME_PER_ACTION, MAX_EPISODE_LENGTH)
    replay_memory = ReplayMemory(REPLAYMEMORY_SIZE, args.window_size, args.input_shape)
    policies = {
        'train_policy': LinearDecayGreedyEpsilonPolicy(1, args.epsilon, LINEAR_DECAY_LENGTH),
        'evaluate_policy': GreedyPolicy(),
    }


    create_network_fn = create_deep_q_network if args.is_duel == 0 else create_duel_q_network
    online_model, online_params = create_model(args.window_size, args.input_shape, num_actions,
                    'online_model', create_network_fn, args.learning_rate,
                    RMSP_DECAY, RMSP_MOMENTUM, RMSP_EPSILON, trainable=True)
    target_model, target_params = create_model(args.window_size, args.input_shape, num_actions,
                    'target_model', create_network_fn, args.learning_rate,
                    RMSP_DECAY, RMSP_MOMENTUM, RMSP_EPSILON, trainable=False)
    update_target_params_ops = [t.assign(s) for s, t in zip(online_params, target_params)]



    agent = DQNAgent(online_model,
                    target_model,
                    replay_memory,
                    policies,
                    args.gamma,
                    4,
                    update_target_params_ops,
                    args.batch_size,
                    args.is_double)

    sess = tf.Session()
    with sess.as_default():
        sess.run(tf.global_variables_initializer())
        # make target_model equal to online_model
        sess.run(update_target_params_ops)
        
        print('Burn in replay_memory')
        agent.evaluate(sess, batch_environment, 16)

    batch_environment.close()

if __name__ == '__main__':
    main()
