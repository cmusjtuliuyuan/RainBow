# RainBow

The OpenAI Gym can be paralleled by the bathEnv.py, which makes the training faster.

You can use the following command to choose which DQN to use:

```
python main.py --is_double 1 --is_duel 1 --is_per 1 --is_distributional 1 --is_noisy 1 --num_step 3
```

The output looks like:

```
Number_of_frame    mean_max_Q    average_reward    variance_reward
```

### RainBow:

https://arxiv.org/abs/1710.02298

### DQN done:

https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf

### Double DQN done:

https://arxiv.org/abs/1509.06461

### Duel DQN done:

https://arxiv.org/abs/1511.06581

DQN, Double DQN, and Duel DQN parts are implemented by Ben and me. https://github.com/bparr/10703

### PER DQN done:

https://arxiv.org/abs/1511.05952

Thanks to: https://jaromiru.com/2016/11/07/lets-make-a-dqn-double-learning-and-prioritized-experience-replay/#fn-444-2

### Distributional DQN done:

https://arxiv.org/abs/1707.06887

The best and most concise implement of Distributional RL loss function in Tensorflow by far in the world!!!
Better and faster than all others' implements that I can find.

### Multi-step done

### NoisyNet done

https://arxiv.org/pdf/1706.10295.pdf

Thanks to: https://github.com/andrewliao11/NoisyNet-DQN/blob/master/tf_util.py
