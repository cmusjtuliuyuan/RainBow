"""Q-networks"""

import tensorflow as tf

# Returns tuple of flat output, flat output size, network_parameters.
def create_conv_network(input_frames, trainable):
    conv1_W = tf.Variable(tf.random_normal([8, 8, 4, 16], stddev=0.1, dtype=tf.float32),
        name='conv1_W', trainable=trainable)
    conv1_b = tf.Variable(tf.zeros([16], dtype=tf.float32),
        name='conv1_b', trainable=trainable)
    conv1 = tf.nn.conv2d(input_frames, conv1_W, strides=[1, 4, 4, 1],
        padding='VALID', name='conv1')
    # (batch size, 20, 20, 16)
    output1 = tf.nn.relu(conv1 + conv1_b, name='output1')

    conv2_W = tf.Variable(tf.random_normal([4, 4, 16, 32], stddev=0.1, dtype=tf.float32),
        name='conv2_W', trainable=trainable)
    conv2_b = tf.Variable(tf.zeros([32], dtype=tf.float32), name='conv2_b',
        dtype=tf.float32, trainable=trainable)
    conv2 = tf.nn.conv2d(output1, conv2_W, strides=[1, 2, 2, 1],
        padding='VALID', name='conv2')
    # (batch size, 9, 9, 32)
    output2 = tf.nn.relu(conv2 + conv2_b, name='output2')

    flat_output2_size = 9 * 9 * 32
    flat_output2 = tf.reshape(output2, [-1, flat_output2_size], name='flat_output2')

    return flat_output2, flat_output2_size, [conv1_W, conv1_b, conv2_W, conv2_b]


# Returns tuple of network, network_parameters.
def create_deep_q_network(input_frames, input_length, num_actions, trainable):
    flat_output, flat_output_size, parameter_list = create_conv_network(input_frames, trainable)
    fc1_W = tf.Variable(tf.random_normal([flat_output_size, 256], stddev=0.1, dtype=tf.float32),
        name='fc1_W', trainable=trainable)
    fc1_b = tf.Variable(tf.zeros([256], dtype=tf.float32), name='fc1_b',
        trainable=trainable)
    # (batch size, 256)
    output3 = tf.nn.relu(tf.matmul(flat_output, fc1_W) + fc1_b, name='output3')

    fc2_W = tf.Variable(tf.random_normal([256, num_actions], stddev=0.1, dtype=tf.float32),
        name='fc2_W', trainable=trainable)
    fc2_b = tf.Variable(tf.zeros([num_actions], dtype=tf.float32), name='fc2_b',
        trainable=trainable)
    # (batch size, num_actions)
    q_network = tf.nn.relu(tf.matmul(output3, fc2_W) + fc2_b, name='q_network')

    parameter_list += [fc1_W, fc1_b, fc2_W, fc2_b]
    return q_network, parameter_list


# Returns tuple of network, network_parameters.
def create_duel_q_network(input_frames, input_length, num_actions, trainable):
    flat_output, flat_output_size, parameter_list = create_conv_network(input_frames, trainable)

    fcV_W = tf.Variable(tf.random_normal([flat_output_size, 512], stddev=0.01, dtype=tf.float32),
        name='fcV_W', trainable=trainable)
    fcV_b = tf.Variable(tf.zeros([512], dtype=tf.float32), name='fcV_b',
        dtype=tf.float32, trainable=trainable)
    outputV = tf.nn.relu(tf.matmul(flat_output, fcV_W) + fcV_b, name='outputV')

    fcV2_W = tf.Variable(tf.random_normal([512, 1], stddev=0.01, dtype=tf.float32),
        name='fcV2_W', trainable=trainable)
    fcV2_b = tf.Variable(tf.zeros([1], dtype=tf.float32), name='fcV2_b',
        trainable=trainable)
    outputV2 = tf.nn.relu(tf.matmul(outputV, fcV2_W) + fcV2_b, name='outputV2')


    fcA_W = tf.Variable(tf.random_normal([flat_output_size, 512], stddev=0.01, dtype=tf.float32),
        name='fcA_W', trainable=trainable)
    fcA_b = tf.Variable(tf.zeros([512], dtype=tf.float32), name='fcA_b',
        trainable=trainable)
    outputA = tf.nn.relu(tf.matmul(flat_output, fcA_W) + fcA_b, name='outputA')

    fcA2_W = tf.Variable(tf.random_normal([512, num_actions], stddev=0.01, dtype=tf.float32),
        name='fcA2_W', trainable=trainable)
    fcA2_b = tf.Variable(tf.zeros([num_actions], dtype=tf.float32), name='fcA2_b',
        trainable=trainable)
    outputA2 = tf.nn.relu(tf.matmul(outputA, fcA2_W) + fcA2_b, name='outputA2')

    q_network = tf.nn.relu(outputV2 + outputA2 - tf.reduce_mean(outputA2), name='q_network')
    
    parameter_list += [fcV_W, fcV_b, fcV2_W, fcV2_b, fcA_W, fcA_b, fcA2_W, fcA2_b]
    return q_network, parameter_list


def create_model(window, input_shape, num_actions, model_name, create_network_fn, trainable):
    """Create the Q-network model."""
    with tf.name_scope(model_name):
        input_frames = tf.placeholder(tf.float32, [None, input_shape[0],
                        input_shape[1], window], name ='input_frames')
        input_length = input_shape[0] * input_shape[1] * window
        q_network, parameter_list = create_network_fn(
            input_frames, input_length, num_actions, trainable)

        mean_max_Q = tf.reduce_mean( tf.reduce_max(q_network, axis=[1]), name='mean_max_Q')
        action = tf.argmax(q_network, axis=1)

        model = {
            'q_values': q_network,
            'input_frames': input_frames,
            'mean_max_Q': mean_max_Q,
            'action': action,
        }
    return model, parameter_list
