import tensorflow as tf
from tensorflow.contrib.layers import flatten


def batch_norm(inputs, is_training):
    """
    This function performs batch normalization on the inputs of the given layer.

    To know more about this, please read :
    Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift
    by Sergey Ioffe, Christian Szegedy
    https://arxiv.org/abs/1502.03167
    """
    decay = 0.999
    epsilon = 1e-03
    #     mean_pop = tf.Variable(tf.zeros(inputs.get_shape().as_list()[1:]), trainable=False)
    mean_pop = tf.Variable(tf.zeros(inputs.get_shape()[-1:]), trainable=False)
    #     var_pop = tf.Variable(tf.ones(inputs.get_shape().as_list()[1:]), trainable=False)
    var_pop = tf.Variable(tf.ones(inputs.get_shape()[-1:]), trainable=False)
    beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
    scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))

    def norm_train(inputs):
        mean_batch, var_batch = tf.nn.moments(inputs, [0, 1, 2])
        mean_train = tf.assign(mean_pop, mean_pop * decay + mean_batch * (1 - decay))
        var_train = tf.assign(var_pop, var_pop * decay + var_batch * (1 - decay))

        with tf.control_dependencies([mean_train, var_train]):
            return tf.nn.batch_normalization(inputs, mean_batch, var_batch, beta, scale, epsilon)

    def norm_test(inputs):
        return tf.nn.batch_normalization(inputs, mean_pop, var_pop, beta, scale, epsilon)

    return tf.cond(is_training, lambda: norm_train(inputs), lambda: norm_test(inputs))


def deep_net(x, is_training, keep_prob):
    """
    This function defines the convolutional neural network(CNN) for the task of the traffic sign recognition.
    It contains 3 convolutional layers and 2 fully connected layers.
    Max pooling is used to reduce the dimensions.
    Each convolutional layer is batch normalized.
    ReLu is used as an activation function.
    Dropout is used at the fully connected layers for the purpose of regularization.

    :param x:
    :param is_training: variable to define whether this function is called during training or testing
    :param keep_prob: variable to define the probability of keeping the neurons in a layer during dropout
    :return: value of the output layer in terms of logits.
    """
    mu = 0
    sigma = 0.1
    weights = {
        "wc1": tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6), mean=mu, stddev=sigma)),
        "wc2": tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean=mu, stddev=sigma)),
        "wc3": tf.Variable(tf.truncated_normal(shape=(5, 5, 16, 32), mean=mu, stddev=sigma)),
        "wf1": tf.Variable(tf.truncated_normal(shape=(1464, 800), mean=mu, stddev=sigma)),
        "wf2": tf.Variable(tf.truncated_normal(shape=(800, 400), mean=mu, stddev=sigma)),
        "wout": tf.Variable(tf.truncated_normal(shape=(400, 43), mean=mu, stddev=sigma)),
    }

    #     bias_out = tf.Variable(tf.zeros(43))
    biases = {
        "bc1": tf.Variable(tf.zeros(6)),
        "bc2": tf.Variable(tf.zeros(16)),
        "bc3": tf.Variable(tf.zeros(32)),
        "bf1": tf.Variable(tf.zeros(800)),
        "bf2": tf.Variable(tf.zeros(400)),
        "bout": tf.Variable(tf.zeros(43))
    }

    conv1 = tf.nn.conv2d(x, weights["wc1"], strides=[1, 1, 1, 1], padding="VALID")
    conv1 = tf.nn.bias_add(conv1, biases["bc1"])
    # conv1 = batch_norm(conv1, is_training)
    conv1 = tf.nn.relu(conv1)
    conv1 = tf.nn.max_pool(conv1, strides=[1, 2, 2, 1], ksize=[1, 2, 2, 1], padding="VALID")
    # print(conv1.get_shape())
    direct_layer = tf.identity(conv1)
    # conv1 = tf.nn.dropout(conv1, keep_prob=keep_prob)

    conv2 = tf.nn.conv2d(conv1, weights["wc2"], strides=[1, 1, 1, 1], padding="VALID")
    conv2 = tf.nn.bias_add(conv2, biases["bc2"])
    # conv2 = batch_norm(conv2, is_training)
    conv2 = tf.nn.relu(conv2)
    # print(conv2.get_shape())
    #     conv2 = tf.nn.max_pool(conv2, strides=[1,2,2,1], ksize=[1,2,2,1], padding="VALID")
    #     conv2 = tf.nn.dropout(conv2, keep_prob=keep_prob)

    conv3 = tf.nn.conv2d(conv2, weights["wc3"], strides=[1, 1, 1, 1], padding="VALID")
    conv3 = tf.nn.bias_add(conv3, biases["bc3"])
    # conv3 = batch_norm(conv3, is_training)
    conv3 = tf.nn.relu(conv3)
    # print(conv3.get_shape())
    conv3 = tf.nn.max_pool(conv3, strides=[1, 2, 2, 1], ksize=[1, 2, 2, 1], padding="VALID")
    #     conv2 = tf.nn.dropout(conv2, keep_prob=keep_prob)
    # print(conv3.get_shape())

    flat1 = flatten(conv3)
    # print(flat1.get_shape())
    flat2 = flatten(direct_layer)
    # print(flat2.get_shape())

    flat = tf.concat(axis=1, values=[flat1, flat2])
    # print(flat.get_shape())
    #     full1 = tf.matmul(flat_layer_part1, weights["wf1"])
    full1 = tf.add(tf.matmul(flat, weights["wf1"]), biases["bf1"])
    #     full_layer1 = tf.nn.bias_add(full_layer1, biases["bf1"])
    #     full_layer1 = batch_norm(full_layer1, is_training)
    full1 = tf.nn.relu(full1)
    full1 = tf.nn.dropout(full1, keep_prob)

    #     full_layer2 = tf.matmul(full_layer1, weights["wf2"])
    full2 = tf.add(tf.matmul(full1, weights["wf2"]), biases["bf2"])
    #     full_layer2 = tf.nn.bias_add(full_layer2, biases["bf2"])
    #     full_layer2 = batch_norm(full_layer2, is_training)
    full2 = tf.nn.relu(full2)
    full2 = tf.nn.dropout(full2, keep_prob)

    logits = tf.add(tf.matmul(full2, weights["wout"]), biases["bout"])
    #     logits_sig = tf.nn.sigmoid(logits)

    return logits



