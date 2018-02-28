import tensorflow as tf

def conv2d_layer(input_tensor, filters, kernel_size=(3,3), name="conv", activation="relu"):
    with tf.variable_scope(name):
        weights = tf.get_variable('conv_weights', [kernel_size[0], kernel_size[1], input_tensor.get_shape()[-1], filters],
                            initializer=tf.truncated_normal_initializer(stddev=0.01))
        conv = tf.nn.conv2d(input_tensor, weights, strides=[1, 2, 2, 1], padding='SAME')
        biases = tf.get_variable('conv_biases', [filters], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
        if activation == "softmax":
            output = tf.nn.softmax(conv)
        elif activation == "relu":
            output = tf.nn.relu(conv)
        elif activation == "sigmoid":
            output = tf.nn.sigmoid(conv)
        return output

def fc_layer(input_tensor, neurons, activation="softmax", name="fc"):
    with tf.variable_scope(name):
        weights = tf.get_variable('fc_weights', [input_tensor.get_shape()[-1], neurons],
                                  initializer=tf.truncated_normal_initializer(stddev=0.01))
        biases = tf.get_variable('fc_biases', [neurons], initializer=tf.constant_initializer(0.0))
        if activation == "softmax":
            output = tf.nn.softmax(tf.matmul(input_tensor, weights) + biases)
        elif activation == "relu":
            output = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)
        elif activation == "sigmoid":
            output = tf.nn.sigmoid(tf.matmul(input_tensor, weights) + biases)
        return output


