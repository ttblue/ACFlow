import math
import tensorflow as tf


def Layernorm(name, norm_axes, inputs):
  mean, var = tf.nn.moments(inputs, norm_axes, keep_dims=True)

  n_neurons = inputs.get_shape().as_list()[3]

  offset = tf.get_variable(name + '.offset', n_neurons,
                           initializer=tf.constant_initializer(0.0))
  scale = tf.get_variable(name + '.scale', n_neurons,
                          initializer=tf.constant_initializer(1.0))

  result = (inputs - mean) / tf.sqrt(var + 1e-5)
  result = result * scale + offset

  return result


def conv2d(input_, output_dim,
           k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
           name="conv2d", padding="SAME", biases=True, init_type="normal"):
  tf.set_random_seed(0)
  with tf.variable_scope(name):
    input_dim = input_.get_shape().as_list()[-1]
    fan_in = input_dim * k_h**2
    fan_out = output_dim * k_h**2 / (d_h**2)

    init_f = tf.truncated_normal_initializer(stddev=stddev, seed=0)
    if init_type == "he":
      filters_stdev = np.sqrt(4. / (fan_in + fan_out))
      init_f = \
          tf.random_uniform_initializer(-np.sqrt(3) *
                                        filters_stdev, np.sqrt(3) * filters_stdev, seed=0)
    elif init_type == "glorot":
      filters_stdev = np.sqrt(2. / (fan_in + fan_out))
      init_f = \
          tf.random_uniform_initializer(-np.sqrt(3) *
                                        filters_stdev, np.sqrt(3) * filters_stdev, seed=0)

    w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                        initializer=init_f)

    conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding=padding)

    if biases:
      biases = tf.get_variable(
          'biases', [output_dim], initializer=tf.constant_initializer(0.0))
      conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

    return conv


def lrelu(x, leak=0.2, name="lrelu"):
  return tf.maximum(x, leak * x)


def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False, init_type="normal"):
  tf.set_random_seed(0)
  shape = input_.get_shape().as_list()

  with tf.variable_scope(scope or "Linear"):
    if init_type == "normal":
      init_f = tf.random_normal_initializer(stddev=stddev, seed=0)
    else:
      st_dev = np.sqrt(2. / (shape[1] + output_size))
      init_f = tf.random_uniform_initializer(-np.sqrt(3)
                                             * st_dev, np.sqrt(3) * st_dev, seed=0)

    matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                             init_f)
    bias = tf.get_variable("bias", [output_size],
                           initializer=tf.constant_initializer(bias_start))
    if with_w:
      return tf.matmul(input_, matrix) + bias, matrix, bias
    else:
      return tf.matmul(input_, matrix) + bias
