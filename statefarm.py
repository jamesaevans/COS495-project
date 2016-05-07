"""Builds the Statefarm network.
Summary of available functions:
 # Compute input images and labels for training. If you would like to run
 # evaluations, use inputs() instead.
 inputs, labels = distorted_inputs()
 # Compute inference on the model inputs to make a prediction.
 predictions = inference(inputs)
 # Compute the total loss of the prediction with respect to the labels.
 loss = loss(predictions, labels)
 # Create a graph to run one step of training with respect to the loss.
 train_op = train(loss, global_step)
"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import re
import sys
import tarfile

from six.moves import urllib
import tensorflow as tf

import statefarm_input as statefarm_input
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages
import numpy as np

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', './tmp/statefarm_data',
                           """Path to the Statefarm data directory.""")

# Global constants describing the Statefarm data set.
IMAGE_SIZE = statefarm_input.IMAGE_SIZE
NUM_CLASSES = statefarm_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = statefarm_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = statefarm_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.

# Constants for resnet architecture
FC_WEIGHT_STDDEV = 0.01
FC_WEIGHT_DECAY = 0.00004
CONV_WEIGHT_STDDEV = 0.1
CONV_WEIGHT_DECAY = 0.00004
BN_DECAY = 0.9997
BN_EPSILON = 0.001
RESNET_VARIABLES = 'resnet_variables'
UPDATE_OPS_COLLECTION = 'resnet_update_ops' # must be grouped with training op
MEAN_BGR = [
    103.062623801, 
    115.902882574,
    123.151630838,
    ]

# If a model is trained with multiple GPU's prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'


def _activation_summary(x):
  """Helper to create summaries for activations.
  Creates a summary that provides a histogram of activations.
  Creates a summary that measure the sparsity of activations.
  Args:
    x: Tensor
  Returns:
    nothing
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.histogram_summary(tensor_name + '/activations', x)
  tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.
  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable
  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    var = tf.get_variable(name, shape, initializer=initializer)
  return var


def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.
  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.
  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.
  Returns:
    Variable Tensor
  """
  var = _variable_on_cpu(name, shape,
                         tf.truncated_normal_initializer(stddev=stddev))
  if wd:
    weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var


def distorted_inputs(inputfile):
  """Construct distorted input for Statefarm training using the Reader ops.
  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  Raises:
    ValueError: If no data_dir
  """
  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')
  data_dir = os.path.join(FLAGS.data_dir, 'statefarm-10-batches-bin')
  return statefarm_input.distorted_inputs(data_dir=data_dir,
                                        batch_size=FLAGS.batch_size,
                                        inputfile=inputfile)

def testing_inputs(validatefile):
  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')
  data_dir = os.path.join(FLAGS.data_dir, 'statefarm-batches-bin')
  return statefarm_input.testing_inputs(data_dir=data_dir,
                                        batch_size=FLAGS.batch_size,
                                        validatefile = validatefile)

def unlabeled_inputs(testfiles):
  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')
  data_dir = os.path.join(FLAGS.data_dir, 'statefarm-batches-bin')
  return statefarm_input.unlabeled_inputs(data_dir=data_dir,
                                        batch_size=FLAGS.batch_size,
                                        testfiles=testfiles)

def inputs(eval_data):
  """Construct input for Statefarm evaluation using the Reader ops.
  Args:
    eval_data: bool, indicating if one should use the train or eval data set.
  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  Raises:
    ValueError: If no data_dir
  """
  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')
  data_dir = os.path.join(FLAGS.data_dir, 'statefarm-batches-bin')
  if eval_data:
    return statefarm_input.testing_inputs(data_dir=data_dir, batch_size=FLAGS.batch_size)

def inference(images):
  """Build the Statefarm model.
  Args:
    images: Images returned from distorted_inputs() or inputs().
  Returns:
    Logits.
  """
  # We instantiate all variables using tf.get_variable() instead of
  # tf.Variable() in order to share variables across multiple GPU training runs.
  # If we only ran this model on a single GPU, we could simplify this function
  # by replacing all instances of tf.get_variable() with tf.Variable().

  with tf.variable_scope('conv1') as scope:
    kernel = _variable_with_weight_decay('weights', shape=[7, 7, 1, 64],
                                         stddev=1e-4, wd=0.0)
    conv = tf.nn.conv2d(images, kernel, [1, 2, 2, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
    bias = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv1)

  # pool1
  pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool1')
  # norm1
  norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm1')

  # conv2
  with tf.variable_scope('conv2') as scope:
    kernel = _variable_with_weight_decay('weights', shape=[5, 5, 64, 64],
                                         stddev=1e-4, wd=0.0)
    conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
    bias = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv2)

  # norm2
  norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm2')
  # pool2
  pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
                         strides=[1, 2, 2, 1], padding='SAME', name='pool2')

  # conv3
  with tf.variable_scope('conv3') as scope:
    kernel = _variable_with_weight_decay('weights', shape=[5, 5, 64, 64],
                                         stddev=1e-4, wd=0.0)
    conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
    bias = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv3)

  # norm2
  norm3 = tf.nn.lrn(conv3, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm3')
  # pool2
  pool3 = tf.nn.max_pool(norm3, ksize=[1, 3, 3, 1],
                         strides=[1, 2, 2, 1], padding='SAME', name='pool2')

  # local3
  with tf.variable_scope('local3') as scope:
    # Move everything into depth so we can perform a single matrix multiply.
    dim = 1
    for d in pool2.get_shape()[1:].as_list():
      dim *= d
    reshape = tf.reshape(pool2, [FLAGS.batch_size, dim])

    weights = _variable_with_weight_decay('weights', shape=[dim, 384],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
    local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
    _activation_summary(local3)

  # local4
  with tf.variable_scope('local4') as scope:
    weights = _variable_with_weight_decay('weights', shape=[384, 192],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
    local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
    _activation_summary(local4)

  # softmax, i.e. softmax(WX + b)
  with tf.variable_scope('softmax_linear') as scope:
    weights = _variable_with_weight_decay('weights', [192, NUM_CLASSES],
                                          stddev=1/192.0, wd=0.0)
    biases = _variable_on_cpu('biases', [NUM_CLASSES],
                              tf.constant_initializer(0.0))
    softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
    _activation_summary(softmax_linear)

  return softmax_linear


def inference_resnet(images, is_training,
              num_classes=10,
              num_blocks=[3, 4, 6, 3],  # defaults to 50-layer network
              preprocess=True,
              bottleneck=True):
    # if preprocess is True, input should be RGB [0,1], otherwise BGR with mean
    # subtracted

    x = images
    #if preprocess:
     #   x = _preprocess(x)

    is_training = tf.convert_to_tensor(is_training,
                                       dtype='bool',
                                       name='is_training')

    with tf.variable_scope('scale1'):
        x = _conv(x, 64, ksize=7, stride=2)
        x = _bn(x, is_training)
        x = _relu(x)

    with tf.variable_scope('scale2'):
        x = _max_pool(x, ksize=3, stride=2)
        x = stack(x, num_blocks[0], 64, bottleneck, is_training, stride=1)

    with tf.variable_scope('scale3'):
        x = stack(x, num_blocks[1], 128, bottleneck, is_training, stride=2)

    with tf.variable_scope('scale4'):
        x = stack(x, num_blocks[2], 256, bottleneck, is_training, stride=2)

    with tf.variable_scope('scale5'):
        x = stack(x, num_blocks[3], 512, bottleneck, is_training, stride=2)

    # post-net
    x = tf.reduce_mean(x, reduction_indices=[1, 2], name="avg_pool")
    with tf.variable_scope('fc'):
        logits = _fc(x, num_units_out=NUM_CLASSES)

    return logits

def _preprocess(rgb):
    """Changes RGB [0,1] valued image to BGR [0,255] with mean subtracted."""
    red, green, blue = tf.split(3, 3, rgb * 255.0) 
    bgr = tf.concat(3, [blue, green, red])
    bgr -= MEAN_BGR
    return bgr

def loss_res(logits, labels, batch_size=None, label_smoothing=0.1):
    if not batch_size:
        batch_size = FLAGS.batch_size

    num_classes = NUM_CLASSES

    # Reshape the labels into a dense Tensor of
    # shape [batch_size, num_classes].
    sparse_labels = tf.reshape(labels, [batch_size, 1])
    indices = tf.reshape(tf.range(batch_size), [batch_size, 1])
    concated = tf.concat(1, [indices, sparse_labels])
    one_hot_labels = tf.sparse_to_dense(concated,
                                        [batch_size, num_classes],
                                        1.0, 0.0)

    if label_smoothing > 0:
      smooth_positives = 1.0 - label_smoothing
      smooth_negatives = label_smoothing / num_classes
      one_hot_labels = one_hot_labels * smooth_positives + smooth_negatives

    loss =  tf.nn.softmax_cross_entropy_with_logits(logits, one_hot_labels)
    return loss


def stack(x, num_blocks, filters_internal, bottleneck, is_training, stride):
    for n in range(num_blocks):
         s = stride if n == 0 else 1
         with tf.variable_scope('block%d' % (n + 1)):
             x = block(x, filters_internal,
                       bottleneck=bottleneck,
                       is_training=is_training,
                       stride=s)
    return x

def block(x, filters_internal, is_training, stride, bottleneck):
    filters_in = x.get_shape()[-1]

    # Note: filters_out isn't how many filters are outputed. 
    # That is the case when bottleneck=False but when bottleneck is 
    # True, filters_internal*4 filters are outputted. filters_internal is how many filters
    # the 3x3 convs output internally.
    if bottleneck:
        filters_out = 4 * filters_internal
    else:
        filters_out = filters_internal

    shortcut = x # branch 1

    if bottleneck:
        with tf.variable_scope('a'):
            x = _conv(x, filters_internal, ksize=1, stride=stride)
            x = _bn(x, is_training)
            x = _relu(x)

        with tf.variable_scope('b'):
            x = _conv(x, filters_internal, ksize=3, stride=1)
            x = _bn(x, is_training)
            x = _relu(x)

        with tf.variable_scope('c'):
            x = _conv(x, filters_out, ksize=1, stride=1)
            x = _bn(x, is_training)
    else:
        with tf.variable_scope('A'):
            x = _conv(x, filters_internal, ksize=3, stride=stride)
            x = _bn(x, is_training)
            x = _relu(x)

        with tf.variable_scope('B'):
            x = _conv(x, filters_out, ksize=3, stride=1)
            x = _bn(x, is_training)

    with tf.variable_scope('shortcut'):
        if filters_out != filters_in or stride != 1:
            shortcut = _conv(shortcut, filters_out, ksize=1, stride=stride)
            shortcut = _bn(shortcut, is_training)

    return _relu(x + shortcut)

def _relu(x):
    return tf.nn.relu(x)
   
def _bn(x, is_training):
    x_shape = x.get_shape()
    params_shape = x_shape[-1:]
    axis = list(range(len(x_shape) - 1))

    beta = _get_variable('beta', params_shape, initializer=tf.zeros_initializer)
    gamma = _get_variable('gamma', params_shape, initializer=tf.ones_initializer)

    moving_mean = _get_variable('moving_mean',
                                params_shape,
                                initializer=tf.zeros_initializer,
                                trainable=False)
    moving_variance = _get_variable('moving_variance',
                                    params_shape,
                                    initializer=tf.ones_initializer,
                                    trainable=False)

    # These ops will only be preformed when training.
    mean, variance = tf.nn.moments(x, axis)
    update_moving_mean = moving_averages.assign_moving_average(
        moving_mean, mean, BN_DECAY)
    update_moving_variance = moving_averages.assign_moving_average(
        moving_variance, variance, BN_DECAY)
    tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_mean)
    tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_variance)

    mean, variance = control_flow_ops.cond(is_training,
            lambda: (mean, variance),
            lambda: (moving_mean, moving_variance))

    x = tf.nn.batch_normalization(
        x, mean, variance, beta, gamma, BN_EPSILON)
    #x.set_shape(inputs.get_shape()) ?? 

    return x

def _fc(x, num_units_out):
    num_units_in = x.get_shape()[1]
    weights_initializer = tf.truncated_normal_initializer(stddev=FC_WEIGHT_STDDEV)

    weights = _get_variable('weights',
                            shape=[num_units_in, num_units_out],
                            initializer=weights_initializer,
                            weight_decay=FC_WEIGHT_STDDEV)
    biases = _get_variable('biases', shape=[num_units_out],
                           initializer=tf.zeros_initializer)
    x = tf.nn.xw_plus_b(x, weights, biases)
    return x

def _get_variable(name, shape, initializer, weight_decay=0.0, dtype='float', trainable=True):
    "A little wrapper around tf.get_variable to do weight decay and add to"
    "resnet collection"
    if weight_decay > 0:
        regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    else:
        regularizer = None
    collections = [ tf.GraphKeys.VARIABLES, RESNET_VARIABLES ]
    return tf.get_variable(name, shape=shape, initializer=initializer, dtype=dtype,
                           regularizer=regularizer, collections=collections,
                           trainable=trainable)

def _conv(x, filters_out, ksize=3, stride=1):
    filters_in = x.get_shape()[-1]
    shape = [ksize, ksize, filters_in, filters_out]
    initializer = tf.truncated_normal_initializer(stddev=CONV_WEIGHT_STDDEV)
    weights = _get_variable('weights', shape=shape, dtype='float',
                            initializer=initializer,
                            weight_decay=CONV_WEIGHT_DECAY)
    return tf.nn.conv2d(x, weights, [1, stride, stride, 1], padding='SAME')
  
        
def _max_pool(x, ksize=3, stride=2):
    return tf.nn.max_pool(x, ksize=[1, ksize, ksize, 1],
         strides=[ 1, stride, stride, 1], padding='SAME')

def predict(logits):
  return tf.nn.softmax(logits)

def loss(logits, labels):
  """Add L2Loss to all the trainable variables.
  Add summary for for "Loss" and "Loss/avg".
  Args:
    logits: Logits from inference().
    labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]
  Returns:
    Loss tensor of type float.
  """
  # Calculate the average cross entropy loss across the batch.
  labels = tf.cast(labels, tf.int64)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits, labels, name='cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
  tf.add_to_collection('losses', cross_entropy_mean)

  # The total loss is defined as the cross entropy loss plus all of the weight
  # decay terms (L2 loss).
  return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _add_loss_summaries(total_loss):
  """Add summaries for losses in Statefarm model.
  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.
  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.scalar_summary(l.op.name +' (raw)', l)
    tf.scalar_summary(l.op.name, loss_averages.average(l))

  return loss_averages_op


def train(total_loss, global_step):
  """Train Statefarm model.
  Create an optimizer and apply to all trainable variables. Add moving
  average for all trainable variables.
  Args:
    total_loss: Total loss from loss().
    global_step: Integer Variable counting the number of training steps
      processed.
  Returns:
    train_op: op for training.
  """
  # Variables that affect learning rate.
  num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
  decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

  # Decay the learning rate exponentially based on the number of steps.
  lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                  global_step,
                                  decay_steps,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)
  tf.scalar_summary('learning_rate', lr)

  # Generate moving averages of all losses and associated summaries.
  loss_averages_op = _add_loss_summaries(total_loss)

  # Compute gradients.
  with tf.control_dependencies([loss_averages_op]):
    opt = tf.train.GradientDescentOptimizer(lr)
    grads = opt.compute_gradients(total_loss)

  # Apply gradients.
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  # Add histograms for trainable variables.
  for var in tf.trainable_variables():
    tf.histogram_summary(var.op.name, var)

  # Add histograms for gradients.
  for grad, var in grads:
    if grad is not None:
      tf.histogram_summary(var.op.name + '/gradients', grad)

  # Track the moving averages of all trainable variables.
  variable_averages = tf.train.ExponentialMovingAverage(
      MOVING_AVERAGE_DECAY, global_step)
  variables_averages_op = variable_averages.apply(tf.trainable_variables())

  with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
    train_op = tf.no_op(name='train')

  return train_op

def evaluation(logits, labels):
  correct_1 = tf.nn.in_top_k(logits, labels, 1)
  correct_2 = tf.nn.in_top_k(logits, labels, 2)
  return tf.reduce_sum(tf.cast(correct_1, tf.int32)), tf.reduce_sum(tf.cast(correct_2, tf.int32))
