from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import statefarm
import sys

FLAGS = tf.app.flags.FLAGS

import statefarm_input
IMAGE_SIZE = statefarm_input.IMAGE_SIZE

tf.app.flags.DEFINE_string('train_dir', './tmp/statefarm_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")


def placeholder_inputs(batch_size):
  images_placeholder = tf.placeholder(tf.float32, shape=(batch_size, IMAGE_SIZE[0], IMAGE_SIZE[1], 1))
  # images_placeholder = tf.placeholder(tf.float32, shape=(batch_size, 200, 200, 3))
  labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))
  return images_placeholder, labels_placeholder

def fill_feed_dict(images_pl, labels_pl, images, labels):
  feed_dict = {
      images_pl: images,
      labels_pl: labels,
  }
  return feed_dict

def do_eval(sess,
            eval_op,
            images_placeholder,
            labels_placeholder,
            images,
            labels
            ):

  # And run one epoch of eval.
  true_count = 0  # Counts the number of correct predictions.
  steps_per_epoch = 2560 // FLAGS.batch_size
  num_examples = 2560
  for step in xrange(steps_per_epoch):
    images_val, labels_val = sess.run([images, labels])
    feed_dict = fill_feed_dict(images_placeholder,
                               labels_placeholder,
                               images_val,
                               labels_val
                               )
    true_count += sess.run(eval_op, feed_dict=feed_dict)
  precision = true_count / num_examples
  print('Validation: Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
        (num_examples, true_count, precision))

def train():
  """Train Statefarm for a number of steps."""
  with tf.Graph().as_default():

    # Extract batches for training and testing
    images_train, labels_train = statefarm.distorted_inputs()
    #images_test, labels_test = statefarm.testing_inputs()

    global_step = tf.Variable(0, trainable=False)

    # Represent input and labels as placeholder to allow evaluation
    # and training runs using same architecture
    images_placeholder, labels_placeholder = placeholder_inputs(FLAGS.batch_size)

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = statefarm.inference(images_placeholder)

    # Calculate loss.
    loss = statefarm.loss(logits, labels_placeholder)

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op = statefarm.train(loss, global_step)

    # Create a saver.
    saver = tf.train.Saver(tf.all_variables())

    # Build the summary operation based on the TF collection of Summaries.
    # summary_op = tf.merge_all_summaries()

    # Build a Graph that evaluates the model on a set of testing/validation
    eval_op = statefarm.evaluation(logits, labels_placeholder)

    # Build an initialization operation to run below.
    init = tf.initialize_all_variables()

    # Start running operations on the Graph.
    sess = tf.Session(config=tf.ConfigProto(
        log_device_placement=FLAGS.log_device_placement))
    sess.run(init)

    # Start the queue runners.
    tf.train.start_queue_runners(sess=sess)

    # summary_writer = tf.train.SummaryWriter(FLAGS.train_dir,
    #                                        graph_def=sess.graph_def)
    print('Graph Initialized, Starting to train...')
    for step in xrange(FLAGS.max_steps):
      start_time = time.time()
      images_val, labels_val = sess.run([images_train, labels_train])
      print ('Loading images... Elapsed time: %.2f' % (time.time() - start_time))
      feed_dict = fill_feed_dict(images_placeholder, labels_placeholder,
                                 images_val, labels_val)
      _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
      print ('Training step... Elapsed time: %.2f' % (time.time() - start_time))
      duration = time.time() - start_time

      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

      if step % 10 == 0:
        num_examples_per_step = FLAGS.batch_size
        examples_per_sec = num_examples_per_step / duration
        sec_per_batch = float(duration)

        format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                      'sec/batch)')
        print (format_str % (datetime.now(), step, loss_value,
                             examples_per_sec, sec_per_batch))
        sys.stdout.flush()

      # Evaluate model on training set here
      # if step % 100 == 0:
      #   do_eval(sess, eval_op, images_placeholder, labels_placeholder,
      #     images_test, labels_test)
        #summary_str = sess.run(summary_op)
        # summary_writer.add_summary(summary_str, step)

      # Save the model checkpoint periodically.
      if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
       checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
       saver.save(sess, checkpoint_path, global_step=step)


def main(argv=None):  # pylint: disable=unused-argument
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)
  train()


if __name__ == '__main__':
  tf.app.run()
