# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from PIL import Image

import os
import time
from scipy import misc

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

# Process images of this size. 
IMAGE_SIZE = (480, 640)

# Global constants describing the Statefarm data set.
NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 10
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10


def read_statefarm(filename_queue):
  """Reads and parses examples from Statefarm data files.
  Recommendation: if you want N-way read parallelism, call this function
  N times.  This will give you N independent Readers reading different
  files & positions within those files, which will give better mixing of
  examples.
  Args:
    filename_queue: A queue of strings with the filenames to read from.
  Returns:
    An object representing a single example, with the following fields:
      height: number of rows in the result (32)
      width: number of columns in the result (32)
      depth: number of color channels in the result (3)
      key: a scalar string Tensor describing the filename & record number
        for this example.
      label: an int32 Tensor with the label in the range 0..9.
      uint8image: a [height, width, depth] uint8 Tensor with the image data
  """

  class StatefarmRecord(object):
    pass
  result = StatefarmRecord()

  # Read a record, getting filenames from the filename_queue.  
  result.key, label = tf.decode_csv(filename_queue.dequeue(), [[""], [""]], " ")

  # Extract raw JPG data as a string
  raw_contents = tf.read_file(result.key)

  # Decode raw data as a PNG. Defaults to uint8 encoding.
  result.uint8image = tf.image.decode_jpeg(raw_contents)

  # TENSORFLOW BUG: image shape not statically determined, so force
  # it to have correct Statefarm dimensions
  result.uint8image.set_shape((IMAGE_SIZE[0], IMAGE_SIZE[1], 3))

  # The first bytes represent the label, which we convert from uint8->int32.
  result.label = tf.cast(tf.string_to_number(label), tf.int32)

  return result


def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size):
  """Construct a queued batch of images and labels.
  Args:
    image: 3-D Tensor of [height, width, 3] of type.float32.
    label: 1-D Tensor of type.int32
    min_queue_examples: int32, minimum number of samples to retain
      in the queue that provides of batches of examples.
    batch_size: Number of images per batch.
  Returns:
    images: Images. 4D tensor of [batch_size, height, width, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  # Create a queue that shuffles the examples, and then
  # read 'batch_size' images + labels from the example queue.
  start_time = time.time()
  num_preprocess_threads = 4 # used 4 for the cluster
  images, label_batch = tf.train.shuffle_batch(
      [image, label],
      batch_size=batch_size,
      num_threads=num_preprocess_threads,
      capacity=min_queue_examples + 3 * batch_size,
      min_after_dequeue=min_queue_examples)

  # Display the training images in the visualizer.
  tf.image_summary('images', images)
  duration = time.time() - start_time

  print('Batch generated! Elapsed time: %.2f', duration)

  return images, tf.reshape(label_batch, [batch_size])


def distorted_inputs(data_dir, batch_size):
  """Construct distorted input for Statefarm training using the Reader ops.
  Args:
    data_dir: Path to the Statefarm data directory.
    batch_size: Number of images per batch.
  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  with open('inputs.txt') as f:
    delimmed = f.readlines()
  delimmed = [l.strip('\n') for l in delimmed]

  # Create a queue that produces the filename, label pairs to read.
  delimmed_queue = tf.train.string_input_producer(delimmed)

  # Read examples from files in the filename queue.
  read_input = read_statefarm(delimmed_queue)
  reshaped_image = tf.cast(read_input.uint8image, tf.float32)

  # Randomly crop a [height, width] section of the image.
  # height = 200
  # width = 200
  # distorted_image = tf.random_crop(reshaped_image, [height, width, 3])

  # Subtract off the mean and divide by the variance of the pixels.
  float_image = tf.image.per_image_whitening(reshaped_image)

  # Convert to grayscale
  bw_image = tf.image.rgb_to_grayscale(float_image)

  # Ensure that the random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                           min_fraction_of_examples_in_queue)
  print ('Filling queue with %d Statefarm images before starting to train. '
         'This will take a few minutes.' % min_queue_examples)

  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_and_label_batch(bw_image, read_input.label,
                                         min_queue_examples, batch_size)


def testing_inputs(data_dir, batch_size):
  """Construct input for Statefarm evaluation using the Reader ops.
  Args:
    eval_data: bool, indicating if one should use the train or eval data set.
    data_dir: Path to the Statefarm data directory.
    batch_size: Number of images per batch.
  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  with open('testing.txt') as f:
    delimmed = f.readlines()
  delimmed = [l.strip('\n') for l in delimmed]

  # Create a queue that produces the filename, label pairs to read.
  delimmed_queue = tf.train.string_input_producer(delimmed)

  # Read examples from files in the filename queue.
  read_input = read_statefarm(delimmed_queue)
  reshaped_image = tf.cast(read_input.uint8image, tf.float32)

  # Image processing for evaluation.
  # Crop the central [height, width] of the image.
  height = 24
  width = 24
  resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,
                                                         width, height)

  # Subtract off the mean and divide by the variance of the pixels.
  float_image = tf.image.per_image_whitening(resized_image)

  # Ensure that the random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_EVAL *
                           min_fraction_of_examples_in_queue)

  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_and_label_batch(float_image, read_input.label,
                                         min_queue_examples, batch_size)