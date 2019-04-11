from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time
import cv2
import csv
import re
from glob import glob as glb

import numpy as np
import tensorflow as tf

import sys

if __name__=="__main__" and __package__ is None:
  sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..'))
  __package__="test"

import model.flow_net as flow_net
from utils.flow_reader import load_flow, load_boundary

import matplotlib.pyplot as plt

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('base_dir', './checkpoints',
                           """dir to store trained net """)
tf.app.flags.DEFINE_bool('display_test', True,
                         """ display the test images """)
tf.app.flags.DEFINE_string('pic_dir', './data/newjzz.jpg',
                           """dir to store trained net """)

TEST_DIR = FLAGS.base_dir


# TEST_DIR = make_checkpoint_path(FLAGS.base_dir, FLAGS)

def tryint(s):
    try:
        return int(s)
    except:
        return s


def alphanum_key(s):
    return [tryint(c) for c in re.split('([0-9]+)', s)]


def evaluate():
    """Run Eval once.
    """
    # get a list of image filenames

    shape = [128, 256]

    with tf.Graph().as_default():
        # Make image placeholder
        boundary_op = tf.placeholder(tf.float32, [1, shape[0], shape[1], 1])

        # Build a Graph that computes the logits predictions from the
        # inference model.
        sflow_p = flow_net.generator(boundary_op, 1.0)

        # Restore the moving average version of the learned variables for eval.
        variables_to_restore = tf.all_variables()
        saver = tf.train.Saver(variables_to_restore)

        sess = tf.Session()

        ckpt = tf.train.get_checkpoint_state(TEST_DIR)

        saver.restore(sess, ckpt.model_checkpoint_path)
        global_step = 1

        graph_def = tf.get_default_graph().as_graph_def(add_shapes=True)


        boundary_np = cv2.imread(FLAGS.pic_dir,0).reshape([1, shape[0], shape[1], 1])
        boundary_np[:,0,:,:] = 255
        boundary_np[:, 127, :, :] = 255
        boundary_np = boundary_np/255.

        # calc logits
        sflow_generated = sess.run(sflow_p, feed_dict={boundary_op: boundary_np})[0]

        if FLAGS.display_test:
            # convert to display
            sflow_plot = sflow_generated
            boundary_concat = np.concatenate([boundary_np], axis=2)
            sflow_plot = np.sqrt(
                np.square(sflow_plot[:, :, 0]) + np.square(sflow_plot[:, :, 1])) - .05 * boundary_concat[0, :, :, 0]
            plt.figure()
            # display it
            plt.imshow(sflow_plot,interpolation='gaussian',cmap=plt.cm.jet)
            plt.colorbar()

            plt.show()


def main(argv=None):  # pylint: disable=unused-argument
    evaluate()


if __name__ == '__main__':
    tf.app.run()
