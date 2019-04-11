""" use the result of discriminator to evaluate the performance of image generation networks(use gan or FCN)
"""
import os.path
import time

import numpy as np
import tensorflow as tf

import sys

if __name__=="__main__" and __package__ is None:
  sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..'))
  __package__="test"

import model.flow_net as flow_net
from utils.experiment_manager import make_checkpoint_path

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('base_dir', '../checkpoints/10-45104',
                           """dir to store GAN net""")
tf.app.flags.DEFINE_string('base_dir1', '/home/jzz/cnn/code/Steady-ori/checkpoints',
                           """dir to store trained net """)
tf.app.flags.DEFINE_integer('batch_size', 8,
                            """ training batch size """)
tf.app.flags.DEFINE_integer('max_steps', 300000,
                            """ max number of steps to train """)
tf.app.flags.DEFINE_float('keep_prob', 0.7,
                          """ keep probability for dropout """)


EPS = 1e-12
# TRAIN_DIR = make_checkpoint_path(FLAGS.base_dir, FLAGS)
TRAIN_DIR = FLAGS.base_dir
TRAIN_DIR1 = FLAGS.base_dir1
print(TRAIN_DIR)


def train():
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    """Train ring_net for a number of steps."""
    with tf.Graph().as_default():

        ema = tf.train.ExponentialMovingAverage(decay=0.9)
        # global_steps = tf.Variable(0, trainable=False)
        # make inputs
        global_step = tf.train.get_or_create_global_step()

        # 学习率每隔1000步降到原有的95%
        lr_ge = tf.train.exponential_decay(FLAGS.learning_rate_ge, global_step=global_step, decay_steps=100000,
                                           decay_rate=0.1, staircase=True,
                                           name='lr_ge')

        lr_di = tf.train.exponential_decay(FLAGS.learning_rate_di, global_step=global_step, decay_steps=50000,
                                           decay_rate=0.1,
                                           staircase=True,
                                           name='lr_ge')
        global_step_op = tf.assign(global_step, global_step + 1)

        boundary, sflow = flow_net.inputs(FLAGS.batch_size)
        # create and unrap network
        sflow_p = flow_net.inference(boundary, FLAGS.keep_prob)
        dis_fake = flow_net.discriminator(boundary, sflow_p, FLAGS.keep_prob)
        dis_true = flow_net.discriminator(boundary, sflow, FLAGS.keep_prob)

        tf.summary.image("predict_fake", tf.image.convert_image_dtype(dis_fake, dtype=tf.uint8))
        tf.summary.image("predict_true", tf.image.convert_image_dtype(dis_true, dtype=tf.uint8))
        # calc ge error
        loss_squ_ge = flow_net.loss_image(sflow_p, sflow)
        loss_cro_ge = tf.reduce_mean(-tf.log(dis_fake + EPS))
        loss_ge = 10000000 * loss_squ_ge + 1 * loss_cro_ge

        tf.summary.scalar("loss_squ_ge", loss_squ_ge)

        tf.summary.scalar("loss_ge", loss_ge)

        # calc dis error
        loss_dis = tf.reduce_mean(-(tf.log(dis_true + EPS) + tf.log(1 - dis_fake + EPS)))

        em_loss_op = ema.apply([loss_cro_ge, loss_dis])
     
        loss_dis_ema = ema.average(loss_dis)
        loss_cro_ge_ema = ema.average(loss_cro_ge)
        tf.summary.scalar("loss_cro_ge", loss_cro_ge_ema)
        # loss_dis = tf.reduce_mean(-(tf.square(dis_true -1 + EPS) + tf.log(1 - dis_fake + EPS)))
        tf.summary.scalar("loss_dis", loss_dis_ema)


        fetches = {"em_loss_op": em_loss_op,
                   "loss_ge": loss_ge,
                   "loss_dis": loss_dis_ema,
                   "loss_cro_ge": loss_cro_ge_ema,
                   "loss_squ_ge": loss_squ_ge,
                   "global_step_op":global_step_op,
                   "global_step": global_step}
        # List of all Variables
        variables = tf.global_variables()

        # Build a saver
        saver = tf.train.Saver(tf.global_variables())
   
        # Summary op
        summary_op = tf.summary.merge_all()

        # Build an initialization operation to run below.
        init = tf.global_variables_initializer()

        # Start running operations on the Graph.
        sess = tf.Session()

        # init if this is the very time training
        sess.run(init)

        # init from GAN checkpoint to get the Discriminator and use Discriminator to evaluate the result of GAN and noGAN 
        saver_restore = tf.train.Saver(variables)
        ckpt = tf.train.get_checkpoint_state(TRAIN_DIR)
        if ckpt is not None:
            print("init from " + TRAIN_DIR)
            try:
                saver_restore.restore(sess, ckpt.model_checkpoint_path)
            except:
                tf.gfile.DeleteRecursively(TRAIN_DIR)
                tf.gfile.MakeDirs(TRAIN_DIR)
                print("there was a problem using variables in checkpoint, random init will be used instead")

        # init from GAN checkpoint or noGAN to evaluate
        variables1 = [var for var in tf.trainable_variables() if var.name.startswith("Ge_")]
        saver_restore1 = tf.train.Saver(variables1)
        ckpt1 = tf.train.get_checkpoint_state(TRAIN_DIR1)
        print("init from " + TRAIN_DIR1)
        saver_restore1.restore(sess, ckpt1.model_checkpoint_path)

        # Start que runner
        tf.train.start_queue_runners(sess=sess)

        # Summary op
        graph_def = sess.graph.as_graph_def(add_shapes=True)
        summary_writer = tf.summary.FileWriter(TRAIN_DIR, graph_def=graph_def)

        for step in range(FLAGS.max_steps):
            t = time.time()
            results = sess.run(fetches, feed_dict={})
            elapsed = time.time() - t

            assert not np.isnan(results["loss_ge"]), 'Model diverged with loss = NaN'

            if step % 100 == 0:
                summary_str = sess.run(summary_op, feed_dict={})
                summary_writer.add_summary(summary_str, results["global_step"])
                print("Step: " + str(results["global_step"]) + "%%loss value at " + "GAN_loss: " + str(
                    results["loss_ge"]) + "   Dis_loss: " + str(results["loss_dis"]))
                print("**loss_squ_ge: " + str(results["loss_squ_ge"]),
                      "  loss_cro_ge_va: " + str(results["loss_cro_ge"]))
                print("time per batch is " + str(elapsed))


def main(argv=None):  # pylint: disable=unused-argument
    train()


if __name__ == '__main__':
    tf.app.run()
