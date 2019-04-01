
import os.path
import time

import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tfdbg
import sys

if __name__=="__main__" and __package__ is None:
  sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..'))
  __package__="train"

import model.flow_net as flow_net
from utils.experiment_manager import make_checkpoint_path

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('base_dir', './checkpoints',
                            """dir to store trained net """)
tf.app.flags.DEFINE_string('dataset_dir', './data/train.tfrecords',
                            """dir of dataset """)
tf.app.flags.DEFINE_integer('batch_size', 8,
                            """ training batch size """)
tf.app.flags.DEFINE_integer('epochs', 800,
                            """ training epochs """)
tf.app.flags.DEFINE_float('keep_prob', 0.7,
                            """ keep probability for dropout """)
tf.app.flags.DEFINE_float('learning_rate_ge', 1e-4,
                            """ keep probability for dropout """)
tf.app.flags.DEFINE_float('learning_rate_di', 1e-4,
                            """ keep probability for dropout """)
tf.app.flags.DEFINE_bool('debug', False,
                           """debug or not """)
tf.app.flags.DEFINE_bool('tensorboard_debug', False,
                           """tensorboard_debug or not """)
tf.app.flags.DEFINE_string('tensorboard_debug_address', None,
                            """tensorboard debug address """)


EPS = 1e-12
# TRAIN_DIR = make_checkpoint_path(FLAGS.base_dir, FLAGS)
TRAIN_DIR = FLAGS.base_dir
print(TRAIN_DIR)

# read dataset
def parser(record):
    keys_to_features = {
        'boundary':tf.FixedLenFeature([],tf.string),
        'sflow':tf.FixedLenFeature([],tf.string)
    }
    parsed = tf.parse_single_example(record, keys_to_features)

    boundary = tf.decode_raw(parsed['boundary'], tf.uint8)
    sflow = tf.decode_raw(parsed['sflow'], tf.float32)
    boundary = tf.reshape(boundary, [128, 256, 1])
    sflow = tf.reshape(sflow, [128, 256, 2])
    boundary = tf.to_float(boundary)
    sflow = tf.to_float(sflow)

    return {"image_raw": boundary}, sflow

# use the .tfrecord file and tfdata to make dataset and return iterator
def flow_inputs(filenames,batch_size,num_epochs):
  dataset = tf.data.TFRecordDataset(filenames)
  dataset = dataset.map(parser,num_parallel_calls=20)
  # dataset = dataset.apply(tf.contrib.data.map_and_batch(parser,batch_size))
  dataset = dataset.batch(batch_size)
  dataset = dataset.shuffle(buffer_size=10000)
  dataset = dataset.repeat(num_epochs)
  dataset = dataset.prefetch(4)

  iterator = dataset.make_initializable_iterator()

  return iterator



def train():

  os.environ['CUDA_VISIBLE_DEVICES'] = '2'
  """Train ring_net for a number of steps."""
  with tf.Graph().as_default():

    # inout and outout placeholder
    boundary = tf.placeholder(tf.float32,shape=[FLAGS.batch_size,128,256,1])
    sflow = tf.placeholder(tf.float32,shape=[FLAGS.batch_size,128,256,2])
    # Forward propagation and utilization of generators and discriminators to get corresponding results
    sflow_p = flow_net.generator(boundary, FLAGS.keep_prob)
    dis_fake = flow_net.discriminator(boundary,sflow_p,FLAGS.keep_prob)
    dis_true = flow_net.discriminator(boundary, sflow, FLAGS.keep_prob)

    tf.summary.image("predict_fake", tf.image.convert_image_dtype(dis_fake, dtype=tf.uint8))
    tf.summary.image("predict_true", tf.image.convert_image_dtype(dis_true, dtype=tf.uint8))
    # Calculate the error of the generator
    loss_squ_ge = flow_net.loss_image(sflow_p, sflow)
    loss_cro_ge = tf.reduce_mean(-tf.log(dis_fake + EPS))
    loss_ge = 10000000 * loss_squ_ge + 1 * loss_cro_ge

    tf.summary.scalar("loss_squ_ge",loss_squ_ge)
    tf.summary.scalar("loss_ge", loss_ge)
    # Calculate the error of the discriminator
    loss_dis = tf.reduce_mean(-(tf.log(dis_true + EPS) + tf.log(1 - dis_fake + EPS)))

    # Calculate the moving average
    ema = tf.train.ExponentialMovingAverage(decay=0.9)
    em_loss_op = ema.apply([loss_cro_ge,loss_dis])
    loss_dis_ema = ema.average(loss_dis)
    loss_cro_ge_ema = ema.average(loss_cro_ge)
    tf.summary.scalar("loss_cro_ge", loss_cro_ge_ema)
    # loss_dis = tf.reduce_mean(-(tf.square(dis_true -1 + EPS) + tf.log(1 - dis_fake + EPS)))
    tf.summary.scalar("loss_dis", loss_dis_ema)

    # define global_step
    global_step = tf.train.get_or_create_global_step()

    # The generator learning rate is reduced to the original 0.1 every 100000 steps.
    lr_ge = tf.train.exponential_decay(FLAGS.learning_rate_ge, global_step=global_step, decay_steps=100000,
                                       decay_rate=0.1, staircase=True,
                                       name='lr_ge')
    # The discriminator learning rate is reduced to the original 0.1 every 100000 steps.
    lr_di = tf.train.exponential_decay(FLAGS.learning_rate_di, global_step=global_step, decay_steps=50000,
                                       decay_rate=0.1,
                                       staircase=True,
                                       name='lr_ge')
    tf.summary.scalar('lr_ge', lr_ge)
    tf.summary.scalar('lr_di', lr_di)
    # train generator
    with tf.name_scope("Ge_train"):
        ge_var = [var for var in tf.trainable_variables() if var.name.startswith("Ge_")]
        ge_train_op = tf.train.AdamOptimizer(lr_ge, 0.5)
        ge_gd = ge_train_op.compute_gradients(loss_ge, var_list=ge_var)
        ge_train = ge_train_op.apply_gradients(ge_gd)

    # train discriminator
    with tf.name_scope("Dis_train"):
        with tf.control_dependencies([ge_train]):
            dis_var = [var for var in tf.trainable_variables() if var.name.startswith("Dis_")]
            dis_train_op = tf.train.AdamOptimizer(lr_di, 0.5)
            dis_gd = dis_train_op.compute_gradients(loss_dis, var_list=dis_var)
            dis_train = dis_train_op.apply_gradients(dis_gd,global_step=global_step)

    fetches = {"dis_train":  dis_train,
               "em_loss_op": em_loss_op,
               "loss_ge":    loss_ge,
               "loss_dis":   loss_dis_ema,
               "loss_cro_ge":loss_cro_ge_ema,
               "loss_squ_ge":loss_squ_ge,
               "global_step":global_step}

    # build a dataset and get the input and labels
    filenames = tf.placeholder(tf.string, shape=[None])
    iterator0 = flow_inputs(filenames, FLAGS.batch_size, FLAGS.epochs)
    features, labels = iterator0.get_next()

    # list of all Variables
    variables = tf.global_variables()

    # build a saver
    saver = tf.train.Saver(tf.global_variables())
    # Summary op
    summary_op = tf.summary.merge_all()

    # build an initialization operation to run below.
    init = tf.global_variables_initializer()

    # open a session
    with tf.Session() as sess:
        # init if this is the very time training
        sess.run(init)
        # init a iterator
        sess.run(iterator0.initializer, feed_dict={filenames: [FLAGS.dataset_dir]})
        # if debug or not
        if FLAGS.debug:
            sess = tfdbg.LocalCLIDebugWrapperSession(sess)
        elif FLAGS.tensorboard_debug:
            sess = tfdbg.TensorBoardDebugWrapperSession(
                sess, FLAGS.tensorboard_debug_address)

        # init from checkpoint
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

        # Summary op
        graph_def = sess.graph.as_graph_def(add_shapes=True)
        summary_writer = tf.summary.FileWriter(TRAIN_DIR, graph_def=graph_def)

        while True:
          # get a batch of input and label
          boundary_t, sflow_t = sess.run([features, labels])
          if boundary_t["image_raw"].shape[0] < FLAGS.batch_size:
              print(boundary_t["image_raw"].shape[0])
              continue

          # Start running main operations on the Graph.
          t = time.time()
          results = sess.run(fetches,feed_dict={boundary:boundary_t["image_raw"],sflow:sflow_t})
          elapsed = time.time() - t

          assert not np.isnan(results["loss_ge"]), 'Model diverged with loss = NaN'
          # write a summary files
          if results["global_step"]%100 == 0:
            summary_str = sess.run(summary_op, feed_dict={boundary:boundary_t["image_raw"],sflow:sflow_t})
            summary_writer.add_summary(summary_str, results["global_step"])
            print("Step: " +str(results["global_step"]) + "loss value at " + "GAN_loss: " + str(results["loss_ge"]) + "  Dis_loss: " + str(results["loss_dis"]))
            print("loss_squ_ge: " + str(results["loss_squ_ge"]),"  loss_cro_ge_va: " + str(results["loss_cro_ge"]))
            print("time per batch is " + str(elapsed))

          # save model
          if results["global_step"]%1000 == 0:
            checkpoint_path = os.path.join(TRAIN_DIR, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=results["global_step"])
            print("saved to " + TRAIN_DIR)

def main(argv=None):
  train()

if __name__ == '__main__':
  tf.app.run()
