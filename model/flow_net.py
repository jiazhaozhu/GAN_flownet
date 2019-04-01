
"""Builds the ring network.

Summary of available functions:

  # Compute pics of the simulation runnig.
  
  # Create a graph to train on.
"""


import tensorflow as tf
import numpy as np
import model.genera as genera
import model.discrimina as discrimina
import input.flow_input as flow_input


FLAGS = tf.app.flags.FLAGS

# Constants describing the training process.
tf.app.flags.DEFINE_string('model', 'res',
                           """ model name to train """)
tf.app.flags.DEFINE_integer('nr_res_blocks', 1,
                           """ nr res blocks """)
tf.app.flags.DEFINE_bool('gated_res', True,
                           """ gated resnet or not """)
tf.app.flags.DEFINE_string('nonlinearity', 'concat_elu',
                           """ nonlinearity used such as concat_elu, elu, concat_relu, relu """)

# """generator"""
def generator(boundary, keep_prob):
  if FLAGS.model == "res": 
    sflow_p = genera.gen(boundary, nr_res_blocks=2, keep_prob=keep_prob, nonlinearity_name=FLAGS.nonlinearity, gated=FLAGS.gated_res)

  return sflow_p


# """discriminator"""
def discriminator(dis_input,target,keep_prob):
    if FLAGS.model == "res":
        dis_output = discrimina.dis(dis_input,target,nr_res_blocks=FLAGS.nr_res_blocks, keep_prob=keep_prob, nonlinearity_name=FLAGS.nonlinearity, gated=FLAGS.gated_res)

    return dis_output

def loss_image(sflow_p, sflow):
  loss = tf.losses.mean_squared_error(sflow, sflow_p)

  return loss


