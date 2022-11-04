"""
This is the module that given long text sequence generates short text sequence
"""
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.util import nest
from tensorflow.python.ops import init_ops
from tensorflow.python.framework import constant_op
from lib.decodertfv2 import attention_decoder
from lib.ops import *

def seq2seq(
            encoder_inputs,
            encoder_length,
            decoder_inputs,
            word_embedding_dim,
            mode,
            latent_dim=250):

    """
        encoder and decoder inputs should be
        word embedding and decoder outputs is
        word embedding
    """
 
    with tf.compat.v1.variable_scope("encoder") as scope:
        fw_cell = tf.compat.v1.nn.rnn_cell.LSTMCell(num_units=latent_dim, state_is_tuple=True)
        bw_cell = tf.compat.v1.nn.rnn_cell.LSTMCell(num_units=latent_dim, state_is_tuple=True)
        
        #bi-lstm encoder
        encoder_outputs,state = tf.compat.v1.nn.bidirectional_dynamic_rnn(
            cell_fw=fw_cell,
            cell_bw=bw_cell,
            dtype=tf.float32,
            sequence_length=encoder_length,
            inputs=encoder_inputs,
            time_major=False
        )

        output_fw, output_bw = encoder_outputs
        state_fw, state_bw = state
        encoder_outputs = tf.concat([output_fw,output_bw],2)
        encoder_state_c = tf.concat((state_fw.c, state_bw.c), 1)
        encoder_state_h = tf.concat((state_fw.h, state_bw.h), 1)
        encoder_state = tf.compat.v1.nn.rnn_cell.LSTMStateTuple(c=encoder_state_c, h=encoder_state_h)


    with tf.compat.v1.variable_scope("decoder") as scope:

        def feed_prev_loop(prev,i):
            return prev

        cell = tf.compat.v1.nn.rnn_cell.LSTMCell(num_units=latent_dim*2, state_is_tuple=True)
        decoder_inputs = batch_to_time_major(decoder_inputs)
        
        #the decoder
        decoder_outputs,decoder_state = attention_decoder(
            decoder_inputs = decoder_inputs,
            initial_state = encoder_state,
            attention_states = encoder_outputs,
            cell = cell,
            output_size = word_embedding_dim,
            loop_function = None if mode=='pretrain' else feed_prev_loop,
            scope = scope
        )
    decoder_batchmajor_outputs = tf.stack(decoder_outputs,axis=1)
    
    return decoder_batchmajor_outputs