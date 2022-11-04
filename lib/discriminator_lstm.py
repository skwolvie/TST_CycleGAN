import tensorflow as tf

def discriminator_lstm(inputs,lstm_length,vocab_size):
    with tf.compat.v1.variable_scope("discriminator_word_embedding") as scope:
        init = tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform")
        discriminator_word_embedding_matrix = tf.compat.v1.get_variable(
            name="word_embedding_matrix",
            shape=[vocab_size,300],
            initializer=init,
            trainable = True
        )
        inputs = tf.nn.embedding_lookup(params=discriminator_word_embedding_matrix,ids=inputs)
    
    cell = tf.compat.v1.nn.rnn_cell.LSTMCell(num_units=500, state_is_tuple=True)
    lstm_outputs, last_states = tf.compat.v1.nn.dynamic_rnn(
        cell=cell,
        dtype=tf.float32,
        sequence_length=lstm_length,
        inputs=inputs)
    with tf.compat.v1.variable_scope("output_project") as scope:
        outputs = tf.contrib.layers.linear(lstm_outputs, 1, scope=scope)
    return tf.squeeze(outputs,axis=2)