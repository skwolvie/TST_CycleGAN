# https://github.com/tensorflow/tensorflow/blob/v1.15.5/tensorflow/contrib/legacy_seq2seq/python/ops/seq2seq.py#L537
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.util import nest
from tensorflow.python.ops import init_ops
from tensorflow.python.framework import constant_op

import tensorflow as tf


class Linear:
    """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.
  Args:
    args: a 2D Tensor or a list of 2D, batch, n, Tensors.
    output_size: int, second dimension of weight variable.
    dtype: data type for variables.
    build_bias: boolean, whether to build a bias variable.
    bias_initializer: starting value to initialize the bias
      (default is all zeros).
    kernel_initializer: starting value to initialize the weight.
  Raises:
    ValueError: if inputs_shape is wrong.
  """

    def __init__(self,
                 args,
                 output_size,
                 build_bias,
                 bias_initializer=None,
                 kernel_initializer=None):
        self._build_bias = build_bias

        if args is None or (nest.is_sequence(args) and not args):
            raise ValueError("`args` must be specified")
        if not nest.is_sequence(args):
            args = [args]
            self._is_sequence = False
        else:
            self._is_sequence = True

        # Calculate the total size of arguments on dimension 1.
        total_arg_size = 0
        shapes = [a.get_shape() for a in args]
        for shape in shapes:
            if shape.ndims != 2:
                raise ValueError("linear is expecting 2D arguments: %s" % shapes)
            if shape.dims[1].value is None:
                raise ValueError("linear expects shape[1] to be provided for shape %s, "
                                 "but saw %s" % (shape, shape[1]))
            else:
                total_arg_size += shape.dims[1].value

        dtype = [a.dtype for a in args][0]

        scope = variable_scope.get_variable_scope()
        with variable_scope.variable_scope(scope) as outer_scope:
            self._weights = variable_scope.get_variable(
                'weights', [total_arg_size, output_size],
                dtype=dtype,
                initializer=kernel_initializer)
            if build_bias:
                with variable_scope.variable_scope(outer_scope) as inner_scope:
                    inner_scope.set_partitioner(None)
                    if bias_initializer is None:
                        bias_initializer = init_ops.constant_initializer(0.0, dtype=dtype)
                    self._biases = variable_scope.get_variable(
                        'bias', [output_size],
                        dtype=dtype,
                        initializer=bias_initializer)

    def __call__(self, args):
        if not self._is_sequence:
            args = [args]

        if len(args) == 1:
            res = math_ops.matmul(args[0], self._weights)
        else:
            # Explicitly creating a one for a minor performance improvement.
            one = constant_op.constant(1, dtype=dtypes.int32)
            res = math_ops.matmul(array_ops.concat(args, one), self._weights)
        if self._build_bias:
            res = nn_ops.bias_add(res, self._biases)
        return res


def attention_decoder(decoder_inputs,
                      initial_state,
                      attention_states,
                      cell,
                      output_size=None,
                      num_heads=1,
                      loop_function=None,
                      dtype=None,
                      scope=None,
                      initial_state_attention=False):
    """RNN decoder with attention for the sequence-to-sequence model.
  In this context "attention" means that, during decoding, the RNN can look up
  information in the additional tensor attention_states, and it does this by
  focusing on a few entries from the tensor. This model has proven to yield
  especially good results in a number of sequence-to-sequence tasks. This
  implementation is based on http://arxiv.org/abs/1412.7449 (see below for
  details). It is recommended for complex sequence-to-sequence tasks.
  Args:
    decoder_inputs: A list of 2D Tensors [batch_size x input_size].
    initial_state: 2D Tensor [batch_size x cell.state_size].
    attention_states: 3D Tensor [batch_size x attn_length x attn_size].
    cell: tf.compat.v1.nn.rnn_cell.RNNCell defining the cell function and size.
    output_size: Size of the output vectors; if None, we use cell.output_size.
    num_heads: Number of attention heads that read from attention_states.
    loop_function: If not None, this function will be applied to i-th output in
      order to generate i+1-th input, and decoder_inputs will be ignored, except
      for the first element ("GO" symbol). This can be used for decoding,
      but also for training to emulate http://arxiv.org/abs/1506.03099.
        Signature -- loop_function(prev, i) = next * prev is a 2D Tensor of
        shape [batch_size x output_size], * i is an integer, the step number
        (when advanced control is needed), * next is a 2D Tensor of shape
        [batch_size x input_size].
    dtype: The dtype to use for the RNN initial state (default: tf.float32).
    scope: VariableScope for the created subgraph; default: "attention_decoder".
    initial_state_attention: If False (default), initial attentions are zero. If
      True, initialize the attentions from the initial state and attention
      states -- useful when we wish to resume decoding from a previously stored
      decoder state and attention states.
  Returns:
    A tuple of the form (outputs, state), where:
      outputs: A list of the same length as decoder_inputs of 2D Tensors of
        shape [batch_size x output_size]. These represent the generated outputs.
        Output i is computed from input i (which is either the i-th element
        of decoder_inputs or loop_function(output {i-1}, i)) as follows.
        First, we run the cell on a combination of the input and previous
        attention masks:
          cell_output, new_state = cell(linear(input, prev_attn), prev_state).
        Then, we calculate new attention masks:
          new_attn = softmax(V^T * tanh(W * attention_states + U * new_state))
        and then we calculate the output:
          output = linear(cell_output, new_attn).
      state: The state of each decoder cell the final time-step.
        It is a 2D Tensor of shape [batch_size x cell.state_size].
  Raises:
    ValueError: when num_heads is not positive, there are no inputs, shapes
      of attention_states are not set, or input size cannot be inferred
      from the input.
  """
    if not decoder_inputs:
        raise ValueError("Must provide at least 1 input to attention decoder.")
    if num_heads < 1:
        raise ValueError("With less than 1 heads, use a non-attention decoder.")
    if attention_states.get_shape()[2] is None:
        raise ValueError("Shape[2] of attention_states must be known: %s" %
                         attention_states.get_shape())
    if output_size is None:
        output_size = cell.output_size

    with variable_scope.variable_scope(
            scope or "attention_decoder", dtype=dtype) as scope:
        dtype = scope.dtype

        batch_size = array_ops.shape(decoder_inputs[0])[0]  # Needed for reshaping.
        attn_length = attention_states.get_shape()[1]
        if attn_length is None:
            attn_length = array_ops.shape(attention_states)[1]
        attn_size = attention_states.get_shape()[2]

        # To calculate W1 * h_t we use a 1-by-1 convolution, need to reshape before.
        hidden = array_ops.reshape(attention_states,
                                   [-1, attn_length, 1, attn_size])
        hidden_features = []
        v = []
        attention_vec_size = attn_size  # Size of query vectors for attention.
        for a in range(num_heads):
            k = variable_scope.get_variable(
                "AttnW_%d" % a, [1, 1, attn_size, attention_vec_size], dtype=dtype)
            hidden_features.append(nn_ops.conv2d(hidden, k, [1, 1, 1, 1], "SAME"))
            v.append(
                variable_scope.get_variable(
                    "AttnV_%d" % a, [attention_vec_size], dtype=dtype))

        state = initial_state

        def attention(query):
            """Put attention masks on hidden using hidden_features and query."""
            ds = []  # Results of attention reads will be stored here.
            if nest.is_sequence(query):  # If the query is a tuple, flatten it.
                query_list = nest.flatten(query)
                for q in query_list:  # Check that ndims == 2 if specified.
                    ndims = q.get_shape().ndims
                    if ndims:
                        assert ndims == 2
                query = array_ops.concat(query_list, 1)
            for a in range(num_heads):
                with variable_scope.variable_scope("Attention_%d" % a):
                    y = Linear(query, attention_vec_size, True)(query)
                    y = array_ops.reshape(y, [-1, 1, 1, attention_vec_size])
                    y = math_ops.cast(y, dtype)
                    # Attention mask is a softmax of v^T * tanh(...).
                    s = math_ops.reduce_sum(v[a] * math_ops.tanh(hidden_features[a] + y),
                                            [2, 3])
                    a = nn_ops.softmax(math_ops.cast(s, dtype=dtypes.float32))
                    # Now calculate the attention-weighted vector d.
                    a = math_ops.cast(a, dtype)
                    d = math_ops.reduce_sum(
                        array_ops.reshape(a, [-1, attn_length, 1, 1]) * hidden, [1, 2])
                    ds.append(array_ops.reshape(d, [-1, attn_size]))
            return ds

        outputs = []
        prev = None
        batch_attn_size = array_ops.stack([batch_size, attn_size])
        attns = [
            array_ops.zeros(batch_attn_size, dtype=dtype) for _ in range(num_heads)
        ]
        for a in attns:  # Ensure the second shape of attention vectors is set.
            a.set_shape([None, attn_size])
        if initial_state_attention:
            attns = attention(initial_state)
        for i, inp in enumerate(decoder_inputs):
            if i > 0:
                variable_scope.get_variable_scope().reuse_variables()
            # If loop_function is set, we use it instead of decoder_inputs.
            if loop_function is not None and prev is not None:
                with variable_scope.variable_scope("loop_function", reuse=True):
                    inp = loop_function(prev, i)
            # Merge input and previous attentions into one vector of the right size.
            input_size = inp.get_shape().with_rank(2)[1]
            if input_size is None:
                raise ValueError("Could not infer input size from input: %s" % inp.name)

            inputs = [inp] + attns
            inputs = [math_ops.cast(e, dtype) for e in inputs]
            x = Linear(inputs, input_size, True)(inputs)
            # Run the RNN.
            cell_output, state = cell(x, state)
            # Run the attention mechanism.
            if i == 0 and initial_state_attention:
                with variable_scope.variable_scope(
                        variable_scope.get_variable_scope(), reuse=True):
                    attns = attention(state)
            else:
                attns = attention(state)

            with variable_scope.variable_scope("AttnOutputProjection"):
                cell_output = math_ops.cast(cell_output, dtype)
                inputs = [cell_output] + attns
                output = Linear(inputs, output_size, True)(inputs)
            if loop_function is not None:
                prev = output
            outputs.append(output)

    return outputs, state


if __name__ == "__main__":
    _outputs, _state = attention_decoder([tf.ones((1, 1))],
                                         tf.ones((1, 1)),
                                         tf.ones((1, 1, 1)),
                                         tf.compat.v1.nn.rnn_cell.BasicRNNCell(1))
    print(_outputs, _state)