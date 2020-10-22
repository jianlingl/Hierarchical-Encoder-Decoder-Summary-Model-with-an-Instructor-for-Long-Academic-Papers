# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# Modifications Copyright 2017 Abigail See
# Modifications Copyright 2018 Arman Cohan
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
import sys
import heapq

"""This file defines the decoder"""

import tensorflow as tf
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import math_ops
from Connect_Layer import Connect_Layer
FLAGS = tf.app.flags.FLAGS

# Note: this function is based on tf.contrib.legacy_seq2seq_attention_decoder, which is now outdated.
# In the future, it would make more sense to write variants on the attention mechanism using the new seq2seq
# library for tensorflow 1.0: https://www.tensorflow.org/api_guides/python/contrib.seq2seq#Attention
def attention_decoder(decoder_inputs,
                      initial_state,
                      cell,

                      encoder_states,
                      extract_state=None,

                      encoder_section_states=None,

                      enc_padding_mask=None,
                      extract_padding_mask=None,
                      enc_section_padding_mask = None,

                      num_words_section=None,

                      initial_state_attention=False,
                      pointer_gen=True,
                      use_coverage=False,
                      prev_coverage=None,
                      temperature=None):
  """
            self.attn_decoder(emb_dec_inputs,  #decode的输入
                              self._dec_in_state,  #上下文向量
                              cell,

                              self._enc_states,  #word级别的state
                              extract_state=self._extract_state,  # extract级别的state

                              encoder_section_states=self._enc_sec_states,  #sec级别的state


                              enc_padding_mask=self._enc_padding_mask,
                              extract_padding_mask=self._extract_padding_mask,
                              enc_section_padding_mask=self._enc_section_padding_mask,


                              num_words_section=self._batch_sections_len,  # section 单词的长度

                              initial_state_attention=(self._hps.mode=="decode"),
                              pointer_gen=self._hps.pointer_gen,
                              use_coverage=self._hps.coverage,
                              prev_coverage=prev_coverage,
                              temperature=self._hps.temperature
                              )

  Args:
    decoder_inputs: A list of 2D Tensors [batch_size x input_size].
    initial_state: 2D Tensor [batch_size x cell.state_size].
    encoder_states: 3D Tensor [batch_size x     seq_len x     encoder_output_size].   !!!!!!!!!!
    cell: rnn_cell.RNNCell defining the cell function and size.
    encoder_section_states: 3D Tensor [batch_size x section_seq_len x encoder_output_size]. Pass None if you don't want hierarchical attentive decoding
    num_words_section: number of words per section [batch_size x section_seq_len]
    enc_padding_mask: 2D Tensor [batch_size x attn_length] containing 1s and 0s; indicates which of the encoder locations are padding (0) or a real token (1).
    enc_section_padding_mask: 3D Tensor [batch_size x num_sections x section_len]
    initial_state_attention:
      Note that this attention decoder passes each decoder input through a linear layer with the previous step's context vector to get a modified version of the input. If initial_state_attention is False, on the first decoder step the "previous context vector" is just a zero vector. If initial_state_attention is True, we use initial_state to (re)calculate the previous step's context vector. We set this to False for train/eval mode (because we call attention_decoder once for all decoder steps) and True for decode mode (because we call attention_decoder once for each decoder step).
    pointer_gen: boolean. If True, calculate the generation probability p_gen for each decoder step.
    use_coverage: boolean. If True, use coverage mechanism.
    prev_coverage:
      If not None, a tensor with shape (batch_size, seq_len). The previous step's coverage vector. This is only not None in decode mode when using coverage.
    simulating the temperature hyperparam for softmax: set to 1.0 for starters

  Returns:
    outputs: A list of the same length as decoder_inputs of 2D Tensors of
      shape [batch_size x cell.output_size]. The output vectors.
    state: The final state of the decoder. A tensor shape [batch_size x cell.state_size].
    attn_dists: A list containing tensors of shape (batch_size,seq_len).
      The attention distributions for each decoder step.
    p_gens: p_gens: List of length input_size, containing tensors of shape [batch_size, 1]. The values of p_gen for each decoder step. Empty list if pointer_gen=False.
    coverage: Coverage vector on the last step computed. None if use_coverage=False.
  """
  tf.logging.info("\n\n@@@@@@@@@ attention @@@@@@@@@@@@@@\n\n")
  print('encoder_states.shape', encoder_states.shape)
  print('decoder_inputs[0].shape', decoder_inputs[0].shape)


  with variable_scope.variable_scope("attention_decoder") as scope:
    batch_size = encoder_states.get_shape()[0].value # if this line fails, it's because the batch size isn't defined
    enc_output_size = encoder_states.get_shape()[2].value # encoder state size, if this line fails, it's because the attention length isn't defined

    tf.logging.info("\n\n@@@@@@@@@@@@@@ shape compare @@@@@@@@@@@@@@\n\n")
    tf.logging.info(encoder_states.get_shape())
    tf.logging.info(extract_state.get_shape())

    # Indicator variable for hierarchical attention
    hier = True if encoder_section_states is not None else False

    # Reshape encoder_states (need to insert a dim)
    encoder_states = tf.expand_dims(encoder_states, axis=2) # now is shape (batch_size, attn_len, 1, enc_output_size)
    extract_state  = tf.expand_dims(extract_state, axis=2)  # now is shape (batch_size, attn_extract_len, 1, enc_output_size)


    # To calculate attention, we calculate
    #   v^T tanh  (W_h h_i + W_s s_t + b_attn)
    # where h_i is an encoder state, and s_t a decoder state.
    # attn_vec_size is the length of the vectors v, b_attn, (W_h h_i) and (W_s s_t).
    # (W_h h_i) is encoder_features, (W_s s_t) + b_att is decoder_features
    # We set it to be equal to the size of the encoder states.
    attention_vec_size = enc_output_size

    # Get the weight matrix W_h and apply it to each encoder state to get (W_h h_i), the encoder features
    # To multiply batch_size number of time_step sizes of encoder states by W_h, we can use conv2d with stride of 1
    W_h = variable_scope.get_variable("W_h", [1, 1, enc_output_size, attention_vec_size])
    encoder_features = nn_ops.conv2d(encoder_states, W_h, [1, 1, 1, 1], "SAME") # shape (batch_size,seq_len,1,attention_vec_size)

    W_h_extract=variable_scope.get_variable("W_h_extract", [1, 1, enc_output_size, attention_vec_size])
    extract_features = nn_ops.conv2d(extract_state, W_h_extract, [1, 1, 1, 1], "SAME")  # shape (batch_size,extract_len,1,attention_vec_size)



    if hier:
      enc_sec_output_size = encoder_section_states.get_shape()[2].value
      encoder_section_states = tf.expand_dims(encoder_section_states, axis=2)

      W_h_s = variable_scope.get_variable("W_h_s", [1, 1, enc_sec_output_size, attention_vec_size])
      encoder_section_features = nn_ops.conv2d(encoder_section_states, W_h_s, [1, 1, 1, 1], "SAME") # shape (batch_size,seq_len,1,attention_vec_size)
      v_sec = variable_scope.get_variable("v_sec", [attention_vec_size])


    # Get the weight vectors v and w_c (w_c is for coverage)
    # v^T tanh  (W_h h_i + W_s s_t + W_c c_t + b_attn)
    # c_t = \sum_{i=1}^{t-1} a^i  (sum of all attention weights in the previous step) shape=(batch_size, seq_len)
    v1 = variable_scope.get_variable("v1", [attention_vec_size])
    v2 = variable_scope.get_variable("v2", [attention_vec_size])
    if use_coverage:
      with variable_scope.variable_scope("coverage"):
        w_c = variable_scope.get_variable("w_c", [1, 1, 1, attention_vec_size])

    if prev_coverage is not None: # for beam search mode with coverage
      # reshape from (batch_size, seq_len) to (batch_size, attn_len, 1, 1)
      prev_coverage = tf.expand_dims(tf.expand_dims(prev_coverage,2),3)

    def attention(decoder_state, coverage=None, num_words_section=None, step=None):
      """Calculate the context vector and attention distribution from the decoder state.

      Args:
        decoder_state: state of the decoder
        coverage: Optional. Previous timestep's coverage vector, shape (batch_size, attn_len, 1, 1).
        num_words_section: number of words in each section (only needed for hierarchical attention)
        [batch_size, num_sections] -- assumes number of sections in the batch is equal (TODO: check sanity)
        step: index of the current decoder step (needed for section attention)

      Returns:
        context_vector: weighted sum of encoder_states
        attn_dist: attention distribution
        coverage: new coverage vector. shape (batch_size, attn_len, 1, 1)
      """
      with variable_scope.variable_scope("Attention"):
        # Pass the decoder state through a linear layer (this is W_s s_t + b_attn in the paper)
        # (W_s s_t) + b_att is decoder_features; s_t = decoder_state
        decoder_features = linear(decoder_state, attention_vec_size, True) # shape (batch_size, attention_vec_size)
        decoder_features = tf.expand_dims(tf.expand_dims(decoder_features, 1), 1) # reshape to (batch_size, 1, 1, attention_vec_size)

        def masked_attention(e_enc,en_extract, enc_padding_mask,extract_padding_mask):
          if enc_section_padding_mask is not None:
            enc_padding_mask = tf.reshape(enc_section_padding_mask, [batch_size, -1])
            enc_padding_mask = tf.cast(enc_padding_mask, tf.float32)

            extract_padding_mask = tf.reshape(extract_padding_mask, [batch_size, -1])
            extract_padding_mask = tf.cast(extract_padding_mask, tf.float32)

          """Take softmax of e then apply enc_padding_mask and re-normalize"""
          attn_dist_enc = nn_ops.softmax(e_enc) # take softmax. shape (batch_size, attn_length)
          attn_dist_enc *= enc_padding_mask # apply mask

          attn_dist_extract = nn_ops.softmax(en_extract) # take softmax. shape (batch_size, extract_attn_length)
          attn_dist_extract *= extract_padding_mask # apply mask


          en_masked_sums = tf.reduce_sum(attn_dist_enc, axis=1) # shape (batch_size)
          extract_masked_sums = tf.reduce_sum(attn_dist_extract, axis=1) # shape (batch_size)

          return attn_dist_enc / tf.reshape(en_masked_sums, [-1, 1]), attn_dist_extract / tf.reshape(extract_masked_sums, [-1, 1]) # re-normalize

        if use_coverage and coverage is not None: # non-first step of coverage
          if hier:
            tf.logging.info("\n%%%%test%%%%\n\n")
            with tf.variable_scope("attention_sections"):
              if FLAGS.fixed_attn:
                tf.logging.debug('running with fixed attn', '\r')
                decoder_features_sec = linear(decoder_state, attention_vec_size, True, scope='Linear--Section-Features') # shape (batch_size, attention_vec_size)
                decoder_features_sec = tf.expand_dims(tf.expand_dims(decoder_features_sec, 1), 1) # reshape to (batch_size, 1, 1, attention_vec_size)
                e_sec = math_ops.reduce_sum(v_sec * math_ops.tanh(encoder_section_features + decoder_features_sec), [2, 3]) # [batch_size x seq_len_sections]
                attn_dist_sec = nn_ops.softmax(e_sec)
              else:
                e_sec = math_ops.reduce_sum(v_sec * math_ops.tanh(encoder_section_features + decoder_features), [2, 3]) # [batch_size x seq_len_sections]
                attn_dist_sec = nn_ops.softmax(e_sec)

            with tf.variable_scope("attention_words"):
              coverage_features = nn_ops.conv2d(coverage, w_c, [1, 1, 1, 1], "SAME") # c has shape (batch_size, seq_len, 1, attention_vec_size)

              # Calculate v^T tanh(W_h h_i + W_s s_t + w_c c_i^t + b_attn)
              e_enc = math_ops.reduce_sum(v1 * math_ops.tanh(encoder_features + decoder_features + coverage_features), [2, 3])  # shape (batch_size,seq_len)
              en_extract=math_ops.reduce_sum(v2*math_ops.tanh(extract_features + decoder_features), [2, 3])

              # Multiply by section weights
              e_enc = tf.reshape(e_enc, [batch_size, -1, num_words_section[0][0]])
              e_enc = tf.multiply(e_enc, attn_dist_sec[:,:,tf.newaxis])
              e_enc = tf.reshape(e_enc, [batch_size,-1])

              en_extract = tf.reshape(en_extract, [batch_size, -1, extract_state.get_shape()[1].value])
              # en_extract = tf.multiply(en_extract, attn_dist_sec[:, :, tf.newaxis])
              en_extract = tf.reshape(en_extract, [batch_size, -1])

#               --- Some hack for reweighting attention (similar to temp for softmax)
              if temperature > 0.0:
                e_enc = e_enc * temperature
                en_extract= en_extract * temperature
                
              attn_dist,attn_dist_extract = masked_attention(e_enc,en_extract, enc_padding_mask,extract_padding_mask)


              coverage += array_ops.reshape(attn_dist, [batch_size, -1, 1, 1]) # shape=(batch_size, seq_len,1,1)


        else:
          # Calculate v^T tanh(W_h h_i + W_s s_t + b_attn)
          if hier:
            with tf.variable_scope("attention_sections"):
              if FLAGS.fixed_attn:
                decoder_features_sec = linear(decoder_state, attention_vec_size, True, scope='Linear--Section-Features') # shape (batch_size, attention_vec_size)
                decoder_features_sec = tf.expand_dims(tf.expand_dims(decoder_features_sec, 1), 1) # reshape to (batch_size, 1, 1, attention_vec_size)
                e_sec = math_ops.reduce_sum(
                  v_sec * math_ops.tanh(encoder_section_features + decoder_features_sec), [2, 3]) # [batch_size x seq_len_sections]
                attn_dist_sec = nn_ops.softmax(e_sec)
              else:
                e_sec = math_ops.reduce_sum(
                  v_sec * math_ops.tanh(encoder_section_features + decoder_features), [2, 3]) # [batch_size x seq_len_sections]
                attn_dist_sec = nn_ops.softmax(e_sec)


            with tf.variable_scope("attention_words"):
              e_enc = math_ops.reduce_sum(v1 * math_ops.tanh(encoder_features + decoder_features), [2, 3]) #[batch_size x seq_len]
              en_extract = math_ops.reduce_sum(v2 * math_ops.tanh(extract_features + decoder_features), [2, 3])

              e_enc = tf.reshape(e_enc, [batch_size, -1, num_words_section[0][0]])
              e_enc = tf.multiply(e_enc, attn_dist_sec[:,:,tf.newaxis])
              e_enc = tf.reshape(e_enc, [batch_size,-1])

              en_extract = tf.reshape(en_extract, [batch_size, -1, extract_state.get_shape()[1].value])
              # en_extract = tf.multiply(en_extract, attn_dist_sec[:, :, tf.newaxis])
              en_extract = tf.reshape(en_extract, [batch_size, -1])


              if temperature > 0.0:
                e_enc = e_enc * temperature
                en_extract = en_extract * temperature
                
              attn_dist,attn_dist_extract= masked_attention(e_enc, en_extract, enc_padding_mask, extract_padding_mask)

        # TODO: coverage for hier

        # Calculate the context vector from attn_dist and encoder_states
        # ecnoder_sates = [batch , seq_len , 1 , encoder_output_size], attn_dist = [batch, seq_len, 1, 1]
        context_vector_en = math_ops.reduce_sum(array_ops.reshape(attn_dist, [batch_size, -1, 1, 1]) * encoder_states, [1, 2]) # shape (batch_size, enc_output_size).
        context_vector_en = array_ops.reshape(context_vector_en, [-1, enc_output_size])

        context_vector_extract = math_ops.reduce_sum(array_ops.reshape(attn_dist_extract, [batch_size, -1, 1, 1]) * extract_state,[1, 2])  # shape (batch_size, enc_output_size).
        context_vector_extract = array_ops.reshape(context_vector_extract, [-1, enc_output_size])

        context_dense=Connect_Layer(enc_output_size, True, tf.nn.relu)
        context_vector=context_dense(context_vector_en, context_vector_extract)

      return context_vector, attn_dist, attn_dist_extract, coverage, attn_dist_sec




    outputs = []
    attn_dists = []
    attn_dist_extracts = []
    attn_dists_sec_list = []
    p_gens = []
    state = initial_state
    coverage = prev_coverage # initialize coverage to None or whatever was passed in
    context_vector = array_ops.zeros([batch_size, enc_output_size])
    context_vector.set_shape([None, enc_output_size])  # Ensure the second shape of attention vectors is set.
    if initial_state_attention: # true in decode mode
      # Re-calculate the context vector from the previous step so that we can pass it through a linear layer with this step's input to get a modified version of the input
      if hier:
        context_vector,  attn_dist, attn_dist_extract, coverage, attn_dist_sec = attention(initial_state, coverage, num_words_section) # in decode mode, this is what updates the coverage vector

    for i, inp in enumerate(decoder_inputs):
      if (i%1) == 0:
        print("Adding attention_decoder timesteps. %i done of %i" % (i+1, len(decoder_inputs)), end='\r')
      if i > 0:
        variable_scope.get_variable_scope().reuse_variables()

      # Merge input and previous attentions into one vector x of the same size as inp
      # inp is [batch_size, input_size]
      input_size = inp.get_shape().with_rank(2)[1]
      if input_size.value is None:
        raise ValueError("Could not infer input size from input: %s" % inp.name)
      x = linear([inp] + [context_vector], input_size, True)

      cell_output, state = cell(x, state)

      # Run the attention mechanism.
      if i == 0 and initial_state_attention:  # always true in decode mode
        with variable_scope.variable_scope(
          variable_scope.get_variable_scope(), reuse=True): # you need this because you've already run the initial attention(...) call
          if hier:
            context_vector, attn_dist, attn_dist_extract, coverage, attn_dist_sec = attention(state, coverage, num_words_section)
      else:
        if hier:
          context_vector,attn_dist, attn_dist_extract,coverage, attn_dist_sec = attention(state, coverage, num_words_section)

      attn_dists.append(attn_dist)
      attn_dist_extracts.append(attn_dist_extract)

      if hier:
        attn_dists_sec_list.append(attn_dist_sec)

      # Calculate p_gen
      if pointer_gen:
        with tf.variable_scope('calculate_pgen'):
          p_gen = linear([context_vector, state.c, state.h, x], 1, True) # a scalar
          p_gen = tf.sigmoid(p_gen)
          p_gens.append(p_gen)

      # Concatenate the cell_output (= decoder state) and the context vector, and pass them through a linear layer
      # This is V[s_t, h*_t] + b in the paper
      with variable_scope.variable_scope("AttnOutputProjection"):
        output = linear([cell_output] + [context_vector], cell.output_size, True)
      outputs.append(output)

    # If using coverage, reshape it
    if coverage is not None:
      coverage = array_ops.reshape(coverage, [batch_size, -1])
    return outputs, state, attn_dists, attn_dist_extracts,p_gens, coverage, attn_dists_sec_list



def linear(args, output_size, bias, bias_start=0.0, scope=None):
  """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.

  Args:
    args: a 2D Tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    bias: boolean, whether to add a bias term or not.
    bias_start: starting value to initialize the bias; 0 by default.
    scope: VariableScope for the created subgraph; defaults to "Linear".

  Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(args[i] * W[i]), where W[i]s are newly created matrices.

  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  """
  if args is None or (isinstance(args, (list, tuple)) and not args):
    raise ValueError("`args` must be specified")
  if not isinstance(args, (list, tuple)):
    args = [args]

  # Calculate the total size of arguments on dimension 1.
  total_arg_size = 0
  shapes = [a.get_shape().as_list() for a in args]
  for shape in shapes:
    if len(shape) != 2:
      raise ValueError("Linear is expecting 2D arguments: %s" % str(shapes))
    if not shape[1]:
      raise ValueError("Linear expects shape[1] of arguments: %s" % str(shapes))
    else:
      total_arg_size += shape[1]

  # Now the computation.
  with tf.variable_scope(scope or "Linear"):
    matrix = tf.get_variable("Matrix", [total_arg_size, output_size])
    if len(args) == 1:
      res = tf.matmul(args[0], matrix)
    else:
      res = tf.matmul(tf.concat(axis=1, values=args), matrix)
    if not bias:
      return res
    bias_term = tf.get_variable(
        "Bias", [output_size], initializer=tf.constant_initializer(bias_start))
  return res + bias_term