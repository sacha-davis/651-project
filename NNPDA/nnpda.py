from abc import ABC

import tensorflow as tf
from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import tanh


class NnpdaCell(tf.compat.v1.nn.rnn_cell.RNNCell, ABC):
    def __init__(self, state_weights, state_bias, action_weights, action_bias, delta, delta_one, input_symbol=None,
                 current_state=None, current_stack=None, state_activation=sigmoid,
                 action_activation=tanh):
        super().__init__(self)
        self.input_symbol = input_symbol              # Input symbol received at time 't'.     shape [Ni x 1]
        self.current_state = current_state            # Internal state at time 't'.            shape [Ns x 1]
        self.current_stack = current_stack            # Stack reading at time 't'.             shape [Nr x 1]
        self.state_activation = state_activation      # State activation function.
        self.action_activation = action_activation    # Action activation function.
        self.state_weights = state_weights            # Weight matrix for internal state.      shape [Ns x Ns x Nr x Ni]
        self.state_bias = state_bias                  # Bias vector for internal state.        shape [Ns x 1]
        self.action_weights = action_weights          # Weight matrix for stack action.        shape [2^Ns x Nr x Ni]
        self.action_bias = action_bias                # Scalar bias for stack action.
        self.delta = delta                            # Delta Matrix
        self.delta_one = delta_one                    # One minus delta matrix

    def __call__(self, **kwargs):
        with tf.compat.v1.variable_scope(type(self).__name__):
            # Equation 5a
            # Tensor dot does a tensor dot product between state weights and input symbol alont the first axis and reduce the sum on the last axis
            WI_s = tf.reduce_sum(tf.tensordot(self.state_weights, self.input_symbol, axes=1), axis=-1)      # The product Ws*I     shape [Ns x Ns x Nr]
            WIR_s = tf.reduce_sum(tf.tensordot(WI_s, self.current_stack, axes=1), axis=-1)                  # The product Ws*I*R   shape [Ns x Ns]
            WIRS = tf.tensordot(WIR_s, self.current_state, axes=1)                                          # The product Ws*I*R*S shape [Ns x 1]
            WIRS_bias = tf.nn.bias_add(WIRS, self.state_bias)                                               # Adding the state bias            shape [Ns x 1]
            next_state = self.state_activation(WIRS_bias)                                                   # Applying the activation function shape [Ns x 1]

            # Equation 5b
            WI_a = tf.reduce_sum(tf.tensordot(self.action_weights, self.input_symbol, axes=1), axis=-1)     # The product Wa*I    shape [2^Ns x Nr]
            WIR_a = tf.reduce_sum(tf.tensordot(WI_a, self.current_stack, axes=1), axis=-1)                  # The product Wa*I*R  shape [2^Ns]

            # Equation 23/24
            Sdelta = tf.multiply(self.delta, tf.transpose(tf.reverse(self.current_state, dims=1)))          # The product delta*S          shape [2^Ns x 1]
            Sdelta_ = tf.multiply(self.delta_one, tf.transpose(tf.reverse(1 - self.current_state, dims=1))) # The product (1-delta)*(1-S)  shape [2^Ns x 1]
            P = tf.reduce_prod(Sdelta + Sdelta_, axis=1)                                                    # P matrix                     shape [2^Ns x 1]

            # Equation 23 and Equation 5b cont'd
            WIRP = tf.reduce_sum(tf.tensordot(WIR_a, P), axis=-1)   # Scalar stack action value
            WIRP_bias = tf.nn.bias_add(WIRP, self.action_bias)         # Adding the scalar action bias
            stack_axn = self.action_activation(WIRP_bias)              # Applying the activation function

        return next_state, stack_axn