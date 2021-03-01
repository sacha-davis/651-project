from abc import ABC

import tensorflow as tf
import numpy as np
from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import tanh
from tensorflow.keras.optimizers import RMSprop

from stack import Stack


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
            WI_s = tf.reduce_sum(input_tensor=tf.tensordot(self.state_weights, self.input_symbol, axes=1), axis=-1)      # The product Ws*I     shape [Ns x Ns x Nr]
            WIR_s = tf.reduce_sum(input_tensor=tf.tensordot(WI_s, self.current_stack, axes=1), axis=-1)                  # The product Ws*I*R   shape [Ns x Ns]
            WIRS = tf.tensordot(WIR_s, self.current_state, axes=1)                                          # The product Ws*I*R*S shape [Ns x 1]
            WIRS_bias = tf.nn.bias_add(WIRS, self.state_bias)                                               # Adding the state bias            shape [Ns x 1]
            next_state = self.state_activation(WIRS_bias)                                                   # Applying the activation function shape [Ns x 1]

            # Equation 5b
            WI_a = tf.reduce_sum(input_tensor=tf.tensordot(self.action_weights, self.input_symbol, axes=1), axis=-1)     # The product Wa*I    shape [2^Ns x Nr]
            WIR_a = tf.reduce_sum(input_tensor=tf.tensordot(WI_a, self.current_stack, axes=1), axis=-1)                  # The product Wa*I*R  shape [2^Ns]

            # Equation 23/24
            Sdelta = tf.multiply(self.delta, tf.transpose(a=tf.reverse(self.current_state, dims=1)))          # The product delta*S          shape [2^Ns x 1]
            Sdelta_ = tf.multiply(self.delta_one, tf.transpose(a=tf.reverse(1 - self.current_state, dims=1))) # The product (1-delta)*(1-S)  shape [2^Ns x 1]
            P = tf.reduce_prod(input_tensor=Sdelta + Sdelta_, axis=1)                                                    # P matrix                     shape [2^Ns x 1]

            # Equation 23 and Equation 5b cont'd
            WIRP = tf.reduce_sum(input_tensor=tf.tensordot(WIR_a, P), axis=-1)   # Scalar stack action value
            WIRP_bias = tf.nn.bias_add(WIRP, self.action_bias)         # Adding the scalar action bias
            stack_axn = self.action_activation(WIRP_bias)              # Applying the activation function

        return next_state, stack_axn


def get_delta(K):
    """This function returns the delta matrix needed calculting Pj = delta*S + (1-delta)*(1-S)

        Args:
            inputs:
                K: Integers below 2^K will be considered
            outputs:
                delta: Matrix containing binary codes of numbers (1, 2^K) each one arranged row-wise.                           shape [2^K x K]
                one_minus_delta: Matrix containing complement of binary codes of numbers (1, 2^K) each one arranged row-wise.   shape [2^K x K]
    """
    delta = np.arange(1, 2 ** K)[:, np.newaxis] >> np.arange(K)[::-1] & 1
    # all_ones = np.array(
    #     [list(np.binary_repr(2 ** int(np.ceil(np.log2(1 + x))) - 1, K)) for x in
    #      range(1, 2 ** K)], dtype=int)
    all_ones = np.array([[1 for _ in range(K)] for _ in range(2**K-1)])
    one_minus_delta = all_ones - delta

    return delta, one_minus_delta


# @tf.function
def define_nnpda(Ns, Ni, Nr, Na, batch_size, num_steps, str_len, optimizer=RMSprop, activation=sigmoid):
    # words = tf.placeholder(tf.float64, [batch_size, Ni, num_steps])       # Placeholder for the inputs in a given batch
    # words = tf.compat.v1.placeholder(tf.float32, [Ni, num_steps])       # Placeholder for the inputs in a given batch
    words = tf.ones([Ni, num_steps], dtype=tf.dtypes.float32)

    tf.print(words)

    # print(words)

    # for iteration/batch in num_steps
    # num steps = number of steps needed to get through every minibatch

    st_desired = tf.compat.v1.placeholder(tf.float32, [Ns, num_steps])  # Placeholder for the desired final state

    Ws = tf.compat.v1.get_variable('Ws', [Ns, Ns, Nr, Ni])    # Weight matrix for computing internal state.
    bs = tf.compat.v1.get_variable('bs', [Ns, 1])             # Bias vector for computing internal state.
    Wa = tf.compat.v1.get_variable('Wa', [2 ** Ns, Nr, Ni])   # Weight matrix for computing stack action.
    ba = tf.compat.v1.get_variable('ba', [1, 1])              # Scalar bias for computing stack action.

    cell = NnpdaCell  # Creating an instance for the NNPDA cell created above.

    initial_state = curr_state = tf.zeros([Ns, 1])  # Initial state of the NNPDA cell
    initial_read = curr_read = tf.zeros([Nr, 1])    # Initial reading of the stack
    delta, one_minus_delta = get_delta(Ns)                      # The delta matrices required to compute P

    sym_stack = Stack()  # Stack for storing the input symbols
    len_stack = Stack()  # Stack for storing the lengths of input symbols

    for i in range(num_steps):  # 200, length of input sequences
        print(i)
        ############# STACK ACTION #############
        # (Default) Pushing for the initial time step
        if i == 0:
            sym_stack.push(words[:, i])  # [20 x 10]
            len_stack.push(tf.norm(tensor=words[:, i], axis=-1))
        # Pushing if At > 0
        elif stack_axn > 0:
            sym_stack.push(words[:, i])
            len_stack.push(stack_axn * tf.norm(tensor=words[:, i], axis=-1))
        # Popping if At < 0
        elif stack_axn < 0:
            len_popped = 0
            # Popping a total of length |At| from the stack
            while (len_popped != -stack_axn):
                # If len(top) > |At|, Updating the length
                if len_stack.peek() > -stack_axn:
                    len_popped += -stack_axn
                    len_stack.update(len_stack.peek() - stack_axn)
                # If len(top) < |At|, Popping the top
                else:
                    len_popped += len_stack.peek()
                    sym_stack.pop()
                    len_stack.pop()
        # No action if At=0
        else:
            continue
        ############# READING THE STACK ##########
        curr_read = tf.zeros([batch_size, Nr, 1])
        len_read = 0
        # Reading a total length '1' from the stack
        while (len_read != 1):
            print(len_stack.peek().eval())
            if len_stack.peek().eval() < 1:
                curr_read += tf.multiply(sym_stack.peek(), len_stack.peek())
                len_read += len_stack.peek()
            else:
                curr_read += sym_stack.peek()
                len_read = 1

        next_state, stack_axn = cell(input_symbol=words[:, i],
                                     state_weights=Ws,
                                     current_state=curr_state,
                                     current_stack=curr_read,
                                     state_activation=sigmoid,
                                     action_activation=tanh,
                                     state_bias=bs,
                                     action_weights=Wa,
                                     action_bias=ba,
                                     delta=delta,
                                     delta_one=one_minus_delta)
        curr_state = next_state

    # Computing the Loss E = (Sf - S(t=T))^2 + (L(t=T))^2
    loss_per_example = tf.square(tf.norm(tensor=st_desired - curr_state)) + tf.square(
        len_stack.peek())
    total_loss = tf.reduce_mean(input_tensor=loss_per_example)

    return total_loss


# a, b = get_delta(4)
# print(a)
# print("---------")
# print(b)

Ns = 20  # state neurons
Ni = 10  # input neurons
Nr = 10  # stack reading neurons
Na = 3  # stack action neurons

batch_size = 20
num_steps = 200
str_len = 10


with tf.compat.v1.Session() as sess:
    define_nnpda(Ns, Ni, Nr, Na, batch_size, num_steps, str_len, optimizer=RMSprop, activation=sigmoid)

# ############################## Training #####################################
# def train_nnpda(num_epoch):
#    with tf.Session as sess:
#        for ep in range(num_epoch):
#
# # This has to be written according to the data format.
