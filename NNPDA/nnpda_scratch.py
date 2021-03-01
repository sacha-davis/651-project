import tensorflow as tf
import numpy as np
from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import tanh
from tensorflow.keras.optimizers import RMSprop

from stack import Stack

Ns = 20  # state neurons
Ni = 10  # input neurons
Nr = 10  # stack reading neurons
Na = 3  # stack action neurons

batch_size = 20
num_steps = 200
str_len = 10


variables_dict = {"Ws": tf.Variable(tf.random.normal([Ns, Ns, Nr, Ni]), name="state_weights"),  # Weight matrix for computing internal state.
                  "bs": tf.Variable(tf.zeros([Ns, 1]), name="state_bias"),  # Bias vector for computing internal state.
                  "Wa": tf.Variable(tf.random.normal([2 ** Ns, Nr, Ni]), name="action_weights"),  # Weight matrix for computing stack action.
                  "ba": tf.Variable(tf.zeros([1, 1]), name="action_bias")}  # Scalar bias for computing stack action.


class NnpdaCell:
    def __init__(self, state_weights, state_bias, action_weights, action_bias, delta, delta_, input_symbol=None,
                 current_state=None, current_stack=None, state_activation=sigmoid,
                 action_activation=tanh):

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
        self.delta_ = delta_                    # One minus delta matrix

        self.nxt = None
        self.axn = None

    def __call__(self, **kwargs):
        print("I'm here!")
        # # Equation 5a
        # print(tf.shape(self.state_weights))
        # print(tf.shape(self.input_symbol))
        # print(tf.shape(npt))
        # print("Ws", tf.shape(self.state_weights))
        # WI_s = tf.reduce_sum(input_tensor=tf.tensordot(self.state_weights, self.input_symbol, axes=1), axis=-1)      # The product Ws*I     shape [Ns x Ns x Nr]
        WI_s = tf.tensordot(self.state_weights, self.input_symbol, axes=1)
        # print("WI_s", tf.shape(WI_s))
        # print("curr_stack", tf.shape(self.current_stack))

        WIR_s = tf.reduce_sum(input_tensor=tf.tensordot(WI_s, self.current_stack, axes=1), axis=-1)                  # The product Ws*I*R   shape [Ns x Ns]
        # WIR_s = tf.tensordot(WI_s, self.current_stack, axes=1)
        # print("WIR_s", tf.shape(WIR_s))

        # print(tf.shape())
        WIRS = tf.tensordot(WIR_s, self.current_state, axes=1)                                          # The product Ws*I*R*S shape [Ns x 1]
        # print("WIRS", tf.shape(WIRS))
        # print("state bias", tf.shape(self.state_bias))
        # print(tf.shape(tf.reshape(self.state_bias, [-1])))
        WIRS_bias = tf.transpose(tf.nn.bias_add(tf.transpose(WIRS), tf.reshape(self.state_bias, [-1])))                                 # Adding the state bias            shape [Ns x 1]
        # print("WIRS bias", tf.shape(WIRS_bias))

        next_state = self.state_activation(WIRS_bias)                                                   # Applying the activation function shape [Ns x 1]

        # ------------------------------------------------fine til here-------------------------------------------------

        # Equation 5b
        # WI_a = tf.reduce_sum(input_tensor=tf.tensordot(self.action_weights, self.input_symbol, axes=1), axis=-1)     # The product Wa*I    shape [2^Ns x Nr]
        WI_a = tf.tensordot(self.action_weights, self.input_symbol,  axes=1)    # The product Wa*I    shape [2^Ns x Nr]
        print("WI_a", tf.shape(WI_a))

        assert False

        WIR_a = tf.reduce_sum(input_tensor=tf.tensordot(WI_a, self.current_stack, axes=1), axis=-1)                  # The product Wa*I*R  shape [2^Ns]
        print("WIR_a", tf.shape(WIR_a))

        # Equation 23/24
        Sdelta = tf.multiply(self.delta, tf.transpose(a=tf.reverse(self.current_state, dims=1)))            # The product delta*S          shape [2^Ns x 1]
        print("Sdelta", tf.shape(Sdelta))

        Sdelta_ = tf.multiply(self.delta_, tf.transpose(a=tf.reverse(1 - self.current_state, dims=1)))      # The product (1-delta)*(1-S)  shape [2^Ns x 1]
        print("Sdelta_", tf.shape(Sdelta_))

        P = tf.reduce_prod(input_tensor=Sdelta + Sdelta_, axis=1)                                                    # P matrix                     shape [2^Ns x 1]
        print("P", tf.shape(P))


        # Equation 23 and Equation 5b continued
        WIRP = tf.reduce_sum(input_tensor=tf.tensordot(WIR_a, P), axis=-1)   # Scalar stack action value
        print("WIRP", tf.shape(WIRP))

        WIRP_bias = tf.nn.bias_add(WIRP, self.action_bias)         # Adding the scalar action bias
        print("WIRP_bias", tf.shape(WIRP_bias))

        stack_axn = self.action_activation(WIRP_bias)              # Applying the activation function
        print("stack_axn", tf.shape(stack_axn))


        # print(stack_axn)
        self.nxt = next_state
        self.axn = stack_axn

        # return next_state, stack_axn

    def rtrn(self):
        return self.nxt, self.axn


def get_delta(k):
    # this function returns the delta matrix needed calculating Pj = delta*S + (1-delta)*(1-S)
    delta = np.arange(1, 2 ** k)[:, np.newaxis] >> np.arange(k)[::-1] & 1
    all_ones = np.array([[1 for _ in range(k)] for _ in range(2**k-1)])
    delta_ = all_ones - delta

    return delta, delta_


def nnpda_cycle(Ns, Ni, Nr, Na, batch_size, num_steps, str_len, optimizer=RMSprop, activation=sigmoid):
    # cell = NnpdaCell
    words = tf.ones([Ni, num_steps], dtype=tf.dtypes.float32)  # [10x200]

    st_desired = tf.Variable(tf.random.normal([Ns, num_steps]))   # Placeholder for the desired final state
    curr_state = tf.zeros([Ns, 1])

    delta, delta_ = get_delta(Ns)

    sym_stack = Stack()  # Stack for storing the input symbols
    len_stack = Stack()  # Stack for storing the lengths of input symbols

    for i in range(num_steps):  # 200, length of input sequences
        print("time step", i)

        ############# STACK ACTION #############
        # (Default) Pushing for the initial time step
        if i == 0:
            sym_stack.push(words[:, i])  # [, 10]
            len_stack.push(tf.norm(tensor=words[:, i], axis=-1))
        # Pushing if At > 0
        elif stack_axn > 0:
            sym_stack.push(words[:, i])
            len_stack.push(stack_axn * tf.norm(tensor=words[:, i], axis=-1))
        # Popping if At < 0
        elif stack_axn < 0:
            len_popped = 0
            # Popping a total of length |At| from the stack
            while len_popped != -stack_axn:
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
        curr_read = tf.zeros([Nr, 1])
        len_read = 0
        # Reading a total length '1' from the stack
        while len_read != 1:
            # print(len_stack.peek())
            if len_stack.peek() < 1:
                curr_read = tf.math.add(curr_read, tf.multiply(sym_stack.peek(), len_stack.peek()))
                # print("current read b4 if", tf.shape(curr_read))
                len_read += len_stack.peek()
            else:
                curr_read = tf.math.add(curr_read, tf.reshape(sym_stack.peek(), [Ni, 1]))

                # print(curr_read)
                # print("current read b4 else", tf.shape(curr_read))

                len_read = 1

        # https://stackoverflow.com/questions/9663562/what-is-the-difference-between-init-and-call
        cell = NnpdaCell(input_symbol=words[:, i],
                         current_state=curr_state,
                         current_stack=curr_read,
                         state_activation=sigmoid,
                         action_activation=tanh,
                         state_weights=variables_dict["Ws"],
                         state_bias=variables_dict["bs"],
                         action_weights=variables_dict["Wa"],
                         action_bias=variables_dict["ba"],
                         delta=delta,
                         delta_=delta_)# .rtrn()
        cell()
        next_state, stack_axn = cell.rtrn()
        print("stack action:", stack_axn)

        curr_state = next_state


nnpda_cycle(Ns, Ni, Nr, Na, batch_size, num_steps, str_len, optimizer=RMSprop, activation=sigmoid)