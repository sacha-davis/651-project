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


variables_dict = {"Ws": tf.Variable(tf.random.normal([Ns, Ns, Nr, Ni]), dtype=tf.dtypes.float32,name="state_weights"),  # Weight matrix for computing internal state.
                  "bs": tf.Variable(tf.ones([Ns, 1]), dtype=tf.dtypes.float32,name="state_bias"),  # Bias vector for computing internal state.
                  "Wa": tf.Variable(tf.random.normal([2 ** Ns, Nr, Ni]), dtype=tf.dtypes.float32,name="action_weights"),  # Weight matrix for computing stack action.
                  "ba": tf.Variable(tf.ones([1, 1]), dtype=tf.dtypes.float32,name="action_bias")}  # Scalar bias for computing stack action.


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
        
        print("In the Call Function")
        # # Equation 5a
        print("____________________")
        print("The following is for the Ws matrix:", end="\n\n")
        print("self.state_weights shape", tf.shape(self.state_weights), "should be Ns x Ns x Nr x Ni",end="\n\n")
        print("self.input_symbol shape", tf.shape(self.input_symbol), "should be Ni x 1",end="\n\n")
        # WI_s = tf.reduce_sum(input_tensor=tf.tensordot(self.state_weights, self.input_symbol, axes=1), axis=-1)      # The product Ws*I     shape [Ns x Ns x Nr]
        WI_s = tf.tensordot(self.state_weights, self.input_symbol, axes=1)                                          # The product Ws*I     shape [Ns x Ns x Nr]
        print("WI_s shape", tf.shape(WI_s), "should be Ns x Ns x Nr",end="\n\n")
        
        
        print("curr_stack shape", tf.shape(self.current_stack), "should be Nr x 1",end="\n\n")
        WIR_s = tf.reduce_sum(input_tensor=tf.tensordot(WI_s, self.current_stack, axes=1), axis=-1)                  # The product Ws*I*R   shape [Ns x Ns]
        # WIR_s = tf.tensordot(WI_s, self.current_stack, axes=1)
        print("WIR_s shape", tf.shape(WIR_s), "should be Ns x Ns",end="\n\n")

        WIRS = tf.tensordot(WIR_s, self.current_state, axes=1)                                          # The product Ws*I*R*S shape [Ns x 1]
        print("WIRS shape", tf.shape(WIRS), "should be Ns x 1",end="\n\n")
        print("state bias shape", tf.shape(tf.reshape(self.state_bias, [-1])), "should be 1D and share a dimension with WIRS",end="\n\n")
        print("tf.nn.bias_add(tf.transpose(WIRS)", tf.transpose(WIRS))
        print("tf.reshape(self.state_bias, [-1]))",tf.reshape(self.state_bias, [-1]))
        # exit()
        WIRS_bias = tf.transpose(tf.nn.bias_add(tf.transpose(WIRS), tf.reshape(self.state_bias, [-1])))                                 # Adding the state bias shape [Ns x 1]
        print("WIRS_bias shape", tf.shape(WIRS_bias), "should be Ns x 1",end="\n\n")

        next_state = self.state_activation(WIRS_bias)                                                   # Applying the activation function shape [Ns x 1]
        print("next_state shape", tf.shape(next_state), "should be NS x 1",end="\n\n")
        print("End of Ws calculations")
        print("____________________")
        # ------------------------------------------------fine til here-------------------------------------------------

        # Equation 5b
        # WI_a = tf.reduce_sum(input_tensor=tf.tensordot(self.action_weights, self.input_symbol, axes=1), axis=-1)     # The product Wa*I    shape [2^Ns x Nr]
        print("____________________")
        print("The following is for the Wa matrix:",end="\n\n")
        WI_a = tf.tensordot(self.action_weights, self.input_symbol,  axes=1)    # The product Wa*I    shape [2^Ns x Nr]
        print("WI_a shape", tf.shape(WI_a), "should be 2^Ns x Nr",end="\n\n")

        WIR_a = tf.reduce_sum(input_tensor=tf.tensordot(WI_a, self.current_stack, axes=1), axis=-1)                  # The product Wa*I*R  shape [2^Ns]
        print("WIR_a shape", tf.shape(WIR_a), "should be 2^Ns",end="\n\n")

        # Equation 23/24
      
        # Information on on these equations:
        # δ inside the product represents the binary values of 0 and 1, 
        # which are determined by the mth bit of the binary number (J-1). 
        # For example, if J-1 = 10, its binary form is 1010, which sets δm: δ1=1, δ2=0, δ3=1 and δ4=0. 
        # The summation of all components of the extended state PJ is equal to one
        print("delta shape", tf.shape(self.delta), "should be ?binary",end="\n\n")
        Sdelta = tf.multiply(self.delta, tf.transpose(a=tf.reverse(self.current_state, axis=[1])))            # The product delta*S          shape [2^Ns x 1]
        print("Sdelta shape", tf.shape(Sdelta), "should be 2^Ns x 1",end="\n\n")

        Sdelta_ = tf.multiply(self.delta_, tf.transpose(a=tf.reverse(1 - self.current_state, axis=[1])))      # The product (1-delta)*(1-S)  shape [2^Ns x 1]
        print("Sdelta_ shape", tf.shape(Sdelta_), "should be 2^Ns x 1",end="\n\n")

        P = tf.reduce_prod(input_tensor=Sdelta + Sdelta_, axis=1)                                                    # P matrix  shape [2^Ns x 1]
        print("P shape", tf.shape(P), "should be 2^Ns x 1",end="\n\n")


        # Equation 23 and Equation 5b continued

        WIRP = tf.tensordot(WIR_a, P, axes=1) # Scalar stack action value (NOTE: WRIP is already a scalar, no reduction needed)
        # WIRP = tf.reduce_sum(input_tensor=it, axis=-1)   # Scalar stack action value
        print("WIRP shape", tf.shape(WIRP), "should be scalar",end="\n\n")
        print("self.action_bias shape", tf.shape(self.action_bias), "should be 1D",end="\n\n")

        # NOTE: Equation 23 has no action bias. Getting the action bias doesn't make sense since
        # WIRP is already a scalar. It's not possible to use bias_add.
        # If we were only using eq 5.b, we'd be finding an action bias with WIRS, like in the state matrix.
        # WIRP_bias = tf.nn.bias_add(WIRP, tf.reshape(self.action_bias, [-1]))  # Adding the scalar action bias
        # print("WIRP_bias shape", tf.shape(WIRP_bias), "should be scalar?",end="\n\n")

        stack_axn = self.action_activation(WIRP)              # Applying the activation function
        print("stack_axn shape", tf.shape(stack_axn), "should be scalar????",end="\n\n")
        
        print("End of Wa calculations")
        print("____________________")

        # print(stack_axn)
        self.nxt = next_state
        self.axn = stack_axn

        assert False

        # return next_state, stack_axn

    def rtrn(self):
        return self.nxt, self.axn


def get_delta(k):
    # this function returns the delta matrix needed calculating Pj = delta*S + (1-delta)*(1-S)
    delta = np.arange(1, (2 ** k)+1)[:, np.newaxis] >> np.arange(k)[::-1] & 1
    all_ones = np.array([[1 for _ in range(k)] for _ in range(2**k)])
    delta_ = all_ones - delta

    return delta, delta_


def nnpda_cycle(Ns, Ni, Nr, Na, batch_size, num_steps, str_len, optimizer=RMSprop, activation=sigmoid):
    # cell = NnpdaCell
    words = tf.ones([Ni, num_steps], dtype=tf.dtypes.float32)  # [10x200]

    st_desired = tf.Variable(tf.random.normal([Ns, num_steps]))   # Placeholder for the desired final state
    curr_state = tf.ones([Ns, 1])

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
        curr_read = tf.ones([Nr, 1])
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