import tensorflow as tf
import numpy as np
from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import tanh
from tensorflow.keras.optimizers import RMSprop

from stack import Stack


class NnpdaCell:
    # def __init__(self, state_weights, state_bias, action_weights, action_bias, delta, delta_, input_symbol=None,
    #              current_state=None, current_stack=None, state_activation=sigmoid,
    #              action_activation=tanh, optimizer=RMSprop, activation=sigmoid):

    def __init__(self, input_symbol=None, current_state=None, current_stack=None, state_activation=sigmoid,
                 action_activation=tanh, optimizer=RMSprop, activation=sigmoid):

        self.input_symbol = input_symbol              # Input symbol received at time 't'.     shape [Ni x 1]
        self.current_state = current_state            # Internal state at time 't'.            shape [Ns x 1]
        self.current_stack = current_stack            # Stack reading at time 't'.             shape [Nr x 1]
        self.state_activation = state_activation      # State activation function.
        self.action_activation = action_activation    # Action activation function.
        # self.state_weights = state_weights    
        self.state_weights = None                     # Weight matrix for internal state.      shape [Ns x Ns x Nr x Ni]
        # self.state_bias = state_bias
        self.state_bias = None                        # Bias vector for internal state.        shape [Ns x 1]
        # self.action_weights = action_weights
        self.action_weights = None                    # Weight matrix for stack action.        shape [2^Ns x Nr x Ni]
        # self.action_bias = action_bias
        self.action_bias = None                       # Scalar bias for stack action.
        self.delta = None                             # Delta Matrix
        self.delta_ = None      
        self.stack_axn = None                         # One minus delta matrix

        self.Ns = 20  # state neurons
        self.Ni = 10  # input neurons
        self.Nr = 10  # stack reading neurons
        self.Na = 3   # stack action neurons

        self.batch_size = 20
        self.str_len = 10
        self.num_steps = 200
        
        self.nxt = None
        self.axn = None
        
        self.initializeVars()


    def initializeVars(self):
        self.state_weights=tf.Variable(tf.random.normal([self.Ns, self.Ns, self.Nr, self.Ni]), dtype=tf.dtypes.float32,name="state_weights")  # Weight matrix for computing internal state.
        self.state_bias=tf.Variable(tf.ones([self.Ns, 1]), dtype=tf.dtypes.float32,name="state_bias")  # Bias vector for computing internal state.
        self.action_weights=tf.Variable(tf.random.normal([2 ** self.Ns, self.Nr, self.Ni]), dtype=tf.dtypes.float32,name="action_weights")  # Weight matrix for computing stack action.
        self.action_bias=tf.Variable(tf.ones([1, 1]), dtype=tf.dtypes.float32,name="action_bias")  # Scalar bias for computing stack action.


    def learnWeightMatrixes(self):
        
        # print("In the Call Function")
        # # Equation 5a
        # print("____________________")
        # print("The following is for the Ws matrix:", end="\n\n")
        # print("self.state_weights shape", tf.shape(self.state_weights), "should be Ns x Ns x Nr x Ni",end="\n\n")
        # print("self.input_symbol shape", tf.shape(self.input_symbol), "should be Ni x 1",end="\n\n")
        # WI_s = tf.reduce_sum(input_tensor=tf.tensordot(self.state_weights, self.input_symbol, axes=1), axis=-1)      # The product Ws*I     shape [Ns x Ns x Nr]
        WI_s = tf.tensordot(self.state_weights, self.input_symbol, axes=1)                                          # The product Ws*I     shape [Ns x Ns x Nr]
        # print("WI_s shape", tf.shape(WI_s), "should be Ns x Ns x Nr",end="\n\n")
        
        
        # print("curr_stack shape", tf.shape(self.current_stack), "should be Nr x 1",end="\n\n")
        WIR_s = tf.reduce_sum(input_tensor=tf.tensordot(WI_s, self.current_stack, axes=1), axis=-1)                  # The product Ws*I*R   shape [Ns x Ns]
        # WIR_s = tf.tensordot(WI_s, self.current_stack, axes=1)
        # print("WIR_s shape", tf.shape(WIR_s), "should be Ns x Ns",end="\n\n")

        WIRS = tf.tensordot(WIR_s, self.current_state, axes=1)                                          # The product Ws*I*R*S shape [Ns x 1]
        # print("WIRS shape", tf.shape(WIRS), "should be Ns x 1",end="\n\n")
        # print("state bias shape", tf.shape(tf.reshape(self.state_bias, [-1])), "should be 1D and share a dimension with WIRS",end="\n\n")
        WIRS_bias = tf.transpose(tf.nn.bias_add(tf.transpose(WIRS), tf.reshape(self.state_bias, [-1])))                                 # Adding the state bias shape [Ns x 1]
        # print("WIRS_bias shape", tf.shape(WIRS_bias), "should be Ns x 1",end="\n\n")

        next_state = self.state_activation(WIRS_bias)                                                   # Applying the activation function shape [Ns x 1]
        # print("next_state shape", tf.shape(next_state), "should be NS x 1",end="\n\n")
        # print("End of Ws calculations")
        # print("____________________")
        # ------------------------------------------------fine til here-------------------------------------------------

        # Equation 5b
        # WI_a = tf.reduce_sum(input_tensor=tf.tensordot(self.action_weights, self.input_symbol, axes=1), axis=-1)     # The product Wa*I    shape [2^Ns x Nr]
        # print("____________________")
        # print("The following is for the Wa matrix:",end="\n\n")
        WI_a = tf.tensordot(self.action_weights, self.input_symbol,  axes=1)    # The product Wa*I    shape [2^Ns x Nr]
        # print("WI_a shape", tf.shape(WI_a), "should be 2^Ns x Nr",end="\n\n")

        WIR_a = tf.reduce_sum(input_tensor=tf.tensordot(WI_a, self.current_stack, axes=1), axis=-1)                  # The product Wa*I*R  shape [2^Ns]
        # print("WIR_a shape", tf.shape(WIR_a), "should be 2^Ns",end="\n\n")

        # Equation 23/24
      
        # Information on on these equations:
        # δ inside the product represents the binary values of 0 and 1, 
        # which are determined by the mth bit of the binary number (J-1). 
        # For example, if J-1 = 10, its binary form is 1010, which sets δm: δ1=1, δ2=0, δ3=1 and δ4=0. 
        # The summation of all components of the extended state PJ is equal to one
        # print("delta shape", tf.shape(self.delta), "should be 2^Ns X 1 (?)",end="\n\n")
        Sdelta = tf.multiply(self.delta, tf.transpose(a=tf.reverse(self.current_state, axis=[1])))            # The product delta*S          shape [2^Ns x 1]
        # print("Sdelta shape", tf.shape(Sdelta), "should be 2^Ns x 1 (?)",end="\n\n")

        Sdelta_ = tf.multiply(self.delta_, tf.transpose(a=tf.reverse(1 - self.current_state, axis=[1])))      # The product (1-delta)*(1-S)  shape [2^Ns x 1]
        # print("Sdelta_ shape", tf.shape(Sdelta_), "should be 2^Ns x 1 (?)",end="\n\n")

        P = tf.reduce_prod(input_tensor=Sdelta + Sdelta_, axis=1)                                                    # P matrix  shape [2^Ns x 1]
        # print("P shape", tf.shape(P), "should be 2^Ns x 1",end="\n\n")


        # Equation 23 and Equation 5b continued

        WIRP = tf.tensordot(WIR_a, P, axes=1) # Scalar stack action value (NOTE: WRIP is already a scalar, no reduction needed)
        # WIRP = tf.reduce_sum(input_tensor=it, axis=-1)   # Scalar stack action value
        # print("WIRP shape", tf.shape(WIRP), "should be scalar",end="\n\n")
        # print("self.action_bias shape", tf.shape(self.action_bias), "should be 1D",end="\n\n")

        # NOTE: Equation 23 has no action bias. Getting the action bias doesn't make sense since
        # WIRP is already a scalar. It's not possible to use bias_add.
        # If we were only using eq 5.b, we'd be finding an action bias with WIRS, like in the state matrix.
        # WIRP_bias = tf.nn.bias_add(WIRP, tf.reshape(self.action_bias, [-1]))  # Adding the scalar action bias
        # print("WIRP_bias shape", tf.shape(WIRP_bias), "should be scalar?",end="\n\n")

        self.stack_axn = self.action_activation(WIRP)              # Applying the activation function
        # print("self.stack_axn shape", tf.shape(self.stack_axn), "should be scalar",end="\n\n")
        
        # print("End of Wa calculations")
        # print("____________________")

        # print(self.stack_axn)
        self.nxt = next_state
        self.axn = self.stack_axn


        # return next_state, stack_axn

    def rtrn(self):
        return self.nxt, self.axn


    def get_delta(self,k):
        # this function returns the delta matrix needed calculating Pj = delta*S + (1-delta)*(1-S)
        delta = np.arange(1, (2 ** k)+1)[:, np.newaxis] >> np.arange(k)[::-1] & 1
        all_ones = np.array([[1 for _ in range(k)] for _ in range(2**k)])
        delta_ = all_ones - delta

        return delta, delta_


    def nnpda_cycle(self, optimizer=RMSprop, activation=sigmoid):
        # cell = NnpdaCell
        words = tf.ones([self.Ni, self.num_steps], dtype=tf.dtypes.float32)  # [10x200]

        st_desired = tf.Variable(tf.random.normal([self.Ns, self.num_steps]))   # Placeholder for the desired final state
        self.current_state = tf.ones([self.Ns, 1])

        self.delta, self.delta_ = self.get_delta(self.Ns)

        sym_stack = Stack()  # Stack for storing the input symbols
        len_stack = Stack()  # Stack for storing the lengths of input symbols

        for i in range(self.num_steps):  # 200, length of input sequences
            print("time step", i)
            print("self.stack_axn", self.stack_axn)
            ############# STACK ACTION #############
            # (Default) Pushing for the initial time step
            if i == 0:
                sym_stack.push(words[:, i])  # [, 10]
                len_stack.push(tf.norm(tensor=words[:, i], axis=-1))
            # Pushing if At > 0
            elif self.stack_axn > 0:
                sym_stack.push(words[:, i])
                len_stack.push(self.stack_axn * tf.norm(tensor=words[:, i], axis=-1))
            # Popping if At < 0
            elif self.stack_axn < 0:
                len_popped = 0
                # Popping a total of length |At| from the stack
                while len_popped != (-1)*self.stack_axn:
                    # If len(top) > |At|, Updating the length
                    if len_stack.peek() > (-1)* self.stack_axn:
                        len_popped += (-1)*self.stack_axn
                        len_stack.update(len_stack.peek() - self.stack_axn)
                    # If len(top) < |At|, Popping the top
                    else:
                        len_popped += len_stack.peek()
                        sym_stack.pop()
                        len_stack.pop()
            # No action if At=0
            else:
                continue

            ############# READING THE STACK ##########
            curr_read = tf.ones([self.Nr, 1])
            len_read = 0
            # Reading a total length '1' from the stack
            while len_read != 1:
                # print(len_stack.peek())
                if len_stack.peek() < 1:
                    curr_read = tf.math.add(curr_read, tf.multiply(sym_stack.peek(), len_stack.peek()))
                    # print("current read b4 if", tf.shape(curr_read))
                    len_read += len_stack.peek()
                else:
                    curr_read = tf.math.add(curr_read, tf.reshape(sym_stack.peek(), [self.Ni, 1]))

                    # print(curr_read)
                    # print("current read b4 else", tf.shape(curr_read))

                    len_read = 1

        
        self.input_symbol=words[:, i]
        # self.current_state=curr_state
        self.current_stack=curr_read
        self.state_activation=sigmoid
        self.action_activation=tanh
        
        self.learnWeightMatrixes()

        # next_state, self.stack_axn = cell.rtrn()
        print("stack action:", self.axn)
        self.stack_axn = self.axn
        self.curr_state = self.nxt
        loss_per_example = tf.square(tf.norm(tensor=st_desired - self.curr_state)) + tf.square(len_stack.peek())
        total_loss = tf.reduce_mean(input_tensor=loss_per_example)
        print("Loss per example", loss_per_example)
        print("Total loss", total_loss)

        return total_loss


cell = NnpdaCell()
cell.nnpda_cycle()