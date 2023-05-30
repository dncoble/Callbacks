import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as backend
import numpy as np
"""
Miscellaneous - functions, regularizers for keras

All these callbacks have been tested for:

TensorFlow 2.10.0
numpy 1.19.5
"""

"""
A generator class for chopping training data from a large sequence. Pass arrays
*args and the last passed element should be y
"""
class TrainingGenerator(keras.utils.Sequence):
    
    def __init__(self, *args, train_len=400):
        self.args = args
        self.train_len = train_len
        self.length = args[0].shape[1]//train_len
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        # F_batch = self.F[:,index*self.train_len:(index+1)*self.train_len,:]
        # lstm_batch = self.lstm_input[:,index*self.train_len:(index+1)*self.train_len,:]
        # return lstm_batch, F_batch
        rtrn = [arg[:,index*self.train_len:(index+1)*self.train_len,:] for arg in self.args]
        
        return rtrn[:-1], rtrn[-1]

"""
The Hoyer regularizer is the ratio of the L1 and L2 norms. It has the effect
of sparsifying the input tensor but does not reduce the tensor's energy.
"""
class HoyerRegularizer:
    def __init__(self, hoyer=0.):
        hoyer = 0 if hoyer is None else hoyer
        self.hoyer = backend.cast_to_floatx(hoyer)
        
    def __call__(self, x):
        regularization = self.hoyer* tf.reduce_sum(tf.abs(x))/tf.reduce_sum(tf.square(x))
        return regularization
    
    def get_config(self):
        return {'hoyer': self.hoyer}

"""
Wrap this around an RNN layer for it to act with statefullness -- carry state
between inferences in the same epoch. The advantage over using stateful=True
is that batch size can be varied. And, set on = True/False, and recompile the
model to turn statefulness on and off.

The provided layer must have return_state=True (but the Statefully layer will
not return state unless its return_state=True)

NOT FULLY BUILT, UNTESTED
"""
class Statefully(keras.layers.RNN):
    
    def __init__(self, layer, 
                       return_sequences=False,
                       return_state=False,
                       go_backwards=False,
                       stateful=False,
                       unroll=False,
                       time_major=False,
                       **kwargs):
        assert layer.return_state == True, "Passed layer must have return_state=True."
        assert layer.stateful == False, "Undefined operations for stateful=True."
        self.layer = layer
        self.cell = layer.cell
        super(Statefully, self).__init__(self.cell, 
                                         return_sequences=False,
                                         return_state=False,
                                         go_backwards=False,
                                         stateful=False,
                                         unroll=False,
                                         time_major=False,
                                         **kwargs)
        
        