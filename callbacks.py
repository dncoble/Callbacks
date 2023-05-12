import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
"""
Callbacks

All these callbacks have been tested for:

TensorFlow 2.10.0
numpy 1.19.5
"""

"""
NanProtection

Protects against gradient explosion producing NaNs. When NaNs are found in
the weights they are replaced by the last saved weights which don't contain
NaNs. It doesn't work that well when NaNs are a recurring problem.
"""
class NanProtection(keras.callbacks.Callback):
    
    def __init__(self, model):
        self.model = model
        super(NanProtection).__init__()
        self.weights = None
    
    def on_train_batch_end(self, batch, logs=None):
        if(self.weights is not None and any([np.isnan(w_).any() or np.isinf(w_).any() for w_ in self.weights])):
           print("critical error reached")
        if(any([np.isnan(w_).any() or np.isinf(w_).any() for w_ in self.model.get_weights()])):
            self.model.set_weights(self.weights)
            for layer in self.model.layers:
                if(layer.stateful):
                    layer.reset_states()
            print("Training encountered NaN values.")
            self.model.stop_training = True
        else:
            self.weights = self.model.get_weights()

"""
StateResetter

Reset all stateful layers in a model at the end of an epoch.
"""
class StateResetter(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        for layer in self.model.layers:
            if(layer.stateful):
                layer.reset_states()

"""
GetTrainData

Saves loss for each batch for the previous epoch.
"""
class GetTrainData(keras.callbacks.Callback):
    
    def __init__(self, epoch_length):
        self.loss = np.zeros((epoch_length,))
    
    def get_loss(self):
        return self.loss
    
    def on_train_batch_end(self, batch, logs=None):
        self.loss[batch] = logs['loss']