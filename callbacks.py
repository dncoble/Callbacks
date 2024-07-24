import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.python.platform import tf_logging as logging
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

"""
StatefullyCallback.

This callback replicates stateful=True for RNNs, but doesn't force a fixed 
batch size. With layers=None, stateful(ly)ness applies to the entire model,
or pass a list of layers that you want stateful(ly)ness to apply to.

NOT YET BUILT
"""
class Statefully(keras.callbacks.Callback):
    
    def __init__(self, layers=None):
        super().__init__()
        # if layers is passed one layer
        if(isinstance(layers), list):
            self.layers = layers
        else:
            self.layers = [layers]
    
    def on_train_begin(self, logs=None):
        # scan the model for layers
        for layer in self.model.layers:
            pass
        pass
        
    def on_train_batch_end(self, batch, logs=None):
        pass
        
    def on_train_batch_begin(self, batch):
        pass

"""
Basically EarlyStopping's restore_best_weights = True. Restore best
weights at the end of training.
"""
class RestoreBestWeights(keras.callbacks.Callback):
    
    def __init__(self, monitor="val_loss", mode="auto"):
        super().__init__()
        
        self.monitor = monitor
        self.best_weights = None
        
        if mode == "min":
            self.monitor_op = np.less
        elif mode == "max":
            self.monitor_op = np.greater
        else:
            if (
                self.monitor.endswith("acc")
                or self.monitor.endswith("accuracy")
                or self.monitor.endswith("auc")
            ):
                self.monitor_op = np.greater
            else:
                self.monitor_op = np.less
        
    
    def _is_improvement(self, monitor_value, reference_value):
        return self.monitor_op(monitor_value, reference_value)
    
    def get_monitor_value(self, logs):
        logs = logs or {}
        monitor_value = logs.get(self.monitor)
        if monitor_value is None:
            logging.warning(
                "Restore best weights conditioned on metric `%s` "
                "which is not available. Available metrics are: %s",
                self.monitor,
                ",".join(list(logs.keys())),
            )
        return monitor_value
    
    def on_train_begin(self, logs=None):
        self.best = np.Inf if self.monitor_op == np.less else -np.Inf
        self.best_weights = None
        self.best_epoch = 0
    
    def on_epoch_end(self, epoch, logs=None):
        current = self.get_monitor_value(logs)
        
        
        if(self.best_weights is None):
            self.best_weights = self.model.get_weights()
        
        if self._is_improvement(current, self.best):
            self.best = current
            self.best_epoch = epoch
            self.best_weights = self.model.get_weights()
        
    def on_train_end(self, logs=None):
        self.model.set_weights(self.best_weights)

'''
Saves any values from logs which it is instructed to.
'''
class SaveBatchValues(keras.callbacks.Callback):
    
    def __init__(self, *value_names, n_batches=None):
        super().__init__()
        
        self.save = {}
        self.n_batches = n_batches
        for name in value_names:
            if(self.n_batches is None):
                self.save[name] = []
            else:
                self.save[name] = [None]*n_batches
    
    def on_train_batch_end(self, batch, logs=None):
        for key, value in logs.items():
            if(key in self.save.keys()):
                if(self.n_batches is None):
                    self.save[key].append(value)
                else:
                    self.save[key][batch] = value
            
    def __getitem__(self, key):
        return self.save[key]