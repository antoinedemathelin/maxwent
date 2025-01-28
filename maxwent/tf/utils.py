import tensorflow as tf
from tensorflow.keras.models import clone_model
from .layers import (DenseMaxWEnt,
                    Conv1DMaxWEnt,
                    Conv2DMaxWEnt,
                    Conv3DMaxWEnt,
                    DropoutOff,
                    SpatialDropout1DOff,
                    SpatialDropout2DOff,
                    SpatialDropout3DOff)


CONSTRUCTORS = {
    "Dense": DenseMaxWEnt,
    "Conv1D": Conv1DMaxWEnt,
    "Conv2D": Conv2DMaxWEnt,
    "Conv3D": Conv3DMaxWEnt,
}

# BatchNorm is not here because 'trainable = False' implies
# 'training = False' in call
NO_TRAINING_LAYER = {
     "Dropout": DropoutOff,
     "SpatialDropout1D": SpatialDropout1DOff,
     "SpatialDropout2D": SpatialDropout2DOff,
     "SpatialDropout3D": SpatialDropout3DOff,
}


def _replace_layer(layer, dropout_off=True):
    class_name = layer.__class__.__name__
    config = layer.get_config()
    config["name"] += "_mwe"
    if class_name in CONSTRUCTORS:
        new_layer = CONSTRUCTORS[class_name].from_config(config)
        new_layer.build(layer.input.shape)
        if hasattr(layer, "kernel") and layer.kernel is not None:
            new_layer.kernel.assign(tf.identity(layer.kernel))
            new_layer.kernel.trainable = False
        if hasattr(layer, "bias") and layer.bias is not None:
            new_layer.bias.assign(tf.identity(layer.bias))
            new_layer.bias.trainable = False
        return new_layer
    else:
        if class_name in NO_TRAINING_LAYER:
            new_layer = NO_TRAINING_LAYER[class_name].from_config(config)
        else:
            new_layer = layer.__class__.from_config(config)
        new_layer.build(layer.input.shape)
        new_layer.set_weights(layer.get_weights())
        new_layer.trainable = False
        return new_layer


def set_maxwent_model(model, dropout_off=False):
    # XXX dropout_off to False per default. Maybe this feature can be removed
    # More layers use Dropout (e.g., Attention layer)
    clone_function = lambda x: _replace_layer(x, dropout_off=dropout_off)
    new_model = clone_model(model, clone_function=clone_function)
    return new_model
