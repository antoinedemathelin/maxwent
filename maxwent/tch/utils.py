import copy
import inspect
import torch
from .layers import LinearMaxWEnt

CONSTRUCTORS = {
    "Linear": LinearMaxWEnt,
}

NO_TRAINING_LAYER = {
}

def _replace_layer(model):
    for name, module in model.named_children():
        class_name = module.__class__.__name__
        if class_name in CONSTRUCTORS:
            init_signature = inspect.signature(module.__init__)
            params = {param.name: getattr(module, param.name) for param in init_signature.parameters.values()
                      if (param.name != 'self' and param.name in module.__dict__)}
            new_module = CONSTRUCTORS[class_name](**params)
            if hasattr(module, "weight") and module.weight is not None:
                new_module.weight.data.copy_(module.weight)
                new_module.weight.requires_grad = False
            if hasattr(module, "bias") and module.bias is not None:
                new_module.bias.data.copy_(module.bias)
                new_module.bias.requires_grad = False
            setattr(model, name, new_module)
        else:  
            for param in module.parameters():
                param.requires_grad = False
            if class_name in NO_TRAINING_LAYER:
                module.eval()
            _replace_layer(module)
    return model


def set_maxwent_model(model, dropout_off=False):
    new_model = copy.deepcopy(model)
    new_model = _replace_layer(new_model)
    return new_model
