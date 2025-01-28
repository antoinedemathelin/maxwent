import os
import importlib
import warnings

_MAXWENT_FRAMEWORK = "tf"

def set_framework(framework: str):
    """
    Set the framework for maxwent between tensorflow ('tf')
    and pytorch ('torch').

    Args:
        framework (str): The framework to use ('tf' or 'torch').

    Raises:
        ValueError: If an invalid framework is specified.
    """
    global _MAXWENT_FRAMEWORK
    if framework not in ["tf", "torch"]:
        raise ValueError("Invalid framework. Choose 'tf' or 'torch'.")
    _MAXWENT_FRAMEWORK = framework

    # Dynamically import the framework-specific functions into the global namespace
    if framework == "tf":
        module_name = "tf"
    else:
        module_name = "tch"
    module = importlib.import_module("maxwent." + module_name)
    globals().update({name: attr for name, attr in module.__dict__.items()
                      if name in module.__all__})


def _import_framework_submodule():
    try:
        import tensorflow
        _tensorflow_installed = True
    except ImportError:
        _tensorflow_installed = False
    try:
        import torch
        _pytorch_installed = True
    except ImportError:
        _pytorch_installed = False
    if _tensorflow_installed:
        set_framework("tf")
        if _pytorch_installed:
            warnings.warn(
                "Both TensorFlow and PyTorch are installed. Defaulting to TensorFlow.\n"
                "You can change the framework by calling `set_framework('torch')` or `set_framework('tf')`.\n"
                "You can also directly import the functions through the 'maxwent._tf' and 'maxwent._torch' modules.",
                UserWarning
            )
    elif _pytorch_installed:
        set_framework("torch")
    else:
        raise ImportError("Neither TensorFlow nor PyTorch is installed.")


_import_framework_submodule()