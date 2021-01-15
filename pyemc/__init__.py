import cupy

if cupy.cuda.is_available():
    from .pyemc import *
else:
    import warnings
    warnings.warn("No CUDA devicec found. Can only use cupy.utils functions")

from .utils import *
