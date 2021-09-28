import cupy
import numpy
import os
from eke import rotmodule
import functools
import inspect
import enum

_NTHREADS = 128
MAX_PHOTON_COUNT = 200000
# _INTERPOLATION = {"nearest_neighbour": 0,
#                   "linear": 1}

class Interpolation(enum.Enum):
    NEAREST = 1
    LINEAR = 2

def type_checked(*type_args):
    def decorator(func):
        func_signature = inspect.signature(func)

        @functools.wraps(func)
        def new_func(*args, **kwargs):

            # Bind arguments and apply defaults. Also figure out which
            # argumets used default values since these will be excempt
            # from type checking
            bound_arguments_no_default = func_signature.bind(*args, **kwargs)
            bound_arguments = func_signature.bind(*args, **kwargs)
            bound_arguments.apply_defaults()
            args = bound_arguments.args
            defaults_used = [i for i, k in enumerate(bound_arguments.arguments.keys()) if k not in bound_arguments_no_default.arguments]

            
            # types = [numpy.dtype(t) for t in type_args]
            for this_type, this_arg, this_index in zip(type_args, args, range(len(type_args))):
                if this_type is None:
                    continue
                if this_index in defaults_used:
                    continue
                elif this_type is "dense":
                    if not isinstance(this_arg, cupy.ndarray) or cupy.int32 != cupy.dtype(this_arg.dtype):
                        raise TypeError(f"Argument {this_index} to {func.__name__} must be dense patterns (cupy int32)")
                elif this_type is "sparse":
                    if (not isinstance(this_arg, dict) or not
                        ("start_indices" in this_arg and cupy.dtype(this_arg["start_indices"].dtype) == cupy.int32 and
                         "indices" in this_arg and cupy.dtype(this_arg["indices"].dtype) == cupy.int32 and
                         "values" in this_arg and cupy.dtype(this_arg["values"].dtype) == cupy.int32)):
                        raise TypeError(f"Argument {this_index} to {func.__name__} must be sparse patterns")
                elif this_type is "sparser":
                    if (not isinstance(this_arg, dict) or not
                        ("start_indices" in this_arg and cupy.dtype(this_arg["start_indices"].dtype) == cupy.int32 and
                         "indices" in this_arg and cupy.dtype(this_arg["indices"].dtype) == cupy.int32 and
                         "values" in this_arg and cupy.dtype(this_arg["values"].dtype) == cupy.int32 and
                         "ones_start_indices" in this_arg and cupy.dtype(this_arg["ones_start_indices"].dtype) == cupy.int32 and
                         "ones_indices" in this_arg and cupy.dtype(this_arg["ones_indices"].dtype) == cupy.int32)):
                        raise TypeError(f"Argument {this_index} to {func.__name__} must be sparse patterns")
                else:
                    if not isinstance(this_arg, cupy.ndarray):
                        raise TypeError(f"Argument {this_index} to {func.__name__} must be a cupy array.")
                    if cupy.dtype(this_type) != cupy.dtype(this_arg.dtype):
                        raise TypeError(f"Argument {this_index} to {func.__name__} is of dtype {cupy.dtype(this_arg.dtype)}, should be {cupy.dtype(this_type)}")
            return func(*args)
        return new_func
    return decorator


class LogFactorialTable:
    def __init__(self):
        # self._create_table(maximum)
        self._table = None
        
    def _create_table(self, maximum):
        table = numpy.zeros(int(maximum+1), dtype="float32")
        table[0] = 0.
        for i in range(1, int(maximum+1)):
            table[i] = table[i-1] + numpy.log(i)
        self._table = cupy.asarray(table, dtype="float32")

    def table(self, maximum):
        if self._table is None or len(self._table) <= maximum:
            self._create_table(maximum)
        return self._table
log_factorial = LogFactorialTable()


class SliceSums:
    def __init__(self, size=100):
        self._array = None
        self._last_size = size

    def _allocate_array(self, size):
        # print(f"Allocate slice sums array of size {size}")
        self._array = cupy.empty(size, dtype="float32")

    def array(self, size=None):
        if size is not None and (self._array is None or len(self._array) < size):
            self._allocate_array(size)
        if size is not None:
            self._last_size = size
        return self._array[:self._last_size]
slice_sums = SliceSums()


def import_cuda_file(file_name, kernel_names, absolute_path=False):
    # nthreads = 128
    threads_code = f"const int NTHREADS = {_NTHREADS};"
    cuda_files_dir = os.path.join(os.path.split(__file__)[0], "cuda")
    header_file = "header.cu"
    with open(os.path.join(cuda_files_dir, header_file), "r") as file_handle:
        header_source = file_handle.read()
    if not absolute_path:
        file_name = os.path.join(cuda_files_dir, file_name)
    with open(file_name, "r") as file_handle:
        main_source = file_handle.read()
    combined_source = "\n".join((header_source, threads_code, main_source))
    module = cupy.RawModule(code=combined_source, options=("--std=c++11", ), name_expressions=kernel_names)
    import sys
    module.compile(log_stream=sys.stdout)
    kernels = {}
    for this_name in kernel_names:
        kernels[this_name] = module.get_function(this_name)
    return kernels


def import_kernels():
    emc_kernels = import_cuda_file("emc_cuda.cu",
                                   ["kernel_expand_model",
                                    "kernel_insert_slices",
                                    "kernel_expand_model_2d",
                                    "kernel_insert_slices_2d",
                                    "kernel_rotate_model"])
    respons_kernels = import_cuda_file("calculate_responsabilities_cuda.cu",
                                       ["kernel_sum_slices",
                                        "kernel_calculate_responsabilities_poisson",
                                        "kernel_calculate_responsabilities_poisson_scaling",
                                        "kernel_calculate_responsabilities_poisson_per_pattern_scaling",
                                        "kernel_calculate_responsabilities_poisson_sparse",
                                        "kernel_calculate_responsabilities_poisson_sparse_scaling",
                                        "kernel_calculate_responsabilities_poisson_sparse_per_pattern_scaling",
                                        "kernel_calculate_responsabilities_poisson_sparser",
                                        "kernel_calculate_responsabilities_poisson_sparser_scaling",
                                        "kernel_calculate_responsabilities_gaussian",
                                        "kernel_calculate_responsabilities_gaussian_scaling",
                                        "kernel_calculate_responsabilities_gaussian_per_pattern_scaling",])
    scaling_kernels = import_cuda_file("calculate_scaling_cuda.cu",
                                       ["kernel_calculate_scaling_poisson",
                                        "kernel_calculate_scaling_poisson_sparse",
                                        "kernel_calculate_scaling_poisson_sparser",
                                        "kernel_calculate_scaling_per_pattern_poisson",
                                        "kernel_calculate_scaling_per_pattern_poisson_sparse"])
    slices_kernels = import_cuda_file("update_slices_cuda.cu",
                                      ["kernel_normalize_slices",
                                       "kernel_update_slices<int>",
                                       "kernel_update_slices<float>",
                                       "kernel_update_slices_scaling<int>",
                                       "kernel_update_slices_scaling<float>",
                                       "kernel_update_slices_per_pattern_scaling<int>",
                                       "kernel_update_slices_per_pattern_scaling<float>",
                                       "kernel_update_slices_sparse",
                                       "kernel_update_slices_sparse_scaling",
                                       "kernel_update_slices_sparse_per_pattern_scaling",
                                       "kernel_update_slices_sparser",
                                       "kernel_update_slices_sparser_scaling"])
    tools_kernels = import_cuda_file("tools.cu",
                                     ["kernel_blur_model"])
    kernels = {**emc_kernels, **respons_kernels, **scaling_kernels, **slices_kernels, **tools_kernels}
    return kernels


def set_nthreads(nthreads):
    global _NTHREADS, kernels
    _NTHREADS = nthreads
    kernels = import_kernels()

kernels = import_kernels()


# def check_model(model):
#     if len(model.shape) != 3:
#         raise ValueError("Model must be a 3D array.")

# def check_slices(slices, length):
#     if len(slices.shape) != 3:
#         raise ValueError("Slices must be a 3D array.")
#     if len(slices) != length:
#         raise ValueError(f"Slices are expected to be of length {length}.")

# def check_rotations(rotations, length):
#     if len(rotations.shape) != 2 or rotations.shape[1] != 4:
#         raise ValueError("Rotations must be a nx4 array.")
#     if len(rotations) != length:
#         raise ValueError(f"Rotations are expected to be of length {length}.")

# def check_coordinates(coordinates, shape):
#     if len(coordinates.shape) != 3 or coordinates.shape[0] != 3 or coordinates.shape[1:] != shape:
#         raise ValueError("coordinates must be 3xXxY array where X and Y are pattern dimensions.")

@type_checked(cupy.float32, cupy.float32, cupy.float32, cupy.float32, None)
def expand_model(model, slices, rotations, coordinates, interpolation=Interpolation.LINEAR):
    if len(slices) != len(rotations):
        raise ValueError("Slices and rotations must be of the same length.")
    if len(model.shape) != 3:
        raise ValueError("Model must be a 3D array.")
    if len(slices.shape) != 3:
        raise ValueError("Slices must be a 3D array.")
    if len(rotations.shape) != 2 or rotations.shape[1] != 4:
        raise ValueError("rotations must be a nx4 array.")
    if len(coordinates.shape) != 3 or coordinates.shape[0] != 3 or coordinates.shape[1:] != slices.shape[1:]:
        raise ValueError("coordinates must be 3xXxY array where X and Y are the dimensions of the slices.")

    number_of_rotations = len(rotations)
    kernels["kernel_expand_model"]((len(rotations), ), (_NTHREADS, ),
                                   (model, model.shape[2], model.shape[1], model.shape[0],
                                    slices, slices.shape[2], slices.shape[1],
                                    rotations, coordinates, interpolation.value))


@type_checked(cupy.float32, cupy.float32, cupy.float32, cupy.float32, cupy.float32, cupy.float32, None)
def insert_slices(model, model_weights, slices, slice_weights, rotations, coordinates, interpolation=Interpolation.LINEAR):
    if len(slices) != len(rotations):
        raise ValueError("slices and rotations must be of the same length.")
    if len(slices) != len(slice_weights):
        raise ValueError("slices and slice_weights must be of the same length.")
    if len(slice_weights.shape) != 1:
        raise ValueError("slice_weights must be one dimensional.")
    if len(model.shape) != 3 or model.shape != model_weights.shape:
        raise ValueError("model and model_weights must be 3D arrays of the same shape")
    if len(slices.shape) != 3:
        raise ValueError("Slices must be a 3D array.")
    if len(rotations.shape) != 2 or rotations.shape[1] != 4:
        raise ValueError("rotations must be a nx4 array.")
    if len(coordinates.shape) != 3 or coordinates.shape[0] != 3 or coordinates.shape[1:] != slices.shape[1:]:
        raise ValueError("coordinates must be 3xXxY array where X and Y are the dimensions of the slices.")

    # interpolation_int = _INTERPOLATION[interpolation]
    number_of_rotations = len(rotations)
    kernels["kernel_insert_slices"]((number_of_rotations, ), (_NTHREADS, ),
                                    (model, model_weights, model.shape[2], model.shape[1], model.shape[0],
                                     slices, slices.shape[2], slices.shape[1], slice_weights,
                                     rotations, coordinates, interpolation.value))


def update_slices(slices, patterns, responsabilities, scalings=None):
    arguments = (slices, patterns, responsabilities) + ((scalings, ) if scalings is not None else ())
    if isinstance(patterns, dict):
        # data is sparse
        if "ones_start_indices" in patterns:
            update_slices_sparser(*arguments)
        else:
            update_slices_sparse(*arguments)
    else:
        # data is dense
        if patterns.dtype == cupy.dtype("int32"):
            update_slices_dense(*arguments)
        else:
            update_slices_dense_float(*arguments)


@type_checked(cupy.float32, "dense", cupy.float32, cupy.float32)
def update_slices_dense(slices, patterns, responsabilities, scalings=None):
    if len(patterns.shape) != 3: raise ValueError("patterns must be a 3D array")
    if len(slices.shape) != 3: raise ValueError("slices must be a 3D array.")
    if patterns.shape[1:] != slices.shape[1:]: raise ValueError("patterns and images must be the same size 2D images")
    if len(responsabilities.shape) != 2 or slices.shape[0] != responsabilities.shape[0] or patterns.shape[0] != responsabilities.shape[1]:
        raise ValueError("responsabilities must have shape nrotations x npatterns")
    if scalings is not None and not (scalings.shape == responsabilities.shape or
                                     (len(scalings.shape) == 1 and scalings.shape[0] == patterns.shape[0])):
        raise ValueError("Scalings must have the same shape as responsabilities")
    number_of_rotations = len(slices)
    if scalings is None:
        kernels["kernel_update_slices<int>"]((number_of_rotations, ), (_NTHREADS, ),
                                        (slices, patterns, patterns.shape[0], patterns.shape[2]*patterns.shape[1],
                                         responsabilities))
    elif len(scalings.shape) == 2:
        # Scaling per pattern and slice pair
        kernels["kernel_update_slices_scaling"]((number_of_rotations, ), (_NTHREADS, ),
                                                (slices, patterns, patterns.shape[0], patterns.shape[2]*patterns.shape[1],
                                                 responsabilities, scalings))
    else:
        # Scaling per pattern
        kernels["kernel_update_slices_per_pattern_scaling"]((number_of_rotations, ), (_NTHREADS, ),
                                                            (slices, patterns, patterns.shape[0], patterns.shape[2]*patterns.shape[1],
                                                             responsabilities, scalings))


@type_checked(cupy.float32, "sparse", cupy.float32, cupy.float32, None)
def update_slices_sparse(slices, patterns, responsabilities, scalings=None, resp_threshold=0.):
    if len(responsabilities.shape) != 2: raise ValueError("responsabilities must have shape nrotations x npatterns")
    if len(patterns["start_indices"].shape) != 1 or patterns["start_indices"].shape[0] != responsabilities.shape[1]+1:
        raise ValueError("start_indices must be a 1d array of length one more than the number of patterns")
    if len(patterns["indices"].shape) != 1 or len(patterns["values"].shape) != 1 or patterns["indices"].shape != patterns["values"].shape:
        raise ValueError("indices and values must have the same shape")
    number_of_patterns = len(patterns["start_indices"])-1
    if len(slices.shape) != 3:
        raise ValueError("slices must be a 3d array")
    if slices.shape[0] != responsabilities.shape[0]:
        raise ValueError("Responsabilities and slices indicate different number of orientations")
    if scalings is not None and not (scalings.shape == responsabilities.shape or
                                     (len(scalings.shape) == 1 and scalings.shape[0] == number_of_patterns)):
        raise ValueError("Scalings must have the same shape as responsabilities")

    number_of_rotations = len(slices)
    number_of_pixels = slices.shape[1]*slices.shape[2] 
    if scalings is None:
        kernels["kernel_update_slices_sparse"]((number_of_rotations, ), (_NTHREADS, ),
                                               (slices, slices.shape[2]*slices.shape[1],
                                                patterns["start_indices"], patterns["indices"], patterns["values"],
                                                number_of_patterns, responsabilities, resp_threshold))
        kernels["kernel_normalize_slices"]((number_of_rotations, ), (_NTHREADS, ),
                                           (slices, responsabilities, number_of_pixels, number_of_patterns))
    elif len(scalings.shape) == 2:
        # Scaling per pattern and slice pair
        kernels["kernel_update_slices_sparse_scaling"]((number_of_rotations, ), (_NTHREADS, ),
                                                       (slices, slices.shape[2]*slices.shape[1],
                                                        patterns["start_indices"], patterns["indices"], patterns["values"],
                                                        number_of_patterns, responsabilities, resp_threshold, scalings))
        kernels["kernel_normalize_slices"]((number_of_rotations, ), (_NTHREADS, ),
                                           (slices, responsabilities, number_of_pixels, number_of_patterns))
    else:
        # Scaling per pattern
        kernels["kernel_update_slices_sparse_per_pattern_scaling"]((number_of_rotations, ), (_NTHREADS, ),
                                                                   (slices, slices.shape[2]*slices.shape[1],
                                                                    patterns["start_indices"], patterns["indices"], patterns["values"],
                                                                    number_of_patterns, responsabilities, scalings))
        kernels["kernel_normalize_slices"]((number_of_rotations, ), (_NTHREADS, ),
                                           (slices, responsabilities, number_of_pixels, number_of_patterns))


@type_checked(cupy.float32, "sparser", cupy.float32, cupy.float32, None)
def update_slices_sparser(slices, patterns, responsabilities, scalings=None, resp_threshold=0.):
    if len(responsabilities.shape) != 2: raise ValueError("responsabilities must have shape nrotations x npatterns")
    if len(patterns["start_indices"].shape) != 1 or patterns["start_indices"].shape[0] != responsabilities.shape[1]+1:
        raise ValueError("start_indices must be a 1d array of length one more than the number of patterns")
    if len(patterns["ones_start_indices"].shape) != 1 or patterns["ones_start_indices"].shape[0] != responsabilities.shape[1]+1:
        raise ValueError("ones_start_indices must be a 1d array of length one more than the number of patterns")
    if len(patterns["indices"].shape) != 1 or len(patterns["values"].shape) != 1 or patterns["indices"].shape != patterns["values"].shape:
        raise ValueError("indices and values must have the same shape")
    number_of_patterns = len(patterns["start_indices"])-1
    if len(slices.shape) != 3:
        raise ValueError("slices must be a 3d array")
    if slices.shape[0] != responsabilities.shape[0]:
        raise ValueError("Responsabilities and slices indicate different number of orientations")
    if scalings is not None and not (scalings.shape == responsabilities.shape or
                                     (len(scalings.shape) == 1 and scalings.shape[0] == number_of_patterns)):
        raise ValueError("Scalings must have the same shape as responsabilities")

    number_of_rotations = len(slices)
    number_of_pixels = slices.shape[1]*slices.shape[2] 
    if scalings is None:
        kernels["kernel_update_slices_sparser"]((number_of_rotations, ), (_NTHREADS, ),
                                                (slices, slices.shape[2]*slices.shape[1],
                                                 patterns["start_indices"], patterns["indices"], patterns["values"],
                                                 patterns["ones_start_indices"], patterns["ones_indices"],
                                                number_of_patterns, responsabilities, resp_threshold))
        kernels["kernel_normalize_slices"]((number_of_rotations, ), (_NTHREADS, ),
                                           (slices, responsabilities, number_of_pixels, number_of_patterns))
    elif len(scalings.shape) == 2:
        # Scaling per pattern and slice pair
        kernels["kernel_update_slices_sparser_scaling"]((number_of_rotations, ), (_NTHREADS, ),
                                                       (slices, slices.shape[2]*slices.shape[1],
                                                        patterns["start_indices"], patterns["indices"], patterns["values"],
                                                        patterns["ones_start_indices"], patterns["ones_indices"],
                                                        number_of_patterns, responsabilities, resp_threshold, scalings))
        kernels["kernel_normalize_slices"]((number_of_rotations, ), (_NTHREADS, ),
                                           (slices, responsabilities, number_of_pixels, number_of_patterns))
    else:
        raise NotImplementedError("Can't use per pattern scalign with sparser format.")
        # Scaling per pattern
        # kernels["kernel_update_slices_sparse_per_pattern_scaling"]((number_of_rotations, ), (_NTHREADS, ),
        #                                                            (slices, slices.shape[2]*slices.shape[1],
        #                                                             patterns["start_indices"], patterns["indices"], patterns["values"],
        #                                                             number_of_patterns, responsabilities, scalings))
        # kernels["kernel_normalize_slices"]((number_of_rotations, ), (_NTHREADS, ),
        #                                    (slices, responsabilities, number_of_pixels, number_of_patterns))

@type_checked(cupy.float32, cupy.float32, cupy.float32, cupy.float32)
def update_slices_dense_float(slices, patterns, responsabilities, scalings=None):
    if len(patterns.shape) != 3: raise ValueError("patterns must be a 3D array")
    if len(slices.shape) != 3: raise ValueError("slices must be a 3D array.")
    if patterns.shape[1:] != slices.shape[1:]: raise ValueError("patterns and images must be the same size 2D images")
    if len(responsabilities.shape) != 2 or slices.shape[0] != responsabilities.shape[0] or patterns.shape[0] != responsabilities.shape[1]:
        raise ValueError("responsabilities must have shape nrotations x npatterns")
    if scalings is not None and not (scalings.shape == responsabilities.shape or
                                     (len(scalings.shape) == 1 and scalings.shape[0] == patterns.shape[0])):
        raise ValueError("Scalings must have the same shape as responsabilities")
    number_of_rotations = len(slices)
    if scalings is None:
        kernels["kernel_update_slices<float>"]((number_of_rotations, ), (_NTHREADS, ),
                                              (slices, patterns, patterns.shape[0], patterns.shape[2]*patterns.shape[1],
                                               responsabilities))
    elif len(scalings.shape) == 2:
        # Scaling per pattern and slice pair
        kernels["kernel_update_slices_scaling_float"]((number_of_rotations, ), (_NTHREADS, ),
                                                      (slices, patterns, patterns.shape[0],
                                                       patterns.shape[2]*patterns.shape[1],
                                                       responsabilities, scalings))
    else:
        # Scaling per pattern
        kernels["kernel_update_slices_per_pattern_scaling_float"]((number_of_rotations, ), (_NTHREADS, ),
                                                            (slices, patterns, patterns.shape[0],
                                                             patterns.shape[2]*patterns.shape[1],
                                                             responsabilities, scalings))


    

def calculate_responsabilities_poisson(patterns, slices, responsabilities, scalings=None):
    arguments = (patterns, slices, responsabilities) + ((scalings, ) if scalings is not None else ())
    if isinstance(patterns, dict):
        # sparse data
        if "ones_start_indices" in patterns:
            calculate_responsabilities_poisson_sparser(*arguments)
        else:
            calculate_responsabilities_poisson_sparse(*arguments)
    else:
        # dense data
        calculate_responsabilities_poisson_dense(*arguments)

@type_checked("dense", cupy.float32, cupy.float32, cupy.float32)
def calculate_responsabilities_poisson_dense(patterns, slices, responsabilities, scalings=None):
    if len(patterns.shape) != 3: raise ValueError("patterns must be a 3D array")
    if len(slices.shape) != 3: raise ValueError("slices must be a 3D array")
    if patterns.shape[1:] != slices.shape[1:]: raise ValueError("patterns and images must be the same size 2D images")
    if len(responsabilities.shape) != 2 or slices.shape[0] != responsabilities.shape[0] or patterns.shape[0] != responsabilities.shape[1]:
        raise ValueError("responsabilities must have shape nrotations x npatterns")
    if scalings is not None and not (scalings.shape == responsabilities.shape or
                                     (len(scalings.shape) == 1 or scalings.shape[0] == patterns.shape[0])):
        raise ValueError("Scalings must have the same shape as responsabilities")
    patterns_max = patterns.max()
    number_of_patterns = len(patterns)
    number_of_rotations = len(slices)
    if scalings is None:
        # print(f"*** call correct kernel ***")
        # from mpi4py import MPI
        # rank = MPI.COMM_WORLD.Get_rank()
        # print(f"{rank}: number_of_patterns = {number_of_patterns}")
        # print(f"{rank}: number_of_rotations = {number_of_rotations}")
        # print(f"{rank}: patterns.shape = {patterns.shape}")
        # print(f"{rank}: patterns.dtype = {patterns.dtype}")
        # print(f"{rank}: patterns.c_cont = {patterns.flags['C_CONTIGUOUS']}")
        # print(f"{rank}: patterns.own = {patterns.flags['OWNDATA']}")
        # print(f"{rank}: patterns.max() = {patterns.max()}")
        # print(f"{rank}: slices.shape = {slices.shape}")
        # print(f"{rank}: slices.dtype = {slices.dtype}")
        # print(f"{rank}: slices.c_cont = {slices.flags['C_CONTIGUOUS']}")
        # print(f"{rank}: slices.own = {slices.flags['OWNDATA']}")
        # print(f"{rank}: responsabilities.shape = {responsabilities.shape}")
        # print(f"{rank}: responsabilities.dtype = {responsabilities.dtype}")
        # print(f"{rank}: responsabilities.c_cont = {responsabilities.flags['C_CONTIGUOUS']}")
        # print(f"{rank}: responsabilities.own = {responsabilities.flags['OWNDATA']}")
        # print(f"{rank}: log_factorial.table. = {responsabilities.flags['OWNDATA']}")
        # import sys
        # sys.exit()
        kernels["kernel_calculate_responsabilities_poisson"]((number_of_patterns, number_of_rotations), (_NTHREADS, ),
                                                             (patterns, slices, slices.shape[2]*slices.shape[1], responsabilities,
                                                              log_factorial.table(patterns_max)))
        # print(f"{rank}: nnans = {cupy.isnan(responsabilities).sum()}")
        # print(f"{rank}: ninfs = {cupy.isinf(responsabilities).sum()}")
        # import sys
        # sys.exit()

    elif len(scalings.shape) == 2:
        # Scaling per pattern and slice pair
        kernels["kernel_calculate_responsabilities_poisson_scaling"]((number_of_patterns, number_of_rotations), (_NTHREADS, ),
                                                                     (patterns, slices, slices.shape[2]*slices.shape[1],
                                                                      scalings, responsabilities,
                                                                      log_factorial.table(patterns_max)))
    else:
        # Scaling per pattern
        kernels["kernel_calculate_responsabilities_poisson_per_pattern_scaling"]((number_of_patterns, number_of_rotations), (_NTHREADS, ),
                                                                                 (patterns, slices, slices.shape[2]*slices.shape[1],
                                                                                  scalings, responsabilities,
                                                                                  log_factorial.table(patterns_max)))


@type_checked("sparse", cupy.float32, cupy.float32, cupy.float32)
def calculate_responsabilities_poisson_sparse(patterns, slices, responsabilities, scalings=None):
    if len(responsabilities.shape) != 2: raise ValueError("responsabilities must have shape nrotations x npatterns")
    if len(patterns["start_indices"].shape) != 1 or patterns["start_indices"].shape[0] != responsabilities.shape[1]+1:
        raise ValueError("start_indices must be a 1d array of length one more than the number of patterns")
    if len(patterns["indices"].shape) != 1 or len(patterns["values"].shape) != 1 or patterns["indices"].shape != patterns["values"].shape:
        raise ValueError("indices and values must have the same shape")
    number_of_patterns = len(patterns["start_indices"])-1
    if len(slices.shape) != 3:
        raise ValueError("slices must be a 3d array")
    if slices.shape[0] != responsabilities.shape[0]:
        raise ValueError("Responsabilities and slices indicate different number of orientations")
    if scalings is not None and not (scalings.shape == responsabilities.shape or
                                     (len(scalings.shape) == 1 or scalings.shape[0] == number_of_patterns)):
        raise ValueError("Scalings must have the same shape as responsabilities")
    
    patterns_max = patterns["values"].max()
    number_of_rotations = len(slices)
    number_of_patterns = len(patterns["start_indices"])-1
    if scalings is None:
        kernels["kernel_sum_slices"]((number_of_rotations, ), (_NTHREADS, ),
                                     (slices, slices.shape[1]*slices.shape[2], slice_sums.array(len(slices))))
        kernels["kernel_calculate_responsabilities_sparse"]((number_of_patterns, number_of_rotations), (_NTHREADS, ),
                                                            (patterns["start_indices"], patterns["indices"], patterns["values"],
                                                             slices, slices.shape[2]*slices.shape[1], responsabilities,
                                                             slice_sums.array(), log_factorial.table(patterns_max)))
    elif len(scalings.shape) == 2:
        kernels["kernel_sum_slices"]((number_of_rotations, ), (_NTHREADS, ),
                                     (slices, slices.shape[1]*slices.shape[2], slice_sums.array(len(slices))))
        kernels["kernel_calculate_responsabilities_sparse_scaling"]((number_of_patterns, number_of_rotations), (_NTHREADS, ),
                                                                    (patterns["start_indices"], patterns["indices"], patterns["values"],
                                                                     slices, slices.shape[2]*slices.shape[1],
                                                                     scalings, responsabilities,
                                                                     slice_sums.array(), log_factorial.table(patterns_max)))
    else:
        kernels["kernel_sum_slices"]((number_of_rotations, ), (_NTHREADS, ),
                                     (slices, slices.shape[1]*slices.shape[2], slice_sums.array(len(slice))))
        kernels["kernel_calculate_responsabilities_sparse_per_pattern_scaling"]((number_of_patterns, number_of_rotations), (_NTHREADS, ),
                                                                                (patterns["start_indices"], patterns["indices"],
                                                                                 patterns["values"],
                                                                                 slices, slices.shape[2]*slices.shape[1], scalings,
                                                                                 responsabilities,
                                                                                 slice_sums.array(), log_factorial.table(patterns_max)))


@type_checked("sparser", cupy.float32, cupy.float32, cupy.float32)
def calculate_responsabilities_poisson_sparser(patterns, slices, responsabilities, scalings=None):
    if len(responsabilities.shape) != 2: raise ValueError("responsabilities must have shape nrotations x npatterns")
    if len(patterns["start_indices"].shape) != 1 or patterns["start_indices"].shape[0] != responsabilities.shape[1]+1:
        raise ValueError("start_indices must be a 1d array of length one more than the number of patterns")
    if len(patterns["ones_start_indices"].shape) != 1 or patterns["ones_start_indices"].shape[0] != responsabilities.shape[1]+1:
        raise ValueError("ones_start_indices must be a 1d array of length one more than the number of patterns")
    if len(patterns["indices"].shape) != 1 or len(patterns["values"].shape) != 1 or patterns["indices"].shape != patterns["values"].shape:
        raise ValueError("indices and values must have the same shape")
    number_of_patterns = len(patterns["start_indices"])-1
    if len(slices.shape) != 3:
        raise ValueError("slices must be a 3d array")
    if slices.shape[0] != responsabilities.shape[0]:
        raise ValueError("Responsabilities and slices indicate different number of orientations")
    if scalings is not None and not (scalings.shape == responsabilities.shape or
                                     (len(scalings.shape) == 1 or scalings.shape[0] == number_of_patterns)):
        raise ValueError("Scalings must have the same shape as responsabilities")
    
    patterns_max = patterns["values"].max()
    number_of_rotations = len(slices)
    number_of_patterns = len(patterns["start_indices"])-1
    if scalings is None:
        kernels["kernel_sum_slices"]((number_of_rotations, ), (_NTHREADS, ),
                                     (slices, slices.shape[1]*slices.shape[2], slice_sums.array(len(slices))))
        kernels["kernel_calculate_responsabilities_sparser"]((number_of_patterns, number_of_rotations), (_NTHREADS, ),
                                                             (patterns["start_indices"], patterns["indices"], patterns["values"],
                                                              patterns["ones_start_indices"], patterns["ones_indices"],
                                                              slices, slices.shape[2]*slices.shape[1], responsabilities,
                                                              slice_sums.array(), log_factorial.table(patterns_max)))
    elif len(scalings.shape) == 2:
        kernels["kernel_sum_slices"]((number_of_rotations, ), (_NTHREADS, ),
                                     (slices, slices.shape[1]*slices.shape[2], slice_sums.array(len(slices))))
        kernels["kernel_calculate_responsabilities_sparser_scaling"]((number_of_patterns, number_of_rotations), (_NTHREADS, ),
                                                                     (patterns["start_indices"], patterns["indices"], patterns["values"],
                                                                      patterns["ones_start_indices"], patterns["ones_indices"],
                                                                      slices, slices.shape[2]*slices.shape[1],
                                                                      scalings, responsabilities,
                                                                      slice_sums.array(), log_factorial.table(patterns_max)))
    else:
        raise NotImplementedError("Can't use per pattern scaling together with sparser format.")
        # kernels["kernel_sum_slices"]((number_of_rotations, ), (_NTHREADS, ),
        #                              (slices, slices.shape[1]*slices.shape[2], calculate_responsabilities_poisson_sparser.slice_sums))
        # kernels["kernel_calculate_responsabilities_sparse_per_pattern_scaling"]((number_of_patterns, number_of_rotations), (_NTHREADS, ),
        #                                                                         (patterns["start_indices"], patterns["indices"],
        #                                                                          patterns["values"],
        #                                                                          slices, slices.shape[2]*slices.shape[1], scalings,
        #                                                                          responsabilities,
        #                                                                          calculate_responsabilities_poisson_sparser.slice_sums,
        #                                                                          calculate_responsabilities_poisson_sparser.log_factorial_table))


def calculate_scaling_poisson(patterns, slices, scaling):
    if isinstance(patterns, dict):
        # patterns are spares
        if "ones_start_indices" in patterns:
            calculate_scaling_poisson_sparser(patterns, slices, scaling)
        else:
            calculate_scaling_poisson_sparse(patterns, slices, scaling)
    else:
        calculate_scaling_poisson_dense(patterns, slices, scaling)


@type_checked("dense", cupy.float32, cupy.float32)
def calculate_scaling_poisson_dense(patterns, slices, scaling):
    if len(patterns.shape) != 3:
        raise ValueError("Patterns must be a 3D array")
    if len(slices.shape) != 3:
        raise ValueError("Slices must be a 3D array")
    if len(scaling.shape) != 2:
        raise ValueError("Slices must be a 2D array")
    if slices.shape[1:] != patterns.shape[1:]:
        raise ValueError("Slices and patterns must be the same shape")
    if scaling.shape[0] != slices.shape[0] or scaling.shape[1] != patterns.shape[0]:
        raise ValueError("scaling must have shape nrotations x npatterns")
    number_of_patterns = len(patterns)
    number_of_rotations = len(slices)
    kernels["kernel_calculate_scaling_poisson"]((number_of_patterns, number_of_rotations), (_NTHREADS, ),
                                                (patterns, slices, scaling, slices.shape[0]*slices.shape[1]))


@type_checked("sparse", cupy.float32, cupy.float32)
def calculate_scaling_poisson_sparse(patterns, slices, scaling):
    if len(patterns["start_indices"].shape) != 1 or patterns["start_indices"].shape[0] != scaling.shape[1]+1:
        raise ValueError("start_indices must be a 1d array of length one more than the number of patterns")
    if len(patterns["indices"].shape) != 1 or len(patterns["values"].shape) != 1 or patterns["indices"].shape != patterns["values"].shape:
        raise ValueError("indices and values must have the same shape")
    if len(slices.shape) != 3:
        raise ValueError("Slices must be a 3D array")
    if len(scaling.shape) != 2:
        raise ValueError("Slices must be a 2D array")
    number_of_patterns = len(patterns["start_indices"])-1
    if scaling.shape[0] != slices.shape[0] or scaling.shape[1] != number_of_patterns:
        raise ValueError("scaling must have shape nrotations x npatterns")
    number_of_rotations = len(slices)
    kernels["kernel_calculate_scaling_poisson_sparse"]((number_of_patterns, number_of_rotations), (_NTHREADS, ),
                                                       (patterns["start_indices"], patterns["indices"], patterns["values"],
                                                        slices, scaling, slices.shape[1]*slices.shape[2]))


@type_checked("sparser", cupy.float32, cupy.float32)
def calculate_scaling_poisson_sparser(patterns, slices, scaling):
    if len(patterns["start_indices"].shape) != 1 or patterns["start_indices"].shape[0] != scaling.shape[1]+1:
        raise ValueError("start_indices must be a 1d array of length one more than the number of patterns")
    if len(patterns["ones_start_indices"].shape) != 1 or patterns["ones_start_indices"].shape[0] != scaling.shape[1]+1:
        raise ValueError("ones_start_indices must be a 1d array of length one more than the number of patterns")
    if len(patterns["indices"].shape) != 1 or len(patterns["values"].shape) != 1 or patterns["indices"].shape != patterns["values"].shape:
        raise ValueError("indices and values must have the same shape")
    if len(slices.shape) != 3:
        raise ValueError("Slices must be a 3D array")
    if len(scaling.shape) != 2:
        raise ValueError("Slices must be a 2D array")
    number_of_patterns = len(patterns["start_indices"])-1
    if scaling.shape[0] != slices.shape[0] or scaling.shape[1] != number_of_patterns:
        raise ValueError("scaling must have shape nrotations x npatterns")
    number_of_rotations = len(slices)
    kernels["kernel_calculate_scaling_poisson_sparser"]((number_of_patterns, number_of_rotations), (_NTHREADS, ),
                                                       (patterns["start_indices"], patterns["indices"], patterns["values"],
                                                        patterns["ones_start_indices"], patterns["ones_indices"],
                                                        slices, scaling, slices.shape[1]*slices.shape[2]))


def calculate_scaling_per_pattern_poisson(patterns, slices, scaling):
    if isinstance(patterns, dict):
        # patterns are spares
        if "ones_start_indices" in patterns:
            raise NotImplementedError("Can't use spraseR format with per pattern scaling.")
        else:
            calculate_scaling_per_pattern_poisson_sparse(patterns, slices, scaling)
    else:
        calculate_scaling_per_pattern_poisson_dense(patterns, slices, scaling)

                                     
@type_checked("dense", cupy.float32, cupy.float32, cupy.float32)    
def calculate_scaling_per_pattern_poisson_dense(patterns, slices, responsabilities, scaling):
    if len(patterns.shape) != 3:
        raise ValueError("Patterns must be a 3D array")
    if len(slices.shape) != 3:
        raise ValueError("Slices must be a 3D array")
    if len(scaling.shape) != 1:
        raise ValueError("Slices must be a 1D array")
    if len(responsabilities.shape) != 2:
        raise ValueError("Slices must be a 2D array")
    if slices.shape[1:] != patterns.shape[1:]:
        raise ValueError("Slices and patterns must be the same shape")
    if scaling.shape[0] != patterns.shape[0]:
        raise ValueError("scaling must have same length as patterns")
    if slices.shape[0] != responsabilities.shape[0] or patterns.shape[0] != responsabilities.shape[1]:
        raise ValueError("Responsabilities must have shape nrotations x npatterns")
    number_of_patterns = len(patterns)
    number_of_rotations = len(slices)
    kernels["kernel_calculate_scaling_per_pattern_poisson"]((number_of_patterns, ), (_NTHREADS, ),
                                                            (patterns, slices, responsabilities, scaling,
                                                             slices.shape[1]*slices.shape[2], number_of_rotations))


@type_checked("sparse", cupy.float32, cupy.float32)    
def calculate_scaling_per_pattern_poisson_sparse(patterns, slices, scaling):
    if len(patterns["start_indices"].shape) != 1 or patterns["start_indices"].shape[0] != scaling.shape[1]+1:
        raise ValueError("start_indices must be a 1d array of length one more than the number of patterns")
    if len(patterns["indices"].shape) != 1 or len(patterns["values"].shape) != 1 or patterns["indices"].shape != patterns["values"].shape:
        raise ValueError("indices and values must have the same shape")
    if len(slices.shape) != 3:
        raise ValueError("Slices must be a 3D array")
    if len(scaling.shape) != 1:
        raise ValueError("Slices must be a 1D array")
    number_of_patterns = len(patterns["start_indices"])-1
    if scaling.shape[0] != number_of_patterns:
        raise ValueError("scaling must have same length as patterns")
    if slices.shape[0] != responsabilities.shape[0] or number_of_patterns != responsabilities.shape[1]:
        raise ValueError("Responsabilities must have shape nrotations x npatterns")
    number_of_patterns = len(patterns["start_indices"]) - 1
    number_of_rotations = len(slices)
    kernels["kernel_calculate_scaling_per_pattern_poisson_sparse"]((number_of_patterns, ), (_NTHREADS, ),
                                                                   (patterns["start_indices"], patterns["indices"], patterns["values"],
                                                                    slices, responsabilities, scaling, slices.shape[1]*slices.shape[2],
                                                                    number_of_rotations))


@type_checked(cupy.float32, cupy.float32, cupy.float32)
def expand_model_2d(model, slices, rotations):
    if len(slices) != len(rotations):
        raise ValueError("Slices and rotations must be of the same length.")
    if len(model.shape) != 2:
        raise ValueError("Model must be a 2D array.")
    if len(slices.shape) != 3:
        raise ValueError("Slices must be a 3D array.")
    if len(rotations.shape) != 1:
        raise ValueError("rotations must be a 1D array.")

    number_of_rotations = len(rotations)
    kernels["kernel_expand_model_2d"]((number_of_rotations, ), (_NTHREADS, ),
                                      (model, model.shape[0], model.shape[1],
                                       slices, slices.shape[1], slices.shape[2],
                                       rotations))


@type_checked(cupy.float32, cupy.float32, cupy.float32, cupy.float32, cupy.float32, None)    
def insert_slices_2d(model, model_weights, slices, slice_weights, rotations, interpolation=Interpolation.LINEAR):
    if len(slices) != len(rotations):
        raise ValueError("slices and rotations must be of the same length.")
    if len(slices) != len(slice_weights):
        raise ValueError("slices and slice_weights must be of the same length.")
    if len(slice_weights.shape) != 1:
        raise ValueError("slice_weights must be one dimensional.")
    if len(model.shape) != 2 or model.shape != model_weights.shape:
        raise ValueError("model and model_weights must be 2D arrays of the same shape")
    if len(slices.shape) != 3:
        raise ValueError("Slices must be a 3D array.")
    if len(rotations.shape) != 1:
        raise ValueError("Rotations must be a 1D array.")

    # interpolation_int = _INTERPOLATION[interpolation]
    number_of_rotations = len(rotations)
    kernels["kernel_insert_slices_2d"]((number_of_rotations, ), (_NTHREADS, ),
                                       (model, model_weights, model.shape[0], model.shape[1],
                                        slices, slices.shape[1], slices.shape[2], slice_weights,
                                        rotations, interpolation.value))


@type_checked(None, cupy.float32, cupy.float32, None)
def assemble_model(patterns, rotations, coordinates, shape=None):
    slice_weights = cupy.ones(len(rotations), dtype="float32")

    if isinstance(patterns, dict):
        raise NotImplementedError("assemble_model does not support sparse data")
    
    # patterns = cupy.asarray(patterns, dtype="float32")
    patterns = cupy.asarray(patterns, dtype="float32")
    rotations = cupy.asarray(rotations, dtype="float32")

    if shape is None:
        shape = ((patterns.shape[1] + patterns.shape[2])//2, )*3
    model = cupy.zeros(shape, dtype="float32")
    model_weights = cupy.zeros(shape, dtype="float32")

    insert_slices(model, model_weights, patterns,
                  slice_weights, rotations, coordinates)
    
    bad_indices = model_weights == 0
    model /= model_weights
    model[bad_indices] = -1

    return model

@type_checked(cupy.float32, None, None)
def blur_model(model, sigma, cutoff):
    tmp_model = cupy.zeros_like(model)
    kernels["kernel_blur_model"]( (model.shape[0], model.shape[1]), (model.shape[2], ),
                                  (tmp_model, model, cupy.float32(sigma), cutoff))
    model[:] = tmp_model[:]
    del tmp_model

@type_checked(cupy.float32, cupy.float32)
def rotate_model(model, rotation):
    if rotation.shape != (4, ):
        raise ValueError("Rotation must be a length-4 array (quaternion)")
    return_model = cupy.zeros_like(model)
    nblocks = (model.shape[0]*model.shape[1]*model.shape[2]-1) // _NTHREADS + 1
    kernels["kernel_rotate_model"]((nblocks, ), (_NTHREADS, ),
                                   (model, return_model,
                                    model.shape[0], model.shape[1], model.shape[2],
                                    rotation))
    return return_model

@type_checked(cupy.float32, cupy.float32, cupy.float32, None)
def calculate_responsabilities_gaussian_dense(patterns, slices, responsabilities):
    if len(patterns.shape) != 3: raise ValueError("patterns must be a 3D array")
    if len(slices.shape) != 3: raise ValueError("slices must be a 3D array")
    if patterns.shape[1:] != slices.shape[1:]: raise ValueError("patterns and images must be the same size 2D images")
    if len(responsabilities.shape) != 2 or slices.shape[0] != responsabilities.shape[0] or patterns.shape[0] != responsabilities.shape[1]:
        raise ValueError("responsabilities must have shape nrotations x npatterns")
    patterns_max = patterns.max()
    number_of_patterns = len(patterns)
    number_of_rotations = len(slices)
    kernels["kernel_calculate_responsabilities_gaussian"]((number_of_patterns, number_of_rotations), (_NTHREADS, ),
                                                          (patterns, slices, slices.shape[2]*slices.shape[1],
                                                           responsabilities))

