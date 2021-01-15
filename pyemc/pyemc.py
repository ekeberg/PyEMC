import cupy
import numpy
import os
from eke import rotmodule

_NTHREADS = 128
MAX_PHOTON_COUNT = 200000
_INTERPOLATION = {"nearest_neighbour": 0,
                  "linear": 1}

def ewald_coordinates(image_shape, wavelength, detector_distance, pixel_size, edge_distance=None):
    if edge_distance is None:
        edge_distance = image_shape[0]/2.
    x_pixels_1d = numpy.arange(image_shape[1], dtype="float32") - image_shape[1]/2. + 0.5
    y_pixels_1d = numpy.arange(image_shape[0], dtype="float32") - image_shape[0]/2. + 0.5
    y_pixels, x_pixels = numpy.meshgrid(y_pixels_1d, x_pixels_1d, indexing="ij")
    x_meters = x_pixels*pixel_size
    y_meters = y_pixels*pixel_size
    radius_meters = numpy.sqrt(x_meters**2 + y_meters**2)

    scattering_angle = numpy.arctan(radius_meters / detector_distance)
    z = -1./wavelength*(1. - numpy.cos(scattering_angle))
    radius_fourier = numpy.sqrt(1./wavelength**2 - (1./wavelength - abs(z))**2)

    x = x_meters * radius_fourier / radius_meters
    y = y_meters * radius_fourier / radius_meters

    x[radius_meters == 0.] = 0.
    y[radius_meters == 0.] = 0.

    output_coordinates = numpy.zeros((3, ) + image_shape)
    output_coordinates[0, :, :] = x
    output_coordinates[1, :, :] = y
    output_coordinates[2, :, :] = z

    # Rescale so that edge pixels match.
    furthest_edge_coordinate = numpy.sqrt(x[0, image_shape[1]//2]**2 + y[0, image_shape[1]//2]**2 + z[0, image_shape[1]//2]**2)
    rescale_factor = edge_distance/furthest_edge_coordinate
    output_coordinates *= rescale_factor
    
    return cupy.asarray(output_coordinates, dtype="float32")

def _log_factorial_table(max_value):
    if max_value > MAX_PHOTON_COUNT:
        raise ValueError("Poisson values can not be used with photon counts higher than {0}".format(MAX_PHOTON_COUNT))
    log_factorial_table = numpy.zeros(int(max_value+1), dtype="float32")
    log_factorial_table[0] = 0.
    for i in range(1, int(max_value+1)):
        log_factorial_table[i] = log_factorial_table[i-1] + numpy.log(i)
    return cupy.asarray(log_factorial_table, dtype="float32")

def chunks(number_of_rotations, chunk_size):
    """Generator for slices to chunk up the data"""
    chunk_starts = numpy.arange(0, number_of_rotations, chunk_size)
    chunk_ends = chunk_starts + chunk_size
    chunk_ends[-1] = number_of_rotations
    chunk_sizes = chunk_ends - chunk_starts
    indices_cpu = [slice(this_chunk_start, this_chunk_end) for this_chunk_start, this_chunk_end
                    in zip(chunk_starts, chunk_ends)]
    indices_gpu = [slice(None, this_chunk_end-this_chunk_start) for this_chunk_start, this_chunk_end
                     in zip(chunk_starts, chunk_ends)]
    for this_indices_cpu, this_indices_gpu in zip(indices_cpu, indices_gpu):
        yield this_indices_cpu, this_indices_gpu

def radial_average(image, mask=None):
    """Calculates the radial average of an array of any shape,
    the center is assumed to be at the physical center."""
    if mask is None:
        mask = numpy.ones(image.shape, dtype='bool8')
    else:
        mask = numpy.bool8(mask)
    axis_values = [numpy.arange(l) - l/2. + 0.5 for l in image.shape]
    radius = numpy.zeros((image.shape[-1]))
    for i in range(len(image.shape)):
        radius = radius + (axis_values[-(1+i)][(slice(0, None), ) + (numpy.newaxis, )*i])**2
    radius = numpy.int32(numpy.sqrt(radius))
    number_of_bins = radius[mask].max() + 1
    radial_sum = numpy.zeros(number_of_bins)
    weight = numpy.zeros(number_of_bins)
    for value, this_radius in zip(image[mask], radius[mask]):
        radial_sum[this_radius] += value
        weight[this_radius] += 1.
    radial_sum[weight > 0] /= weight[weight > 0]
    radial_sum[weight == 0] = numpy.nan
    return radial_sum        

def import_cuda_file(file_name, kernel_names):
    # nthreads = 128
    threads_code = f"const int NTHREADS = {_NTHREADS};"
    cuda_files_dir = os.path.join(os.path.split(__file__)[0], "cuda")
    header_file = "header.cu"
    with open(os.path.join(cuda_files_dir, header_file), "r") as file_handle:
        header_source = file_handle.read()
    with open(os.path.join(cuda_files_dir, file_name), "r") as file_handle:
        main_source = file_handle.read()
    combined_source = "\n".join((header_source, threads_code, main_source))
    # print(combined_source)
    # import sys; sys.exit()
    module = cupy.RawModule(code=combined_source)
    kernels = {}
    for this_name in kernel_names:
        kernels[this_name] = module.get_function(this_name)
    return kernels

def import_kernels():
    emc_kernels = import_cuda_file("emc_cuda.cu",
                                   ["kernel_expand_model",
                                    "kernel_insert_slices"])
    respons_kernels = import_cuda_file("calculate_responsabilities_cuda.cu",
                                       ["kernel_sum_slices",
                                        "kernel_calculate_responsabilities_poisson",
                                        "kernel_calculate_responsabilities_poisson_scaling",
                                        "kernel_calculate_responsabilities_poisson_per_pattern_scaling",
                                        "kernel_calculate_responsabilities_sparse",
                                        "kernel_calculate_responsabilities_sparse_scaling",
                                        "kernel_calculate_responsabilities_sparse_per_pattern_scaling"])
    scaling_kernels = import_cuda_file("calculate_scaling_cuda.cu",
                                       ["kernel_calculate_scaling_poisson",
                                        "kernel_calculate_scaling_poisson_sparse",
                                        "kernel_calculate_scaling_per_pattern_poisson",
                                        "kernel_calculate_scaling_per_pattern_poisson_sparse"])
    slices_kernels = import_cuda_file("update_slices_cuda.cu",
                                      ["kernel_normalize_slices",
                                       "kernel_update_slices",
                                       "kernel_update_slices_scaling",
                                       "kernel_update_slices_per_pattern_scaling",
                                       "kernel_update_slices_sparse",
                                       "kernel_update_slices_sparse_scaling",
                                       "kernel_update_slices_sparse_per_pattern_scaling"])
    kernels = {**emc_kernels, **respons_kernels, **scaling_kernels, **slices_kernels}
    return kernels

def set_nthreads(nthreads):
    global _NTHREADS, kernels
    _NTHREADS = nthreads
    kernels = import_kernels()

kernels = import_kernels()
    
def expand_model(model, slices, rotations, coordinates):
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
                                    rotations, coordinates))

    
def insert_slices(model, model_weights, slices, slice_weights, rotations, coordinates, interpolation="linear"):
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

    interpolation_int = _INTERPOLATION[interpolation]
    number_of_rotations = len(rotations)
    kernels["kernel_insert_slices"]((number_of_rotations, ), (_NTHREADS, ),
                                    (model, model_weights, model.shape[2], model.shape[1], model.shape[0],
                                     slices, slices.shape[2], slices.shape[1], slice_weights,
                                     rotations, coordinates, interpolation_int))


def update_slices(slices, patterns, responsabilities, scalings=None):
    if isinstance(patterns, dict):
        # data is sparse
        update_slices_sparse(slices, patterns, responsabilities, scalings)
    else:
        # data is dense
        update_slices_dense(slices, patterns, responsabilities, scalings)


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
        kernels["kernel_update_slices"]((number_of_rotations, ), (_NTHREADS, ),
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


def update_slices_sparse(slices, patterns, responsabilities, scalings=None, resp_threshold=0.):
    if (not "indices" in patterns or
        not "values" in patterns or
        not "start_indices" in patterns):
        raise ValueError("patterns must contain the keys indcies, values and start_indices")
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

        
def calculate_responsabilities_poisson(patterns, slices, responsabilities, scalings=None):
    if isinstance(patterns, dict):
        # sparse data
        calculate_responsabilities_poisson_sparse(patterns, slices, responsabilities, scalings)
    else:
        # dense data
        calculate_responsabilities_poisson_dense(patterns, slices, responsabilities, scalings)
        
def calculate_responsabilities_poisson_dense(patterns, slices, responsabilities, scalings=None):
    if len(patterns.shape) != 3: raise ValueError("patterns must be a 3D array")
    if len(slices.shape) != 3: raise ValueError("slices must be a 3D array")
    if patterns.shape[1:] != slices.shape[1:]: raise ValueError("patterns and images must be the same size 2D images")
    if len(responsabilities.shape) != 2 or slices.shape[0] != responsabilities.shape[0] or patterns.shape[0] != responsabilities.shape[1]:
        raise ValueError("responsabilities must have shape nrotations x npatterns")
    if (calculate_responsabilities_poisson_dense.log_factorial_table is None or
        len(calculate_responsabilities_poisson_dense.log_factorial_table) <= patterns.max()):
        calculate_responsabilities_poisson_dense.log_factorial_table = _log_factorial_table(patterns.max())
    if scalings is not None and not (scalings.shape == responsabilities.shape or
                                     (len(scalings.shape) == 1 or scalings.shape[0] == patterns.shape[0])):
        raise ValueError("Scalings must have the same shape as responsabilities")
    number_of_patterns = len(patterns)
    number_of_rotations = len(slices)
    if scalings is None:
        kernels["kernel_calculate_responsabilities_poisson"]((number_of_patterns, number_of_rotations), (_NTHREADS, ),
                                                             (patterns, slices, slices.shape[2]*slices.shape[1], responsabilities,
                                                              calculate_responsabilities_poisson_dense.log_factorial_table))
    elif len(scalings.shape) == 2:
        # Scaling per pattern and slice pair
        kernels["kernel_calculate_responsabilities_poisson_scaling"]((number_of_patterns, number_of_rotations), (_NTHREADS, ),
                                                                     (patterns, slices, slices.shape[2]*slices.shape[1],
                                                                      scalings, responsabilities,
                                                                      calculate_responsabilities_poisson_dense.log_factorial_table))
    else:
        # Scaling per pattern
        kernels["kernel_calculate_responsabilities_poisson_per_pattern_scaling"]((number_of_patterns, number_of_rotations), (_NTHREADS, ),
                                                                                 (patterns, slices, slices.shape[2]*slices.shape[1],
                                                                                  scalings, responsabilities,
                                                                                  calculate_responsabilities_poisson_dense.log_factorial_table))
calculate_responsabilities_poisson_dense.log_factorial_table = None


def calculate_responsabilities_poisson_sparse(patterns, slices, responsabilities, scalings=None):
    if not isinstance(patterns, dict):
        raise ValueError("patterns must be a dictionary containing the keys: indcies, values and start_indices")
    if ("indices" not in patterns or
        "values" not in patterns or
        "start_indices" not in patterns):
        raise ValueError("patterns must contain the keys indcies, values and start_indices")
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
    
    if (calculate_responsabilities_poisson_sparse.log_factorial_table is None or
        len(calculate_responsabilities_poisson_sparse.log_factorial_table) <= patterns["values"].max()):
        calculate_responsabilities_poisson_sparse.log_factorial_table = _log_factorial_table(patterns["values"].max())
    
    if (calculate_responsabilities_poisson_sparse.slice_sums is None or
        len(calculate_responsabilities_poisson_sparse.slice_sums) != len(slices)):
        calculate_responsabilities_poisson_sparse.slice_sums = cupy.empty(len(slices), dtype="float32")

    number_of_rotations = len(slices)
    number_of_patterns = len(patterns["start_indices"])-1
    if scalings is None:
        kernels["kernel_sum_slices"]((number_of_rotations, ), (_NTHREADS, ),
                                     (slices, slices.shape[1]*slices.shape[2], calculate_responsabilities_poisson_sparse.slice_sums))
        kernels["kernel_calculate_responsabilities_sparse"]((number_of_patterns, number_of_rotations), (_NTHREADS, ),
                                                            (patterns["start_indices"], patterns["indices"], patterns["values"],
                                                             slices, slices.shape[2]*slices.shape[1], responsabilities,
                                                             calculate_responsabilities_poisson_sparse.slice_sums,
                                                             calculate_responsabilities_poisson_sparse.log_factorial_table))
    elif len(scalings.shape) == 2:
        kernels["kernel_sum_slices"]((number_of_rotations, ), (_NTHREADS, ),
                                     (slices, slices.shape[1]*slices.shape[2], calculate_responsabilities_poisson_sparse.slice_sums))
        kernels["kernel_calculate_responsabilities_sparse_scaling"]((number_of_patterns, number_of_rotations), (_NTHREADS, ),
                                                                    (patterns["start_indices"], patterns["indices"], patterns["values"],
                                                                     slices, slices.shape[2]*slices.shape[1],
                                                                     scalings, responsabilities,
                                                                     calculate_responsabilities_poisson_sparse.slice_sums,
                                                                     calculate_responsabilities_poisson_sparse.log_factorial_table))
    else:
        kernels["kernel_sum_slices"]((number_of_rotations, ), (_NTHREADS, ),
                                     (slices, slices.shape[1]*slices.shape[2], calculate_responsabilities_poisson_sparse.slice_sums))
        kernels["kernel_calculate_responsabilities_sparse_per_pattern_scaling"]((number_of_patterns, number_of_rotations), (_NTHREADS, ),
                                                                                (patterns["start_indices"], patterns["indices"],
                                                                                 patterns["values"],
                                                                                 slices, slices.shape[2]*slices.shape[1], scalings,
                                                                                 responsabilities,
                                                                                 calculate_responsabilities_poisson_sparse.slice_sums,
                                                                                 calculate_responsabilities_poisson_sparse.log_factorial_table))
calculate_responsabilities_poisson_sparse.log_factorial_table = None
calculate_responsabilities_poisson_sparse.slice_sums = None


def calculate_scaling_poisson(patterns, slices, scaling):
    if isinstance(patterns, dict):
        # patterns are spares
        calculate_scaling_poisson_sparse(patterns, slices, scaling)
    else:
        calculate_scaling_poisson_dense(patterns, slices, scaling)

        
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


def calculate_scaling_poisson_sparse(patterns, slices, scaling):
    if not isinstance(patterns, dict):
        raise ValueError("patterns must be a dictionary containing the keys: indcies, values and start_indices")
    if ("indices" not in patterns or
        "values" not in patterns or
        "start_indices" not in patterns):
        raise ValueError("patterns must contain the keys indcies, values and start_indices")
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


def calculate_scaling_per_pattern_poisson(patterns, slices, scaling):
    if isinstance(patterns, dict):
        # patterns are spares
        calculate_scaling_per_pattern_poisson_sparse(patterns, slices, scaling)
    else:
        calculate_scaling_per_pattern_poisson_dense(patterns, slices, scaling)

    
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

    
def calculate_scaling_per_pattern_poisson_sparse(patterns, slices, scaling):
    if not isinstance(patterns, dict):
        raise ValueError("patterns must be a dictionary containing the keys: indcies, values and start_indices")
    if ("indices" not in patterns or
        "values" not in patterns or
        "start_indices" not in patterns):
        raise ValueError("patterns must contain the keys indcies, values and start_indices")
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
