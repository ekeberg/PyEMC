import cupy
import h5py
import numpy
import warnings


def ewald_coordinates(image_shape, wavelength, detector_distance, pixel_size,
                      edge_distance=None, output_type="cupy"):
    if edge_distance is None:
        edge_distance = image_shape[0]/2.
    x_pixels_1d = (numpy.arange(image_shape[1], dtype="float64") -
                   image_shape[1]/2. + 0.5)
    y_pixels_1d = (numpy.arange(image_shape[0], dtype="float64") -
                   image_shape[0]/2. + 0.5)
    y_pixels, x_pixels = numpy.meshgrid(y_pixels_1d, x_pixels_1d,
                                        indexing="ij")
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

    output_coordinates = numpy.zeros((3, ) + image_shape, dtype="float32")
    output_coordinates[0, :, :] = numpy.float32(x)
    output_coordinates[1, :, :] = numpy.float32(y)
    output_coordinates[2, :, :] = numpy.float32(z)

    # Rescale so that edge pixels match.
    furthest_edge_coordinate = numpy.sqrt(x[0, image_shape[1]//2]**2 +
                                          y[0, image_shape[1]//2]**2 +
                                          z[0, image_shape[1]//2]**2)
    rescale_factor = edge_distance/furthest_edge_coordinate
    output_coordinates *= rescale_factor

    if output_type.lower() == "numpy":
        output_module = numpy
    elif output_type.lower() == "cupy":
        if not cupy.cuda.is_available():
            warnings.warn("in function ewald_coordinates: Trying to use "
                          "output_type cupy with no available CUDA devices. "
                          "Reverting to numpy")
            output_module = numpy
        else:
            output_module = cupy

    return output_module.asarray(output_coordinates, dtype="float32")


def read_sparse_data(file_name, file_key=None, start_index=0, end_index=-1,
                     output_type="numpy"):
    with h5py.File(file_name, "r") as file_handle:
        if file_key is None:
            group = file_handle
        else:
            group = file_handle[file_key]
        all_start_indices = group["start_indices"][...]
        value_start_index = all_start_indices[start_index]
        if end_index == len(all_start_indices)-1 or end_index == -1:
            value_end_index = -1
        else:
            value_end_index = all_start_indices[end_index]
        # if end_index == -1:
        #     end_index = len(all_start_indices)-1

        if output_type.lower() == "numpy":
            output_module = numpy
        elif output_type.lower() == "cupy":
            output_module = cupy
        else:
            raise ValueError(f"Argument output_array must be either numpy "
                             f"or cupy. Can't recognize: {output_type}")

        start_indices = (all_start_indices[start_index:end_index+1] -
                         all_start_indices[start_index])
        indices = group["indices"][value_start_index:value_end_index]
        values = group["values"][value_start_index:value_end_index]

        patterns = {
            "start_indices": output_module.asarray(start_indices,
                                                   dtype="int32"),
            "indices": output_module.asarray(indices,
                                             dtype="int32"),
            "values": output_module.asarray(values,
                                            dtype="int32"),
            "shape": tuple(group["shape"][...])}
        return patterns


def read_sparser_data(file_name, file_key=None, start_index=0, end_index=-1,
                      output_type="numpy"):
    with h5py.File(file_name, "r") as file_handle:
        if file_key is None:
            group = file_handle
        else:
            group = file_handle[file_key]

        all_start_indices = group["start_indices"][...]
        all_ones_start_indices = group["ones_start_indices"][...]
        value_start_index = all_start_indices[start_index]
        ones_start_index = all_ones_start_indices[start_index]
        if end_index == len(all_start_indices)-1 or end_index == -1:
            value_end_index = -1
            ones_end_index = -1
        else:
            value_end_index = all_start_indices[end_index]
            ones_end_index = all_ones_start_indices[end_index]

        if output_type.lower() == "numpy":
            output_module = numpy
        elif output_type.lower() == "cupy":
            output_module = cupy
        else:
            raise ValueError(f"Argument output_array must be either numpy "
                             f"or cupy. Can't recognize: {output_type}")

        start_indices = (all_start_indices[start_index:end_index+1] -
                         all_start_indices[start_index])
        indices = group["indices"][value_start_index:value_end_index]
        values = group["values"][value_start_index:value_end_index]
        ones_start_indices = (all_ones_start_indices[start_index:end_index+1] -
                              all_ones_start_indices[start_index])
        ones_indices = group["ones_indices"][ones_start_index:ones_end_index]

        patterns = {
            "start_indices": output_module.asarray(start_indices,
                                                   dtype="int32"),
            "indices": output_module.asarray(indices,
                                             dtype="int32"),
            "values": output_module.asarray(values,
                                            dtype="int32"),
            "ones_start_indices": output_module.asarray(ones_start_indices,
                                                        dtype="int32"),
            "ones_indices": output_module.asarray(ones_indices,
                                                  dtype="int32"),
            "shape": tuple(group["shape"][...])}
        return patterns


def read_dense_data(file_name, file_key=None, start_index=0, end_index=-1,
                    output_type="numpy"):
    if output_type.lower() == "numpy":
        output_module = numpy
    elif output_type.lower() == "cupy":
        output_module = cupy
    else:
        raise ValueError(f"Argument output_array must be either numpy or "
                         f"cupy. Can't recognize: {output_type}")

    with h5py.File(file_name, "r") as file_handle:
        patterns = file_handle[file_key][start_index:end_index, ...]
        if (
                patterns.dtype == numpy.dtype("int32") or
                patterns.dtype == numpy.dtype("int64")
        ):
            patterns = output_module.asarray(patterns, dtype="int32")
        elif (
                patterns.dtype == numpy.dtype("float32") or
                patterns.dtype == numpy.dtype("float64")
        ):
            patterns = output_module.asarray(patterns, dtype="float32")
        else:
            raise ValueError(f"Can't read data of type {patterns.dtype}")
    return patterns


def radial_average(image, mask=None):
    """Calculates the radial average of an array of any shape,
    the center is assumed to be at the physical center."""
    if mask is None:
        mask = numpy.ones(image.shape, dtype='bool8')
    else:
        mask = numpy.bool8(mask)
    axis_values = [numpy.arange(s) - s/2. + 0.5 for s in image.shape]
    radius = numpy.zeros((image.shape[-1]))
    for i in range(len(image.shape)):
        radius = radius + (axis_values[-(1+i)][(slice(0, None), ) +
                                               (numpy.newaxis, )*i])**2
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


def init_model_radial_average_old(patterns, randomness=0.):
    """Simple function to create a random start. The new array will have
    a side similar to the second axis of the patterns"""

    pattern_radial_average = radial_average(patterns.mean(axis=0))
    side = patterns.shape[1]
    x = numpy.arange(side) - side/2 + 0.5

    r_int = numpy.int32(numpy.sqrt(x[:, numpy.newaxis, numpy.newaxis]**2 +
                                   x[numpy.newaxis, :, numpy.newaxis]**2 +
                                   x[numpy.newaxis, numpy.newaxis, :]**2))
    r_int_copy = r_int.copy()
    r_int[r_int >= len(pattern_radial_average)] = 0

    model = pattern_radial_average[numpy.int32(r_int)]
    model *= 1 - randomness + 2*randomness * numpy.random.random((side, )*3)
    model[r_int_copy >= len(pattern_radial_average)] = -1.
    return model


def init_model_radial_average(patterns, randomness=0.):
    """Simple function to create a random start. The new array will have
    a side similar to the second axis of the patterns"""

    patterns = patterns.copy()
    patterns_weights = patterns >= 0.
    patterns[~patterns_weights] = 0
    patterns = patterns.sum(axis=0)
    patterns_weights = patterns_weights.sum(axis=0)

    pattern_radial_average = radial_average(patterns)
    weight_radial_average = radial_average(patterns_weights)
    positive_weights = weight_radial_average[weight_radial_average > 0]
    pattern_radial_average[weight_radial_average > 0] /= positive_weights
    pattern_radial_average[weight_radial_average <= 0] = -1.

    side = patterns.shape[1]
    x = numpy.arange(side) - side/2 + 0.5

    r_int = numpy.int32(numpy.sqrt(x[:, numpy.newaxis, numpy.newaxis]**2 +
                                   x[numpy.newaxis, :, numpy.newaxis]**2 +
                                   x[numpy.newaxis, numpy.newaxis, :]**2))
    r_int_copy = r_int.copy()
    r_int[r_int >= len(pattern_radial_average)] = 0

    model = pattern_radial_average[numpy.int32(r_int)]
    model *= 1 - randomness + 2*randomness * numpy.random.random((side, )*3)
    model[r_int_copy >= len(pattern_radial_average)] = -1.
    return model


def chunks(number_of_rotations, chunk_size):
    """Generator for slices to chunk up the data"""
    chunk_starts = numpy.arange(0, number_of_rotations, chunk_size)
    chunk_ends = chunk_starts + chunk_size
    chunk_ends[-1] = number_of_rotations
    indices_cpu = [slice(this_chunk_start, this_chunk_end)
                   for this_chunk_start, this_chunk_end
                   in zip(chunk_starts, chunk_ends)]
    indices_gpu = [slice(None, this_chunk_end-this_chunk_start)
                   for this_chunk_start, this_chunk_end
                   in zip(chunk_starts, chunk_ends)]
    for this_indices_cpu, this_indices_gpu in zip(indices_cpu, indices_gpu):
        yield this_indices_cpu, this_indices_gpu


def images_to_sparse(patterns):
    """Convert a stack of diffraction patterns to sparse format"""
    number_of_patterns = len(patterns)
    number_of_lit_pixels = (patterns > 0).sum()
    start_indices = numpy.zeros(number_of_patterns+1, dtype="int32")
    indices = numpy.zeros(number_of_lit_pixels, dtype="int32")
    values = numpy.zeros(number_of_lit_pixels, dtype="int32")
    counter = 0

    for index_pattern, this_pattern in enumerate(patterns):
        if index_pattern % 100 == 0:
            print(f"{index_pattern} patterns done")
        flat_pattern = this_pattern.flatten()
        start_indices[index_pattern] = counter
        for index, value in enumerate(flat_pattern):
            if value > 0:
                indices[counter] = index
                values[counter] = value
                counter += 1
    start_indices[-1] = counter
    return {"start_indices": start_indices,
            "indices": indices,
            "values": values,
            "shape": patterns.shape[1:]}


def images_to_sparser(patterns):
    """Convert a stack of diffraction patterns to sparseR format"""
    number_of_patterns = len(patterns)
    # number_of_lit_pixels = (patterns > 0).sum()
    number_of_ones = (patterns == 1).sum()
    number_of_largers = (patterns > 1).sum()
    ones_start_indices = numpy.zeros(number_of_patterns+1, dtype="int32")
    start_indices = numpy.zeros(number_of_patterns+1, dtype="int32")
    ones_indices = numpy.zeros(number_of_ones, dtype="int32")
    indices = numpy.zeros(number_of_largers, dtype="int32")
    values = numpy.zeros(number_of_largers, dtype="int32")

    ones_counter = 0
    largers_counter = 0
    for index_pattern, this_pattern in enumerate(patterns):
        if index_pattern % 100 == 0:
            print(f"{index_pattern} patterns done")
        flat_pattern = this_pattern.flatten()
        ones_start_indices[index_pattern] = ones_counter
        start_indices[index_pattern] = largers_counter
        for index, value in enumerate(flat_pattern):
            if value == 1:
                ones_indices[ones_counter] = index
                ones_counter += 1
            if value > 1:
                indices[largers_counter] = index
                values[largers_counter] = value
                largers_counter += 1
    ones_start_indices[-1] = ones_counter
    start_indices[-1] = largers_counter

    return {"ones_start_indices": ones_start_indices,
            "start_indices": start_indices,
            "ones_indices": ones_indices,
            "indices": indices,
            "values": values,
            "shape": patterns.shape[1:]}
