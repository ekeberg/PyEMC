import cupy
import h5py
import numpy

def read_sparse_data(file_name, file_key=None, start_index=0, end_index=-1, output_type="numpy"):
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
            value_end_index = all_start_indices[end_index+1]
        # if end_index == -1:
        #     end_index = len(all_start_indices)-1

        if output_type.lower() == "numpy":
            output_module = numpy
        elif output_type.lower() == "cupy":
            output_module = cupy
        else:
            raise ValueError(f"Argument output_array must be either numpy or cupy. Can't recognize: {output_array}")
            
        patterns = {"start_indices": output_module.asarray(all_start_indices[start_index:end_index+1] - all_start_indices[start_index], dtype="int32"),
                    "indices": output_module.asarray(group["indices"][value_start_index:value_end_index], dtype="int32"),
                    "values": output_module.asarray(group["values"][value_start_index:value_end_index], dtype="int32"),
                    "shape": tuple(group["shape"][...])}

        # patterns = {"start_indices": numpy.int32(all_start_indices[start_index:end_index+1]),
        #             "indices": numpy.int32(group["indices"][value_start_index:value_end_index]),
        #             "values": numpy.float32(group["values"][value_start_index:value_end_index]),
        #             "shape": tuple(group["shape"][...])}
        return patterns

def read_dense_data(file_name, file_key=None, start_index=0, end_index=-1, output_type="numpy"):
    if output_type.lower() == "numpy":
        output_module = numpy
    elif output_type.lower() == "cupy":
        output_module = cupy
    else:
        raise ValueError(f"Argument output_array must be either numpy or cupy. Can't recognize: {output_array}")

    with h5py.File(file_name, "r") as file_handle:
        patterns = output_module.asarray(file_handle[file_key][start_index:end_index, ...], dtype="int32")
    return patterns

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

def init_model_radial_average(patterns, randomness=0.):
    """Simple function to create a random start. The new array will have
    a side similar to the second axis of the patterns"""
    pattern_mean = patterns.mean(axis=0)
    pattern_radial_average = radial_average(patterns.mean(axis=0))
    side = patterns.shape[1]
    x = numpy.arange(side) - side/2 + 0.5

    r_int = numpy.int32(numpy.sqrt(x[:, numpy.newaxis, numpy.newaxis]**2 +
                                   x[numpy.newaxis, :, numpy.newaxis]**2 +
                                   x[numpy.newaxis, numpy.newaxis, :]**2))
    r_int_copy = r_int.copy()
    r_int[r_int >= len(pattern_radial_average)] = 0
    
    model = pattern_radial_average[numpy.int32(r_int)]
    model *= 1. - randomness + 2. * randomness * numpy.random.random((side, )*3)
    model[r_int_copy >= len(pattern_radial_average)] = -1.    
    return model
