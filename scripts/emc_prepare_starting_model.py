import numpy
import h5py
from eke import sphelper
from eke import tools
import pyemc
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("input_file", type=str)
parser.add_argument("output_file", type=str)
parser.add_argument("--input_key", type=str, default="patterns")
parser.add_argument("--number_of_patterns", type=int, default=None)
# parser.add_argument("--output_key", type=str, default=None)
args = parser.parse_args()


def init_model_radial_average_sparse(patterns, model_shape):
    print("start creating model")
    radial_index = numpy.int32(tools.radial_distance(patterns["shape"]))
    radial_sum = numpy.zeros(radial_index.max()+1)
    radial_count = numpy.zeros(radial_index.max()+1)
    # Only count the values in the sum

    indices_cpu = numpy.int32(patterns["indices"])
    values_cpu = numpy.float32(patterns["values"])

    for i, v in zip(indices_cpu, values_cpu):
        radial_sum[radial_index.flat[i]] += v
    # The weight needs all pixels, but is the same for every pattern (assuming identical (or no) masks)
    for i in radial_index:
        radial_count[i] += 1
    radial_count *= len(patterns["start_indices"])

    radial_average = radial_sum / radial_count
    
    radial_index_3d = numpy.int32(tools.radial_distance(model_shape))
    radial_index_3d_copy = radial_index_3d.copy()
    radial_index_3d_copy[radial_index_3d >= len(radial_average)] = 0
    
    model = radial_average[radial_index_3d_copy]
    model[radial_index_3d_copy >= model_shape[0]/2.] = -1.
    
    return numpy.array(model, dtype=numpy.dtype("float32"))
    
def mask_sparse_data(patterns, mask):
    mask = numpy.bool8(mask)
    # loop through masked out pixels
    good_values = numpy.ones(patterns["indices"].shape, dtype="bool8")
    pattern_indices_cpu = numpy.float32(patterns["indices"])
    for pixel_index, mask_value in enumerate(mask.flat):
        # create bool array with indices to remove
        if not mask_value:
            good_values[pattern_indices_cpu == pixel_index] = False

    good_values_gpu = numpy.bool8(good_values)
    # prune values and indices with the above array
    patterns["indices"] = patterns["indices"][good_values_gpu]
    patterns["values"] = patterns["values"][good_values_gpu]
    # 
    patterns["start_indices"] = numpy.int32([good_values[:this_start_index].sum() for this_start_index in patterns["start_indices"]])

    
# patterns_file = "/home/ekeberg/Work/Projects/python_emc/mpi/ribosome_sparse.h5"
# patterns_key = "noisy_1e4"
# number_of_patterns = 10000

with h5py.File(args.input_file, "r") as file_handle:
    patterns_handle = file_handle[args.input_key]
    if isinstance(patterns_handle, h5py.Group):
        print("Data is sparse")
        sparse_data = True
    else:
        print("Data is dense")
        sparse_data = False

if args.number_of_patterns is None:
    args.number_of_patterns = -1
        
if sparse_data:
    patterns = pyemc.read_sparse_data(args.input_file, args.input_key, 0, args.number_of_patterns, output_type="numpy")
    model_shape = (patterns["shape"][0], )*3
    model = init_model_radial_average_sparse(patterns, model_shape)
else:
    patterns = pyemc.read_dense_data(args.input_file, args.input_key, 0, args.number_of_patterns, output_type="numpy")
    model = pyemc.init_model_radial_average(patterns, randomness=0.)
    
sphelper.save_spimage(model, args.output_file)
