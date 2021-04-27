import numpy
import h5py
from eke import tools
import argparse
import pyemc

parser = argparse.ArgumentParser()
parser.add_argument("input_file", type=str)
parser.add_argument("output_file", type=str)
parser.add_argument("--input_key", type=str, default="patterns")
parser.add_argument("--output_key", type=str, default=None)
parser.add_argument("--sparser", action="store_true", default=False)
args = parser.parse_args()

if args.output_key is None:
    args.output_key = args.input_key

parameters = {}

with h5py.File(args.input_file, "r") as file_handle:
    patterns = file_handle[args.input_key][...]
    # patterns = file_handle[args.input_key][:300]
    parameters_group = file_handle["parameters"]
    for key, value in parameters_group.items():
        parameters[key] = value[...]
    if "rotations" in file_handle.keys():
        rotations = file_handle["rotations"][...]
    else:
        rotations = None
    if "states" in file_handle.keys():
        states = file_handle["states"][...]
    else:
        states = None

    if "scaling" in file_handle.keys():
        scaling = file_handle["scaling"][...]
    else:
        scaling = None
        
#mask = ~tools.circular_mask(patterns.shape[1], 7.) * tools.circular_mask(patterns.shape[1], 32.)
#patterns[:, ~mask] = -1.

if args.sparser:
    sparse_patterns = pyemc.images_to_sparser(patterns)

    with h5py.File(args.output_file, "a") as file_handle:
        output_group = file_handle.create_group(args.output_key)
        output_group.create_dataset("ones_start_indices", data=numpy.array(sparse_patterns["ones_start_indices"]))
        output_group.create_dataset("start_indices", data=numpy.array(sparse_patterns["start_indices"]))
        output_group.create_dataset("ones_indices", data=numpy.array(sparse_patterns["ones_indices"]))
        output_group.create_dataset("indices", data=numpy.array(sparse_patterns["indices"]))
        output_group.create_dataset("values", data=numpy.array(sparse_patterns["values"]))
        output_group.create_dataset("shape", data=sparse_patterns["shape"])
        if "parameters" not in file_handle.keys():
            parameters_group = file_handle.create_group("parameters")
            for key, value in parameters.items():
                parameters_group.create_dataset(key, data=value)
else:
    sparse_patterns = pyemc.images_to_sparse(patterns)

    with h5py.File(args.output_file, "a") as file_handle:
        output_group = file_handle.create_group(args.output_key)
        output_group.create_dataset("start_indices", data=numpy.array(sparse_patterns["start_indices"]))
        output_group.create_dataset("indices", data=numpy.array(sparse_patterns["indices"]))
        output_group.create_dataset("values", data=numpy.array(sparse_patterns["values"]))
        output_group.create_dataset("shape", data=sparse_patterns["shape"])
        if "parameters" not in file_handle.keys():
            parameters_group = file_handle.create_group("parameters")
            for key, value in parameters.items():
                parameters_group.create_dataset(key, data=value)

with h5py.File(args.output_file, "a") as file_handle:
    if rotations is not None and "rotations" not in file_handle.keys():
        file_handle.create_dataset("rotations", data=rotations)
    if states is not None and "states" not in file_handle.keys():
        file_handle.create_dataset("states", data=states)
    if scaling is not None and "scaling" not in file_handle.keys():
        file_handle.create_dataset("scaling", data=scaling)



    # return {"ones_start_indices": ones_start_indices,
    #         "start_indices": start_indices,
    #         "ones_indices": ones_indices,
    #         "indices": indices,
    #         "values": values,
    #         "shape": patterns.shape}
        
