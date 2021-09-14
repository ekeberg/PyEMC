import numpy
import h5py
import cupy
import pyemc
import argparse
import re
from eke import conversions
from eke import sphelper


parser = argparse.ArgumentParser()
parser.add_argument("patterns", type=str)
parser.add_argument("rotations", type=str)
parser.add_argument("photon_energy", type=float)
parser.add_argument("detector_distance", type=float)
parser.add_argument("pixel_size", type=float)
parser.add_argument("output_file", type=str)
parser.add_argument("--number_of_patterns", type=int, default=0)

args = parser.parse_args()

wavelength = conversions.ev_to_m(args.photon_energy)


if args.number_of_patterns != 0:
    raise NotImplementedError("Can't specify number of patterns")

def split_file_and_key(string):
    file_name, key = re.search("^(.+\.h5)(/.+)?$", string).groups()
    return file_name, key

def pattern_shape(patterns):
    if isinstance(patterns, dict):
        return patterns["shape"]
    else:
        return patterns.shape[1:]

def get_number_of_patterns(file_name, key):
    with h5py.File(file_name, "r") as file_handle:
        data_handle = file_handle[key]
        if isinstance(data_handle, h5py.Dataset):
            return len(data_handle)
        else:
            return len(data_handle["start_indices"])-1
        
        

#print(args.input_file)
patterns_file, patterns_key = split_file_and_key(args.patterns)
rotations_file, rotations_key = split_file_and_key(args.rotations)

if patterns_key == None or rotations_key == None:
    raise ValueError(f"Must provide a location in the hdf5 file: file.h5/location")


patterns_reader = pyemc.DataReader(number_of_patterns=get_number_of_patterns(patterns_file, patterns_key))
patterns = patterns_reader.read_patterns(patterns_file, patterns_key)
patterns = cupy.asarray(patterns, dtype="float32")

with h5py.File(rotations_file, "r") as file_handle:
    rotations = file_handle[rotations_key][...]
rotations = cupy.asarray(rotations, dtype="float32")
    
coordinates = pyemc.ewald_coordinates(pattern_shape(patterns), wavelength, args.detector_distance, args.pixel_size)

assembled = pyemc.assemble_model(patterns, rotations, coordinates).get()

sphelper.save_spimage(assembled, args.output_file)
