#!/bin/env python

import os
import numpy
import cupy
import h5py
import pyemc
import rotsampling
import time
from eke import sphelper
from eke import tools
from eke import shell_functions
from eke import conversions
from eke import compare_rotations

pyemc.set_nthreads(256)

wavelength = conversions.ev_to_nm(8000)
detector_distance = 0.2
pixel_size = 2*200e-6

side = 128 # of diffraction pattern
# number_of_patterns = 10000
number_of_patterns = 10000
rotation_sampling_density = 6 # See Veit Elsers paper
chunk_size = 100
number_of_iterations = 100
# This is a regularization parameter. Standar is 1 but needs to be
# lower to handle data with high signal.
# alpha = 0.001
alpha = 4e-4

print(f"number_of_patterns = {number_of_patterns}")

output_dir = "output_n6"
shell_functions.mkdir_p(output_dir)

mask = cupy.array(tools.circular_mask(side) * ~tools.circular_mask(side, 7))
# mask = cupy.array(tools.circular_mask(side))

# Read data
# file_name = "patterns_preprocessed_new_0000_0029.h5"
# data_directory = "/home/ekeberg/Work/Projects/python_emc/mixed_set/Phytochrome/Range/data_new"
file_name = "ribosome_fixed.h5"
data_directory = "/home/august/emc"
patterns_file = os.path.join(data_directory, file_name)
print(patterns_file)
with h5py.File(patterns_file, "r") as file_handle:
    patterns = cupy.asarray(file_handle["patterns"][:number_of_patterns], dtype="float32")
    correct_rotations = file_handle["rotation"][:number_of_patterns]
patterns[:, ~mask] = -1 # Ugly inverse of mask because of afnumpy
print("pattern type", patterns.dtype)
print("read patterns")
print(f"patterns_max = {patterns.max()}")

# Create the rotation sampling (according to original EMC paper)
rotations, rotation_weights = rotsampling.rotsampling(rotation_sampling_density, return_weights=True)
rotations = cupy.asarray(rotations, dtype="float32")
number_of_rotations = len(rotations)
print(f"number_of_rotations = {number_of_rotations}")

# Create and save the starting model. This is all the randomness that goes in to the algorithm.
model = pyemc.init_model_radial_average(patterns, 0.1)
initial_model = model.get().copy() # Convert from afnumpy to standard numpy array before saving
sphelper.save_spimage(initial_model, os.path.join(output_dir, "model_init.h5"))

# Setup Ewald-sphere coordinates
coordinates = pyemc.ewald_coordinates((side, )*2, wavelength, detector_distance, pixel_size)

# Setup arrays that will be used during the run. Some will have a
# smaller version for the GPU and a full version for the CPU.
slices = cupy.zeros((chunk_size, side, side), dtype="float32")
responsabilities_cpu = numpy.zeros((number_of_rotations, number_of_patterns), dtype="float32")
responsabilities_gpu = cupy.zeros((chunk_size, number_of_patterns), dtype="float32")
rotations_gpu = cupy.zeros((chunk_size, 4), dtype="float32")
model_weights = cupy.zeros(model.shape, dtype="float32")


history = {"model": [],
           "likelihood": []}
for iteration in range(number_of_iterations):
    start_time = time.time()

    # First expand the model and calculate where each pattern fits the
    # best.
    for this_indices_cpu, this_indices_gpu in pyemc.chunks(number_of_rotations, chunk_size):
        # print(f"this_indices_cpu = {this_indices_cpu}")
        # print(f"responsabilities_gpu.mean() = {responsabilities_gpu.mean()}")
        rotations_gpu[this_indices_gpu] = cupy.array(rotations[this_indices_cpu])
        pyemc.expand_model(model, slices[this_indices_gpu],
                           rotations_gpu[this_indices_gpu], coordinates)
        pyemc.calculate_responsabilities_poisson(patterns, slices[this_indices_gpu],
                                                 responsabilities_gpu[this_indices_gpu])
        # responsabilities_cpu[this_indices_cpu, :] = responsabilities_gpu[this_indices_gpu, :].get()
        responsabilities_cpu[this_indices_cpu, :] = responsabilities_gpu[this_indices_gpu, :].get()

    history["likelihood"].append(responsabilities_cpu.sum())
    print("likelihood: {0}".format(responsabilities_cpu.sum()))

    # Rescale and normalize responsabilities (done on CPU since the
    # entire matrix doesn't fit on GPU.
    responsabilities_cpu *= alpha
    responsabilities_cpu -= responsabilities_cpu.max(axis=0)[numpy.newaxis, :]
    numpy.exp(responsabilities_cpu, out=responsabilities_cpu)
    responsabilities_cpu *= rotation_weights[:, numpy.newaxis]
    responsabilities_cpu /= responsabilities_cpu.sum(axis=0)[numpy.newaxis, :]

    # The best responsability essentially tells us the rotation that
    # each pattern fits best in. Here we save these rotations.
    best_responsability = responsabilities_cpu.max(axis=0)
    best_resp_index = responsabilities_cpu.argmax(axis=0)
    best_rotations = rotations.get()[best_resp_index, :]
    with h5py.File(os.path.join(output_dir, "best_rot_{0:04}.h5".format(iteration)), "w") as file_handle:
        file_handle.create_dataset("best_rotations", data=best_rotations)
    # rot_error = compare_rotations.absolute_orientation_error(correct_rotations, best_rotations)
    rot_error = compare_rotations.relative_orientation_error(correct_rotations, best_rotations, shallow_ewald=True)
    print(f"{180/numpy.pi*rot_error} degrees", flush=True)

    

    # Set model and weights to zero before new compression step
    model[:] = 0.    
    model_weights[:] = 0.

    # Compress the model, which essentially means that we construct a
    # model from the new orientaitons of each pattern.
    for this_indices_cpu, this_indices_gpu in pyemc.chunks(number_of_rotations, chunk_size):
        responsabilities_gpu[this_indices_gpu, :] = cupy.array(responsabilities_cpu[this_indices_cpu, :])
        slice_weights = responsabilities_gpu[this_indices_gpu, :].sum(axis=1)
        rotations_gpu[this_indices_gpu] = cupy.array(rotations[this_indices_cpu])
        pyemc.update_slices(slices[this_indices_gpu], patterns, responsabilities_gpu[this_indices_gpu])
        pyemc.insert_slices(model, model_weights, slices[this_indices_gpu],
                            slice_weights[this_indices_gpu],
                            rotations_gpu[this_indices_gpu], coordinates)
        
    # Normalize the model with the weights. Need to take care of
    # pixels with zero weight.
    bad_indices = model_weights == 0.
    model /= model_weights
    model[bad_indices] = -1.
    history["model"].append(model.get())
    if (numpy.isnan(model).sum() > 0):
        print(f"Warning, we have {numpy.isnan(model).sum()} nan in model. Setting them to -1")
        model[cupy.isnan(model)] = -1
    sphelper.save_spimage(model.get(), os.path.join(output_dir, "model_{0:04}.h5".format(iteration)))

    end_time = time.time()

    # Output a few metrics
    print(("iteration {iteration}: likelihood = {likelihood}, best_resp = {best_resp}\n"
           "iteration took {time} seconds").format(iteration=iteration,
                                                   time=end_time-start_time,
                                                   likelihood=history["likelihood"][-1],
                                                   best_resp=best_responsability.mean()))
