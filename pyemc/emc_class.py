import os
try:
    from mpi4py import MPI
except ImportError:
    pass
import numpy
import h5py
# import emc_mpi_tools
import rotsampling
import cupy
from . import pyemc
from . import mpi as mpi_module
from . import utils

# def dataset_format(patterns):
#     if isinstance(patterns, dict) or isinstance(patterns, h5py.Group):
#         if "ones_start_indices" in patterns:
#             return "sparser"
#         else:
#             return "sparse"
#     else:
#         return "dense"

    
class DataReader:
    def __init__(self, mpi=None, number_of_patterns=None):
        if mpi == None and number_of_patterns == None:
            raise ValueError("Must specify either mpi or number_of_patterns")
        if mpi is not None:
            self._mpi = mpi
        else:
            self._mpi = mpi_module.MpiDistNoMpi()
            self._mpi.set_number_of_patterns(number_of_patterns)

    def read_patterns(self, file_name, file_loc):
        with h5py.File(file_name, "r") as file_handle:
            file_location = file_handle[file_loc]
            data_type = pyemc.pattern_type(patterns)
            # data_type = dataset_format(file_location)
        load_functions = {pyemc.PatternType.DENSE: utils.read_dense_data,
                          pyemc.PatternType.SPARSE: utils.read_sparse_data,
                          pyemc.PatternType.SPARSER: utils.read_sparser_data}
        data = load_functions[data_type](file_name, file_loc, self._mpi.pattern_slice().start, self._mpi.pattern_slice().stop)
        return data


class EMC:
    def __init__(self, patterns, mask, start_model, coordinates, n, rescale=False, mpi=None, quiet=False, two_dimensional=False):
        # Initialize MPI

        if mpi is not None:
            self._mpi = mpi
        else:
            self._mpi = mpi_module.MpiDistNoMpi()

        self._rescale = bool(rescale)

        self._quiet = bool(quiet)

        self._two_dimensional = bool(two_dimensional)
            
        # Numpy arrays used for mpi communications
        self._mpi_buffers = {}

        self._chunk_size = 1000

        if self._two_dimensional:
            self.set_rotsampling_2d(n)
        else:
            self.set_n(n)
        self.set_model(start_model)
        if not self._two_dimensional:
            self.set_coordinates(coordinates)
        else:
            self._coordinates = None
        self.set_patterns(patterns)
        self.set_mask(mask)
        self.set_alpha("static", 1)
        self._slices = None
        self._responsabilities_cpu = None
        self._responsabilities = None
        self._best_resp_rot_index = None

        self.current_iteration = 0

    def set_n(self, n):
        # Update rotations, weights, number_of_rotations, responsabilities_cpu, scaling_cpu
        all_rotations, all_rotation_weights = rotsampling.rotsampling(n, return_weights=True)
        self._all_rotations = numpy.float32(all_rotations)
        self._mpi.set_number_of_rotations(len(self._all_rotations))
        self._rotations = cupy.asarray(self._all_rotations[self._mpi.rotation_slice()], dtype="float32")
        self._rotation_weights_cpu = numpy.float32(all_rotation_weights[self._mpi.rotation_slice()])
        self._number_of_rotations = len(self._rotations)

    def set_rotsampling_2d(self, number_of_rotations):
        self._all_rotations = numpy.linspace(0, 2*numpy.pi, number_of_rotations)
        self._mpi.set_number_of_rotations(len(self._all_rotations))
        self._rotations = cupy.asarray(self._all_rotations[self._mpi.rotation_slice()], dtype="float32")
        self._rotation_weights_cpu = numpy.ones(len(self._all_rotations), dtype="float32")
        self._number_of_rotations = len(self._rotations)

    def set_model(self, model):
        # Update model, number_of_models, model_send/recv
        # Interpret starting model
        if hasattr(model, "shape"):
            # This is probably a numpy array
            self._model = [cupy.asarray(model, dtype="float32")]
        else:
            # This should be a list of arrays
            for m in model[1:]:
                if m.shape != model[0].shape:
                    raise ValueError("Models are not all the same shape")
            self._model = [cupy.asarray(m, dtype="float32") for m in model]
        self._model_weight = [cupy.zeros_like(m, dtype="float32") for m in self._model]
        self._number_of_models = len(self._model)
        if self._mpi.mpi_on:
            self._mpi_buffers["model_1"] = numpy.zeros(self._model[0].shape, dtype="float32")
            self._mpi_buffers["model_2"] = numpy.zeros(self._model[0].shape, dtype="float32")

    def set_coordinates(self, coordinates):
        # Update coordinates, slices
        # Convert coordinates
        self._coordinates = cupy.asarray(coordinates, dtype="float32")        
        
    def set_patterns(self, patterns):
        # Update patterns, number of patterns, number_of_patterns, responsabilities_cpu, scaling_cpu
        # Interpret patterns
        # Is it sparse or not
        # data_type = dataset_format(patterns)
        data_type = pyemc.pattern_type(patterns)
        # Convert types
        if data_type is pyemc.PatternType.SPARSE:
            self._patterns = {"indices": cupy.asarray(patterns["indices"], dtype="int32"),
                              "values": cupy.asarray(patterns["values"], dtype="int32"),
                              "start_indices": cupy.asarray(patterns["start_indices"], dtype="int32"),
                              "shape": patterns["shape"]}
            self._pattern_shape = patterns["shape"]
            self._number_of_patterns = len(self._patterns["start_indices"])-1
        elif data_type is pyemc.PatternType.SPARSER:
            self._patterns = {"indices": cupy.asarray(patterns["indices"], dtype="int32"),
                              "values": cupy.asarray(patterns["values"], dtype="int32"),
                              "start_indices": cupy.asarray(patterns["start_indices"], dtype="int32"),
                              "ones_indices": cupy.asarray(patterns["ones_indices"], dtype="int32"),
                              "ones_start_indices": cupy.asarray(patterns["ones_start_indices"], dtype="int32"),
                              "shape": patterns["shape"]}
            self._pattern_shape = patterns["shape"]
            self._number_of_patterns = len(self._patterns["start_indices"])-1
        elif data_type is pyemc.PatternType.DENSE:
            self._patterns = cupy.asarray(patterns, dtype="int32")
            self._pattern_shape = patterns.shape[1:]
            self._number_of_patterns = len(self._patterns)
        elif data_type is pyemc.PatternType.DENSEFLOAT:
            self._patterns = cupy.asarray(patterns, dtype="float32")
            self._pattern_shape = patterns.shape[1:]
            self._number_of_patterns = len(self._patterns)
        else:
            raise ValueError("Unsupported pattern format")
        # Figure out how many

        if self._mpi.mpi_on:
            self._mpi_buffers["resp_1"] = numpy.zeros(self._number_of_patterns, dtype="float32")
            self._mpi_buffers["resp_2"] = numpy.zeros(self._number_of_patterns, dtype="float32")
            self._mpi_buffers["resp_3"] = numpy.zeros(self._number_of_patterns, dtype="float32")
            self._mpi_buffers["resp_master"] = (numpy.zeros((self._mpi.rot_size(), self._number_of_patterns),
                                                            dtype="float32")
                                                if self._mpi.is_rot_master() else None)
            self._mpi_buffers["resp_index_master"] = (numpy.zeros((self._mpi.rot_size(), self._number_of_patterns),
                                                                  dtype="int64")
                                                      if self._mpi.is_rot_master() else None)
            if self._two_dimensional:
                self._mpi_buffers["all_rotations"] = (numpy.zeros(self._mpi.total_number_of_patterns, dtype="float32")
                                                      if self._mpi.is_master() else None)
            else:
                self._mpi_buffers["all_rotations"] = (numpy.zeros((self._mpi.total_number_of_patterns, 4), dtype="float32")
                                                      if self._mpi.is_master() else None)
            self._mpi_buffers["all_conformations"] = (numpy.zeros((self._mpi.total_number_of_patterns), dtype="int32")
                                                      if self._mpi.is_master() else None)

    def set_mask(self, mask):
        # Convert mask
        self._mask = cupy.asarray(mask, dtype="bool8")
        self._mask_inv = ~self._mask
        
    def set_alpha(self, method, *params):
        if method is "adaptive":
            self._alpha_method = {"method": method,
                                  "speed": params[0]}
        elif method is "static":
            self._alpha_method = {"method": method,
                                  "value": params[0]}
        else:
            raise ValueError("Alpha method must be 'adaptive' or 'static'")

    
    def _alpha_adaptive(self, target_resp_diff):
        epsilon = 1e-6
        if self._mpi.mpi_on:
            resp_global_max = numpy.empty(self._number_of_patterns, dtype="float32")
            resp_global_sum = numpy.empty(self._number_of_patterns, dtype="float32")
            self._mpi.comm_rot.Reduce(self._responsabilities_cpu.max(axis=0), resp_global_max, op=MPI.MAX, root=0)
            self._mpi.comm_rot.Reduce(self._responsabilities_cpu.sum(axis=0), resp_global_sum, op=MPI.SUM, root=0)
        else:
            resp_global_max = self._responsabilities_cpu.max(axis=0)
            resp_global_sum = self._responsabilities_cpu.sum(axis=0)

        if self._mpi.is_rot_master():
            alpha = (target_resp_diff / (epsilon + abs(resp_global_max - resp_global_sum /
                                                       (self._mpi.total_number_of_rotations*self._number_of_models))))
        else:
            alpha = None

        if self._mpi.mpi_on:
            return numpy.float32(self._mpi.comm_rot.bcast(alpha, root=0))
        else:
            return alpha

    def _chunks(self):
        chunk_generator = utils.chunks(self._number_of_rotations, self._chunk_size)
        return chunk_generator

    def calculate_responsabilities(self, slice_big_array, slice_small_array):
        pyemc.calculate_responsabilities_poisson(self._patterns,
                                                 self._slices[slice_small_array],
                                                 self._responsabilities[slice_small_array],
                                                 scalings=self._scaling[slice_small_array] if self._rescale else None)

    def update_slices(self, slice_big_array, slice_small_array):
        pyemc.update_slices(self._slices[slice_small_array],
                            self._patterns,
                            self._responsabilities[slice_small_array],
                            scalings=self._scaling[slice_small_array] if self._rescale else None)

    def apply_alpha(self):
        if self._alpha_method["method"] is "adaptive":
            alpha = self._alpha_adaptive(self._alpha_method["speed"] * (self.current_iteration+1))
            if self._mpi.is_master() and not self._quiet:
                print(f"alpha mean = {alpha.mean()}, std = {alpha.std()}", flush=True)
        elif self._alpha_method["method"] is "static":
            alpha = self._alpha_method["value"]
        
        self._responsabilities_cpu *= alpha

    def model_postprocessing(self):
        pass
    
    def iteration(self):
        if self._mpi.is_master() and not self._quiet:
            print(f"Start iteration {self.current_iteration}", flush=True)

        # Check that all the sizes match.
        # Mask, patterns, slices, coordinates
        if self._two_dimensional:
            if self._mask.shape != self._pattern_shape:
                raise ValueError("Sizes of mask and patterns don't match.")
        else:
            if self._mask.shape != self._pattern_shape or self._mask.shape != self._coordinates.shape[1:]:
                raise ValueError("Sizes of mask, patterns and coordinates don't match.")

        # If slices, resp or scaling is of wrong size, recreate it.
        if (self._slices is None or self._slices.shape[0] != self._chunk_size or
            self._slices.shape[1:] != self._pattern_shape):
            self._slices = cupy.zeros((self._chunk_size, ) + self._pattern_shape, dtype="float32")
        resp_cpu_shape = (self._number_of_rotations * self._number_of_models, self._number_of_patterns)
        if self._responsabilities_cpu is None or self._responsabilities_cpu.shape != resp_cpu_shape:
            self._responsabilities_cpu = numpy.zeros(resp_cpu_shape, dtype="float32")
            if self._rescale:
                self._scaling_cpu = numpy.ones(resp_cpu_shape, dtype="float32")
        resp_shape = (self._chunk_size, self._number_of_patterns)
        if self._responsabilities is None or self._responsabilities.shape != resp_shape:
            self._responsabilities = cupy.zeros(resp_shape, dtype="float32")
            if self._rescale:
                self._scaling = cupy.ones(resp_shape, dtype="float32")

        for model_index in range(self._number_of_models):
            if self._mpi.is_master() and not self._quiet:
                print(f"Loop 1, model {model_index}", flush=True)
            for slice_big_array, slice_small_array in self._chunks():
                if self._two_dimensional:
                    pyemc.expand_model_2d(self._model[model_index],
                                          self._slices[slice_small_array],
                                          self._rotations[slice_big_array])
                else:
                    pyemc.expand_model(self._model[model_index],
                                       self._slices[slice_small_array],
                                       self._rotations[slice_big_array],
                                       self._coordinates)
                self._slices[:, self._mask_inv] = -1
                if self._rescale:
                    pyemc.calculate_scaling_poisson(self._patterns,
                                                    self._slices[slice_small_array],
                                                    self._scaling[slice_small_array])
                self.calculate_responsabilities(slice_big_array, slice_small_array)

                self._responsabilities_cpu[model_index*self._number_of_rotations+slice_big_array.start:
                                           model_index*self._number_of_rotations+slice_big_array.stop, :] = self._responsabilities[slice_small_array].get()
                if self._rescale:
                    # print(f"{self._mpi.rank()}: copy scaling")
                    self._scaling_cpu[model_index*self._number_of_rotations+slice_big_array.start:
                                      model_index*self._number_of_rotations+slice_big_array.stop, :] = self._scaling[slice_small_array].get()


        self.apply_alpha()

        # if self._mpi.is_master(): print("Share max")
        if self._mpi.mpi_on:
            self._mpi_buffers["resp_1"][...] = self._responsabilities_cpu.max(axis=0)
            self._mpi.comm_rot.Allreduce(self._mpi_buffers["resp_1"], self._mpi_buffers["resp_2"], op=MPI.MAX)
            resp_max = self._mpi_buffers["resp_2"]
        else:
            resp_max = self._responsabilities_cpu.max(axis=0)

        # if self._mpi.is_master(): print("Subtract max")
        self._responsabilities_cpu -= resp_max[numpy.newaxis, :]

        # if self._mpi.is_master(): print("Exp")
        numpy.exp(self._responsabilities_cpu, out=self._responsabilities_cpu)

        # if self._mpi.is_master(): print("Rot weight multiply")
        for model_index in range(self._number_of_models):
            self._responsabilities_cpu[model_index*self._number_of_rotations:
                                       (model_index+1)*self._number_of_rotations] *= self._rotation_weights_cpu[:, numpy.newaxis]

        # if self._mpi.is_master(): print("Share sum")
        if self._mpi.mpi_on:
            self._mpi_buffers["resp_1"][...] = self._responsabilities_cpu.sum(axis=0)
            self._mpi.comm_rot.Allreduce(self._mpi_buffers["resp_1"], self._mpi_buffers["resp_2"], op=MPI.SUM)
            resp_sum = self._mpi_buffers["resp_2"]
        else:
            resp_sum = self._responsabilities_cpu.sum(axis=0)

        # if self._mpi.is_master(): print("Normalize")
        self._responsabilities_cpu /= resp_sum[numpy.newaxis, :]

        # if self._mpi.is_master(): print("Zero models")
        for this_model in self._model:
            this_model[...] = 0
        for this_model_weight in self._model_weight:
            this_model_weight[...] = 0

        for model_index in range(self._number_of_models):
            if self._mpi.is_master() and not self._quiet:
                print(f"Loop 2, model {model_index}", flush=True)
            for slice_big_array, slice_small_array in self._chunks():
                # print(f"Loop 2, model {model_index}, rotations {slice_big_array.start} - {slice_big_array.stop}")
                self._responsabilities[slice_small_array] = cupy.array(self._responsabilities_cpu[model_index*self._number_of_rotations+
                                                                                                  slice_big_array.start:
                                                                                                  model_index*self._number_of_rotations+
                                                                                                  slice_big_array.stop])
                if self._rescale:
                    self._scaling[slice_small_array] = cupy.array(self._scaling_cpu[model_index*self._number_of_rotations+
                                                                                    slice_big_array.start:
                                                                                    model_index*self._number_of_rotations+
                                                                                    slice_big_array.stop])
                slice_weights = self._responsabilities[slice_small_array].sum(axis=1)
                self.update_slices(slice_big_array, slice_small_array)
                self._slices[:, self._mask_inv] = -1
                if self._two_dimensional:
                    pyemc.insert_slices_2d(self._model[model_index],
                                           self._model_weight[model_index],
                                           self._slices[slice_small_array],
                                           slice_weights,
                                           self._rotations[slice_big_array])
                else:
                    pyemc.insert_slices(self._model[model_index],
                                        self._model_weight[model_index],
                                        self._slices[slice_small_array],
                                        slice_weights,
                                        self._rotations[slice_big_array],
                                        self._coordinates)

            if self._mpi.mpi_on:
                self._mpi_buffers["model_1"][...] = self._model[model_index].get()
                self._mpi.comm.Allreduce(self._mpi_buffers["model_1"], self._mpi_buffers["model_2"], op=MPI.SUM)
                self._model[model_index][...] = cupy.asarray(self._mpi_buffers["model_2"], dtype="float32")
            
                self._mpi_buffers["model_1"][...] = self._model_weight[model_index].get()
                self._mpi.comm.Allreduce(self._mpi_buffers["model_1"], self._mpi_buffers["model_2"], op=MPI.SUM)
                self._model_weight[model_index][...] = cupy.asarray(self._mpi_buffers["model_2"], dtype="float32")
            else:
                pass # No need to average models when MPI is off.

            bad_indices = self._model_weight[model_index] == 0
            self._model[model_index] /= self._model_weight[model_index]
            self._model[model_index][bad_indices] = -1.

            # Scaling normalization. Should probably be optional
            if self._rescale:
                self._model[model_index] /= self._model[model_index][~bad_indices].mean()

        self.model_postprocessing()
                
        self._best_resp_rot_index = None
        self.current_iteration += 1

    def get_model(self, output_list=False):
        if len(self._model) == 1 and not output_list:
            return self._model[0].get()
        else:
            return [m.get() for m in self._model]

    def _update_best_resp_index(self):
        if self._mpi.mpi_on:
            self._mpi.comm_rot.Gather(self._responsabilities_cpu.max(axis=0), self._mpi_buffers["resp_master"], root=0)
            self._mpi.comm_rot.Gather(self._responsabilities_cpu.argmax(axis=0), self._mpi_buffers["resp_index_master"], root=0)

            if self._mpi.is_rot_master():
                best_resp_rot_index = []
                for i in range(self._number_of_patterns):
                    rot_rank = self._mpi_buffers["resp_master"][:, i].argmax()
                    # print(f"{self._mpi.pattern_rank()}: rot_rank = {rot_rank}")
                    # this_index = mpi.local_to_global_rotation_index(rot_rank, (self._mpi_buffers["resp_index_master"][rot_rank, i] %
                    #                                                            self._mpi.number_of_rotations[rot_rank]))
                    this_index = (self._mpi.local_to_global_rotation_index(rot_rank, (self._mpi_buffers["resp_index_master"][rot_rank, i] %
                                                                                      self._mpi.number_of_rotations[rot_rank])) +
                                  (self._mpi_buffers["resp_index_master"][rot_rank, i] //
                                   self._mpi.number_of_rotations[rot_rank])*self._mpi.total_number_of_rotations)
                    best_resp_rot_index.append(this_index)
                self._best_resp_rot_index = numpy.int32(best_resp_rot_index)
            else:
                self._best_resp_rot_index = True
        else:
            self._best_resp_rot_index = self._responsabilities_cpu.argmax(axis=0)
        
    def get_best_rotations(self):
        if self._best_resp_rot_index is None:
            self._update_best_resp_index()

        if self._mpi.is_rot_master():
            # best_rotations = self._all_rotations[self._best_resp_rot_index % self._number_of_rotations, :]
            best_rotations = self._all_rotations[self._best_resp_rot_index % self._mpi.total_number_of_rotations, :]

            if self._mpi.mpi_on:
                # print(f"{self._mpi.rank()}: {best_rotations}, {self._mpi_buffers['all_rotations']}")
                # print(f"{self._mpi.rank()}: {best_rotations.shape}, {self._mpi_buffers['all_rotations'].shape}")
                self._mpi.comm_pattern.Gather(best_rotations, self._mpi_buffers["all_rotations"], root=0)
                if self._mpi.is_master():
                    return self._mpi_buffers["all_rotations"]
            else:
                return best_rotations
        return None

    def get_best_conformations(self):
        if self._best_resp_rot_index is None:
            self._update_best_resp_index()

        if self._mpi.is_rot_master():
            best_conformations = self._best_resp_rot_index // self._mpi.total_number_of_rotations

            if self._mpi.mpi_on:
                self._mpi.comm_pattern.Gather(best_conformations, self._mpi_buffers["all_conformations"], root=0)
                if self._mpi.is_master():
                    return self._mpi_buffers["all_conformations"]
                else:
                    return best_conformations
            else:
                return best_conformations
        return None

    def get_best_scaling(self):
        if not self._rescale:
            raise ValueError("Scaling is turned off")
        if self._best_resp_rot_index is None:
            self._update_best_resp_index()

        # Get the scaling for the best LOCAL responsability
        best_scaling_send = numpy.float32([self._scaling_cpu[self._responsabilities_cpu[:, i].argmax(), i] for i in range(self._number_of_patterns)])

        if self._mpi.mpi_on:
            # Gather all the above scalings to rot_master
            scaling_recv = numpy.empty((self._mpi.rot_size(), self._number_of_patterns), dtype="float32")
            self._mpi.comm_rot.Gather(best_scaling_send, scaling_recv, root=0)
        
            if self._mpi.is_rot_master():
                # Take the scaling for the node with best GLOBAL responsability
                best_scaling = numpy.float32([scaling_recv[self._mpi_buffers["resp_master"][:, i].argmax(), i] for i in range(self._number_of_patterns)])

                # Gather scalings for all patterns onto the master node
                all_scaling = numpy.empty(self._mpi.total_number_of_patterns, dtype="float32")
                self._mpi.comm_pattern.Gather(best_scaling, all_scaling, root=0)

                if self._mpi.is_master():
                    return all_scaling
        else:
            return numpy.int32([self._scaling_cpu[:, i] for i in self._best_resp_rot_index])
        return None
    
    def get_average_best_resp(self):
        if self._mpi.mpi_on:
            self._mpi.comm_rot.Reduce(self._responsabilities_cpu.max(axis=0), self._mpi_buffers["resp_2"], op=MPI.MAX, root=0)
            if self._mpi.is_rot_master():
                best_resp_mean = self._mpi.comm_pattern.reduce(self._mpi_buffers["resp_2"].sum(), op=MPI.SUM)
                if self._mpi.is_master():
                    return best_resp_mean / self._mpi.total_number_of_patterns
            return None
        else:
            return self._responsabilities_cpu.max(axis=0).mean()

