import numpy
import h5py
import rotsampling
import cupy
from . import pyemc
from . import mpi as mpi_module
from . import utils


class DataReader:
    def __init__(self, mpi=None, number_of_patterns=None):
        if mpi is None and number_of_patterns is None:
            raise ValueError("Must specify either mpi or number_of_patterns")
        if mpi is not None:
            self._mpi = mpi
        else:
            self._mpi = mpi_module.MpiDistNoMpi()
            self._mpi.set_number_of_patterns(number_of_patterns)

    def dataset_format(self, file_location):
        if isinstance(file_location, h5py.Dataset):
            if numpy.issubdtype(file_location.dtype, numpy.integer):
                return pyemc.PatternType.DENSE
            elif numpy.issubdtype(file_location.dtype, numpy.floating):
                return pyemc.PatternType.DENSEFLOAT
            else:
                raise ValueError(f"Unsupported dataset of type {file_location.dtype}")
        else:
            if "ones_indices" in file_location:
                return pyemc.PatternType.SPARSER
            elif "indices" in file_location:
                return pyemc.PatternType.SPARSE
            else:
                raise ValueError("Unsupported dataset")

    def read_patterns(self, file_name, file_loc):
        with h5py.File(file_name, "r") as file_handle:
            file_location = file_handle[file_loc]
            # data_type = pyemc.pattern_type(file_location)
            data_type = self.dataset_format(file_location)
        load_functions = {pyemc.PatternType.DENSE: utils.read_dense_data,
                          pyemc.PatternType.DENSEFLOAT: utils.read_dense_data,
                          pyemc.PatternType.SPARSE: utils.read_sparse_data,
                          pyemc.PatternType.SPARSER: utils.read_sparser_data}
        data = load_functions[data_type](file_name, file_loc,
                                         self._mpi.pattern_slice().start,
                                         self._mpi.pattern_slice().stop)
        return data


class EMC:
    def __init__(self, patterns, mask, start_model, coordinates, n,
                 rescale=False, mpi=None, quiet=False, two_dimensional=False):
        # Initialize MPI

        if mpi is not None:
            self._mpi = mpi
        else:
            self._mpi = mpi_module.MpiDistNoMpi()

        if self._mpi.mpi_on:
            self._mpi_flags = {"SUM": mpi_module.MPI.SUM,
                               "MAX": mpi_module.MPI.MAX}

        self._rescale = bool(rescale)

        self._quiet = bool(quiet)

        self._two_dimensional = bool(two_dimensional)

        # Numpy arrays used for mpi communications
        self._mpi_buffers = {}

        self._chunk_size = 1000

        self._interpolation = pyemc.Interpolation.LINEAR

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
        self._resp_cpu = None
        self._resp = None
        self._best_resp_rot_index = None

        self.current_iteration = 0

    def set_n(self, n):
        # Update rotations, weights, number_of_rotations,
        # resp_cpu, scaling_cpu
        all_rotations, all_rotation_weights = rotsampling.rotsampling(
            n, return_weights=True)
        self._all_rotations = numpy.float32(all_rotations)
        self._mpi.set_number_of_rotations(len(self._all_rotations))
        my_rotations = self._all_rotations[self._mpi.rotation_slice()]
        my_weights = all_rotation_weights[self._mpi.rotation_slice()]
        self._rotations = cupy.asarray(my_rotations, dtype="float32")
        self._rotation_weights_cpu = numpy.float32(my_weights)
        self._number_of_rotations = len(self._rotations)

    def set_rotsampling_2d(self, number_of_rotations):
        self._all_rotations = numpy.linspace(0, 2*numpy.pi,
                                             number_of_rotations)
        self._mpi.set_number_of_rotations(len(self._all_rotations))
        my_rotations = self._all_rotations[self._mpi.rotation_slice()]
        my_weights = len(self._all_rotations)
        self._rotations = cupy.asarray(my_rotations, dtype="float32")
        self._rotation_weights_cpu = numpy.ones(my_weights, dtype="float32")
        self._number_of_rotations = len(self._rotations)

    def set_interpolation(self, interpolation):
        try:
            self._interpolation = pyemc.Interpolation(interpolation)
        except ValueError:
            if interpolation.lower() == "nearest":
                self._interpolation = pyemc.Interpolation.NEAREST
            elif interpolation.lower() == "linear":
                self._interpolation = pyemc.Interpolation.LINEAR
            elif interpolation.lower() == "sinc":
                self._interpolation = pyemc.Interpolation.SINC
            else:
                raise ValueError(f"{interpolation} is not a valid "
                                 "interpolation")
        
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
        self._model_weight = [cupy.zeros_like(m, dtype="float32")
                              for m in self._model]
        self._number_of_models = len(self._model)
        if self._mpi.mpi_on:
            self._mpi_buffers["model_1"] = numpy.zeros(self._model[0].shape,
                                                       dtype="float32")
            self._mpi_buffers["model_2"] = numpy.zeros(self._model[0].shape,
                                                       dtype="float32")

    def set_coordinates(self, coordinates):
        # Update coordinates, slices
        # Convert coordinates
        self._coordinates = cupy.asarray(coordinates, dtype="float32")

    def set_patterns(self, patterns):
        # Update patterns, number of patterns, number_of_patterns,
        # resp_cpu, scaling_cpu
        # Interpret patterns
        # Is it sparse or not
        # data_type = dataset_format(patterns)
        data_type = pyemc.pattern_type(patterns)
        # Convert types
        if data_type is pyemc.PatternType.SPARSE:
            self._patterns = {
                "indices": cupy.asarray(patterns["indices"],
                                        dtype="int32"),
                "values": cupy.asarray(patterns["values"],
                                       dtype="int32"),
                "start_indices": cupy.asarray(patterns["start_indices"],
                                              dtype="int32"),
                "shape": patterns["shape"]}
            self._pattern_shape = patterns["shape"]
            self._number_of_patterns = len(self._patterns["start_indices"])-1
        elif data_type is pyemc.PatternType.SPARSER:
            self._patterns = {
                "indices": cupy.asarray(patterns["indices"],
                                        dtype="int32"),
                "values": cupy.asarray(patterns["values"],
                                       dtype="int32"),
                "start_indices": cupy.asarray(patterns["start_indices"],
                                              dtype="int32"),
                "ones_indices": cupy.asarray(patterns["ones_indices"],
                                             dtype="int32"),
                "ones_start_indices": cupy.asarray(
                    patterns["ones_start_indices"], dtype="int32"),
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
            self._mpi_buffers["resp_1"] = numpy.zeros(self._number_of_patterns,
                                                      dtype="float32")
            self._mpi_buffers["resp_2"] = numpy.zeros(self._number_of_patterns,
                                                      dtype="float32")
            self._mpi_buffers["resp_3"] = numpy.zeros(self._number_of_patterns,
                                                      dtype="float32")
            self._mpi_buffers["resp_master"] = (
                numpy.zeros((self._mpi.rot_size(),
                             self._number_of_patterns),
                            dtype="float32")
                if self._mpi.is_rot_master() else None)
            self._mpi_buffers["resp_index_master"] = (
                numpy.zeros((self._mpi.rot_size(),
                             self._number_of_patterns),
                            dtype="int64")
                if self._mpi.is_rot_master() else None)
            if self._two_dimensional:
                self._mpi_buffers["all_rotations"] = (
                    numpy.zeros(self._mpi.total_number_of_patterns,
                                dtype="float32")
                    if self._mpi.is_master() else None)
            else:
                self._mpi_buffers["all_rotations"] = (
                    numpy.zeros((self._mpi.total_number_of_patterns, 4),
                                dtype="float32")
                    if self._mpi.is_master() else None)
            self._mpi_buffers["all_conformations"] = (
                numpy.zeros((self._mpi.total_number_of_patterns),
                            dtype="int32")
                if self._mpi.is_master() else None)

    def set_mask(self, mask):
        # Convert mask
        self._mask = cupy.asarray(mask, dtype="bool8")
        self._mask_inv = ~self._mask

    def set_alpha(self, method, *params):
        if method == "adaptive":
            self._alpha_method = {"method": method,
                                  "speed": params[0]}
        elif method == "static":
            self._alpha_method = {"method": method,
                                  "value": params[0]}
        else:
            raise ValueError("Alpha method must be 'adaptive' or 'static'")

    def _alpha_adaptive(self, target_resp_diff):
        epsilon = 1e-6
        if self._mpi.mpi_on:
            resp_global_max = numpy.empty(self._number_of_patterns,
                                          dtype="float32")
            resp_global_sum = numpy.empty(self._number_of_patterns,
                                          dtype="float32")
            self._mpi.comm_rot.Reduce(self._resp_cpu.max(axis=0),
                                      resp_global_max,
                                      op=self._mpi_flags["MAX"], root=0)
            self._mpi.comm_rot.Reduce(self._resp_cpu.sum(axis=0),
                                      resp_global_sum,
                                      op=self._mpi_flags["SUM"], root=0)
        else:
            resp_global_max = self._resp_cpu.max(axis=0)
            resp_global_sum = self._resp_cpu.sum(axis=0)

        if self._mpi.is_rot_master():
            number_of_states = (self._mpi.total_number_of_rotations *
                                self._number_of_models)
            average_diff = abs(resp_global_max - resp_global_sum /
                               number_of_states)
            alpha = (target_resp_diff / (epsilon + average_diff))
        else:
            alpha = None

        if self._mpi.mpi_on:
            return numpy.float32(self._mpi.comm_rot.bcast(alpha, root=0))
        else:
            return alpha

    def _chunks(self):
        chunk_generator = utils.chunks(self._number_of_rotations,
                                       self._chunk_size)
        return chunk_generator

    def calculate_resp(self, slice_big, slice_small):
        scalings = self._scaling[slice_small] if self._rescale else None
        pyemc.calculate_responsabilities_poisson(
            self._patterns,
            self._slices[slice_small],
            self._resp[slice_small],
            scalings=scalings)

    def update_slices(self, slice_big, slice_small):
        scalings = self._scaling[slice_small] if self._rescale else None
        pyemc.update_slices(self._slices[slice_small],
                            self._patterns,
                            self._resp[slice_small],
                            scalings=scalings)

    def apply_alpha(self):
        if self._alpha_method["method"] == "adaptive":
            alpha_strength = (self._alpha_method["speed"] *
                              (self.current_iteration+1))
            alpha = self._alpha_adaptive(alpha_strength)
            if self._mpi.is_master() and not self._quiet:
                print(f"alpha mean = {alpha.mean()}, std = {alpha.std()}",
                      flush=True)
        elif self._alpha_method["method"] == "static":
            alpha = self._alpha_method["value"]
        self._resp_cpu *= alpha

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
            if (
                    self._mask.shape != self._pattern_shape or
                    self._mask.shape != self._coordinates.shape[1:]
            ):
                raise ValueError("Sizes of mask, patterns and coordinates "
                                 "don't match.")

        # If slices, resp or scaling is of wrong size, recreate it.
        if (
                self._slices is None or
                self._slices.shape[0] != self._chunk_size or
                self._slices.shape[1:] != self._pattern_shape
        ):
            slices_shape = (self._chunk_size, ) + self._pattern_shape
            self._slices = cupy.zeros(slices_shape, dtype="float32")

        resp_cpu_shape = (self._number_of_rotations * self._number_of_models,
                          self._number_of_patterns)
        if (
                self._resp_cpu is None or
                self._resp_cpu.shape != resp_cpu_shape
        ):
            self._resp_cpu = numpy.zeros(resp_cpu_shape,
                                         dtype="float32")
            if self._rescale:
                self._scaling_cpu = numpy.ones(resp_cpu_shape, dtype="float32")
        resp_shape = (self._chunk_size, self._number_of_patterns)
        if (
                self._resp is None or
                self._resp.shape != resp_shape
        ):
            self._resp = cupy.zeros(resp_shape, dtype="float32")
            if self._rescale:
                self._scaling = cupy.ones(resp_shape, dtype="float32")

        for model_index, this_model in enumerate(self._model):
            if self._mpi.is_master() and not self._quiet:
                print(f"Loop 1, model {model_index}", flush=True)
            for slice_big, slice_small in self._chunks():
                if self._two_dimensional:
                    pyemc.expand_model_2d(this_model,
                                          self._slices[slice_small],
                                          self._rotations[slice_big],
                                          interpolation=self._interpolation)
                else:
                    pyemc.expand_model(this_model,
                                       self._slices[slice_small],
                                       self._rotations[slice_big],
                                       self._coordinates,
                                       interpolation=self._interpolation)
                self._slices[:, self._mask_inv] = -1
                if self._rescale:
                    pyemc.calculate_scaling_poisson(
                        self._patterns,
                        self._slices[slice_small],
                        self._scaling[slice_small])
                self.calculate_resp(slice_big,
                                    slice_small)

                model_slice = slice(model_index*self._number_of_rotations +
                                    slice_big.start,
                                    model_index*self._number_of_rotations +
                                    slice_big.stop)
                resp_cpu = self._resp[slice_small].get()
                self._resp_cpu[model_slice, :] = resp_cpu
                if self._rescale:
                    scaling_cpu = self._scaling[slice_small].get()
                    self._scaling_cpu[model_slice, :] = scaling_cpu

        self.apply_alpha()

        # if self._mpi.is_master(): print("Share max")
        if self._mpi.mpi_on:
            self._mpi_buffers["resp_1"][...] = self._resp_cpu.max(axis=0)
            self._mpi.comm_rot.Allreduce(self._mpi_buffers["resp_1"],
                                         self._mpi_buffers["resp_2"],
                                         op=self._mpi_flags["MAX"])
            resp_max = self._mpi_buffers["resp_2"]
        else:
            resp_max = self._resp_cpu.max(axis=0)

        # if self._mpi.is_master(): print("Subtract max")
        self._resp_cpu -= resp_max[numpy.newaxis, :]

        # if self._mpi.is_master(): print("Exp")
        numpy.exp(self._resp_cpu, out=self._resp_cpu)

        # if self._mpi.is_master(): print("Rot weight multiply")
        for model_index in range(self._number_of_models):
            model_slice = slice(model_index*self._number_of_rotations,
                                (model_index+1)*self._number_of_rotations)
            self._resp_cpu[model_slice] *= self._rotation_weights_cpu[
                :, numpy.newaxis]

        # if self._mpi.is_master(): print("Share sum")
        if self._mpi.mpi_on:
            self._mpi_buffers["resp_1"][...] = self._resp_cpu.sum(axis=0)
            self._mpi.comm_rot.Allreduce(self._mpi_buffers["resp_1"],
                                         self._mpi_buffers["resp_2"],
                                         op=self._mpi_flags["SUM"])
            resp_sum = self._mpi_buffers["resp_2"]
        else:
            resp_sum = self._resp_cpu.sum(axis=0)

        # if self._mpi.is_master(): print("Normalize")
        self._resp_cpu /= resp_sum[numpy.newaxis, :]

        # if self._mpi.is_master(): print("Zero models")
        for this_model in self._model:
            this_model[...] = 0
        for this_model_weight in self._model_weight:
            this_model_weight[...] = 0

        # for model_index in range(self._number_of_model):
        for loop_values in enumerate(zip(self._model, self._model_weight)):
            model_index, (this_model, this_model_weight) = loop_values
            if self._mpi.is_master() and not self._quiet:
                print(f"Loop 2, model {model_index}", flush=True)
            for slice_big, slice_small in self._chunks():
                model_slice = slice(model_index * self._number_of_rotations +
                                    slice_big.start,
                                    model_index * self._number_of_rotations +
                                    slice_big.stop)

                self._resp[slice_small] = cupy.asarray(
                    self._resp_cpu[model_slice], dtype="float32")
                if self._rescale:
                    self._scaling[slice_small] = cupy.asarray(
                        self._scaling_cpu[model_slice], dtype="float32")

                slice_weights = self._resp[slice_small].sum(axis=1)
                self.update_slices(slice_big, slice_small)
                self._slices[:, self._mask_inv] = -1
                if self._two_dimensional:
                    pyemc.insert_slices_2d(this_model,
                                           this_model_weight,
                                           self._slices[slice_small],
                                           slice_weights,
                                           self._rotations[slice_big],
                                           interpolation=self._interpolation)
                else:
                    pyemc.insert_slices(this_model,
                                        this_model_weight,
                                        self._slices[slice_small],
                                        slice_weights,
                                        self._rotations[slice_big],
                                        self._coordinates,
                                        interpolation=self._interpolation)

            if self._mpi.mpi_on:
                self._mpi_buffers["model_1"][...] = this_model.get()
                self._mpi.comm.Allreduce(self._mpi_buffers["model_1"],
                                         self._mpi_buffers["model_2"],
                                         op=self._mpi_flags["SUM"])
                this_model[...] = cupy.asarray(self._mpi_buffers["model_2"],
                                               dtype="float32")

                self._mpi_buffers["model_1"][...] = this_model_weight.get()
                self._mpi.comm.Allreduce(self._mpi_buffers["model_1"],
                                         self._mpi_buffers["model_2"],
                                         op=self._mpi_flags["SUM"])
                this_model_weight[...] = cupy.asarray(
                    self._mpi_buffers["model_2"], dtype="float32")
            else:
                pass  # No need to average models when MPI is off.

            bad_indices = this_model_weight == 0
            this_model /= this_model_weight
            this_model[bad_indices] = -1.

            # Scaling normalization. Should probably be optional
            if self._rescale:
                this_model /= this_model[~bad_indices].mean()

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
            self._mpi.comm_rot.Gather(self._resp_cpu.max(axis=0),
                                      self._mpi_buffers["resp_master"],
                                      root=0)
            self._mpi.comm_rot.Gather(self._resp_cpu.argmax(axis=0),
                                      self._mpi_buffers["resp_index_master"],
                                      root=0)

            if self._mpi.is_rot_master():
                best_resp_rot_index = []
                for i in range(self._number_of_patterns):
                    rot_rank = self._mpi_buffers["resp_master"][:, i].argmax()
                    local_rot_index = (
                        self._mpi_buffers["resp_index_master"][rot_rank, i]
                        % self._mpi.number_of_rotations[rot_rank])
                    rot_index = self._mpi.local_to_global_rotation_index(
                        rot_rank, local_rot_index)
                    model_index = (
                        self._mpi_buffers["resp_index_master"][rot_rank, i]
                        // self._mpi.number_of_rotations[rot_rank])
                    this_index = (rot_index
                                  + model_index
                                  * self._mpi.total_number_of_rotations)
                    best_resp_rot_index.append(this_index)
                self._best_resp_rot_index = numpy.int32(best_resp_rot_index)
            else:
                self._best_resp_rot_index = True
        else:
            self._best_resp_rot_index = self._resp_cpu.argmax(axis=0)

    def get_best_rotations(self):
        if self._best_resp_rot_index is None:
            self._update_best_resp_index()

        if self._mpi.is_rot_master():
            index_in_model = (self._best_resp_rot_index
                              % self._mpi.total_number_of_rotations)
            best_rotations = self._all_rotations[index_in_model, :]

            if self._mpi.mpi_on:
                self._mpi.comm_pattern.Gather(
                    best_rotations, self._mpi_buffers["all_rotations"], root=0)
                if self._mpi.is_master():
                    return self._mpi_buffers["all_rotations"]
            else:
                return best_rotations
        return None

    def get_best_conformations(self):
        if self._best_resp_rot_index is None:
            self._update_best_resp_index()

        if self._mpi.is_rot_master():
            best_conformations = (self._best_resp_rot_index
                                  // self._mpi.total_number_of_rotations)

            if self._mpi.mpi_on:
                self._mpi.comm_pattern.Gather(
                    best_conformations,
                    self._mpi_buffers["all_conformations"],
                    root=0)
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
        best_scaling_send = numpy.float32([
            self._scaling_cpu[self._resp_cpu[:, i].argmax(), i]
            for i in range(self._number_of_patterns)])

        if self._mpi.mpi_on:
            # Gather all the above scalings to rot_master
            scaling_recv = numpy.empty((self._mpi.rot_size(),
                                        self._number_of_patterns),
                                       dtype="float32")
            self._mpi.comm_rot.Gather(best_scaling_send, scaling_recv, root=0)

            if self._mpi.is_rot_master():
                # Take the scaling for the node with best GLOBAL responsability
                process_best_resps = self._mpi_buffers["resp_master"]
                best_scaling = numpy.float32([
                    scaling_recv[process_best_resps[:, i].argmax(), i]
                    for i in range(self._number_of_patterns)])

                # Gather scalings for all patterns onto the master node
                all_scaling = numpy.empty(self._mpi.total_number_of_patterns,
                                          dtype="float32")
                self._mpi.comm_pattern.Gather(best_scaling, all_scaling,
                                              root=0)

                if self._mpi.is_master():
                    return all_scaling
        else:
            return numpy.int32([self._scaling_cpu[:, i]
                                for i in self._best_resp_rot_index])
        return None

    def get_average_best_resp(self):
        if self._mpi.mpi_on:
            self._mpi.comm_rot.Reduce(self._resp_cpu.max(axis=0),
                                      self._mpi_buffers["resp_2"],
                                      op=self._mpi_flags["MAX"],
                                      root=0)
            if self._mpi.is_rot_master():
                best_resp_mean = self._mpi.comm_pattern.reduce(
                    self._mpi_buffers["resp_2"].sum(),
                    op=self._mpi_flags["SUM"])
                if self._mpi.is_master():
                    return best_resp_mean / self._mpi.total_number_of_patterns
            return None
        else:
            return self._resp_cpu.max(axis=0).mean()

