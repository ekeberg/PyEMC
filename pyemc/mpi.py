import numpy
import os
from mpi4py import MPI
import socket
# import emc
# import afnumpy
from eke import tools

class Mpi:
    def __init__(self):
        self.mpi_on = True
        self.comm = MPI.COMM_WORLD

    def size(self):
        return self.comm.Get_size()

    def rank(self):
        return self.comm.Get_rank()

    def is_master(self):
        return self.rank() == 0

    def distribute_gpus(self):
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(MPI.COMM_WORLD.Get_rank() % 4)
        this_host = socket.gethostname()
        all_hosts = self.comm.allgather(this_host)
        my_gpu_index = [i for i, h in enumerate(all_hosts) if h == this_host].index(self.rank())
        print(f"{self.rank()}: {this_host}: {my_gpu_index}")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(my_gpu_index)

        
class MpiPatternDist(Mpi):
    def __init__(self):
        super().__init__()
        
    def set_number_of_patterns(self, number_of_patterns):
        self.total_number_of_patterns = number_of_patterns
        if self.total_number_of_patterns % self.size() == 0:
            self.pattern_index_start = numpy.arange(self.size())*self.total_number_of_patterns//self.size()
            self.pattern_index_end = (numpy.arange(self.size())+1)*self.total_number_of_patterns//self.size()            
        else:
            self.pattern_index_start = numpy.arange(self.size())*(self.total_number_of_patterns//self.size()+1)
            self.pattern_index_end = (numpy.arange(self.size())+1)*(self.total_number_of_patterns//self.size()+1)
            self.pattern_index_end[-1] = total_number_of_patterns
        self.number_of_patterns = self.pattern_index_end - self.pattern_index_start
        
    def local_number_of_patterns(self):
        return self.number_of_patterns[self.rank()]

    def pattern_slice(self):
        return slice(self.pattern_index_start[self.rank()], self.pattern_index_end[self.rank()])


class MpiPatternDistNoMpi(MpiPatternDist):
    def __init__(self):
        super().__init__()
        self.mpi_on = False
    
    def set_number_of_patterns(self, number_of_patterns):
        self.total_number_of_patterns = number_of_patterns

    def size(self):
        return 1

    def rank(self):
        return 0

    def is_master(self):
        return True

    def local_number_of_patterns(self):
        return self.total_number_of_patterns

    def pattern_slice(self):
        return slice(0, self.total_number_of_patterns)

    def distribute_gpus(self):
        pass


class MpiRotationDist(Mpi):
    def __init__(self):
        super().__init__()

    def set_number_of_rotations(self, number_of_rotations):
        self.total_number_of_rotations = number_of_rotations
        if self.total_number_of_rotations % self.size() == 0:
            self.rotation_index_start = numpy.arange(self.size())*self.total_number_of_rotations//self.size()
            self.rotation_index_end = (numpy.arange(self.size())+1)*self.total_number_of_rotations//self.size()            
        else:
            self.rotation_index_start = numpy.arange(self.size())*(self.total_number_of_rotations//self.size()+1)
            self.rotation_index_end = (numpy.arange(self.size())+1)*(self.total_number_of_rotations//self.size()+1)
            self.rotation_index_end[-1] = self.total_number_of_rotations
        self.number_of_rotations = self.rotation_index_end - self.rotation_index_start

    def local_number_of_rotations(self):
        return self.number_of_rotations[self.rank()]

    def rotation_slice(self):
        return slice(self.rotation_index_start[self.rank()], self.rotation_index_end[self.rank()])

    def local_to_global_index(self, rank, index):
        return self.number_of_rotations[:rank].sum() + index
        


class MpiRotationDistNoMpi(MpiRotationDist):
    def __init__(self):
        super().__init__()
        self.mpi_on = False
    
    def set_number_of_rotations(self, number_of_rotations):
        self.total_number_of_rotations = number_of_rotations
        self.number_of_rotations = numpy.array([self.total_number_of_rotations])

    def size(self):
        return 1

    def rank(self):
        return 0

    def is_master(self):
        return True

    def local_number_of_rotations(self):
        return self.total_number_of_rotations

    def rotation_slice(self):
        return slice(0, self.total_number_of_rotations)

    def distribute_gpus(self):
        pass


class MpiDist(Mpi):
    def __init__(self, rot_size, pattern_size):
        super().__init__()
        self._rot_size = rot_size
        self._pattern_size = pattern_size
        self._size = self._rot_size*self._pattern_size
        self.comm = MPI.COMM_WORLD.Create_cart(dims=(self._rot_size, self._pattern_size),
                                               periods=[False, False],
                                               reorder=False)
        self.comm_rot = self.comm.Sub(remain_dims=[True, False])
        self.comm_pattern = self.comm.Sub(remain_dims=[False, True])

    def is_rot_master(self):
        return self.rot_rank() == 0

    def is_pattern_master(self):
        return self.pattern_rank() == 0
        
    def rot_size(self):
        return self.comm_rot.Get_size()

    def pattern_size(self):
        return self.comm_pattern.Get_size()

    def rot_rank(self):
        return self.comm_rot.Get_rank()

    def pattern_rank(self):
        return self.comm_pattern.Get_rank()

    def set_number_of_rotations(self, number_of_rotations):
        self.total_number_of_rotations = number_of_rotations
        if self.total_number_of_rotations % self.rot_size() == 0:
            self.rotation_index_start = numpy.arange(self.rot_size())*self.total_number_of_rotations//self.rot_size()
            self.rotation_index_end = (numpy.arange(self.rot_size())+1)*self.total_number_of_rotations//self.rot_size()            
        else:
            self.rotation_index_start = numpy.arange(self.rot_size())*(self.total_number_of_rotations//self.rot_size()+1)
            self.rotation_index_end = (numpy.arange(self.rot_size())+1)*(self.total_number_of_rotations//self.rot_size()+1)
            self.rotation_index_end[-1] = self.total_number_of_rotations
        self.number_of_rotations = self.rotation_index_end - self.rotation_index_start

    def set_number_of_patterns(self, number_of_patterns):
        self.total_number_of_patterns = number_of_patterns
        if self.total_number_of_patterns % self.pattern_size() == 0:
            self.pattern_index_start = numpy.arange(self.pattern_size())*self.total_number_of_patterns//self.pattern_size()
            self.pattern_index_end = (numpy.arange(self.pattern_size())+1)*self.total_number_of_patterns//self.pattern_size()
        else:
            self.pattern_index_start = numpy.arange(self.pattern_size())*(self.total_number_of_patterns//self.pattern_size()+1)
            self.pattern_index_end = (numpy.arange(self.pattern_size())+1)*(self.total_number_of_patterns//self.pattern_size()+1)
            self.pattern_index_end[-1] = total_number_of_patterns
        self.number_of_patterns = self.pattern_index_end - self.pattern_index_start

    def local_number_of_rotations(self):
        return self.number_of_rotations[self.rot_rank()]

    def local_number_of_patterns(self):
        return self.number_of_patterns[self.pattern_rank()]

    def rotation_slice(self):
        return slice(self.rotation_index_start[self.rot_rank()], self.rotation_index_end[self.rot_rank()])

    def pattern_slice(self):
        return slice(self.pattern_index_start[self.pattern_rank()], self.pattern_index_end[self.pattern_rank()])

    def local_to_global_rotation_index(self, rank, index):
        return self.number_of_rotations[:rank].sum() + index

    def local_to_global_pattern_index(self, rank, index):
        return self.number_of_patterns[:rank].sum() + index


class MpiDistNoMpi(MpiDist):
    def __init__(self):
        super().__init__(1, 1)
        self.mpi_on = False
    
    def set_number_of_rotations(self, number_of_rotations):
        self.total_number_of_rotations = number_of_rotations
        self.number_of_rotations = numpy.array([self.total_number_of_rotations])

    def set_number_of_patterns(self, number_of_patterns):
        self.total_number_of_patterns = number_of_patterns
        self.number_of_patterns = numpy.array([self.total_number_of_patterns])

    def size(self):
        return 1

    def rank(self):
        return 0

    def rot_size(self):
        return 1

    def pattern_size(self):
        return 1

    def rot_rank(self):
        return 0

    def pattern_rank(self):
        return 0

    def is_master(self):
        return True

    def local_number_of_rotations(self):
        return self.total_number_of_rotations

    def local_number_of_patterns(self):
        return self.total_number_of_patterns
    
    def rotation_slice(self):
        return slice(0, self.total_number_of_rotations)

    def rotation_slice(self):
        return slice(0, self.total_number_of_patterns)
    
    def distribute_gpus(self):
        pass

    
def init_model_radial_average(mpi, patterns, randomness=0.):
    """Simple function to create a random start. The new array will have
    a side similar to the second axis of the patterns"""
    
    pattern_mean = numpy.array(patterns.mean(axis=0))
    if mpi.is_master():
        pattern_mean_all = numpy.zeros_like(pattern_mean)
    else:
        pattern_mean_all = None
    mpi.comm.Reduce(pattern_mean, pattern_mean_all, root=0, op=MPI.SUM)
    if mpi.is_master():
        pattern_mean_all /= mpi.size()
        pattern_radial_average = tools.radial_average(numpy.array(patterns.mean(axis=0)))
        side = patterns.shape[1]
        x = numpy.arange(side) - side/2 + 0.5

        r_int = numpy.int32(numpy.sqrt(x[:, numpy.newaxis, numpy.newaxis]**2 +
                                       x[numpy.newaxis, :, numpy.newaxis]**2 +
                                       x[numpy.newaxis, numpy.newaxis, :]**2))
        r_int_copy = r_int.copy()
        r_int[r_int >= len(pattern_radial_average)] = 0

        model = numpy.float32(pattern_radial_average[numpy.int32(r_int)])
        model *= 1. - randomness + 2. * randomness * numpy.random.random((side, )*3)
        model[r_int_copy >= len(pattern_radial_average)] = -1.
    else:
        model = numpy.zeros((patterns.shape[1], )*3, dtype="float32")
    mpi.comm.Bcast(model, root=0)
    return model
