import numpy
import os
from mpi4py import MPI
import socket
from eke import tools

from . import mpi


class MpiDist(mpi.MpiDistBase):
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
        self.mpi_on = True

    def is_master(self):
        return self.rank() == 0
        
    def is_rot_master(self):
        return self.rot_rank() == 0

    def is_pattern_master(self):
        return self.pattern_rank() == 0

    def size(self):
        return self.comm.Get_size()
    
    def rot_size(self):
        return self.comm_rot.Get_size()

    def pattern_size(self):
        return self.comm_pattern.Get_size()

    def rank(self):
        return self.comm.Get_rank()
    
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

    def distribute_gpus(self):
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(MPI.COMM_WORLD.Get_rank() % 4)
        this_host = socket.gethostname()
        all_hosts = self.comm.allgather(this_host)
        my_gpu_index = [i for i, h in enumerate(all_hosts) if h == this_host].index(self.rank())
        print(f"{self.rank()}: {this_host}: {my_gpu_index}")
        import cupy
        cupy.cuda.runtime.setDevice(my_gpu_index)
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(my_gpu_index)

        # os.environ["NVIDIA_VISIBLE_DEVICES"] = str(my_gpu_index)
        # os.environ["CUDA_VISIBLE_DEVICES"] = "0"

        # os.environ["CUDA_VISIBLE_DEVICES"] = str(my_gpu_index)
        # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
