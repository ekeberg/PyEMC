import numpy
import os


def mpi_is_running():
    if (
            "MPI_LOCALNRANKS" in os.environ or
            "OMPI_COMM_WORLD_SIZE" in os.environ
    ):
        return True
    else:
        return False


class MpiBase:
    def __init__(self):
        pass
        # self.mpi_on = True
        # self.comm = MPI.COMM_WORLD

    def size(self):
        pass

    def rank(self):
        pass

    def is_master(self):
        pass

    def distribute_gpus(self):
        pass


class MpiDistBase(MpiBase):
    def __init__(self):
        super().__init__()

    def set_number_of_rotations(self, number_of_rotations):
        pass

    def set_number_of_patterns(self, number_of_rotations):
        pass

    def rot_size(self):
        pass

    def pattern_size(self):
        pass

    def rot_rank(self):
        pass

    def pattern_rank(self):
        pass

    def is_rot_master(self):
        pass

    def is_pattern_master(self):
        pass

    def local_number_of_rotations(self):
        pass

    def local_number_of_patterns(self):
        pass

    def rotation_slice(self):
        pass

    def pattern_slice(self):
        pass


class MpiDistNoMpi(MpiDistBase):
    def __init__(self):
        super().__init__()
        self.mpi_on = False

    def set_number_of_rotations(self, number_of_rotations):
        self.total_number_of_rotations = number_of_rotations
        self.number_of_rotations = numpy.array(
            [self.total_number_of_rotations])

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

    def is_rot_master(self):
        return True

    def is_pattern_master(self):
        return True

    def local_number_of_rotations(self):
        return self.total_number_of_rotations

    def local_number_of_patterns(self):
        return self.total_number_of_patterns

    def rotation_slice(self):
        return slice(0, self.total_number_of_rotations)

    def pattern_slice(self):
        return slice(0, self.total_number_of_patterns)

    def distribute_gpus(self):
        pass


def get_default_mpi():
    raise NotImplementedError("Sorry")


# This import sadly must happen after the above definitions, since
# they are in turn needed from _mpi.

if mpi_is_running():
    from mpi4py import MPI
    from ._mpi import *

# from ._mpi import *
