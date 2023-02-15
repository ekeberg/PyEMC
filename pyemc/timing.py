import time

class Timer:
    """Keep track of multiple timers"""
    def __init__(self):
        self._records = defaultdict(lambda: 0)
        self._start = defaultdict(lambda: None)

    def start(self, name):
        """Start timer with the given name. Create if it doesn't exist"""
        self._start[name] = time.time()

    def stop(self, name):
        """Stop timer with the given name."""
        if self._start[name] is None:
            raise ValueError(f"Trying to stop inactive timer: {name}")
        self._records[name] += time.time() - self._start[name]
        self._start[name] = None

    def get_total(self):
        """Get a dictionary with all accumulated times"""
        return self._records

    def print_per_process(self, mpi):
        """"Provide an mpi object to return times for that process
        specifically"""
        for this_rank in range(mpi.size()):
            if mpi.rank() == this_rank:
                print(f"Timing {this_rank}:")
                for n, v in timer.get_total().items():
                    print(f"{n}: {v}")
                print("")
            mpi.comm.Barrier()

    def print_single(self):
        """Print the result of get_total()"""
        print("Timing:")
        for n, v in timer.get_total().items():
            print(f"{n}: {v}")

    def print_total(self, mpi=None):
        """If no mpi object is given, same as print_single. Otherwise the
        result is summed up for all processes."""
        if not mpi.mpi_on:
            self.print_single()
            return

        if mpi.is_master():
            print("Timing total:")
        for n, v in timer.get_total().items():
            tot_v = mpi.comm.reduce(v, root=0)
            if mpi.is_master():
                print(f"{n}: {tot_v}")


timer = Timer()


def timed(func):
    """Decorator stating that the total time spent in this function should
    be keept track of"""
    func_signature = inspect.signature(func)

    @functools.wraps(func)
    def new_func(*args, **kwargs):
        bound_arguments = func_signature.bind(*args, **kwargs)
        bound_arguments.apply_defaults()
        args = bound_arguments.args

        timer.start(func.__name__)
        ret = func(*args)
        cupy.cuda.stream.get_current_stream().synchronize()
        timer.stop(func.__name__)

        return ret
    return new_func


def print_timing(mpi=None):
    if mpi is None or mpi.mpi_on:
        timer.print_single()
    else:
        timer.print_total(mpi)
