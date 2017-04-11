# -*- coding: utf-8 -*-

from .analysis import postprocess
from .backends import MemoryBackend
try:
    from ._dnest4 import sample as _sample
    from ._dnest4 import MPISampler
except ImportError:
    _sample = None

__all__ = ["DNest4Sampler"]


class DNest4Sampler(object):

    def __init__(self, model, backend=None, MPISampler=None):
        if _sample is None:
            raise ImportError("You must build the Cython extensions to use "
                              "the Python API")
        self._model = model
        if backend is None:
            backend = MemoryBackend()
        self.backend = backend

        self.mpi_sampler = MPISampler

    def sample(self, max_num_levels, **kwargs):
        self.backend.reset()

        if self.mpi_sampler is None:
            for sample in _sample(self._model, max_num_levels, **kwargs):
                self.backend.write_particles(
                    sample["samples"], sample["sample_info"]
                )
                self.backend.write_levels(sample["levels"])
                yield sample
        else:
            for sample in self.mpi_sampler.sample(self._model, max_num_levels, **kwargs):
                self.backend.write_particles(
                    sample["samples"], sample["sample_info"]
                )
                self.backend.write_levels(sample["levels"])
                yield sample


    def run(self, num_steps, max_num_levels, **kwargs):
        if num_steps <= 0:
            raise ValueError("Invalid number of steps")
        kwargs["num_steps"] = int(num_steps)
        for _ in self.sample(max_num_levels, **kwargs):
            pass

    def postprocess(self, **kwargs):
        return postprocess(self.backend, **kwargs)
