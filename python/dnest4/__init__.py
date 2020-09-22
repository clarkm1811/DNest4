# -*- coding: utf-8 -*-

__version__ = "0.1.5"

try:
    __DNEST4_SETUP__
except NameError:
    __DNEST4_SETUP__ = False

if not __DNEST4_SETUP__:
    __all__ = ["DNest4Sampler", "MPISampler", "postprocess", "analysis", "my_loadtxt, loadtxt_rows"]

    from . import analysis
    from .sampler import DNest4Sampler
    from ._dnest4 import MPISampler
    from .deprecated import postprocess, postprocess_abc
    from .loading import my_loadtxt, loadtxt_rows
    from .utils import randh
    from .utils import wrap
