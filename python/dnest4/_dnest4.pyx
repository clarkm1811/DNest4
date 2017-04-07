# distutils: language = c++
from __future__ import division

cimport cython
from libcpp.vector cimport vector
from cython.operator cimport dereference

import time
import numpy as np
cimport numpy as np
np.import_array()

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

MPI = None

cdef extern from "DNest4.h" namespace "DNest4":

    cdef cppclass Options:
        Options ()
        Options (
            unsigned int num_particles,
            unsigned int new_level_interval,
            unsigned int save_interval,
            unsigned int thread_steps,
            unsigned int max_num_levels,
            double lam,
            double beta,
            unsigned int max_num_saves
        )

    cdef cppclass Sampler[T]:
        Sampler()
        Sampler(unsigned int num_threads, double compression,
                const Options options, unsigned save_to_disk)

        # Setup, running, etc.
        void initialise(unsigned int first_seed) except +
        void run() except +
        void increase_max_num_saves(unsigned int increment)
        void mcmc_thread(unsigned int thread)
        void process_threads()

        # Interface.
        int size()
        T* particle(unsigned int i)
        vector[Level] get_levels()
        vector[unsigned] get_level_assignments()
        vector[LikelihoodType] get_log_likelihoods()
        void set_levels(const vector[Level] level_vec)
        void copy_levels(unsigned int thread)
        vector[Level] get_levels_copy (unsigned int thread )
        void set_levels_copy(vector[Level] level_vec, unsigned int thread)

    cdef cppclass LikelihoodType:
        double get_value()
        double get_tiebreaker()

    cdef cppclass Level:
        Level()
        Level(double like_value, double like_tiebreaker,
              unsigned int visits,
              unsigned int exceeds,
              unsigned int accepts,
              unsigned int tries,
              double log_X
        )
        LikelihoodType get_log_likelihood()
        unsigned int get_visits()
        unsigned int get_exceeds()
        unsigned int get_accepts()
        unsigned int get_tries()
        double get_log_X()


cdef extern from "PyModel.h":

    cdef cppclass PyModel:
        void set_py_self (object py_self)
        object get_py_self ()
        int get_exception ()
        object get_npy_coords ()
        void set_coords (object coords)


class DNest4Error(Exception):
    pass

class MPISampler(object):

  def __init__(self, comm=None, debug=False):

    global MPI
    try:
      import mpi4py.MPI
      MPI = mpi4py.MPI
    except ImportError:
      raise ImportError("Please install mpi4py")

    if comm is None:
      self.comm = MPI.COMM_WORLD
    else:
      self.comm = comm

    self.rank = self.comm.Get_rank()
    self.size = self.comm.Get_size() - 1

    self.debug = debug

    #tags for MPI communication
    self.tags = self.enum('READY', 'SET_LEVELS', 'RUN_THREAD', 'CLOSE')


  def enum(self, *sequential, **named):
      """Handy way to fake an enumerated type in Python
      http://stackoverflow.com/questions/36932/how-can-i-represent-an-enum-in-python
      """
      enums = dict(zip(sequential, range(len(sequential))), **named)
      return type('Enum', (), enums)

  def is_master(self):
    """
    Is the current process the master?
    """
    return self.rank == 0

  def wait(self, model,
    unsigned int max_num_levels,

    int num_steps=-1,
    unsigned int num_per_step=10000,

    # sampler args
    # unsigned int num_particles=1,
    unsigned int new_level_interval=10000,
    unsigned int thread_steps=100,

    double lam=5.0,
    double beta=100.0,

    # "command line" arguments
    seed=None,
    double compression=np.exp(1.0),
  ):

    """
    If this isn't the master process, wait for instructions.
    """
    if self.is_master():
       raise RuntimeError("Master node told to await jobs.")

    num_particles = self.size

    # Check the model.
    if not hasattr(model, "from_prior") or not callable(model.from_prior):
       raise ValueError("DNest4 models must have a callable 'from_prior' method")
    if not hasattr(model, "perturb") or not callable(model.perturb):
       raise ValueError("DNest4 models must have a callable 'perturb' method")
    if not hasattr(model, "log_likelihood") or not callable(model.log_likelihood):
       raise ValueError("DNest4 models must have a callable 'log_likelihood' method")

    # Set up the options.
    if (num_per_step <= 0 or num_particles <= 0 or new_level_interval <= 0
           or max_num_levels <= 0 or thread_steps <= 0):
       raise ValueError("'num_per_step', 'num_particles', "
                        "'new_level_interval', and "
                        "'max_num_levels' must all be positive")
    if lam <= 0.0 or beta < 0.0:
       raise ValueError("'lam' and 'beta' must be non-negative")
    cdef Options options = Options(
       num_particles, new_level_interval, num_per_step, thread_steps,
       max_num_levels, lam, beta, 1
    )

    # Declarations.
    cdef int i, j, n, error
    cdef Sampler[PyModel] sampler = Sampler[PyModel](1, compression, options, 0)
    cdef vector[Level] levels

    # Initialize the particles.
    n = sampler.size()
    for j in range(n):
       particle = sampler.particle(j)
       particle.set_py_self(model)
       error = particle.get_exception()
       if error != 0:
           raise DNest4Error(error)

    status = MPI.Status()

    while True:
      # Event loop.
      # Sit here and await instructions.
      if self.debug:
        print("Worker {0} waiting for task.".format(self.rank))

      # Blocking receive to wait for instructions.
      task = self.comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
      if self.debug:
        print("Worker {0} got task {1} with tag {2}.".format(self.rank, task, status.tag))

      tag = status.Get_tag()

      if tag == self.tags.SET_LEVELS:
        #Set level info
        sampler_status = task

        #recreate level vector
        n_levels = len(result["levels"])
        levels.resize(n_levels)
        for j in range(n_levels):
          levels[j] = Level(result["levels"][j]["log_likelihood"], result["levels"][j]["tiebreaker"],
                result["levels"][j]["visits"],
                result["levels"][j]["exceeds"],
                result["levels"][j]["accepts"],
                result["levels"][j]["tries"],
                result["levels"][j]["log_X"])

        sampler.set_levels(levels)
        self.comm.isend(None, dest=0, tag=status.tag)

      elif tag == self.tags.RUN_THREAD:
        #Give the particle the most up to date coords
        particle_coords = task
        part_idx = self.rank-1
        particle = sampler.particle(part_idx)
        particle.set_coords(particle_coords)

        #run it for thread_steps iterations
        #but don't just use run_thread, cause that does bookkeeping i want to do on the rank0 side
        sampler.copy_levels(part_idx)

        #run mcmc
        sampler.mcmc_thread(part_idx)

        # Loop over levels and save them.
        levels = sampler.get_levels_copy(part_idx)
        n = levels.size()

        result = dict()
        result["levels"] = np.empty(n, dtype=[
            ("log_X", np.float64), ("log_likelihood", np.float64),
            ("tiebreaker", np.float64), ("accepts", np.uint16),
            ("tries", np.uint16), ("exceeds", np.uint16),
            ("visits", np.uint16)
        ])
        for j in range(n):
            level = levels[j]
            result["levels"][j]["log_X"] = level.get_log_X()
            result["levels"][j]["log_likelihood"] = \
                level.get_log_likelihood().get_value()
            result["levels"][j]["tiebreaker"] = \
                level.get_log_likelihood().get_tiebreaker()
            result["levels"][j]["accepts"] = level.get_accepts()
            result["levels"][j]["tries"] = level.get_tries()
            result["levels"][j]["exceeds"] = level.get_exceeds()
            result["levels"][j]["visits"] = level.get_visits()

        #return your thread's copy_of_levels and particle info
        particle_and_levels = ( sampler.particle(part_idx).get_npy_coords(),  result)

        self.comm.isend(particle_and_levels, dest=0, tag=status.tag)
      elif tag == self.tags.CLOSE:
        if self.debug:
            print("Worker {0} told to quit.".format(self.rank))
        break

  def sample(self,
      model,
      unsigned int max_num_levels,

      int num_steps=-1,
      unsigned int num_per_step=10000,

      # sampler args
      # unsigned int num_particles=1,
      unsigned int new_level_interval=10000,
      unsigned int thread_steps=100,

      double lam=5.0,
      double beta=100.0,

      # "command line" arguments
      seed=None,
      double compression=np.exp(1.0),
  ):

    #everything is set up now if you're a worker, so wait for instructions
    if not self.is_master():
      self.wait(model, max_num_levels, num_steps=num_steps, num_per_step=num_per_step, new_level_interval=new_level_interval, thread_steps=thread_steps, lam=lam, beta=beta, seed=seed, compression=compression)
      return

    num_particles = self.size

    # Check the model.
    if not hasattr(model, "from_prior") or not callable(model.from_prior):
        raise ValueError("DNest4 models must have a callable 'from_prior' method")
    if not hasattr(model, "perturb") or not callable(model.perturb):
        raise ValueError("DNest4 models must have a callable 'perturb' method")
    if not hasattr(model, "log_likelihood") or not callable(model.log_likelihood):
        raise ValueError("DNest4 models must have a callable 'log_likelihood' method")

    # Set up the options.
    if (num_per_step <= 0 or num_particles <= 0 or new_level_interval <= 0
            or max_num_levels <= 0 or thread_steps <= 0):
        raise ValueError("'num_per_step', 'num_particles', "
                         "'new_level_interval', and "
                         "'max_num_levels' must all be positive")
    if lam <= 0.0 or beta < 0.0:
        raise ValueError("'lam' and 'beta' must be non-negative")
    cdef Options options = Options(
        num_particles, new_level_interval, num_per_step, thread_steps,
        max_num_levels, lam, beta, 1
    )

    # Declarations.
    cdef int i, j, n, error
    cdef Sampler[PyModel] sampler = Sampler[PyModel](1, compression, options, 0)
    cdef vector[vector[Level]] copies_of_levels

    # Initialize the particles.
    n = sampler.size()
    for j in range(n):
        particle = sampler.particle(j)
        particle.set_py_self(model)
        error = particle.get_exception()
        if error != 0:
            raise DNest4Error(error)

    # Initialize the sampler.
    if seed is None:
        seed = time.time()
    cdef unsigned int seed_ = int(abs(seed))
    sampler.initialise(seed_)
    n = sampler.size()
    for j in range(n):
        particle = sampler.particle(j)
        error = particle.get_exception()
        if error != 0:
            raise DNest4Error(error)

    #First, distribute out the levels
    status_dict = read_status(sampler)
    requests = []
    for i in range(self.size):
        r = self.comm.isend(status_dict, dest=i + 1, tag=self.tags.SET_LEVELS)
        requests.append(r)

    MPI.Request.waitall(requests)

    i = 0
    while num_steps < 0 or i < num_steps:

        #tell each thread to run its own particle
        requests = []
        for j in range(n):
            r = self.comm.isend(j, dest=j + 1, tag=self.tags.RUN_THREAD)
            requests.append(r)

        MPI.Request.waitall(requests)

        #re-collect information about levels & particles
        results = []
        for i in range(n):
          worker = i+1
          if self.debug:
              print("Master waiting for worker {0} with tag {1}"
                    .format(worker, i))
          result = self.comm.recv(source=worker, tag=MPI.ANY_TAG)

          ( particle_coords, level_info ) = result
          particle = sampler.particle(i)
          particle.set_coords(particle_coords)

          n_levels = len(level_info["levels"])
          copies_of_levels[i].resize(n_levels)
          for j in range(n_levels):
            copies_of_levels[i][j] = Level(level_info["levels"][j]["log_likelihood"], level_info["levels"][j]["tiebreaker"],
                  level_info["levels"][j]["visits"],
                  level_info["levels"][j]["exceeds"],
                  level_info["levels"][j]["accepts"],
                  level_info["levels"][j]["tries"],
                  level_info["levels"][j]["log_X"])

          sampler.set_levels_copy(copies_of_levels[i], i)

        #now do the usual book-keeping done in run_thread
        sampler.process_threads()

        result = read_status(sampler, step=i)
        # Yield items as a generator.
        yield result

        # Hack to continue running.
        sampler.increase_max_num_saves(1)
        i += 1

  # def set_levels(self, result):
  #   cdef vector[Level] new_levels
  #
  #   #Set the level information
  #   n = result["levels"].size()
  #   new_levels.resize(n)
  #   for j in range(n):
  #     new_levels[j] =  Level(result["levels"][j]["log_likelihood"], result["levels"][j]["tiebreaker"], result["levels"][j]["visits"], result["levels"][j]["exceeds"], result["levels"][j]["accepts"], result["levels"][j]["tries"], result["levels"][j]["log_X"])
  #
  #   self.sampler.set_levels(new_levels)

  # def set_particle(self, params, part_idx):
  #   #Only thing that should matter is the numpy "params" array
  #   particle = sampler.particle(part_idx)
  #   particle.set_coords(params)

  def close(self):
    """
    Just send a message off to all the pool members which contains
    the special :class:`_close_pool_message` sentinel.
    """
    if self.is_master():
        for i in range(self.size):
            self.comm.isend(None, dest=i + 1, tag=self.tags.CLOSE)

  def __enter__(self):
      return self

  def __exit__(self, *args):
      self.close()

cdef read_status(Sampler[PyModel] sampler, step=0):

      n = sampler.size()
      result = dict(step=step)
      level_assignments = sampler.get_level_assignments()
      log_likelihoods = sampler.get_log_likelihoods()
      samples = []
      sample_info = []
      for j in range(n):
          # Errors?
          particle = sampler.particle(j)
          error = particle.get_exception()
          if error != 0:
              raise DNest4Error(error)

          # Results.
          samples.append(particle.get_npy_coords())
          sample_info.append((
              level_assignments[j],
              log_likelihoods[j].get_value(),
              log_likelihoods[j].get_tiebreaker(),
          ))

      # Convert the sampling results to arrays.
      result["samples"] = np.array(samples)
      result["sample_info"] = np.array(sample_info, dtype=[
          ("level_assignment", np.uint16),
          ("log_likelihood", np.float64),
          ("tiebreaker", np.float64),
      ])

      # Loop over levels and save them.
      levels = sampler.get_levels()
      n = levels.size()
      result["levels"] = np.empty(n, dtype=[
          ("log_X", np.float64), ("log_likelihood", np.float64),
          ("tiebreaker", np.float64), ("accepts", np.uint16),
          ("tries", np.uint16), ("exceeds", np.uint16),
          ("visits", np.uint16)
      ])
      for j in range(n):
          level = levels[j]
          result["levels"][j]["log_X"] = level.get_log_X()
          result["levels"][j]["log_likelihood"] = \
              level.get_log_likelihood().get_value()
          result["levels"][j]["tiebreaker"] = \
              level.get_log_likelihood().get_tiebreaker()
          result["levels"][j]["accepts"] = level.get_accepts()
          result["levels"][j]["tries"] = level.get_tries()
          result["levels"][j]["exceeds"] = level.get_exceeds()
          result["levels"][j]["visits"] = level.get_visits()

      return result

def sample(
    model,

    unsigned int max_num_levels,

    int num_steps=-1,
    unsigned int num_per_step=10000,

    # sampler args
    unsigned int num_particles=1,
    unsigned int new_level_interval=10000,
    unsigned int thread_steps=1,

    double lam=5.0,
    double beta=100.0,

    # "command line" arguments
    seed=None,
    double compression=np.exp(1.0),
):
    """
    Sample using a DNest4 model.

    :param model:
        A model class satisfying the DNest4 model protocol. This must
        implement the ``from_prior``, ``perturb``, and ``log_likelihood``
        methods.

    :param max_num_levels:
        The maximum number of levels to create.

    :param num_steps: (optional)
        The number of MCMC iterations (of ``num_per_step`` moves) to run. By
        default, this will run forever.

    :param num_particles: (optional)
        The number of particles in the ensemble. (default: ``1``)

    :param new_level_interval: (optional)
        The number of moves to run before creating a new level.
        (default: ``10000``)

    :param thread_steps: (optional)
        Pretty much irrelevant for the Python API. Kept for compatibility.
        (default: ``1``)

    :param lam: (optional)
        Backtracking scale length. (default: ``5.0``)

    :param beta: (optional)
        Strength of effect to force histogram to equal push.
        (default: ``100.0``)

    :param seed: (optional)
        Seed for the C++ random number generator.

    :param compression: (optional)
        The target compression factor. (default: ``np.exp(1)``)

    """
    # Check the model.
    if not hasattr(model, "from_prior") or not callable(model.from_prior):
        raise ValueError("DNest4 models must have a callable 'from_prior' method")
    if not hasattr(model, "perturb") or not callable(model.perturb):
        raise ValueError("DNest4 models must have a callable 'perturb' method")
    if not hasattr(model, "log_likelihood") or not callable(model.log_likelihood):
        raise ValueError("DNest4 models must have a callable 'log_likelihood' method")

    # Set up the options.
    if (num_per_step <= 0 or num_particles <= 0 or new_level_interval <= 0
            or max_num_levels <= 0 or thread_steps <= 0):
        raise ValueError("'num_per_step', 'num_particles', "
                         "'new_level_interval', and "
                         "'max_num_levels' must all be positive")
    if lam <= 0.0 or beta < 0.0:
        raise ValueError("'lam' and 'beta' must be non-negative")
    cdef Options options = Options(
        num_particles, new_level_interval, num_per_step, thread_steps,
        max_num_levels, lam, beta, 1
    )

    # Declarations.
    cdef int i, j, n, error
    cdef Sampler[PyModel] sampler = Sampler[PyModel](1, compression, options, 0)
    cdef PyModel* particle
    cdef vector[Level] levels
    cdef Level level
    cdef vector[unsigned] level_assignments
    cdef vector[LikelihoodType] log_likelihoods

    # Initialize the particles.
    n = sampler.size()
    for j in range(n):
        particle = sampler.particle(j)
        particle.set_py_self(model)
        error = particle.get_exception()
        if error != 0:
            raise DNest4Error(error)

    # Initialize the sampler.
    if seed is None:
        seed = time.time()
    cdef unsigned int seed_ = int(abs(seed))
    sampler.initialise(seed_)
    n = sampler.size()
    for j in range(n):
        particle = sampler.particle(j)
        error = particle.get_exception()
        if error != 0:
            raise DNest4Error(error)

    i = 0
    while num_steps < 0 or i < num_steps:
        sampler.run()

        # Loop over particles, build the results list, and check for errors.
        n = sampler.size()
        result = dict(step=i)
        level_assignments = sampler.get_level_assignments()
        log_likelihoods = sampler.get_log_likelihoods()
        samples = []
        sample_info = []
        for j in range(n):
            # Errors?
            particle = sampler.particle(j)
            error = particle.get_exception()
            if error != 0:
                raise DNest4Error(error)

            # Results.
            samples.append(particle.get_npy_coords())
            sample_info.append((
                level_assignments[j],
                log_likelihoods[j].get_value(),
                log_likelihoods[j].get_tiebreaker(),
            ))

        # Convert the sampling results to arrays.
        result["samples"] = np.array(samples)
        result["sample_info"] = np.array(sample_info, dtype=[
            ("level_assignment", np.uint16),
            ("log_likelihood", np.float64),
            ("tiebreaker", np.float64),
        ])

        # Loop over levels and save them.
        levels = sampler.get_levels()
        n = levels.size()
        result["levels"] = np.empty(n, dtype=[
            ("log_X", np.float64), ("log_likelihood", np.float64),
            ("tiebreaker", np.float64), ("accepts", np.uint16),
            ("tries", np.uint16), ("exceeds", np.uint16),
            ("visits", np.uint16)
        ])
        for j in range(n):
            level = levels[j]
            result["levels"][j]["log_X"] = level.get_log_X()
            result["levels"][j]["log_likelihood"] = \
                level.get_log_likelihood().get_value()
            result["levels"][j]["tiebreaker"] = \
                level.get_log_likelihood().get_tiebreaker()
            result["levels"][j]["accepts"] = level.get_accepts()
            result["levels"][j]["tries"] = level.get_tries()
            result["levels"][j]["exceeds"] = level.get_exceeds()
            result["levels"][j]["visits"] = level.get_visits()

        # Yield items as a generator.
        yield result

        # Hack to continue running.
        sampler.increase_max_num_saves(1)
        i += 1
