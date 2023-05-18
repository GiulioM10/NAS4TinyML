# GM 05/17/23
import time
import numpy as np
from space import Space
from individual import Individual
import types
from typing import List
from analyzer import Analyzer

class Search:
  def __init__(self, space: Space, max_params: float, max_flops: float, analyzer: Analyzer) -> None:
    """Class to search over a certain space

    Args:
        space (Space): Space over wich toperform search
        max_params (float): Maximum number of desired parameters
        max_flops (float): Maximum number of flops allowed  
        analyzer (Analyzer): An analyzer to manage the results
    """
    self.space = space
    self.max_params = max_params
    self.max_flops = max_flops
    self.analyzer = analyzer

  def is_feasible(self, individual: Individual) -> bool:
    """Return wether an Individual satisfies the constraints

    Args:
        individual (Individual): The individial to check

    Returns:
        bool: Wether the individual is feasible or not
    """
    cost = individual.get_cost_info()
    if cost['flops'] <= self.max_flops and cost['params'] <= self.max_params:
        return True
    return False

  def population_init(self, N: int) -> List[Individual]:
    """Get a valid intial population

    Args:
        N (int): Size of population

    Returns:
        List[Individual]: The population
    """
    population = []
    while len(population) < N:
      new = self.space.get_random_population(1)[0]
      if self.is_feasible(new):
        population.append(new)
    return population
  
  def return_top_k(self, population: List[Individual], k: int):
    """Return the k most fitted indivisuals of the population

    Args:
        population (List[Individual]): Population among which we wanto to choose the best
        k (int): How many idivisuals are returned

    Returns:
        List: List containing the k most performing individuals
    """
    individuals = [individual for individual in population if individual.has_metrics]
    for individual in individuals:
      individual.rank = []

    for metric_n in self.space.metrics:
      individuals.sort(key = lambda x: -x.get_metrics()[metric_n])
      for index, individual in enumerate(individuals):
        individual.rank.append(index + 1)
    
    for individual in individuals:
      individual.rank = np.mean(individual.rank)

    individuals.sort(key = lambda x: x.rank)

    return np.array(individuals)[:k]
  
  def return_best(self, population: List[Individual]):
    """Return the individual(s) with the best rank

    Args:
        population (List[Individual]): Population among which we wanto to choose the best

    Returns:
        List: List containing the best individual(s)
    """
    individuals = [individual for individual in population if individual.has_metrics]
    for individual in individuals:
      individual.rank = []

    for metric_n in self.space.metrics:
      individuals.sort(key = lambda x: -x.get_metrics()[metric_n])
      for index, individual in enumerate(individuals):
        individual.rank.append(index + 1)
    
    for individual in individuals:
      individual.rank = np.mean(individual.rank)

    individuals.sort(key = lambda x: x.rank)

    best = [individuals[0]]
    for i in range(len(individuals) - 1):
        if individuals[i + 1].rank == individuals[i].rank:
            best.append(individuals[i + 1])
        else:
            break
    return best
  
  def mutate(self, individuals: List[Individual],
             R: int = 1, P: int =1,
             generation: int = 0,
             skip_downsampling: bool = True) -> List[Individual]:
    """Generate mutated individual starting from a population

    Args:
        individuals (List[Individual]): The ancestors to mutate
        R (int, optional): Number of nucleotides to muate. Defaults to 1.
        P (int, optional): Number of different mutations. Defaults to 1.
        generation (int, optional): The generation of the individuals. Defaults to 0.
        skip_downsampling (bool, optional): Wether to consider downsampling or not. Defaults to True.

    Returns:
        List[Individual]: Mutated individuals
    """
    new_individuals = []
    for _ in range(P):
      for individual in individuals:
        gen = [gene[:] for gene in individual.genotype]
        ugo = self.space.mutation(gen, R, skip_downsampling)
        ugo.set_generation(generation)
        new_individuals.append(ugo)
    return new_individuals

  def random(self, max_time: float, save_time: float, load_from: str = None):
    """Perform a random search on the search space

    Args:
        max_time (float): Max search time
        save_time (float): After how much time a snapshot is taken
        load_from (str, optional): Path of the file containing the info to be loaded. If None
        the experiments starts from the beggining. Defaults to None.

    Raises:
        Exception: The experiment has already ended
        Exception: The loaded data come from a different type of experiment
    """
    elaps = 0
    loaded = False
    start = time.time()
    while elaps < max_time:
        elaps_save_time = 0
        individuals = []
        if load_from is not None and not loaded:
          _, prev_best, experiment_age, search_alg, _, _ = self.analyzer.load_experiments_result(self.space, load_from)
          max_time = max_time - experiment_age
          if search_alg != "Random":
            raise Exception("Search procedures do not match")
          if max_time <= 0:
            raise Exception("Experiment already concluded")
          for best_ind in prev_best:
            individuals.append(best_ind)
          loaded = True
        start_save_time = time.time()
        while elaps_save_time < save_time:
            new_network = self.population_init(1)[0]
            new_network.set_metrics()
            individuals.append(new_network)
            end_save_time = time.time()
            elaps_save_time = (end_save_time - start_save_time)/60
        # return top performing networks and save them
        best = self.return_best(individuals)
        end = time.time()
        elaps = (end - start)/60
        self.analyzer.snapshot_experiment(best, [], elaps, "Random")
        prev_best = best
    self.analyzer.snapshot_experiment(best, [], elaps, "Random")
    print("End")
    
  def freeREAminus(self, max_time: float, number_steps_save: int, pop_size: int = 25, sample_size:int = 5, load_from:str = None):
    """Regularized evolutionary algorithm for individual selection

    Args:
        max_time (float): Duration of the experiment
        number_steps_save (int): After how many steps we wish to save the best individual
        pop_size (int, optional): How many individuals are alive at each step. Defaults to 25.
        sample_size (int, optional): How many alive individuals are chosen at each step. Defaults to 5.
        load_from (str, optional): Path of the file from which the experiment is loaded. Defaults to None.

    Raises:
        Exception: The search procedures do not match
        Exception: Experiment already concluded
        Exception: Population sizes do not match
    """
    loaded = False
    if load_from is not None:
      results, prev_best, elapsed_time, search_alg, l_pop_size, gen =\
        self.analyzer.load_experiments_result(self.space, load_from)
      loaded = True
    if not loaded:
      start = time.time()
      population = self.population_init(pop_size)
      end = time.time(); elaps = (end - start)/60
      print("Initialization done in {} minutes".format(elaps))
      self.analyzer.snapshot_experiment([], population, elaps, "FreeREA", 0)
      step = 0; prev_best = []
    else:
      step = gen
      max_time = max_time - elapsed_time
      if search_alg != "FreeREA":
        raise Exception("Search procedures do not match")
      if max_time <= 0:
        raise Exception("Experiment already concluded")
      if pop_size != l_pop_size:
        raise Exception("Loaded and expressed pop sizes do not match")
      population = results
      start = time.time(); elaps = elapsed_time
      
    while elaps <= max_time:
      while True:
        sampled = np.random.choice(population, sample_size)
        for individual in sampled:
          if not(individual.has_metrics):
            individual.set_metrics()
        parent = self.return_top_k(sampled, 1)
        offspring = self.mutate(parent, 1, 1, step + 1)[0]
        if self.is_feasible(offspring):
          break
      population.append(offspring)
      population.pop(0)
      step += 1
      end = time.time(); elaps = (end - start)/60
      # Every tot generations update the best and save a snapshot of the experiment
      if step % number_steps_save == 0:
        pool = population + prev_best
        best = self.return_best(pool)
        best[0].print_structure()
        print(best[0].generation, best[0].get_metrics(), best[0].get_cost_info())
        self.analyzer.snapshot_experiment(best, population, elaps, "FreeREA", step)
        prev_best = best
    self.analyzer.snapshot_experiment(best, population, elaps, "FreeREA", step)
    print("End")
    
    
