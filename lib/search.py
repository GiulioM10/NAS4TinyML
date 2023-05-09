import time
import numpy as np
from space import Space
from individual import Individual
import types
from typing import List
from analyzer import Analyzer

class Search:
  def __init__(self, space: Space, max_params: float, max_flops: float, analyzer: Analyzer) -> None:
     self.space = space
     self.max_params = max_params
     self.max_flops = max_flops
     self.analyzer = analyzer

  def _is_feasible(self, individual: Individual) -> bool:
    cost = individual.get_cost_info()
    if cost['flops'] <= self.max_flops and cost['params'] <= self.max_params:
        return True
    return False

  def population_init(self, N: int) -> List[Individual]:
    population = []
    while len(population) < N:
      new = self.space.get_random_population(1)[0]
      if self._is_feasible(new):
        population.append(new)
    return population
  
  def return_top_k(self, population, k):
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
  
  def return_best(self, population):
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

  def random(self, max_time: float, save_time: float, prev_best: Individual = None):
    elaps = 0
    start = time.time()
    while elaps < max_time:
        elaps_save_time = 0
        individuals = []
        if prev_best is not None:
          for best_ind in prev_best:
            individuals.append(best_ind)
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
        self.analyzer.snapshot_experiment(best, elaps)
        prev_best = best
        for prev_best_ind in prev_best:
            print(prev_best_ind.rank)
            print("-------------")
    print("End")
