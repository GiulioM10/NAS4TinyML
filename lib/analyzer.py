from individual import Individual
import numpy as np
import json
from typing import List, Tuple
import torch

class Analyzer():
    def __init__(self, output_file: str) -> None:
        """This class implements utilities to save progress during the NAS procedure

        Args:
            output_file (str): The path of the json file where the outputs are stored
        """
        self.output_file = output_file
    
    def snapshot_experiment(self, population: List[Individual], elapsed_time: float) -> None:
        """Save current state of an experiment i.e. all of the members of the current population
        and the currently elapsed time since experiment begun

        Args:
            population (List[Individual]): The networks to be saved
            elapsed_time (float): Minutes since beginning of the experiment

        Returns:
            None: The data is stored in a json file
        """
        output = {}
        output["pop_size"] = len(population)
        output["elapsed_minutes"] = elapsed_time

        def convert_int64(obj):
            if isinstance(obj, np.int64):
                return int(obj)
            else:
                return obj
    
        output["population"] = []
        for individual in population:
            indi = {}
            indi["genome"] = individual.genotype
            indi["cost_info"] = individual.get_cost_info()
            indi["metrics"] = individual.get_metrics()
            output["population"].append(indi)
        json_string = json.dumps(output, default=convert_int64)
        with open(self.output_file, 'w') as out_file:
            out_file.write(json_string)

    def load_experiments_result(self, file_path: str = None, device: torch.device = None) -> Tuple:
        """Load the Networks stored in a json file as well as how much time the experiment run

        Args:
            file_path (str, optional): The path of the json file containing the information. Defaults to None.
            device (torch.device, optional): The device we want to assign the networks. Defaults to None.

        Returns:
            Tuple: A lists of Individuals, the elapsed time and the length of the individual list
        """
        if file_path is None:
            file_path = self.output_file
        
        with open(file_path, 'r') as open_file:
            data = json.load(open_file)
        results = []
        for individual in data["population"]:
            ind = Individual(individual["genome"], None, device)
            ind.set_cost_info(individual["cost_info"])
            ind.set_metrics(individual["metrics"])
            results.append(ind)
        elapsed_time = data["elapsed_minutes"]
        pop_size = data["pop_size"]
        return results, elapsed_time, pop_size
