from individual import Individual
import numpy as np
import json
from typing import List

class Analyzer():
    def __init__(self, output_file: str) -> None:
        self.output_file = output_file
    
    def snapshot_experiment(self, population: List[Individual], elapsed_time: float) -> None:
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

    def load_experiments_result(self, file_path: str = None, device = None):
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
        return results
