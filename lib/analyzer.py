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
        output["elapsed_hours"] = 0
        output["elapsed_hours"] = elapsed_time
        print(elapsed_time)

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
        print(json_string)
        with open(self.output_file, 'w') as out_file:
            out_file.write(json_string)
