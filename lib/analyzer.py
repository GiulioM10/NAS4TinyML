from individual import Individual
import json
from typing import List

class Analyzer():
    def __init__(self, output_file: str) -> None:
        self.output_file = output_file
    
    def snapshot_experiment(self, population: List[Individual], elapsed_time: float) -> None:
        output = {}
        output["pop_size"] = len(population)
        output["population"] = []
        for individual in population:
            indi = {}
            indi["genome"] = individual.genotype
            indi["cost_info"] = individual.get_cost_info()
            indi["metrics"] = individual.get_metrics()
            output["population"].append(indi)
        output["elapsed_time"] = elapsed_time/3600
        json_string = json.dumps(output)
        print(json_string)
        with open(self.output_file, 'w') as out_file:
            json.dump(json_string, out_file)
            out_file.close()
