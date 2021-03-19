from src.genom import Genom, crossover_genes
from src.registry import Registry
import numpy as np


class Population:
    def __init__(self, input_size: int, output_size: int, population_size: int):
        self.registry = Registry(input_size, output_size)

        self.population_size = population_size
        self.population = []

        for _ in range(self.population_size):
            self.population.append(Genom(self.registry))

        x = np.random.random(25)
        for g in self.population:
            g.set_next_nodes()
            ret = g.forward(x)
            print(ret)

    def update(self):
        """ evaluate all genoms, genetic operations, next population """
        pass


pop = Population(25, 5, 100)
