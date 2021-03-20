from src.genom import Genom, crossover_genes
from src.registry import Registry

from test_case.classification_problem import input_set, inside


class Population:
    def __init__(self, input_size: int, output_size: int, population_size: int, fitness_fn):
        self.registry = Registry(input_size, output_size)
        self.fitness_fn = fitness_fn

        self.population_size = population_size
        self.population = []
        self.scores = []

        for _ in range(self.population_size):
            self.population.append(Genom(self.registry))

    def update(self):
        """ evaluate all genoms, genetic operations, next population """

        # reset scores
        self.scores = []
        pass

    def evaluate_nets(self):
        in_data = input_set
        for g in self.population:
            score = 0
            g.set_next_nodes()
            for i in range(in_data.shape[0]):
                pred = g.forward(in_data[i])
                ret = self.fitness_fn(*in_data[i], pred)
                score += ret
            print(score)
            self.scores.append(score)


pop = Population(2, 2, 100, inside)
pop.evaluate_nets()
