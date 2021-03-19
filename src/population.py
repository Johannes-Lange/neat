from src.genom import Genom, crossover_genes
from src.registry import Registry

from test_case.classification_problem import input_set, inside_circle


class Population:
    def __init__(self, input_size: int, output_size: int, population_size: int, fitness_fn):
        self.registry = Registry(input_size, output_size)
        self.fitness_fn = fitness_fn

        self.population_size = population_size
        self.population = []

        for _ in range(self.population_size):
            self.population.append(Genom(self.registry))

        # # temp
        # x = np.random.random(25)
        # for g in self.population:
        #     g.mutate_add_node()
        #     g.set_next_nodes()
        #     ret = g.forward(x)
        #     print(ret)
        # print(self.registry.connections)

    def update(self):
        """ evaluate all genoms, genetic operations, next population """
        pass


pop = Population(2, 1, 100, inside_circle)
