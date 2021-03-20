from src.genom import Genom, crossover_genes, distance
from src.registry import Registry

from test_case.classification_problem import input_set, inside

"""
Score has to be >= 0!
"""

D_THRESHOLD = 2.


class Population:
    def __init__(self, input_size: int, output_size: int, population_size: int, fitness_fn):
        self.registry = Registry(input_size, output_size)
        self.fitness_fn = fitness_fn

        self.population_size = population_size
        self.population = []
        self.scores = []

        self.species = []
        self.new_size = 0

        for _ in range(self.population_size):
            self.population.append(Genom(self.registry))

    def update(self):
        """ evaluate all genoms, genetic operations, next population """
        self.evaluate_nets()
        self.speciate()
        self.explicit_fitness_sharing()

        # reset for next generation
        self.scores = []
        self.species = []
        self.new_size = 0

    def evaluate_nets(self):
        in_data = input_set
        for g in self.population:
            score = 0
            g.set_next_nodes()
            for i in range(in_data.shape[0]):
                pred = g.forward(in_data[i])
                ret = self.fitness_fn(*in_data[i], pred)
                score += ret
            # print(score)
            self.scores.append(score)

    def speciate(self):
        # save genoms in species as tuple (genom, score)
        c = 0
        self.species = []
        for g, s in zip(self.population, self.scores):
            succ = False
            if len(self.species) == 0:
                succ = True
                self.species.append([[g, s]])
            else:
                for subspec in self.species:
                    delta = distance(g, subspec[0][0])
                    if delta < D_THRESHOLD:
                        succ = True
                        subspec.append([g, s])
                        break
            if not succ:
                self.species.append([[g, s]])

    def explicit_fitness_sharing(self):
        # norm each genoms fitness with size of its subspecies
        for subspec in self.species:
            members = len(subspec)
            for i in range(len(subspec)):
                subspec[i][1] /= members

        # calculate over all fitness average
        score_sum, pop_sz = 0, 0
        for subspec in self.species:
            for (_, score) in subspec:
                pop_sz += 1
                score_sum += score
        mean_entire = score_sum / pop_sz

        # New species sizes
        new_size = []
        for subspec in self.species:
            sz_new = 1/mean_entire * sum([sc for (_, sc) in subspec])
            new_size.append(sz_new)
        self.new_size = new_size


pop = Population(2, 2, 100, inside)
pop.update()
