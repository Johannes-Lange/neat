from src.population import Population

pop = Population(2, 2, 150, max_hidden=10, fitness_fn=None)

for _ in range(1000):
    pop.update()
