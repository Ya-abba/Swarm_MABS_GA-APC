import numpy as np

class ACO:
    def __init__(self, num_ants, num_iterations, num_dimensions, bounds, fitness_function, alpha=1, beta=2, evaporation_rate=0.5):
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.num_dimensions = num_dimensions
        self.bounds = bounds
        self.fitness_function = fitness_function
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate

        self.pheromone = np.ones((num_dimensions, num_dimensions))
        self.best_solution = None
        self.best_score = float('inf')

    def optimize(self):
        for _ in range(self.num_iterations):
            solutions = []
            scores = []

            for ant in range(self.num_ants):
                solution = np.random.uniform(self.bounds[0], self.bounds[1], self.num_dimensions)
                score = self.fitness_function(solution)
                solutions.append(solution)
                scores.append(score)

                for i in range(len(solution) - 1):
                    self.pheromone[i][i + 1] += 1.0 / score

            best_ant = np.argmin(scores)
            if scores[best_ant] < self.best_score:
                self.best_score = scores[best_ant]
                self.best_solution = solutions[best_ant]

            self.pheromone *= self.evaporation_rate
        return self.best_solution, self.best_score
