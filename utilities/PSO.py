import numpy as np

class PSO:
    def __init__(self, num_particles, num_dimensions, bounds, fitness_function, max_velocity=1.0, inertia=0.5, cognitive=1.5, social=1.5):
        self.num_particles = num_particles
        self.num_dimensions = num_dimensions
        self.bounds = bounds
        self.fitness_function = fitness_function
        self.max_velocity = max_velocity
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social

        self.positions = np.random.uniform(bounds[0], bounds[1], (num_particles, num_dimensions))
        self.velocities = np.zeros((num_particles, num_dimensions))
        self.best_positions = np.copy(self.positions)
        self.best_scores = np.full(num_particles, float('inf'))
        self.global_best_position = None
        self.global_best_score = float('inf')

    def optimize(self, iterations):
        for _ in range(iterations):
            for i in range(self.num_particles):
                score = self.fitness_function(self.positions[i])
                if score < self.best_scores[i]:
                    self.best_scores[i] = score
                    self.best_positions[i] = self.positions[i]
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = self.positions[i]

            r1, r2 = np.random.rand(self.num_particles, self.num_dimensions), np.random.rand(self.num_particles, self.num_dimensions)
            self.velocities = self.inertia * self.velocities + \
                              self.cognitive * r1 * (self.best_positions - self.positions) + \
                              self.social * r2 * (self.global_best_position - self.positions)
            self.velocities = np.clip(self.velocities, -self.max_velocity, self.max_velocity)
            self.positions = np.clip(self.positions + self.velocities, self.bounds[0], self.bounds[1])
        return self.global_best_position, self.global_best_score
