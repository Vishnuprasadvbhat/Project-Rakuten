import numpy as np

class Particle:
    def __init__(self, dimension):
        self.position = np.random.uniform(-1, 1, dimension)
        self.velocity = np.random.uniform(-0.5, 0.5, dimension)
        self.best_position = self.position.copy()
        self.best_score = float('inf')

    def update_velocity(self, global_best_position, inertia, cognitive_const, social_const):
        cognitive_velocity = cognitive_const * np.random.random() * (self.best_position - self.position)
        social_velocity = social_const * np.random.random() * (global_best_position - self.position)
        self.velocity = inertia * self.velocity + cognitive_velocity + social_velocity

    def update_position(self):
        self.position += self.velocity

    def evaluate(self, objective_function):
        score = objective_function(self.position)
        if score < self.best_score:
            self.best_score = score
            self.best_position = self.position.copy()
        return score

def pso(objective_function, dimension, num_particles=30, max_iter=100, inertia=0.7, cognitive_const=1.5, social_const=2.0):
    particles = [Particle(dimension) for _ in range(num_particles)]
    global_best_position = None
    global_best_score = float('inf')

    for _ in range(max_iter):
        for particle in particles:
            score = particle.evaluate(objective_function)
            if score < global_best_score:
                global_best_score = score
                global_best_position = particle.position.copy()

        for particle in particles:
            particle.update_velocity(global_best_position, inertia, cognitive_const, social_const)
            particle.update_position()

    return global_best_position, global_best_score

# Example usage:

def objective_function(weights):
    # This should be your actual objective function, for example, 
    # mean squared error for a neural network, etc.
    return np.sum(weights ** 2)

dimension = 10  # For example, if you are optimizing 10 weights
best_weights, best_score = pso(objective_function, dimension)

print(f'Best Weights: {best_weights}')
print(f'Best Score: {best_score}')
