import numpy as np
from model import train_network_forward_propagation



def pso_iris(num_particles, num_iterations, hidden_layer_size, X_train_data, y_train_data):
    # PSO parameters
    num_dimensions = 4 * hidden_layer_size + hidden_layer_size * 3
    positions = np.random.rand(num_particles, num_dimensions) - 0.5  # Initialize positions
    velocities = np.zeros_like(positions)  # Initialize velocities
    pbest_positions = np.copy(positions)
    pbest_scores = np.array([train_network_forward_propagation(p, hidden_layer_size, X_train_data, y_train_data) for p in positions])
    gbest_position = pbest_positions[np.argmin(pbest_scores)]
    gbest_score = np.min(pbest_scores)


    # PSO loop
    w = 0.5  # inertia
    c1 = 2  # cognitive parameter
    c2 = 2  # social parameter
    for i in range(num_iterations):
        for j in range(num_particles):
            r1, r2 = np.random.rand(2)
            velocities[j] = w * velocities[j] + c1 * r1 * (pbest_positions[j] - positions[j]) + c2 * r2 * (gbest_position - positions[j])
            positions[j] += velocities[j]
            
            current_score = train_network_forward_propagation(positions[j], hidden_layer_size, X_train_data, y_train_data)
            if current_score < pbest_scores[j]:
                pbest_scores[j] = current_score
                pbest_positions[j] = positions[j]
        
        # Update global best
        current_gbest_score = np.min([train_network_forward_propagation(p, hidden_layer_size, X_train_data, y_train_data) for p in positions])
        if current_gbest_score < gbest_score:
            gbest_score = current_gbest_score
            gbest_position = positions[np.argmin([train_network_forward_propagation(p, hidden_layer_size, X_train_data, y_train_data) for p in positions])]

        print(f"Iteration {i+1} - Best Loss: {gbest_score}")

    print("Optimal Weights Found by PSO:", gbest_position)
    return gbest_position