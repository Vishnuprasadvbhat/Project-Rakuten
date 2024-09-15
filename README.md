

# Particle Swarm Optimization (PSO)

Particle Swarm Optimization (PSO) is an optimization algorithm inspired by the collective behavior of biological systems. Unlike Backpropagation, PSO does not rely on gradients, making it a metaheuristic approach that searches large solution spaces without guaranteeing an optimal solution. PSO operates through three key components: particles, constants, and iterations.

## Particles

<p align="center">
  <img src="https://miro.medium.com/v2/resize:fit:640/format:webp/1*tF8zXKlWSTwOfjChLSKgzQ.png" alt="alt" width="70%" />
</p>

The first key element in PSO is the particle. At the start, a defined number of particles is initialized, each representing a potential solution. When optimizing a neural network, each particle corresponds to a specific set of weights.


<br>


<p align="center">
  <img src="https://editor.analyticsvidhya.com/uploads/98419Fig3.png" alt="alt" width="80%" />
</p>
These properties dictate the particle’s direction in the solution space.

1. **Velocity**: Initialized randomly, it controls the particle's movement.
2. **Personal Best Solution (pbest)**: The best solution that the particle has discovered so far.
3. **Global Best Solution (gbest)**: The best solution discovered by the entire swarm.



## Constants


PSO uses three constants that influence particle movement:



1. **Cognitive Coefficient (c1)**: Influences how much the particle is pulled toward its personal best solution.
2. **Social Coefficient (c2)**: Influences how much the particle is attracted to the global best solution.
3. **Inertia (w)**: Controls how much of the particle's previous velocity is retained.

The particle’s next velocity is determined by combining its current velocity, the personal best (pbest), and the global best (gbest), along with two random factors (r1 and r2). This balance guides the particle between its own experience and the swarm's collective knowledge.

## Iterations

<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/ParticleSwarmArrowsAnimation.gif/330px-ParticleSwarmArrowsAnimation.gif" alt="alt" width="80%" />
</p>
Iterations represent the number of times particles update their velocities and move through the solution space. In each iteration, particles adjust their positions based on their velocity, personal best, and global best solutions, refining the search for the optimal solution.

As the process continues, particles converge toward the best solution, with each iteration improving the swarm's overall performance.

## References:
<ul>

- [**ScienceDirect**](https://www.sciencedirect.com/science/article/abs/pii/S2210650218309246)
- [**PSO Algorithm**](https://en.wikipedia.org/wiki/Particle_swarm_optimization)
