import numpy as np

# Define the fitness function to be maximized (e.g., a simple function)
def fitness_function(x):
    return -x**2 + 4*x - 5

# Genetic Algorithm parameters
population_size = 100
num_generations = 50
mutation_rate = 0.1

# Initialize a random population
population = np.random.rand(population_size)

# Genetic Algorithm loop
for generation in range(num_generations):
    # Evaluate fitness of each individual in the population
    fitness_scores = [fitness_function(ind) for ind in population]
    
    # Select parents for reproduction
    parents = np.random.choice(population, size=population_size//2, p=fitness_scores/np.sum(fitness_scores))
    
    # Perform crossover and mutation
    offspring = []
    for i in range(len(parents) - 1):
        crossover_point = np.random.randint(1, len(parents[i]))
        child = np.concatenate((parents[i][:crossover_point], parents[i+1][crossover_point:]))
        offspring.append(child)
    
    # Apply mutation
    for i in range(len(offspring)):
        if np.random.rand() < mutation_rate:
            mutation_position = np.random.randint(len(offspring[i]))
            offspring[i][mutation_position] = np.random.rand()
    
    # Replace old population with new offspring
    population = np.concatenate((parents, offspring))

# Find the best solution in the final population
best_solution = population[np.argmax([fitness_function(ind) for ind in population])]
print("Best solution:", best_solution)
print("Best fitness:", fitness_function(best_solution))
