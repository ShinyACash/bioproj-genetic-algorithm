import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

## Simulation parameters
num_creatures = 10
num_food_sources = 10
mutation_rate = 0.01
max_generations = 100
env_size = 100
steps_per_gen = 50
initial_learning_rate = 0.01  
learning_rate_increase = 0.005  


creatures = np.random.rand(num_creatures, 2) * env_size
food_sources = np.random.rand(num_food_sources, 2) * env_size


def fitness(creature, food_sources):
    distances = np.linalg.norm(food_sources - creature, axis=1)
    return 1 / np.min(distances)

# Selection (roulette wheel selection)
def selection(fitness_scores):
    total_fitness = np.sum(fitness_scores)
    probabilities = fitness_scores / total_fitness
    return np.random.choice(len(fitness_scores), p=probabilities)


def crossover(parent1, parent2):
    crossover_point = np.random.randint(len(parent1))
    child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
    child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
    return child1, child2


def mutation(individual, mutation_rate):
    for i in range(len(individual)):
        if np.random.rand() < mutation_rate:
            individual[i] += np.random.randn() * 0.1
    return individual


fig, ax = plt.subplots()

def update(frame):
    global creatures, food_sources, learning_rate
    
    learning_rate = min(initial_learning_rate + (frame // steps_per_gen) * learning_rate_increase, 1.0)

    # Creature movement and food consumption
    for i in range(num_creatures):
        closest_food = np.argmin(np.linalg.norm(food_sources - creatures[i], axis=1))
        direction_to_food = food_sources[closest_food] - creatures[i]
        direction_to_food /= np.linalg.norm(direction_to_food)  

        random_direction = np.random.randn(2) * 0.1
        movement = (1 - learning_rate) * random_direction + learning_rate * direction_to_food

        creatures[i] += movement
        creatures[i] = np.clip(creatures[i], 0, env_size)

        if np.linalg.norm(food_sources[closest_food] - creatures[i]) < 1:
            food_sources[closest_food] = np.random.rand(2) * env_size

    ax.clear()
    ax.scatter(creatures[:, 0], creatures[:, 1], s=20, c='blue')
    ax.scatter(food_sources[:, 0], food_sources[:, 1], s=50, c='red')
    ax.set_xlim(0, env_size)
    ax.set_ylim(0, env_size)
    ax.set_title(f"Generation: {frame // steps_per_gen + 1}")

    # Selection, crossover, and mutation after each generation
    if frame % steps_per_gen == steps_per_gen - 1:
        fitness_scores = np.array([fitness(creature, food_sources) for creature in creatures])
        new_generation = []
        for _ in range(num_creatures):
            parent1 = creatures[selection(fitness_scores)]
            parent2 = creatures[selection(fitness_scores)]
            child1, child2 = crossover(parent1, parent2)
            new_generation.append(mutation(child1, mutation_rate))
            new_generation.append(mutation(child2, mutation_rate))
        creatures = np.array(new_generation)

ani = animation.FuncAnimation(fig, update, frames=max_generations * steps_per_gen, interval=10)
plt.show()