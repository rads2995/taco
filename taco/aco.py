import numpy as np
import numpy.typing as npt

class aco:

    num_cities: int = 0     # Number of cities to visit
    num_ants: int = 0       # Number of ants
    num_iter: int = 100     # Number of iterations
    alpha: float = 1        # Control influence of tau 
    beta: float = 1         # Control influence of eta 
    rho: float = 0.1        # Pheromone evaporation coefficient
    delta_tau: float = 1    # Pheromone deposited by ant
    distance_matrix: npt.NDArray[np.float64]
    pheromone_matrix: npt.NDArray[np.float64]
    desire_matrix: npt.NDArray[np.float64]

    best_path = None
    best_length = float('inf')
    pheromones: int = 0

    def __init__(self, distance_matrix: npt.NDArray[np.float64], num_ants: int, num_iter: int, alpha: float, beta: float, rho: float, delta_tau: float) -> None:
              
        self.distance_matrix = distance_matrix
        self.num_cities = len(distance_matrix)
        self.num_ants = num_ants
        self.num_iter = num_iter
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.delta_tau = delta_tau 
        self.pheromone_matrix = np.ones(np.shape(self.distance_matrix))
        
        with np.errstate(divide = 'ignore'): 
            self.desire_matrix = np.where(self.distance_matrix != 0, 1/self.distance_matrix, 0)

    def run(self):
        ant_paths = []
        path_lengths = []

        # Simulate all ants
        for _ in range(self.num_ants):
            path:int = [np.random.randint(self.num_cities)] # Start at random city
            visited = set(path)

            # Build a complete tour
            while len(path) < self.num_cities:
                current_city = path[-1]
                probabilities = []

                for city in range(self.num_cities):
                    if city not in visited:
                        pheromone = self.pheromone_matrix[current_city, city] ** self.alpha
                        heuristic = (1 / self.distance_matrix[current_city, city]) ** self.beta
                        probabilities.append(pheromone * heuristic)
                    else:
                        probabilities.append(0)

                probabilities = np.array(probabilities)
                probabilities /= probabilities.sum()  # Normalize
                next_city = np.random.choice(range(self.num_cities), p=probabilities)
                path.append(next_city)
                visited.add(next_city)
         
            ant_paths.append(path)

            # Calculate path length
            length = sum(self.distance_matrix[path[i], path[i+1]] for i in range(self.num_cities - 1))
            length += self.distance_matrix[path[-1], path[0]]  # Return to start
            path_lengths.append(length)

        # Update best path if a shorter one is found

        shortest_index = np.argmin(path_lengths)

        if path_lengths[shortest_index] < self.best_length:
            self.best_length = path_lengths[shortest_index]
            self.best_path = ant_paths[shortest_index]


        # Update pheromones

        self.pheromones *= (1 - self.rho)  # Evaporation

        for path, length in zip(ant_paths, path_lengths):

            pheromone_addition = self.delta_tau / length

            for i in range(self.num_cities - 1):

                self.pheromones[path[i], path[i+1]] += pheromone_addition

                self.pheromones[path[i+1], path[i]] += pheromone_addition  # Symmetric

            # Add self.pheromones for returning to the starting city

            self.pheromones[path[-1], path[0]] += pheromone_addition

            self.pheromones[path[0], path[-1]] += pheromone_addition

        return best_path, best_length
