import numpy as np
import numpy.typing as npt

class aco:

    num_cities: int     # Number of cities to visit
    num_ants: int       # Number of ants
    num_iter: int       # Number of iterations
    alpha: float        # Control influence of tau 
    beta: float         # Control influence of eta 
    rho: float          # Pheromone evaporation coefficient
    delta_tau: float    # Pheromone deposited by ant
    distance_matrix: npt.NDArray[np.float64]     
    pheromone_matrix: npt.NDArray[np.float64]   # Matrix tau   
    # desire_matrix: npt.NDArray[np.float64]      # Matrix eta

    best_path = None
    best_length = float('inf')

    def __init__(self, distance_matrix: npt.NDArray[np.float64], num_ants: int = 20, 
                       num_iter: int = 100, alpha: float = 1, beta: float = 1,
                       rho: float = 0.5, delta_tau: float = 1) -> None:
              
        self.distance_matrix = distance_matrix
        self.num_cities = len(distance_matrix)
        self.num_ants = num_ants
        self.num_iter = num_iter
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.delta_tau = delta_tau 
        self.pheromone_matrix = np.ones(np.shape(self.distance_matrix))
        
        # with np.errstate(divide = 'ignore'): 
        #     self.desire_matrix = np.where(self.distance_matrix != 0, 1/self.distance_matrix, 0)

    def run(self):
        best_distance = float('inf')
        best_path = None

        for iteration in range(self.num_iter):
            all_paths = []
            all_distances = []

            for ant in range(self.num_ants):
                path = self.construct_solution()
                distance = self.calculate_distance(path)
                
                all_paths.append(path)
                all_distances.append(distance)

                if distance < best_distance:
                    best_distance = distance
                    best_path = path
            
            self.update_pheromones(all_paths, all_distances)
        
        return best_path, best_distance

    def construct_solution(self):
        path = []
        unvisited = set(range(self.num_cities))
        current_city = np.random.choice(self.num_cities)
        path.append(current_city)
        unvisited.remove(current_city)

        while unvisited:
            probabilities = self.calculate_transition_probabilities(current_city, unvisited)
            next_city = np.random.choice(list(unvisited), p=probabilities)
            path.append(next_city)
            unvisited.remove(next_city)
            current_city = next_city

        return path
    
    def calculate_transition_probabilities(self, current_city, unvisited):
        pheromones = self.pheromone_matrix[current_city, list(unvisited)]
        distances = self.distance_matrix[current_city, list(unvisited)]
        heuristic_values = 1 / (distances + 1e-10)  # Avoid division by zero

        numerator = (pheromones ** self.alpha) * (heuristic_values ** self.beta)
        denominator = np.sum(numerator)
        probabilities = numerator / denominator

        return probabilities
    
    def calculate_distance(self, path):
        total_distance = 0
        for i in range(len(path) - 1):
            total_distance += self.distance_matrix[path[i], path[i + 1]]
        total_distance += self.distance_matrix[path[-1], path[0]]  # Return to the start city
        return total_distance
    
    def update_pheromones(self, all_paths, all_distances):
        self.pheromone_matrix *= (1 - self.rho)  # Evaporation

        for path, distance in zip(all_paths, all_distances):
            pheromone_to_add = self.delta_tau / distance
            for i in range(len(path) - 1):
                self.pheromone_matrix[path[i], path[i + 1]] += pheromone_to_add
            self.pheromone_matrix[path[-1], path[0]] += pheromone_to_add  # Closing the loop
