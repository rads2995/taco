# The fundamental package for scientific computing with Python (including type information)
import numpy as np
import numpy.typing as npt

class taco:

    num_cities: int     # Number of cities to visit
    num_ants: int       # Number of ants in total
    num_iter: int       # Number of max iterations
    alpha: float        # Influence of pheromone matrix
    beta: float         # Influence of desirability matrix 
    rho: float          # Pheromone evaporation coefficient
    Q: float            # Constant for pheromone deposited by ant
    distance_matrix: npt.NDArray[np.float64]    # Distance matrix
    tau_matrix: npt.NDArray[np.float64]         # Amount of pheromone deposited  
    eta_matrix: npt.NDArray[np.float64]         # Desirability of state transition

    # Default constructor (default values are okay except for the number of ants)
    def __init__(self, distance_matrix: npt.NDArray[np.float64], num_ants: int = 50, 
                       num_iter: int = 1000, alpha: float = 1.0, beta: float = 2.0,
                       rho: float = 0.5, Q: float = 1.0) -> None:

        self.distance_matrix = distance_matrix
        self.num_cities = np.shape(distance_matrix)[0]
        self.num_ants = num_ants
        self.num_iter = num_iter
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q 
        self.tau_matrix = np.ones(np.shape(self.distance_matrix))
        # Desirability of state transition (x,y) is typically 1/distance(x,y)
        # Note: we want to supress any warnings about dividing by zero even when it doesn't happen
        with np.errstate(divide = 'ignore'): 
            self.eta_matrix = np.where(self.distance_matrix != 0, 1/self.distance_matrix, 0)

    def run(self) -> tuple[list[int], float]:
        best_distance: float = float('inf')     # Our best distance starts as positive infinity
        best_path: list[int] = []               # Our best path starts as an empty list

        for _ in range(self.num_iter):
            all_paths: list[list[int]] = []
            all_distances: list[float] = []

            for _ in range(self.num_ants):
                path: list[int] = self.construct_solution()
                distance: float = self.calculate_distance(path)
                
                all_paths.append(path)
                all_distances.append(distance)

                if distance < best_distance:
                    best_distance = distance
                    best_path = path
            
            self.update_pheromones(all_paths, all_distances)
        
        return best_path, best_distance

    def construct_solution(self) -> list[int]:
        path: list[int] = []
        
        # Append current city from generated random sample from all cities
        current_city: int = int(np.random.choice(self.num_cities))
        path.append(current_city)
        
        # Create list of unvisited cities from number of cities and remove current city
        unvisited: list[int] = list(range(self.num_cities))
        unvisited.remove(current_city)

        # For all unvisited cities, we calculate the probabilities and select the next city
        while unvisited:
            numerator = (self.tau_matrix[current_city, unvisited] ** self.alpha) * (self.eta_matrix[current_city, unvisited] ** self.beta)
            denominator = np.sum(numerator)
            probabilities: npt.NDArray[np.float64] = numerator / denominator
            
            next_city: int = int(np.random.choice(unvisited, p=probabilities))
            path.append(next_city)
            unvisited.remove(next_city)
            current_city = next_city

        return path
    
    def calculate_distance(self, path: list[int]) -> float:
        total_distance: float = 0
        for i in range(len(path) - 1):
            total_distance += self.distance_matrix[path[i], path[i + 1]]
        
        # Return to the start city
        total_distance += self.distance_matrix[path[-1], path[0]]
        
        return float(total_distance)
    
    def update_pheromones(self, all_paths: list[list[int]], all_distances: list[float]) -> None:
        # Global pheromone updating rule
        self.tau_matrix *= (1 - self.rho)

        # Update pheromones to all paths matching their respective distances
        for path, distance in zip(all_paths, all_distances):
            pheromone_to_add = self.Q / distance
            for i in range(len(path) - 1):
                self.tau_matrix[path[i], path[i + 1]] += pheromone_to_add
            # Close the loop by returning to the initial node/city
            self.tau_matrix[path[-1], path[0]] += pheromone_to_add
