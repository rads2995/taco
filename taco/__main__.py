import argparse
import pathlib

import numpy as np

from .taco import taco

parser = argparse.ArgumentParser(
    prog='taco', 
    description='Solves the Traveling Salesman Problem (TSP) using Ant Colony Optimization (ACO).',
    )
parser.add_argument('file', 
    action='store', 
    nargs='?', 
    default=pathlib.Path(__file__).parent.joinpath('data','tsp_data.csv'),
    type=pathlib.Path,
    help='Path to comma-separated values (.csv) file containing distance matrix.')
args = parser.parse_args()

if args.file.suffix != '.csv':
    print("Error: only paths to comma-separated values (.csv) files are supported.")
else:
    dist_matrix_file = np.loadtxt(args.file,
    dtype=str,
    delimiter=',',
    encoding='utf-8-sig')
    dist_matrix = dist_matrix_file[np.any(dist_matrix_file != '', axis=1)].astype(np.float64)

taquito: taco = taco(distance_matrix=dist_matrix, num_ants=np.shape(dist_matrix)[0], num_iter=1000, alpha=1, beta=2, rho=0.5, Q=1.0)

best_path, best_distance = taquito.run()

print('TSP Instance Name: {:}'.format(args.file.name))
print('Objective: {:.6f}'.format(best_distance))
print('Best Route:')
print('{:d}'.format(best_path[0]), end=' ')
for index in range(1, len(best_path) - 1):
    print('-> {:d}'.format(best_path[index]), end=' ')
print('-> {:d} -> {:d}'.format(best_path[-1], best_path[0]))
