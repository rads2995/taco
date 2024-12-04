import argparse
import pathlib

import numpy as np

from taco.taco import taco

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

taquito: taco = taco(dist_matrix, num_ants=50, num_iter=1000, alpha=1, beta=2, rho=0.5, Q=0.5)
print(taquito.run())
