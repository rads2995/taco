import unittest
import pathlib

import numpy as np

from taco.taco import taco

class TestData(unittest.TestCase):

    num_iter: int = 1000
    alpha: float = 1
    beta: float = 2
    rho: float = 0.5
    Q: float = 1.0
    filepath: pathlib.Path = pathlib.Path(__file__).parent.parent.joinpath('taco','data')

    def test_tsp_data(self):

        dist_matrix_file = np.loadtxt(self.filepath.joinpath('tsp_data.csv'),
        dtype=str,
        delimiter=',',
        encoding='utf-8-sig')
        dist_matrix = dist_matrix_file[np.any(dist_matrix_file != '', axis=1)].astype(np.float64)
        taquito: taco = taco(distance_matrix=dist_matrix, num_ants=np.shape(dist_matrix)[0], num_iter=self.num_iter, alpha=self.alpha, beta=self.beta, rho=self.rho, Q=self.Q)
        taquito.run()
        self.assertAlmostEqual(float(taquito.run()[1]), 51.0)

    def test_google_or_tools(self):

        dist_matrix_file = np.loadtxt(self.filepath.joinpath('google_or_tools.csv'),
        dtype=str,
        delimiter=',',
        encoding='utf-8-sig')
        dist_matrix = dist_matrix_file[np.any(dist_matrix_file != '', axis=1)].astype(np.float64)
        taquito: taco = taco(distance_matrix=dist_matrix, num_ants=np.shape(dist_matrix)[0], num_iter=self.num_iter, alpha=self.alpha, beta=self.beta, rho=self.rho, Q=self.Q)
        taquito.run()
        self.assertAlmostEqual(float(taquito.run()[1]), 7293.0)

    def test_all_data(self):

        for root, dirs, files in self.filepath.walk(on_error=print):
            for iterator in files:
                dist_matrix_file = np.loadtxt(self.filepath.joinpath(iterator),
                dtype=str,
                delimiter=',',
                encoding='utf-8-sig')
                dist_matrix = dist_matrix_file[np.any(dist_matrix_file != '', axis=1)].astype(np.float64)
                taquito: taco = taco(distance_matrix=dist_matrix, num_ants=np.shape(dist_matrix)[0], num_iter=self.num_iter, alpha=self.alpha, beta=self.beta, rho=self.rho, Q=self.Q)
                best_path, best_distance = taquito.run()

                print(' ')
                print('TSP Instance Name: {:}'.format(iterator))
                print('Objective: {:.6f}'.format(best_distance))
                print('Best Route:')
                print('{:d}'.format(best_path[0]), end=' ')
                for index in range(1, len(best_path) - 1):
                    print('-> {:d}'.format(best_path[index]), end=' ')
                print('-> {:d} -> {:d}'.format(best_path[-1], best_path[0]))

if __name__ == '__main__':
    unittest.main()
