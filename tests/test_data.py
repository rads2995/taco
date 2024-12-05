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

    def test_small_cities(self):

        dist_matrix_file = np.loadtxt(self.filepath.joinpath('small_cities.csv'),
        dtype=str,
        delimiter=',',
        encoding='utf-8-sig')
        dist_matrix = dist_matrix_file[np.any(dist_matrix_file != '', axis=1)].astype(np.float64)
        taquito: taco = taco(distance_matrix=dist_matrix, num_ants=np.shape(dist_matrix)[0], num_iter=self.num_iter, alpha=self.alpha, beta=self.beta, rho=self.rho, Q=self.Q)
        taquito.run()
        self.assertAlmostEqual(float(taquito.run()[1]), 1248.0)

    def test_medium_cities(self):

        dist_matrix_file = np.loadtxt(self.filepath.joinpath('medium_cities.csv'),
        dtype=str,
        delimiter=',',
        encoding='utf-8-sig')
        dist_matrix = dist_matrix_file[np.any(dist_matrix_file != '', axis=1)].astype(np.float64)
        taquito: taco = taco(distance_matrix=dist_matrix, num_ants=np.shape(dist_matrix)[0], num_iter=self.num_iter, alpha=self.alpha, beta=self.beta, rho=self.rho, Q=self.Q)
        taquito.run()
        self.assertAlmostEqual(float(taquito.run()[1]), 1194.0)

    def test_big_cities(self):

        dist_matrix_file = np.loadtxt(self.filepath.joinpath('big_cities.csv'),
        dtype=str,
        delimiter=',',
        encoding='utf-8-sig')
        dist_matrix = dist_matrix_file[np.any(dist_matrix_file != '', axis=1)].astype(np.float64)
        taquito: taco = taco(distance_matrix=dist_matrix, num_ants=np.shape(dist_matrix)[0], num_iter=self.num_iter, alpha=self.alpha, beta=self.beta, rho=self.rho, Q=self.Q)
        taquito.run()
        self.assertAlmostEqual(float(taquito.run()[1]), 27603.0)

    def test_burma14(self):

        dist_matrix_file = np.loadtxt(self.filepath.joinpath('3burma14.csv'),
        dtype=str,
        delimiter=',',
        encoding='utf-8-sig')
        dist_matrix = dist_matrix_file[np.any(dist_matrix_file != '', axis=1)].astype(np.float64)
        taquito: taco = taco(distance_matrix=dist_matrix, num_ants=np.shape(dist_matrix)[0], num_iter=self.num_iter, alpha=self.alpha, beta=self.beta, rho=self.rho, Q=self.Q)
        taquito.run()
        self.assertAlmostEqual(float(taquito.run()[1]), 3323.0)

    def test_gr17(self):

        dist_matrix_file = np.loadtxt(self.filepath.joinpath('4gr17.csv'),
        dtype=str,
        delimiter=',',
        encoding='utf-8-sig')
        dist_matrix = dist_matrix_file[np.any(dist_matrix_file != '', axis=1)].astype(np.float64)
        taquito: taco = taco(distance_matrix=dist_matrix, num_ants=np.shape(dist_matrix)[0], num_iter=self.num_iter, alpha=self.alpha, beta=self.beta, rho=self.rho, Q=self.Q)
        taquito.run()
        self.assertAlmostEqual(float(taquito.run()[1]), 2085.0)

    def test_gr21(self):

        dist_matrix_file = np.loadtxt(self.filepath.joinpath('5gr21.csv'),
        dtype=str,
        delimiter=',',
        encoding='utf-8-sig')
        dist_matrix = dist_matrix_file[np.any(dist_matrix_file != '', axis=1)].astype(np.float64)
        taquito: taco = taco(distance_matrix=dist_matrix, num_ants=np.shape(dist_matrix)[0], num_iter=self.num_iter, alpha=self.alpha, beta=self.beta, rho=self.rho, Q=self.Q)
        taquito.run()
        self.assertAlmostEqual(float(taquito.run()[1]), 2707.0)

    def test_gr24(self):

        dist_matrix_file = np.loadtxt(self.filepath.joinpath('5gr24.csv'),
        dtype=str,
        delimiter=',',
        encoding='utf-8-sig')
        dist_matrix = dist_matrix_file[np.any(dist_matrix_file != '', axis=1)].astype(np.float64)
        taquito: taco = taco(distance_matrix=dist_matrix, num_ants=np.shape(dist_matrix)[0], num_iter=self.num_iter, alpha=self.alpha, beta=self.beta, rho=self.rho, Q=self.Q)
        taquito.run()
        self.assertAlmostEqual(float(taquito.run()[1]), 1272.0)

    def test_bays29(self):

        dist_matrix_file = np.loadtxt(self.filepath.joinpath('6bays29.csv'),
        dtype=str,
        delimiter=',',
        encoding='utf-8-sig')
        dist_matrix = dist_matrix_file[np.any(dist_matrix_file != '', axis=1)].astype(np.float64)
        taquito: taco = taco(distance_matrix=dist_matrix, num_ants=np.shape(dist_matrix)[0], num_iter=self.num_iter, alpha=self.alpha, beta=self.beta, rho=self.rho, Q=self.Q)
        taquito.run()
        self.assertAlmostEqual(float(taquito.run()[1]), 2020.0)

    def test_fri26(self):

        dist_matrix_file = np.loadtxt(self.filepath.joinpath('6fri26.csv'),
        dtype=str,
        delimiter=',',
        encoding='utf-8-sig')
        dist_matrix = dist_matrix_file[np.any(dist_matrix_file != '', axis=1)].astype(np.float64)
        taquito: taco = taco(distance_matrix=dist_matrix, num_ants=np.shape(dist_matrix)[0], num_iter=self.num_iter, alpha=self.alpha, beta=self.beta, rho=self.rho, Q=self.Q)
        taquito.run()
        self.assertAlmostEqual(float(taquito.run()[1]), 937.0)

    def test_dantzig42(self):

        dist_matrix_file = np.loadtxt(self.filepath.joinpath('9dantzig42.csv'),
        dtype=str,
        delimiter=',',
        encoding='utf-8-sig')
        dist_matrix = dist_matrix_file[np.any(dist_matrix_file != '', axis=1)].astype(np.float64)
        taquito: taco = taco(distance_matrix=dist_matrix, num_ants=np.shape(dist_matrix)[0], num_iter=self.num_iter, alpha=self.alpha, beta=self.beta, rho=self.rho, Q=self.Q)
        taquito.run()
        self.assertAlmostEqual(float(taquito.run()[1]), 699.0)

if __name__ == '__main__':
    unittest.main()
