# TACO
Travelling salesman problem (TSP) using the Ant Colony Optimization (ACO) algorithm.

## How to Install
After you download the project's directory, you may install it running the following command:
```shell
python -m pip install -e taco
```
Or, if you happen to be inside of the project's directory:
```shell
python -m pip install -e .
```

## How to Run
Once the project is installed, you can run it as:
```shell
python -m taco
```
This will use the default dataset that exists in the following path: `taco/data/tsp_data.csv`

**note:** you can run the module without installing it by performing this command, but it is important to make sure that all dependencies are satisfied, such as Numpy.

If instead you would like to use a different mission, you can either provide your own `.csv` file that contains a distance matrix, or select any from the data directory. You can do this as follows:
```shell
python -m taco taco/data/google_or_tools.csv
``` 

## How to Test
If you would like to run all of the available datasets in the data directories, you can do so as follows:
```shell
python -m unittest -v
```
You may also run a specific test-case, such as the following:
```shell
python -m unittest -v tests.test_data.TestData.test_tsp_data
```

Objective value results from all datasets have been verified using the [optimal solutions for symmetric TSPs](http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/STSP.html).
