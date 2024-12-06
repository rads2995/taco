# TACO
Travelling salesman problem (TSP) using the Ant Colony Optimization (ACO) algorithm.

**Note:** the starting node/city is always random, but the path, as well as the objective distance value, are always the same.

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
Once the project is installed, you can run it from anywhere with:
```shell
python -m taco
```
This will use the default dataset that exists in the following path: `taco/data/tsp_data.csv`

**Note:** you can run the module without installing it, but it is important to make sure that all dependencies are satisfied, such as Numpy. If desired, you can install all required dependencies in the `requirements.txt` file with the following command:
```shell
python -m pip install -r requirements.txt
```

Finally, if you would like to use a different example, you can either provide your own `.csv` file that contains a distance matrix in the expected format, or select any from the `data` directory. If you find one that you like, you can run it as follows:
```shell
python -m taco taco/data/google_or_tools.csv
``` 

## How to Test
If you have a lot of time and would like to run all of the available examples in the `data` directory, you can do so as follows:
```shell
python -m unittest -v
```
You may also run a specific test-case of interest, such as the following:
```shell
python -m unittest -v tests.test_data.TestData.test_tsp_data
```

Resulting objective values from all examples have been verified using the [optimal solutions for symmetric TSPs](http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/STSP.html). 

## Disclaimer

All credits go to [TSPLIB group](http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/) and the respective authors for each example that I used to test my software.
