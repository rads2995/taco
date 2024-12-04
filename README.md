# taco
Travelling salesman problem (TSP) using the Ant Colony Optimization (ACO) algorithm.

Results have been verified using the [optimal solutions for symmetric TSPs](http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/STSP.html)


```shell
find taco/data -name *.csv -type f -print0 | xargs -0 -t -L 1 python -m taco
```
