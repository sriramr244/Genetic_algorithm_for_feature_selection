# Objectives

- Feature selection using genetic algorithm

# Installation
```
pip install FeatureSelectionUsingGA

from fsga import Genetic_algorithm
```

# Running

- Create a object of the class `Genetic_algorithm`
- Important Arguments
``` 
input_data_x -> Training features (pandas.core.frame.DataFrame)
input_data_y -> Targets (pandas.core.frame.DataFrame)
max_population -> Population size (integer greater than zero)
crossover_prob -> Probability of crossover (float 0<crossover_prb<1 )
mutation_r -> mutation rate (float 0<mutation_r<1 )
stop_by_f -> flag to stop the evolution when fitness value is reached (bool)
stop_fitness -> value of fitness value (float 0<stop_fitness<1)
fitness_func -> Callable function that input arguments train features, train targets, validation features, validation targets 
For example:
    def estimate(T_x, T_y, V_x, V_y):
        clf = svm.SVC(C = 1, kernel = 'poly', gamma = 'auto')
        clf.fit(T_x, T_y)
        return clf.score(V_x, V_y)
```
This estimate function can be passed as the argument for fitness_func
- To run the evolution 
```
GA_F = Genetic_algorithm(..Arguments...)
GA_F.evolve(__no_of_generations__)
```
- To plot the movement of solutions
```
GA_F.plot(_path_to_save_the_figure_)
```