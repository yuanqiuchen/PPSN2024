from gplearn.genetic import SymbolicRegressor
from gplearn.functions import add2, sub2, mul2, div2, sin1, cos1, log1, sqrt1, exp1
import random
import numpy as np

regressor = SymbolicRegressor(population_size=1000, init_depth=(2, 5),
                        generations=10000, stopping_criteria=1e-10,
                        function_set=[add2, sub2, mul2, div2, sqrt1, log1, sin1, cos1, exp1],
                        p_crossover=0.7, p_subtree_mutation=0.,
                        p_hoist_mutation=0., p_point_mutation=0.3,
                        max_samples=1.0, verbose=1,
                        parsimony_coefficient=0.1,
                        n_jobs=1,
                        const_range=(-1, 1),
                        random_state=random.randint(1, 100), 
                        low_memory=True)
    
X = np.array([[1],[2],[3]])
y = np.array([0.84,1.02,1.12])

regressor.fit(X, y)