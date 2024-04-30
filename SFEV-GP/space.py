from gplearn._program import _Program
from gplearn.functions import _Function
import random
from gplearn.functions import _sympol_map
from gplearn.functions import add2, sub2, mul2, div2, sqrt1, log1, sin1, cos1, exp1
from gplearn.fitness import mean_square_error

class Space(object):
    def __init__(self, logo, regressor, population_size, init_depth, function_set, init_method, n_features, const_range, feature_names=None, ) -> None:
        self.logo = logo
        self.regressor = regressor
        self.population = []
        self.fitness = None
        self.population_size = population_size
        self.num_map = {
            'add': 0, 
            'sub': 0, 
            'mul': 0, 
            'div': 0, 
            'sqrt': 0, 
            'log': 0, 
            'sin': 0, 
            'cos': 0, 
            'exp': 0
        }
        self.num_sum = 0
        self.prob_map = {
            'add': 1/9, 
            'sub': 1/9, 
            'mul': 1/9, 
            'div': 1/9, 
            'sqrt': 1/9, 
            'log': 1/9, 
            'sin': 1/9, 
            'cos': 1/9, 
            'exp': 1/9
        }
        self.function_dist = [0.11] * 9
        self.new_programs = None
        self.init_depth = init_depth
        self.function_set = function_set
        self.init_method = init_method
        self.n_features = n_features
        self.const_range = const_range
        self.program = None
        self.feature_names = feature_names
        self.prior_generations = 0
        self.generations = self.regressor.generations
        self.ev_prob = None
        self.new_programs = None
        self.visit_num = 1
        self.miu = 0
        self.ucb = 0

    def addProgram(self, program):
        self.population.append(program)
        self.size += 1

    def removeProgram(self, program):
        self.population.remove(program)
        self.size -= 1
    
    def getSize(self):
        return len(self.population)

    def updateDistribution(self):
        new_population = self.regressor.current_top100 + self.regressor.sorted_population
        if new_population != None:
            for _program in new_population:
                if isinstance(_program, list):
                    for _function in _program:
                        if isinstance(_function, _Function):
                            function_name = _function.name
                            self.num_map[function_name] += 1
                            self.num_sum += 1
                else:
                    for _function in _program.program:
                        if isinstance(_function, _Function):
                            function_name = _function.name
                            self.num_map[function_name] += 1
                            self.num_sum += 1
        for item in self.num_map:
            prob = self.num_map[item] / self.num_sum
            self.prob_map[item] = prob
            self.function_dist = list(self.prob_map.values())

    def get_distribution(self):
        print("prob_map：", self.prob_map)

    def get_expression(self):
        """Get a real formula with functions and terminals."""
        terminals = [0]
        output = ''
        stack = []
        X_num = []
        for i, node in enumerate(self.program):
            if isinstance(node, _Function):
                terminals.append(node.arity)
                if node.arity == 2:
                    if node.name not in 'addsubmuldiv':
                        output += node.name + '('
                    else:
                        output += '('
                    stack.append(node)
                elif node.arity == 1:
                    output += node.name + '('
                else:
                    print("node arity error!")
            else:
                if isinstance(node, int):
                    if self.feature_names is None:
                        output += 'x%s' % node
                        if node not in X_num:
                            X_num.append(node)
                    else:
                        output += self.feature_names[node]
                else:
                    if node<0:
                        output += '(%.3f)' % node
                    else:
                        output += '%.3f' % node
                terminals[-1] -= 1
                while terminals[-1] == 0:
                    terminals.pop()
                    terminals[-1] -= 1
                    output += ')'
                if i != len(self.program) - 1:
                    if stack[-1].name in 'addsubmuldiv':
                        output += _sympol_map[f'{stack[-1].name}']
                        stack.pop()
        return output

    def buildProgramWithDist(self, function_dist, random_state):
        """Build a program according to distribution.

        Parameters
        ----------
        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        program : list
            The flattened tree representation of the program.

        """
        if self.init_method == 'half and half':
            method = ('full' if random_state.randint(0, 2) else 'grow')
        else:
            method = self.init_method
        max_depth = random_state.randint(*self.init_depth)
        prob_function = function_dist
        result = random.choices(self.function_set, weights=prob_function, k=1)[0]
        function = random.choices(self.function_set, weights=prob_function, k=1)[0]
        self.num_map[function.name] += 1
        self.num_sum += 1
        program = [function]
        terminal_stack = [function.arity]

        while terminal_stack:
            depth = len(terminal_stack)
            choice = self.n_features + len(self.function_set)
            choice = random_state.randint(0, choice - 1)
            if (depth < max_depth) and (method == 'full' or
                                        choice <= len(self.function_set)):
                function = random.choices(self.function_set, weights=prob_function, k=1)[0]
                self.num_map[function.name] += 1
                self.num_sum += 1
                program.append(function)
                terminal_stack.append(function.arity)
            else:
                if self.const_range is not None:
                    terminal = random_state.randint(0, self.n_features)
                else:
                    terminal = random_state.randint(0, self.n_features - 1)
                if terminal == self.n_features:
                    terminal = random_state.uniform(*self.const_range)
                    if self.const_range is None:
                        raise ValueError('A constant was produced with '
                                         'const_range=None.')
                program.append(terminal)
                terminal_stack[-1] -= 1
                while terminal_stack[-1] == 0:
                    terminal_stack.pop()
                    if not terminal_stack:
                        self.program = program
                        temp_program = _Program(function_set=[add2, sub2, mul2, div2, sqrt1, log1, sin1, cos1, exp1], arities={2:[add2, sub2, mul2, div2],1:[sqrt1, log1, sin1, cos1, exp1]}, init_depth=(2,5), init_method='half and half', n_features=self.n_features, const_range=(-1,1), metric=mean_square_error, p_point_replace=0.2, parsimony_coefficient=0.1, random_state=random)
                        temp_program.program = program
                        return temp_program
                    terminal_stack[-1] -= 1
        return None

    def newPrograms(self, X, y, num):
        self.new_programs = []
        while len(self.new_programs) < num:
            program = self.buildProgramWithDist(self.function_dist, random_state=random)
            self.new_programs.append(program)
        self.updateDistribution()
        return self.new_programs