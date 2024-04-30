# from ._program import _Program
# from individual import Individual
from gplearn._program import _Program
import random
import numpy as np
from gplearn.functions import add2, sub2, mul2, div2, sqrt1, log1, sin1, cos1, exp1
from gplearn.fitness import mean_square_error
from gplearn.genetic import SymbolicRegressor
from space import Space
from ev import EV
from dataset import getData
import math
import argparse


def main(path, runNum, fileNum):

    space_num = 100
    all_time = 0
    lamda = 0.5
    top = None

    data_path = path
    X, y, n_features = getData(data_path,2)

    space_list = []
    seed = 1
    for i in range(0, space_num):
        regressor = SymbolicRegressor(population_size=900, init_depth=(2, 5),
                        generations=10000, stopping_criteria=1e-10,
                        function_set=[add2, sub2, mul2, div2, sqrt1, log1, sin1, cos1, exp1],
                        p_crossover=0.7, p_subtree_mutation=0.,
                        p_hoist_mutation=0., p_point_mutation=0.3,
                        max_samples=1.0, verbose=1,
                        parsimony_coefficient=0.1,
                        n_jobs=1,
                        const_range=(-1, 1),
                        random_state=seed+runNum, 
                        low_memory=True)
        seed += 1        
        space = Space(logo=i+1, regressor=regressor, population_size=900, init_depth=(2,5), function_set=[add2, sub2, mul2, div2, sqrt1, log1, sin1, cos1, exp1], init_method='half and half', n_features=n_features, const_range=(-1, 1))
        space_list.append(space)

    evprob_list = [1.0/space_num] * space_num
    ucb_list = [0] * space_num
    target_list = [0] * space_num

    for space in space_list:
        space.regressor.run_num = 10
        space.regressor.space_logo = space.logo
        space.regressor.population_size = 1000
        space.regressor.prior_generations = space.prior_generations
        space.regressor.fit(X,y, runNum=runNum, fileNum=fileNum, top=top)
        top = space.regressor.top
        space.population = space.regressor.population
        space.fitness = space.regressor.fitness
        space.prior_generations = space.regressor.prior_generations
        space.regressor.population_size = 900

    judge = True
    for iter in range(0, 1000):
        if len(space_list) == 0:
            break
        selected_index = target_list.index(max(target_list))
        selected_space = space_list[selected_index]
        space_list[selected_index].visit_num += 1
        space_list[selected_index].regressor.space_visit = space_list[selected_index].visit_num
        all_time += 1
        new_programs = selected_space.newPrograms(X, y, 1000)
        for item in new_programs:
            item.fitness_ = item.raw_fitness(X,y,sample_weight=None)
            item.raw_fitness_ = item.fitness_
        for item in new_programs:
            if np.isnan(item.raw_fitness_) or math.isinf(item.raw_fitness_):
                new_programs.remove(item)
        def compare(program):
            return program.raw_fitness_
        new_programs.sort(key=compare)
        selected_space.population = new_programs[0:100]
        random.shuffle(selected_space.population)
        selected_space.regressor.population = selected_space.population
        selected_space.regressor.space_logo = selected_space.logo
        selected_space.regressor.prior_generations = selected_space.prior_generations
        selected_space.regressor.fit(X,y, runNum=runNum, fileNum=fileNum, top=top)
        top = space.regressor.top
        selected_space.prior_generations = selected_space.regressor.prior_generations
        selected_space.updateDistribution()
        selected_space.get_distribution()
        try:
            ev = EV(selected_space)
            ev.updateParams()
        except:
            continue
        sum_fitness = []
        for space in space_list:
            for program in space.regressor.current_top100:
                sum_fitness.append(program.raw_fitness_)
        sum_fitness.sort()
        aim_fitness = sum_fitness[0]*0.1
        evprob = ev.get_evprob(aim_fitness)
        if np.isnan(evprob):
            evprob = evprob_list[selected_index] * 0.9
        selected_space.ev_prob = evprob
        evprob_list[selected_index] = evprob

        for index, selected_space in enumerate(space_list):
            selected_space.ev_prob = evprob_list[index]
            selected_space.miu = 1.0 / selected_space.regressor.history_best.raw_fitness_
            selected_space.ucb = selected_space.miu + lamda * math.sqrt(2.0 * np.log(all_time) / selected_space.visit_num)
            ucb_list[index] = selected_space.ucb
            target_list[index] = selected_space.ev_prob * 0.8 +  selected_space.ucb * 0.2
            if selected_space.regressor.prior_generations == 1000:
                remove_index = space_list.index(selected_space)
                space_list.remove(selected_space)
                evprob_list.remove(evprob_list[remove_index])
                ucb_list.remove(ucb_list[remove_index])
                target_list.remove(target_list[remove_index])

        outPath = r"/home/qc/test2/new/EVGP6/result/"
        space_file = open(outPath + str(runNum) + '_space_evgp_korns_three_' + str(fileNum) + '.txt', 'a', encoding='utf-8')
        space_file.write(str(space_list[selected_index].logo) + " " + str(evprob_list[selected_index]) + " " + str(ucb_list[selected_index]) + " " + str(target_list[selected_index]) + " " + str(space_list[selected_index].visit_num) + " " + str(space_list[selected_index].prob_map))
        space_file.write('\n')
        space_file.close()


if __name__ == "__main__":
    path = r"/home/qc/test2/data/korns/three_v/"
    runNum = 1
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--fileNum', default=0, type=int)
    args = argparser.parse_args()
    fileNum = str(args.fileNum)
    dataPath = str(path) + str(fileNum) + '.txt'
    while (runNum <= 30):
        main(path=dataPath, runNum=runNum, fileNum=fileNum)
        runNum += 1