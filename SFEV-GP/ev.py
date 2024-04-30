# ev model

import numpy as np

class EV(object):
    def __init__(self, space) -> None:
        self.threshold = None
        self.space = space
        self.ev_samples = None
        self.ev_mean = None
        self.ev_sigma = None
        self.ev_ksi = None
        self.lambda1 = None
        self.lambda2 = None
        self.sample_num = None
        self.ev_function = None
        self.ev_prob = None
        self.judge_fitness = None

        self.threshold = np.mean([program.raw_fitness_ for program in self.space.regressor.population])
        self.epsilon = 1e-5

        if self.ev_samples == None:
            self.get_samples(self.space)
            self.sample_num = len(self.ev_samples)

        if self.ev_mean == None:
            self.ev_mean = np.mean(self.ev_samples)
            self.lambda1 = self.ev_mean
        
        if self.ev_ksi == None:
            self.get_ev_ksi()

        if self.ev_sigma == None:
            self.ev_sigma = (1 - self.ev_ksi) * self.lambda1

        if self.ev_function == None:
            self.ev_function =  self.ev_function

    def add_space(self, space):
        self.space = space

    def get_ev_ksi(self):
        sum = 0
        for i in range(0, self.sample_num):
            for j in range(0, self.sample_num):
                if i > j:
                    temp_ = self.ev_samples[i] - self.ev_samples[j]
                    sum += temp_
        self.lambda2 = sum / (self.sample_num * (self.sample_num - 1))
        if self.lambda2 == 0:
            self.ev_ksi = 2 - (self.lambda1 / self.epsilon)
        else:
            self.ev_ksi = 2 - (self.lambda1 / self.lambda2)

    def get_samples(self, space):
        samples = []
        fitnesses = [program.raw_fitness_ for program in space.regressor.current_top100]
        for fitness in fitnesses:
            if fitness <= self.threshold:
                samples.append(fitness)
        samples.sort(reverse=False)
        self.ev_samples = samples
        self.sample_num = len(self.ev_samples)
        return self.ev_samples

    def updateParams(self):
        self.get_samples(self.space)
        self.sample_num = len(self.ev_samples)
        self.ev_mean = np.mean(self.ev_samples)
        self.lambda1 = self.ev_mean
        self.get_ev_ksi()
        self.ev_sigma = (1 - self.ev_ksi) * self.lambda1
        self.ev_function =  self.ev_function

    def get_evfunction(self):
        return self.get_evprob

    def get_evprob(self, judge_fitness):
        self.judge_fitness = judge_fitness
        self.ev_prob = 1 - (1 + self.ev_ksi * (judge_fitness / self.ev_sigma)) ** (-(1 / self.ev_ksi))
        return self.ev_prob
