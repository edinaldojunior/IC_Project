from __future__ import division
import copy
from functools import partial
from benchmark import BinaryProblems as bp
from swarm.decorators import timer
import numpy as np
import pandas as pd
import os


# This code was based on in the following references:
# [1] "A discrete binary version of the particle swarm algorithm" published in 1997 by J Kennedy and RC Eberhart


class BinaryParticle(object):
    BINARY_BASE = 2

    def __init__(self, dim, maximize=True):
        self.dim = dim
        self.pos = BinaryParticle.__initialize_position(dim)
        self.speed = np.zeros((1, dim), dtype=np.float32).reshape(dim)
        self.cost = -np.inf if maximize else np.inf
        self.train_acc = 0.0
        self.test_acc = 0.0
        self.features = 0.0
        self.pbest_pos = self.pos
        self.pbest_cost = self.cost

    def update_components(self, w, c1, c2, v_max, gbest):
        r1 = np.random.random(self.dim)
        r2 = np.random.random(self.dim)
        self.speed = w * self.speed + c1 * r1 * (self.pbest_pos - self.pos) + \
                     c2 * r2 * (gbest.pos - self.pos)
        self.restrict_vmax(v_max)
        self.update_pos()

    def restrict_vmax(self, v_max):
        self.speed[self.speed > v_max] = v_max
        self.speed[self.speed < -v_max] = -v_max

    def update_pos(self):
        probs = map(BinaryParticle.__sgm, self.speed)
        prob = np.random.random(self.dim)
        self.pos[probs > prob] = 1
        self.pos[probs < prob] = 0

    @staticmethod
    def __sgm(v):
        return 1 / (1 + np.exp(-v))

    @staticmethod
    def __initialize_position(dim):
        return np.random.randint(BinaryParticle.BINARY_BASE, size=dim)


class BPSO(object):
    def __init__(self, objective_function, pop_size=1000, max_iter=5000, lb_w=0.4, up_w=0.9, c1=2.05, c2=2.05,
                 v_max=100000, maximize=True, simulation_id=1):
        self.name = "BPSO"
        self.c1 = c1
        self.c2 = c2
        self.w = up_w
        self.lb_w = lb_w
        self.up_w = up_w
        self.v_max = min(v_max, 100000)
        self.dim = objective_function.dim
        self.objective_function = objective_function
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.maximize = maximize
        self.op = max if maximize else min

        self.optimum_cost_tracking_eval = []
        self.optimum_cost_tracking_iter = []

        self.optimum_train_acc_tracking_eval = []
        self.optimum_train_acc_tracking_iter = []

        self.optimum_test_acc_tracking_eval = []
        self.optimum_test_acc_tracking_iter = []

        self.optimum_features_tracking_eval = []
        self.optimum_features_tracking_iter = []

        self.best_agent = None
        self.simulation_id = simulation_id
        
    def __eval_track_update(self):
        self.optimum_cost_tracking_eval.append(self.best_agent.cost)
        self.optimum_train_acc_tracking_eval.append(self.best_agent.train_acc)
        self.optimum_test_acc_tracking_eval.append(self.best_agent.test_acc)
        self.optimum_features_tracking_eval.append(self.best_agent.features)

    def __iter_track_update(self):
        self.optimum_cost_tracking_iter.append(self.best_agent.cost)
        self.optimum_train_acc_tracking_iter.append(self.best_agent.train_acc)
        self.optimum_test_acc_tracking_iter.append(self.best_agent.test_acc)
        self.optimum_features_tracking_iter.append(self.best_agent.features)

    def __init_swarm(self):
        self.w = self.up_w
        self.swarm = []
        self.best_agent = None

        self.optimum_cost_tracking_eval = []
        self.optimum_cost_tracking_iter = []

        self.optimum_train_acc_tracking_eval = []
        self.optimum_train_acc_tracking_iter = []

        self.optimum_test_acc_tracking_eval = []
        self.optimum_test_acc_tracking_iter = []

        self.optimum_features_tracking_eval = []
        self.optimum_features_tracking_iter = []

        for _ in range(self.pop_size):
            self.swarm.append(BinaryParticle(self.dim, self.maximize))

    def __evaluate_swarm(self):
        for particle in self.swarm:
            self.__evaluate(self.objective_function, self.op, particle)        
        #evaluate = partial(BPSO.__evaluate, self, self.objective_function, self.op)
        #map(evaluate, self.swarm)

    def __select_best_particle(self):
        current_optimal = copy.deepcopy(self.op(self.swarm, key=lambda p: p.cost))
        if not self.best_agent:
            self.best_agent = copy.deepcopy(current_optimal)
            return

        if current_optimal.cost > self.best_agent.cost:
            self.best_agent = copy.deepcopy(current_optimal)

    def __update_components(self):
        update = partial(BPSO.__update_swarm_components, self.up_w, self.c1, self.c2,
                         self.v_max, self.best_agent) #Global topology
        map(update, self.swarm)

    def __update_inertia_weight(self, itr):
        self.w = self.up_w - (float(itr) / self.max_iter) * (self.up_w - self.lb_w)

    def __evaluate(self, fitness, op, particle):
        particle.cost, particle.test_acc, particle.train_acc, particle.features = fitness.evaluate(particle.pos)
        #self.__eval_track_update()
        particle.pbest_cost = op(particle.cost, particle.pbest_cost)
        if particle.pbest_cost == particle.cost:
            particle.pbest_pos = particle.pos

    @staticmethod
    def __update_swarm_components(w, c1, c2, vmax, gbest, particle):
        particle.update_components(w, c1, c2, vmax, gbest)

    @timer
    def optimize(self):
        self.__init_swarm()
        self.__select_best_particle()

        for itr in range(self.max_iter):
            self.__update_components()
            self.__evaluate_swarm()
            self.__select_best_particle()
            self.__update_inertia_weight(itr)
            self.__iter_track_update()
            #print('LOG: Iter: {} - Cost: {} - Train Acc: {} -Test Acc: {} -Feat: {}'.format(itr, self.best_agent.cost,
            #                                                                                self.best_agent.train_acc,
            #                                                                                self.best_agent.test_acc,
            #                                                                                self.best_agent.features))
        
        df = pd.DataFrame(self.optimum_cost_tracking_iter, columns=['fit'])
        output_path = 'output/'
        if not os.path.exists(output_path):
            os.mkdir(output_path)

        filename = f'simu_bpso_{func_fitness.name}_{self.dim}_{self.simulation_id}.csv'
        df.to_csv(os.path.join(output_path, filename), index_label='it')
        return self.optimum_cost_tracking_iter

if __name__ == '__main__':
    #func_fitness = bp.OneMax(dimensions)
    #func_fitness = bp.ZeroMax(dimensions)  
    MAX_SIMU = 10  
    max_iter = 1000
    dimensions = 50
    func_fitness_list = [bp.ZeroMax(dimensions), bp.OneMax(dimensions), bp.KNAPSACK(dimensions)] 
    agents = 30

    for func_fitness in func_fitness_list:
        simulation_id = 1
        print(func_fitness.name)       
        bfs = []
        while simulation_id <= MAX_SIMU:            
            bpso = BPSO(objective_function = func_fitness, pop_size = agents, max_iter = max_iter,
                     simulation_id=simulation_id, lb_w = 0.1, up_w = 0.8, c1 = 0.72984 * 2.05, 
                     c2 = 0.72984 * 2.05, v_max = 1000)
            bf = bpso.optimize()
            bfs.append(bf[-1])
            print (f"({simulation_id:02d}) - FIT={bf[-1]}")
            simulation_id+=1
        
        print(np.mean(bfs))