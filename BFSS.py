import copy
import numpy as np


# This code was based on in the following references:
# [1] "Binary Fish School Search applied to feature selection: Application to ICU readmissions" published in 2014 by
# Sargo, Souza and Bastos-Filho


class Fish(object):
    def __init__(self, dim):
        self.pos = np.random.choice([0, 1], size=(dim,))
        self.cost = np.nan
        self.train_acc = 0.0
        self.test_acc = 0.0
        self.features = 0.0
        self.delta_cost = np.nan
        self.weight = np.nan


class BFSS(object):
    def __init__(self, objective_function, n_iter, school_size, step_ind_init, step_ind_final, thres_c, thres_v,
                 min_w, w_scale):
        self.name = "BFSS"
        self.objective_function = objective_function
        self.dim = objective_function.dim
        self.minf = objective_function.minf
        self.maxf = objective_function.maxf
        self.n_iter = n_iter
        self.school_size = school_size

        self.step_ind_init = step_ind_init
        self.step_ind_curr = step_ind_init
        self.step_ind_final = step_ind_final

        self.thres_c = thres_c
        self.thres_v = thres_v

        self.min_w = min_w
        self.w_scale = w_scale
        self.prev_weight_school = 0.0
        self.curr_weight_school = 0.0
        self.best_agent = None

        self.optimum_cost_tracking_eval = []
        self.optimum_cost_tracking_iter = []

        self.optimum_train_acc_tracking_eval = []
        self.optimum_train_acc_tracking_iter = []

        self.optimum_test_acc_tracking_eval = []
        self.optimum_test_acc_tracking_iter = []

        self.optimum_features_tracking_eval = []
        self.optimum_features_tracking_iter = []

    def eval_track_update(self):
        self.optimum_cost_tracking_eval.append(self.best_agent.cost)
        self.optimum_train_acc_tracking_eval.append(self.best_agent.train_acc)
        self.optimum_test_acc_tracking_eval.append(self.best_agent.test_acc)
        self.optimum_features_tracking_eval.append(self.best_agent.features)

    def iter_track_update(self):
        self.optimum_cost_tracking_iter.append(self.best_agent.cost)
        self.optimum_train_acc_tracking_iter.append(self.best_agent.train_acc)
        self.optimum_test_acc_tracking_iter.append(self.best_agent.test_acc)
        self.optimum_features_tracking_iter.append(self.best_agent.features)

    def __gen_weight(self):
        return self.w_scale / 2.0

    def __init_fss(self):
        self.best_agent = None
        self.school = []
        self.optimum_cost_tracking_eval = []
        self.optimum_cost_tracking_iter = []

        self.optimum_train_acc_tracking_eval = []
        self.optimum_train_acc_tracking_iter = []

        self.optimum_test_acc_tracking_eval = []
        self.optimum_test_acc_tracking_iter = []

        self.optimum_features_tracking_eval = []
        self.optimum_features_tracking_iter = []

    def __init_fish(self):
        fish = Fish(self.dim)
        fish.weight = self.__gen_weight()
        fish.cost, fish.test_acc, fish.train_acc, fish.features = self.objective_function.evaluate(fish.pos)
        self.eval_track_update()
        return fish

    def __init_school(self):
        self.best_agent = Fish(self.dim)
        self.best_agent.cost = -np.inf
        self.curr_weight_school = 0.0
        self.prev_weight_school = 0.0
        self.school = []

        for idx in range(self.school_size):
            fish = self.__init_fish()
            self.school.append(fish)
            self.curr_weight_school += fish.weight
            if self.best_agent.cost < fish.cost:
                self.best_agent = copy.copy(fish)
        self.prev_weight_school = self.curr_weight_school
        self.iter_track_update()

    def max_delta_cost(self):
        max_ = -np.inf
        for fish in self.school:
            if max_ < fish.delta_cost:
                max_ = fish.delta_cost
        return max_

    def total_school_weight(self):
        self.prev_weight_school = self.curr_weight_school
        self.curr_weight_school = 0.0
        for fish in self.school:
            self.curr_weight_school += fish.weight

    def calculate_barycenter(self):
        barycenter = np.zeros((self.dim,), dtype=np.float)
        density = 0.0

        for fish in self.school:
            density += fish.weight
            for dim in range(self.dim):
                barycenter[dim] += (fish.pos[dim] * fish.weight)
        for dim in range(self.dim):
            barycenter[dim] = barycenter[dim] / density

        return barycenter

    def update_steps(self, curr_iter):
        self.step_ind_curr = self.step_ind_init - (self.step_ind_init - self.step_ind_final) * (curr_iter / self.n_iter)

    def update_best_fish(self):
        for fish in self.school:
            if self.best_agent.cost < fish.cost:
                self.best_agent = copy.copy(fish)

    def feeding(self):
        for fish in self.school:
            if self.max_delta_cost():
                fish.weight = fish.weight + (fish.delta_cost / self.max_delta_cost())
            if fish.weight > self.w_scale:
                fish.weight = self.w_scale
            elif fish.weight < self.min_w:
                fish.weight = self.min_w

    def individual_movement(self):
        for fish in self.school:
            new_pos = copy.deepcopy(fish.pos)
            for dim in range(self.dim):
                u = np.random.random()
                if u < self.step_ind_curr:
                    new_pos[dim] = int(not new_pos[dim])
            cost, test_acc, train_acc, features = self.objective_function.evaluate(new_pos)
            self.eval_track_update()
            if cost > fish.cost:
                fish.delta_cost = cost - fish.cost
                fish.cost = cost
                fish.pos = new_pos
                fish.test_acc = test_acc
                fish.train_acc = train_acc
                fish.features = features
            else:
                fish.delta_cost = 0

    def collective_instinctive_movement(self):
        cost_eval_enhanced = np.zeros((self.dim,), dtype=np.float)
        density = 0.0
        for fish in self.school:
            density += fish.delta_cost
            for dim in range(self.dim):
                cost_eval_enhanced[dim] += (fish.pos[dim] * fish.delta_cost)
        for dim in range(self.dim):
            if density != 0:
                cost_eval_enhanced[dim] = cost_eval_enhanced[dim] / density

        max_i = max(cost_eval_enhanced)
        new_pos = np.zeros((self.dim,), dtype=np.float)
        for dim in range(self.dim):
            if cost_eval_enhanced[dim] >= self.thres_c * max_i:
                new_pos[dim] = 1

        for fish in self.school:
            for dim in np.random.choice(range(self.dim), self.dim):
                if fish.pos[dim] != new_pos[dim]:
                    fish.pos[dim] = new_pos[dim]
                    break

    def collective_volitive_movement(self):
        self.total_school_weight()
        barycenter = self.calculate_barycenter()

        max_i = max(barycenter)
        bin_baricenter = np.zeros((self.dim,), dtype=np.float)

        for dim in range(self.dim):
            if barycenter[dim] >= self.thres_v * max_i:
                bin_baricenter[dim] = 1

        for fish in self.school:
            if self.curr_weight_school > self.prev_weight_school:
                for dim in np.random.choice(range(self.dim), self.dim):
                    if fish.pos[dim] != bin_baricenter[dim]:
                        fish.pos[dim] = int(not (bin_baricenter[dim]))
                        break
            else:
                for dim in np.random.choice(range(self.dim), self.dim):
                    if fish.pos[dim] == bin_baricenter[dim]:
                        fish.pos[dim] = int(not (bin_baricenter[dim]))
                        break

            fish.cost, fish.test_acc, fish.train_acc, fish.features = self.objective_function.evaluate(fish.pos)
            self.eval_track_update()

    def optimize(self):
        self.__init_fss()
        self.__init_school()

        for i in range(self.n_iter):
            self.individual_movement()
            self.update_best_fish()
            self.feeding()
            self.collective_instinctive_movement()
            self.collective_volitive_movement()
            self.update_steps(i)
            self.update_best_fish()
            self.iter_track_update()
            # print('LOG: Iter: {} - Cost: {} - Train Acc: {} - Test Acc: {} - Feat: {}'.format(i, self.best_agent.cost,
            #                                                                                   self.best_agent.train_acc,
            #                                                                                   self.best_agent.test_acc,
            #                                                                                   self.best_agent.features))
