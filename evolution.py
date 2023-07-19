from deap import base
from deap import creator
from deap import tools
import random
import time
import pandas as pd
import numpy as np
from helper import configure_logger


class CPHMaxCindexGenetic:
    def __init__(self, exp, log_dir,  hall_of_fame_size=3, cross_over_p=1.0, ind_mutation_p=1.0, bit_mutation_p=0.01, tournsize=10, correct_labels_p=0.5, seed=20, find_in='train'):
        random.seed(seed)
        self.logger = configure_logger(logdir=log_dir, name='Genetic_Search')
        self.correct_labels_p = correct_labels_p
        self.exp = exp #self.load_exp(random_state_seed=seed)
        self.find_in = find_in
        self.log_dfs = []
        # self.best_test_hist = []
        # self.best_train_hist = []
        self.hall_of_fame_size = hall_of_fame_size
        # if find_in == 'train':
        #     self.ind_size = self.exp.e_train.sum()
        #     self.target_e = self.exp.new_unknown_true_e_train[self.exp.e_train == 1]
        # elif find_in== 'test':
        #     self.ind_size = self.exp.e_test.sum()
        #     self.target_e = self.exp.new_unknown_true_e_test[self.exp.e_test == 1]
        # else: # 'EM'
        #     if self.exp.latest_task == 'train_find_in_test':
        #         self.ind_size = self.exp.e_test.sum()
        #         self.target_e = self.exp.new_unknown_true_e_test[self.exp.e_test == 1]
        #     else:
        #         self.ind_size = self.exp.e_train.sum()
        #         self.target_e = self.exp.new_unknown_true_e_train[self.exp.e_train == 1]

        self._update_ind_target_and_size()




        self.bit_mutation_p = bit_mutation_p
        self.ind_mutation_p = ind_mutation_p
        self.cross_over_p = cross_over_p
        self.tournsize = tournsize
        self.max_possible_fitness = 1
        #self.model_class = model_class

        self._init_toolbox()

    def _init_toolbox(self):
        try:

            del creator.FitnessMax
            del creator.Individual
        except:
            pass
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        self.toolbox = base.Toolbox()
        self.toolbox.register("individual_creator", self.generate_indiv, creator.Individual, self.ind_size, self.correct_labels_p)
        self.toolbox.register("population_creator", tools.initRepeat, list, self.toolbox.individual_creator)

        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", tools.mutShuffleIndexes, indpb=self.bit_mutation_p)
        self.toolbox.register("select", tools.selTournament, tournsize=self.tournsize)
        self.toolbox.register("evaluate", self.evaluate)

        if self.hall_of_fame_size >0:
            self.hof = tools.HallOfFame(self.hall_of_fame_size)
        else:
            self.hof = None

        # prepare the statistics object:
        self.stats = tools.Statistics(lambda ind: ind.fitness.values)
        self.stats.register("max", np.max, axis=0)
        self.stats.register("avg", np.mean, axis=0)

    def _update_ind_target_and_size(self):
        if self.find_in == 'train':
            self.ind_size = self.exp.e_train.sum()
            self.target_e = self.exp.new_unknown_true_e_train[self.exp.e_train == 1]
        elif self.find_in== 'test':
            self.ind_size = self.exp.e_test.sum()
            self.target_e = self.exp.new_unknown_true_e_test[self.exp.e_test == 1]
        else: # 'EM'
            if self.exp.latest_task == 'train_find_in_test':
                self.ind_size = self.exp.e_test.sum()
                self.target_e = self.exp.new_unknown_true_e_test[self.exp.e_test == 1]
            else:
                self.ind_size = self.exp.e_train.sum()
                self.target_e = self.exp.new_unknown_true_e_train[self.exp.e_train == 1]

    @staticmethod
    def generate_indiv(ind_cls, size, p):
        # print(p)
        n_ones = int(p * size)
        n_zeros = int(size - n_ones)

        # print('n_zeros type:', type(n_zeros))
        # print('n_ones type:', type(n_ones))
        base_e = np.concatenate([np.ones(n_ones), np.zeros(n_zeros)]).astype(int)
        np.random.shuffle(base_e)
        ind = ind_cls(base_e)  # ind_cls(random.uniform()<p for _ in range(size))
        return ind

    def load_exp(self, random_state_seed):
        pass

    def evaluate(self, individual):
        ci = 0
        if self.find_in == 'train':
            ci = self.exp.train_score_some_e_events_part_only(individual)
        elif self.find_in == 'test':
            ci = self.exp.score_some_e_events_part_only_on_correct_model(individual)
        elif self.find_in == 'EM':
            ci = self.exp.score_some_e_events_part_only_on_latest_model(individual)
        else:
            print('Not specified task')
        return (ci,)

    @staticmethod
    def _get_ind_accuracy(individual, target_e):
        ind_arr = np.array(individual)
        acc = (ind_arr == target_e).mean()
        return acc

    @staticmethod
    def _get_ind_ones_accuracy(individual, target_e):
        ind_arr = np.array(individual)
        ones_acc = ((ind_arr == 1) & (target_e == 1)).sum() / target_e.sum()
        return ones_acc
    @staticmethod
    def _get_ind_zeros_accuracy(individual, target_e):
        ind_arr = np.array(individual)
        zeros_acc = ((ind_arr == 0) & (target_e == 0)).sum() / (len(target_e) - target_e.sum())
        return zeros_acc

    def _get_best_induvidual(self, population, best_fitness):
        # evaluated_individuals = [ind for ind in population if ind.fitness.valid]
        # best_ind = self.toolbox.clone(population[0])
        fitness_values = [ind.fitness.values[0] for ind in population]
        max_fitness = np.max(fitness_values)
        avg_fitness = np.mean(fitness_values)
        best_index = fitness_values.index(max(fitness_values))
        best_ind = self.toolbox.clone(population[best_index])

        best_ind_acc, best_ind_ones_acc, best_ind_zeros_acc = self._get_ind_accuracy(best_ind, self.target_e), self._get_ind_ones_accuracy(best_ind, self.target_e), self._get_ind_zeros_accuracy(best_ind, self.target_e)
        accs = [self._get_ind_accuracy(ind, self.target_e) for ind in population]
        ones_accs = [self._get_ind_ones_accuracy(ind, self.target_e) for ind in population]
        zeros_accs = [self._get_ind_zeros_accuracy(ind, self.target_e) for ind in population]
        avg_acc = np.mean(accs)
        avg_ones_acc = np.mean(ones_accs)
        avg_zeros_acc = np.mean(zeros_accs)
        if max_fitness > best_fitness:
            updated = True
        else:
            updated = False
        return best_ind, max_fitness, avg_fitness, best_ind_acc, avg_acc, best_ind_ones_acc, avg_ones_acc, best_ind_zeros_acc, avg_zeros_acc, updated

    def _get_offsprings(self, population):
        offspring = self.toolbox.select(population, len(population))
        offspring = [self.toolbox.clone(ind) for ind in offspring]

        for i, (child1, child2) in enumerate(zip(offspring[::2], offspring[1::2])):
            if random.random() <= self.cross_over_p:
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < self.ind_mutation_p:
                self.toolbox.mutate(mutant)
                del mutant.fitness.values

        return offspring

    def _evalulate_population(self, population):
        fresh_individuals = [ind for ind in population if not ind.fitness.valid]
        fresh_fitness_values = list(map(self.toolbox.evaluate, fresh_individuals))
        for individual, fitness_value in zip(fresh_individuals, fresh_fitness_values):
            individual.fitness.values = fitness_value

    def search(self, population_size, max_generations=10, max_nochange=1):
        best_fitness = -1 * np.inf

        no_change_counter = 0
        generation_counter = 0
        log_df = pd.DataFrame(columns=('generation', 'time', 'best_fitness', 'average_fitness', 'acc', 'avg_acc', 'random_acc', 'ones_acc', 'avg_ones_acc', 'random_ones_acc', 'zeros_acc', 'avg_zeros_acc', 'random_zeros_acc', 'best_solution'))

        population = self.toolbox.population_creator(n=population_size)

        start = time.time()
        self._evalulate_population(population)

        # add the best to the hof
        if self.hof is not None:
            self.hof.update(population)

        best_ind, max_fitness, avg_fitness, best_ind_acc, avg_acc, best_ind_ones_acc, avg_ones_acc, best_ind_zeros_acc, avg_zeros_acc, updated = self._get_best_induvidual(population, best_fitness)
        #best_ind_arr = np.array(best_ind)
        #acc = (best_ind_arr == self.exp.new_unknown_true_e_train[self.exp.e_train == 1]).mean()
        #ones_acc = ((best_ind_arr == 1) & (self.exp.new_unknown_true_e_train[self.exp.e_train == 1] == 1)).sum()/self.exp.new_unknown_true_e_train.sum()
        random_ones_acc = self.correct_labels_p
        random_zeros_acc = 1 - self.correct_labels_p
        random_acc = self.correct_labels_p**2 + (1-self.correct_labels_p)**2

        end = time.time()
        dt = (end - start)
        self.logger.info('Gen,\tMxCI,\tAvCI,\tAc,\tAvAc,\tRAc,\t1Ac,\tAv1Ac,\tR1Ac,\t0Ac,\tAv0Ac,\tR0Ac,\tETime m')

        if updated:
            no_change_counter = 0
            best_fitness = max_fitness
            row = [generation_counter, dt, np.round(100 * best_fitness, 2), np.round(100 * avg_fitness, 2),
                   best_ind_acc, avg_acc, random_acc, best_ind_ones_acc, avg_ones_acc, random_ones_acc, best_ind_zeros_acc, avg_zeros_acc, random_zeros_acc, best_ind]
            self.logger.info(
                '{},\t{:.2f},\t{:.2f},\t{:.2f},\t{:.2f},\t{:.2f},\t{:.2f},\t{:.2f},\t{:.2f},\t{:.2f},\t{:.2f},\t{:.2f},\t{:.2f}'.format(
                    generation_counter,
                    (100 * best_fitness),
                    (100 * avg_fitness),
                    best_ind_acc,
                    avg_acc,
                    random_acc,
                    best_ind_ones_acc,
                    avg_ones_acc,
                    random_ones_acc,
                    best_ind_zeros_acc,
                    avg_zeros_acc,
                    random_zeros_acc,
                    (dt / 60)))
            log_df.loc[len(log_df)] = row

        while (generation_counter < max_generations) and (no_change_counter < max_nochange) and (
                best_fitness < self.max_possible_fitness):
            generation_counter += 1
            offspring = self._get_offsprings(population)
            self._evalulate_population(offspring)

            if self.hof is not None:
                offspring.extend(self.hof.items)

            population[:] = tools.selBest(population + offspring, population_size)

            if self.hof is not None:
                self.hof.update(population)

            best_ind, max_fitness, avg_fitness, best_ind_acc, avg_acc, best_ind_ones_acc, avg_ones_acc, best_ind_zeros_acc, avg_zeros_acc, updated = self._get_best_induvidual(population, best_fitness)
            #best_ind_arr = np.array(best_ind)
            #acc = (best_ind_arr == self.exp.new_unknown_true_e_train[self.exp.e_train == 1]).mean()
            #ones_acc = ((best_ind_arr == 1) & (self.exp.new_unknown_true_e_train[self.exp.e_train == 1] == 1)).sum() / self.exp.new_unknown_true_e_train.sum()
            #print(max_fitness, avg_fitness, updated)
            if updated:
                no_change_counter = 0
                best_fitness = max_fitness
                end = time.time()
                dt = (end - start)
                row = [generation_counter, dt, np.round(100 * best_fitness, 2), np.round(100 * avg_fitness, 2),
                       best_ind_acc, avg_acc, random_acc, best_ind_ones_acc, avg_ones_acc, random_ones_acc, best_ind_zeros_acc, avg_zeros_acc, random_zeros_acc, best_ind]
                log_df.loc[len(log_df)] = row
                self.logger.info(
                    '{},\t{:.2f},\t{:.2f},\t{:.2f},\t{:.2f},\t{:.2f},\t{:.2f},\t{:.2f},\t{:.2f},\t{:.2f},\t{:.2f},\t{:.2f},\t{:.2f}'.format(
                        generation_counter,
                        (100 * best_fitness),
                        (100 * avg_fitness),
                        best_ind_acc,
                        avg_acc,
                        random_acc,
                        best_ind_ones_acc,
                        avg_ones_acc,
                        random_ones_acc,
                        best_ind_zeros_acc,
                        avg_zeros_acc,
                        random_zeros_acc,
                        (dt / 60)))
            else:
                #print('No Change')
                no_change_counter += 1
        end = time.time()
        dt = (end - start) / 60
        self.logger.info('Time Elapsed: {:.2f} m'.format(dt))

        self.population = population
        self.log_df = log_df
        return self.population[0]

    def search_EM(self, population_size, max_generations=10, max_nochange=1, max_iters=10):
        self.accs_train, self.accs_test, self.ones_accs_train, self.ones_accs_test, self.zeros_accs_train, self.zeros_accs_test, self.ones_p_train, self.ones_p_test = [], [], [], [], [], [], [], []
        for i in range(max_iters):
            self.logger.info(f'Iter {i} --------------------------------------------------------')
            self._init_toolbox()
            self.exp.update_latest_model()
            # print('test', self.exp.latest_e_test)
            # print('train', self.exp.latest_e_train)
            if self.exp.latest_task=='train_find_in_test':
                self.logger.info('Model on Train - Search on Test')
            else:
                self.logger.info('Model on Test - Search on Train')
            self._update_ind_target_and_size()
            best_e = self.search(population_size, max_generations, max_nochange)
            if self.exp.latest_task=='train_find_in_test':
                #print('Model on Train - Search on Test')
                self.exp.best_test_hist.append(best_e)
                # prev_e_test = self.exp.latest_e_test[self.exp.e_test==1]
                # accum_e_test = (best_e and prev_e_test)
                accum_e_test_ps = np.array(self.exp.best_test_hist).mean(axis=0)
                cut_val = np.percentile(accum_e_test_ps, 100 - self.correct_labels_p*100) #self._get_cut_val(accum_e_test_ps, self.correct_labels_p) # 1 - np.percentile(accum_e_test_ps, self.correct_labels_p*100)
                accum_e_test = list((accum_e_test_ps >= cut_val).astype(int))
                acc = self._get_ind_accuracy(accum_e_test, self.target_e)
                ones_acc = self._get_ind_ones_accuracy(accum_e_test, self.target_e)
                zeros_acc = self._get_ind_zeros_accuracy(accum_e_test, self.target_e)
                ones_p = np.array(accum_e_test).mean()
                ci = self.exp.score_some_e_events_part_only_on_latest_model_ci(accum_e_test)
                self.logger.info(f'CI {ci}, Test Accum Acc: {acc}, Ones_Acc: {ones_acc}, Zeros_Acc: {zeros_acc}, Ones_p {ones_p}')
                self.accs_test.append(acc), self.ones_accs_test.append(ones_acc), self.zeros_accs_test.append(zeros_acc), self.ones_p_test.append(ones_p)
                self.exp.update_e_and_flip_task(accum_e_test)
            else:
                #print('Model on Test - Search on Train')
                self.exp.best_train_hist.append(best_e)
                # prev_e_train = self.exp.latest_e_train[self.exp.e_train == 1]
                # accum_e_train = (best_e and prev_e_train)
                accum_e_train_ps = np.array(self.exp.best_train_hist).mean(axis=0)
                cut_val = np.percentile(accum_e_train_ps, 100 - self.correct_labels_p*100) #self._get_cut_val(accum_e_train_ps, self.correct_labels_p) #1 - np.percentile(accum_e_train_ps, self.correct_labels_p*100)
                accum_e_train = list((accum_e_train_ps >= cut_val).astype(int))
                acc = self._get_ind_accuracy(accum_e_train, self.target_e)
                ones_acc = self._get_ind_ones_accuracy(accum_e_train, self.target_e)
                zeros_acc = self._get_ind_zeros_accuracy(accum_e_train, self.target_e)
                ones_p = np.array(accum_e_train).mean()
                ci = self.exp.score_some_e_events_part_only_on_latest_model_ci(accum_e_train)
                self.logger.info(f'CI {ci}, Train Accum Acc: {acc}, Ones_Acc: {ones_acc}, Zeros_Acc: {zeros_acc}, Ones_p {ones_p}')
                self.accs_train.append(acc), self.ones_accs_train.append(ones_acc), self.zeros_accs_train.append(zeros_acc), self.ones_p_train.append(ones_p)
                self.exp.update_e_and_flip_task(accum_e_train)



            self.log_dfs.append(self.log_df.copy())

            #print(best_e)

            self.logger.info('------------------------------------------------------------------')


